"""FastAPI application factory and entry point.

Usage::

    # Development server
    uv run uvicorn inference_proxy.main:app --host 0.0.0.0 --port 8000

    # Programmatic access (tests)
    from inference_proxy.main import create_app
    app = create_app()
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from inference_proxy.api.admin import admin_router
from inference_proxy.api.chat import chat_router
from inference_proxy.api.dashboard import dashboard_router
from inference_proxy.api.middleware import RequestLoggingMiddleware
from inference_proxy.api.routes import router
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.logging import configure_logging
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_from_etcd
from inference_proxy.discovery.watcher import run_watcher
from inference_proxy.huggingface.catalog import ModelCatalogService
from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.provisioning.ssh_client import SSHClient
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.quads.client import QUADSClient
from inference_proxy.quads.poller import QUADSPoller
from inference_proxy.quads.schedule_enforcer import ScheduleEnforcer
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.resilience.health_checker import run_health_checker
from inference_proxy.resilience.shutdown import ShutdownMiddleware
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics

logger = structlog.get_logger()


def _initial_load(etcd_client: EtcdClient, registry: NodeRegistry) -> None:
    """Fetch all nodes from etcd and populate the registry.

    Per D-05: synchronous initial fetch is acceptable during startup.
    Per D-09: if etcd is unavailable, start with an empty registry and
    log a warning -- the gateway remains responsive but routing will
    fail until nodes appear via the watch thread.

    Args:
        etcd_client: The etcd client wrapper.
        registry: The node registry to populate.
    """
    try:
        results = etcd_client.get_prefix()
        count = 0
        for value_bytes, metadata in results:
            raw_key: bytes | str = metadata["key"]
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            node = node_from_etcd(key, value_bytes, etcd_client.prefix)
            if node is not None:
                registry.add(node)
                count += 1
        logger.info("initial node load complete", node_count=count)
    except Exception:
        logger.warning(
            "etcd unavailable at startup, starting with empty registry",
            exc_info=True,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional settings instance. When ``None`` (the default),
            settings are loaded from the environment via ``get_settings()``.
            Pass an explicit instance in tests to avoid hitting real
            etcd during lifespan startup.

    Returns:
        A fully configured FastAPI application with registered routes.
    """
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan: logging, service discovery, and shutdown.

        Startup:
            1. Configure structured logging
            2. Create etcd client and node registry
            3. Fetch initial node list from etcd (per D-05)
            4. Start watch thread for real-time updates (per D-03)
            5. Store registry in ``app.state`` for dependency injection (per D-07)

        Shutdown:
            1. Signal the watch thread to stop via ``threading.Event`` (per D-10)
            2. Join the watch thread with timeout
        """
        # D-08/D-10: HuggingFace startup guards — before any HF usage
        import os as _os

        _os.environ["HF_HUB_DISABLE_XET"] = "1"
        if TYPE_CHECKING:
            from huggingface_hub.utils.tqdm import disable_progress_bars
        else:
            from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()

        configure_logging(
            json_output=resolved_settings.logging.json_output,
            log_level=getattr(
                logging, resolved_settings.logging.level.upper(), logging.INFO
            ),
        )
        etcd_client = EtcdClient(resolved_settings.etcd)
        registry = NodeRegistry()

        _initial_load(etcd_client, registry)

        stop_event = threading.Event()
        watch_thread = threading.Thread(
            target=run_watcher,
            args=(etcd_client, registry, stop_event),
            daemon=True,
        )
        watch_thread.start()

        app.state.registry = registry

        circuit_breaker_registry = CircuitBreakerRegistry(
            threshold=resolved_settings.resilience.circuit_breaker_threshold,
        )
        app.state.circuit_breaker_registry = circuit_breaker_registry

        health_thread = threading.Thread(
            target=run_health_checker,
            args=(
                registry,
                circuit_breaker_registry,
                stop_event,
                resolved_settings.resilience.health_check_interval,
                resolved_settings.resilience.health_check_failure_threshold,
            ),
            daemon=True,
        )
        health_thread.start()

        connection_tracker = ConnectionTracker()
        node_selector = NodeSelector(registry, connection_tracker)
        app.state.node_selector = node_selector

        request_metrics = RequestMetrics()
        app.state.request_metrics = request_metrics

        ssh_client = SSHClient(resolved_settings.ssh)

        llmfit_runner = LLMFitRunner(
            ssh_client=ssh_client, settings=resolved_settings.llmfit
        )
        app.state.llmfit_runner = llmfit_runner

        # Model catalog from NFS-mounted HuggingFace cache
        cache_path = Path(resolved_settings.huggingface.cache_dir)
        if not cache_path.is_dir():
            raise RuntimeError(
                f"HuggingFace cache directory does not exist: {cache_path}"
            )
        catalog_service = ModelCatalogService(
            cache_dir=resolved_settings.huggingface.cache_dir
        )
        app.state.catalog_service = catalog_service

        # Download service — token extracted via get_secret_value() per D-04
        token = (
            resolved_settings.huggingface.api_token.get_secret_value()
            if resolved_settings.huggingface.api_token
            else None
        )
        download_service = DownloadService(
            cache_dir=resolved_settings.huggingface.cache_dir, token=token
        )
        app.state.download_service = download_service

        if resolved_settings.redfish.bmc_username is not None:
            redfish_http = httpx.AsyncClient(
                auth=httpx.BasicAuth(
                    username=resolved_settings.redfish.bmc_username,
                    password=resolved_settings.redfish.bmc_password.get_secret_value(),  # type: ignore[union-attr]
                ),
                verify=resolved_settings.redfish.verify_ssl,
                timeout=httpx.Timeout(
                    connect=resolved_settings.redfish.connect_timeout,
                    read=resolved_settings.redfish.read_timeout,
                    write=10.0,
                    pool=10.0,
                ),
            )
            redfish_client = RedfishClient(
                redfish_http,
                bmc_host_template=resolved_settings.redfish.bmc_host_template,
                system_id=resolved_settings.redfish.system_id,
                poll_timeout=resolved_settings.redfish.power_poll_timeout,
                poll_interval=resolved_settings.redfish.power_poll_interval,
            )
            app.state.redfish_client = redfish_client
            logger.info("redfish client initialized")
        else:
            app.state.redfish_client = None
            redfish_http = None
            logger.info("redfish disabled (no bmc_username configured)")

        log_buffer = ProvisioningLogBuffer()

        provisioner = NodeProvisioner(
            ssh_client=ssh_client,
            etcd_client=etcd_client,
            settings=resolved_settings.provisioning,
            registry=registry,
            connection_tracker=connection_tracker,
            redfish_client=app.state.redfish_client,
            log_buffer=log_buffer,
        )
        app.state.provisioner = provisioner

        if resolved_settings.quads.base_url is not None:
            quads_http = httpx.AsyncClient(
                timeout=httpx.Timeout(resolved_settings.quads.timeout),
                verify=resolved_settings.quads.verify_ssl,
            )
            quads_client = QUADSClient(quads_http, resolved_settings.quads.base_url)
            app.state.quads_client = quads_client
            quads_poller = QUADSPoller(
                quads_client, resolved_settings.quads.poll_interval
            )
            quads_poller.start()
            app.state.quads_poller = quads_poller

            schedule_enforcer = ScheduleEnforcer(
                client=quads_client,
                registry=registry,
                provisioner=provisioner,
                lookahead_hours=resolved_settings.quads.schedule_lookahead_hours,
                check_interval=resolved_settings.quads.schedule_check_interval,
            )
            schedule_enforcer.start()
            app.state.schedule_enforcer = schedule_enforcer
        else:
            app.state.quads_client = None
            app.state.quads_poller = None
            app.state.schedule_enforcer = None
            quads_http = None

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=resolved_settings.proxy.connect_timeout,
                read=resolved_settings.proxy.read_timeout,
                write=resolved_settings.proxy.write_timeout,
                pool=resolved_settings.proxy.pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=resolved_settings.proxy.max_connections,
                max_keepalive_connections=resolved_settings.proxy.max_keepalive_connections,
                keepalive_expiry=resolved_settings.proxy.keepalive_expiry,
            ),
        )
        proxy_client = ProxyClient(http_client)
        app.state.proxy_client = proxy_client

        app.state.shutting_down = False

        yield

        app.state.shutting_down = True
        logger.info(
            "graceful shutdown initiated",
            timeout=resolved_settings.gateway.graceful_shutdown_timeout,
        )
        await asyncio.sleep(resolved_settings.gateway.graceful_shutdown_timeout)

        await http_client.aclose()
        if app.state.schedule_enforcer is not None:
            await app.state.schedule_enforcer.stop()
        if app.state.quads_poller is not None:
            await app.state.quads_poller.stop()
        if quads_http is not None:
            await quads_http.aclose()
        if redfish_http is not None:
            await redfish_http.aclose()
        stop_event.set()
        watch_thread.join(timeout=10)
        health_thread.join(timeout=10)

    application = FastAPI(
        title="QUADS LLM Inference Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(ShutdownMiddleware)
    application.add_middleware(RequestLoggingMiddleware)

    @application.get("/health")
    async def health() -> JSONResponse:
        """Return gateway health status with registered node count."""
        registry: NodeRegistry = application.state.registry
        return JSONResponse(
            content={
                "status": "ok",
                "nodes_registered": len(registry.get_all()),
            }
        )

    application.include_router(router)
    application.include_router(admin_router)
    application.include_router(dashboard_router)
    application.include_router(chat_router)

    static_dir = Path(__file__).resolve().parent / "static"
    application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return application


app = create_app()
