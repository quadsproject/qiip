"""FastAPI application factory and entry point.

Usage::

    # Development server
    uv run uvicorn inference_proxy.main:create_app --factory --host 0.0.0.0 --port 8080

    # Programmatic access (tests)
    from inference_proxy.main import create_app
    app = create_app()
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

# huggingface_hub snapshots this setting during import. It must be set before
# any application module imports the catalog or downloader packages.
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
from inference_proxy.discovery.node_leases import NodeLeaseManager
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.relaunch_recovery import reconcile_interrupted_relaunch
from inference_proxy.discovery.serializer import node_from_etcd
from inference_proxy.discovery.watcher import EtcdWatcher
from inference_proxy.huggingface.artifacts import GGUFArtifactIndex
from inference_proxy.huggingface.catalog import ModelCatalogService
from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.models.endpoint import EndpointPolicy
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
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics

logger = structlog.get_logger()


async def _safe_async_cleanup(
    resource: str, cleanup: Callable[[], Awaitable[None]]
) -> None:
    """Run one async cleanup without masking the initiating failure."""
    try:
        await cleanup()
    except Exception:
        logger.warning(
            "application_resource_cleanup_failed",
            resource=resource,
            exc_info=True,
        )


def _safe_sync_cleanup(resource: str, cleanup: Callable[[], None]) -> None:
    """Run one synchronous cleanup without masking the initiating failure."""
    try:
        cleanup()
    except Exception:
        logger.warning(
            "application_resource_cleanup_failed",
            resource=resource,
            exc_info=True,
        )


async def _stop_discovery_workers(
    stop_event: threading.Event,
    watcher: EtcdWatcher,
    threads: list[threading.Thread],
) -> None:
    """Stop discovery workers before closing the etcd client they share."""
    stop_event.set()
    _safe_sync_cleanup("etcd watcher", watcher.stop)
    for thread in reversed(threads):
        try:
            # Lifespan shutdown runs only after Uvicorn has drained requests.
            # A direct bounded join avoids creating another executor worker
            # while the event loop itself is shutting down.
            thread.join(timeout=10)
        except Exception:
            logger.warning(
                "application_resource_cleanup_failed",
                resource=f"thread:{thread.name}",
                exc_info=True,
            )
            continue
        if thread.is_alive():
            logger.warning("application_thread_did_not_stop", thread=thread.name)


def _initial_load(
    etcd_client: EtcdClient,
    registry: NodeRegistry,
    endpoint_policy: EndpointPolicy,
) -> bool:
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
        snapshot = etcd_client.get_snapshot()
        count = 0
        for record in snapshot.records:
            reconciled = reconcile_interrupted_relaunch(
                etcd_client,
                record,
                endpoint_policy,
            )
            if reconciled is None:
                continue
            node = node_from_etcd(
                reconciled.key,
                reconciled.value,
                etcd_client.prefix,
                endpoint_policy=endpoint_policy,
            )
            if node is not None:
                registry.add(node)
                count += 1
        logger.info("initial node load complete", node_count=count)
        return True
    except Exception:
        logger.warning(
            "etcd unavailable at startup, starting with empty registry",
            exc_info=True,
        )
        return False


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
        """Own startup rollback and reverse-order application cleanup."""
        if TYPE_CHECKING:
            from huggingface_hub.utils.tqdm import disable_progress_bars
        else:
            from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        configure_logging(
            json_output=resolved_settings.logging.json_output,
            log_level=getattr(logging, resolved_settings.logging.level),
        )

        async with AsyncExitStack() as resources:
            etcd_client = EtcdClient(resolved_settings.etcd)
            resources.callback(_safe_sync_cleanup, "etcd client", etcd_client.close)
            registry = NodeRegistry()
            lease_manager = NodeLeaseManager(etcd_client)
            endpoint_policy = resolved_settings.routing.endpoint_policy()

            allowlist_fields = {
                "allowed_endpoint_hosts",
                "allowed_endpoint_networks",
                "allowed_endpoint_ports",
            }
            if not resolved_settings.routing.model_fields_set & allowlist_fields:
                logger.warning(
                    "backend endpoint allowlist is unset; "
                    "loopback-only secure defaults are active",
                    allowed_hosts=resolved_settings.routing.allowed_endpoint_hosts,
                    allowed_networks=(
                        resolved_settings.routing.allowed_endpoint_networks
                    ),
                    allowed_ports=resolved_settings.routing.allowed_endpoint_ports,
                )

            initial_load_complete = _initial_load(
                etcd_client,
                registry,
                endpoint_policy,
            )

            stop_event = threading.Event()
            etcd_watcher = EtcdWatcher(
                etcd_client,
                registry,
                stop_event,
                endpoint_policy,
                lease_manager,
                reconcile_startup_relaunches=not initial_load_complete,
            )
            worker_threads: list[threading.Thread] = []
            resources.push_async_callback(
                _stop_discovery_workers,
                stop_event,
                etcd_watcher,
                worker_threads,
            )
            watch_thread = threading.Thread(
                target=etcd_watcher.run,
                name="etcd-watcher",
                daemon=True,
            )
            watch_thread.start()
            worker_threads.append(watch_thread)

            app.state.registry = registry

            circuit_breaker_registry = CircuitBreakerRegistry(
                threshold=resolved_settings.resilience.circuit_breaker_threshold,
            )
            registry.register_remove_listener(circuit_breaker_registry.remove)
            app.state.circuit_breaker_registry = circuit_breaker_registry

            connection_tracker = ConnectionTracker()
            node_selector = NodeSelector(registry, connection_tracker)
            app.state.node_selector = node_selector

            health_thread = threading.Thread(
                target=run_health_checker,
                name="health-checker",
                args=(
                    registry,
                    circuit_breaker_registry,
                    stop_event,
                    resolved_settings.resilience.health_check_interval,
                    resolved_settings.resilience.health_check_failure_threshold,
                ),
                kwargs={
                    "connection_tracker": connection_tracker,
                    "lease_manager": lease_manager,
                },
                daemon=True,
            )
            health_thread.start()
            worker_threads.append(health_thread)

            request_metrics = RequestMetrics()
            app.state.request_metrics = request_metrics

            ssh_client = SSHClient(resolved_settings.ssh)

            llmfit_runner = LLMFitRunner(
                ssh_client=ssh_client, settings=resolved_settings.llmfit
            )
            app.state.llmfit_runner = llmfit_runner

            cache_path = Path(resolved_settings.huggingface.cache_dir)
            if not cache_path.is_dir():
                raise RuntimeError(
                    f"HuggingFace cache directory does not exist: {cache_path}"
                )
            artifact_index = GGUFArtifactIndex(
                cache_path,
                shared_root=resolved_settings.huggingface.shared_root,
            )
            app.state.artifact_index = artifact_index
            catalog_service = ModelCatalogService(
                cache_dir=resolved_settings.huggingface.cache_dir,
                artifact_index=artifact_index,
            )
            app.state.catalog_service = catalog_service

            token = (
                resolved_settings.huggingface.api_token.get_secret_value()
                if resolved_settings.huggingface.api_token
                else None
            )
            download_service = DownloadService(
                cache_dir=resolved_settings.huggingface.cache_dir,
                token=token,
                artifact_index=artifact_index,
            )
            app.state.download_service = download_service
            resources.push_async_callback(
                _safe_async_cleanup,
                "download service",
                download_service.shutdown,
            )

            redfish_username = resolved_settings.redfish.bmc_username
            redfish_password = resolved_settings.redfish.bmc_password
            if redfish_username is not None and redfish_password is not None:
                redfish_http = httpx.AsyncClient(
                    verify=resolved_settings.redfish.verify_ssl,
                    timeout=httpx.Timeout(
                        connect=resolved_settings.redfish.connect_timeout,
                        read=resolved_settings.redfish.read_timeout,
                        write=10.0,
                        pool=10.0,
                    ),
                )
                resources.push_async_callback(
                    _safe_async_cleanup,
                    "redfish HTTP client",
                    redfish_http.aclose,
                )
                redfish_client = RedfishClient(
                    redfish_http,
                    bmc_host_template=resolved_settings.redfish.bmc_host_template,
                    system_id=resolved_settings.redfish.system_id,
                    hostname_policy=endpoint_policy,
                    auth=httpx.BasicAuth(
                        username=redfish_username,
                        password=redfish_password.get_secret_value(),
                    ),
                    poll_timeout=resolved_settings.redfish.power_poll_timeout,
                    poll_interval=resolved_settings.redfish.power_poll_interval,
                )
                app.state.redfish_client = redfish_client
                logger.info("redfish client initialized")
            else:
                app.state.redfish_client = None
                logger.info("redfish disabled (BMC credentials not configured)")

            log_buffer = ProvisioningLogBuffer(
                max_entries_per_host=(
                    resolved_settings.provisioning.log_max_entries_per_host
                ),
                max_bytes_per_host=(
                    resolved_settings.provisioning.log_max_bytes_per_host
                ),
                max_entry_bytes=resolved_settings.provisioning.log_max_entry_bytes,
                max_completed_hosts=(
                    resolved_settings.provisioning.log_max_completed_hosts
                ),
            )

            hf_token = resolved_settings.huggingface.api_token
            provisioner = NodeProvisioner(
                ssh_client=ssh_client,
                etcd_client=etcd_client,
                settings=resolved_settings.provisioning,
                llmfit_settings=resolved_settings.llmfit,
                endpoint_policy=endpoint_policy,
                registry=registry,
                connection_tracker=connection_tracker,
                circuit_breaker_registry=circuit_breaker_registry,
                redfish_client=app.state.redfish_client,
                log_buffer=log_buffer,
                hf_token=hf_token.get_secret_value() if hf_token else None,
                nfs_export=resolved_settings.huggingface.nfs_export,
                artifact_index=artifact_index,
            )
            app.state.provisioner = provisioner
            resources.push_async_callback(
                _safe_async_cleanup,
                "provisioner",
                provisioner.shutdown,
            )

            if resolved_settings.quads.base_url is not None:
                quads_server_timezone = resolved_settings.quads.server_timezone
                if quads_server_timezone is None:
                    raise RuntimeError(
                        "QUADS server timezone missing after settings validation"
                    )
                quads_http = httpx.AsyncClient(
                    timeout=httpx.Timeout(resolved_settings.quads.timeout),
                    verify=resolved_settings.quads.verify_ssl,
                )
                resources.push_async_callback(
                    _safe_async_cleanup,
                    "QUADS HTTP client",
                    quads_http.aclose,
                )
                quads_client = QUADSClient(
                    quads_http,
                    resolved_settings.quads.base_url,
                    server_timezone=quads_server_timezone,
                )
                app.state.quads_client = quads_client
                quads_poller = QUADSPoller(
                    quads_client,
                    resolved_settings.quads.poll_interval,
                )
                quads_poller.start()
                app.state.quads_poller = quads_poller
                resources.push_async_callback(
                    _safe_async_cleanup,
                    "QUADS poller",
                    quads_poller.stop,
                )

                schedule_enforcer = ScheduleEnforcer(
                    client=quads_client,
                    registry=registry,
                    provisioner=provisioner,
                    lookahead_hours=(resolved_settings.quads.schedule_lookahead_hours),
                    check_interval=(resolved_settings.quads.schedule_check_interval),
                )
                schedule_enforcer.start()
                app.state.schedule_enforcer = schedule_enforcer
                resources.push_async_callback(
                    _safe_async_cleanup,
                    "schedule enforcer",
                    schedule_enforcer.stop,
                )
            else:
                app.state.quads_client = None
                app.state.quads_poller = None
                app.state.schedule_enforcer = None

            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=resolved_settings.proxy.connect_timeout,
                    read=resolved_settings.proxy.read_timeout,
                    write=resolved_settings.proxy.write_timeout,
                    pool=resolved_settings.proxy.pool_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=resolved_settings.proxy.max_connections,
                    max_keepalive_connections=(
                        resolved_settings.proxy.max_keepalive_connections
                    ),
                    keepalive_expiry=resolved_settings.proxy.keepalive_expiry,
                ),
            )
            resources.push_async_callback(
                _safe_async_cleanup,
                "proxy HTTP client",
                http_client.aclose,
            )
            proxy_client = ProxyClient(http_client)
            app.state.proxy_client = proxy_client

            try:
                yield
            finally:
                logger.info("application shutdown initiated")

    application = FastAPI(
        title="QUADS LLM Inference Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.dependency_overrides[get_settings] = lambda: resolved_settings

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
