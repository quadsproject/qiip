"""Application lifecycle regression tests.

These tests drive the real FastAPI lifespan directly, and one test runs it
under a real Uvicorn server so shutdown ordering is verified at the server
boundary rather than by setting application state manually.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import uvicorn

from inference_proxy.config.settings import QUADSSettings, Settings
from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.main import create_app


def _etcd_client(*, order: list[str] | None = None) -> MagicMock:
    client = MagicMock()
    client.get_prefix.return_value = []
    client.prefix = "/nodes/"
    if order is not None:
        client.put.side_effect = lambda *_args: order.append("state-write")
        client.close.side_effect = lambda: order.append("etcd-close")
    return client


def _bounded_health_worker(*args: object, **_kwargs: object) -> None:
    """Stand in for the health loop while still requiring its stop signal."""
    stop_event = args[2]
    if not isinstance(stop_event, threading.Event):
        raise TypeError("health worker did not receive a threading.Event")
    stop_event.wait(timeout=1)


def _without_legacy_shutdown_delay(settings: Settings) -> Settings:
    """Keep old-main comparisons behavioral instead of waiting 30 seconds."""
    if "graceful_shutdown_timeout" not in type(settings.gateway).model_fields:
        return settings
    gateway = settings.gateway.model_copy(update={"graceful_shutdown_timeout": 0})
    return settings.model_copy(update={"gateway": gateway})


def test_lifespan_cancels_provision_before_etcd_close(
    tmp_path: Path,
) -> None:
    """C2: cancellation records FAILED while etcd is still available."""
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_provision_shutdown_probe.py")),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_lifespan_stops_enforcer_before_provisioner_and_etcd(
    test_settings: Settings,
) -> None:
    """C2: task producers stop before task cancellation and state storage."""
    order: list[str] = []
    etcd = _etcd_client(order=order)
    provisioner = MagicMock()
    provisioner.shutdown = AsyncMock(side_effect=lambda: order.append("provisioner"))
    poller = MagicMock()
    poller.stop = AsyncMock(side_effect=lambda: order.append("poller"))
    enforcer = MagicMock()
    enforcer.stop = AsyncMock(side_effect=lambda: order.append("enforcer"))
    quads_http = MagicMock()
    quads_http.aclose = AsyncMock(side_effect=lambda: order.append("quads-http"))
    proxy_http = MagicMock()
    proxy_http.aclose = AsyncMock(side_effect=lambda: order.append("proxy-http"))
    quads = QUADSSettings(
        base_url="http://quads.example.com",
        server_timezone="America/New_York",
    )
    settings = _without_legacy_shutdown_delay(
        test_settings.model_copy(update={"quads": quads})
    )

    with (
        patch("inference_proxy.main.EtcdClient", return_value=etcd),
        patch("inference_proxy.main.EtcdWatcher"),
        patch(
            "inference_proxy.main.run_health_checker",
            new=_bounded_health_worker,
        ),
        patch("inference_proxy.main.NodeProvisioner", return_value=provisioner),
        patch("inference_proxy.main.QUADSPoller", return_value=poller),
        patch("inference_proxy.main.ScheduleEnforcer", return_value=enforcer),
        patch(
            "inference_proxy.main.httpx.AsyncClient",
            side_effect=[quads_http, proxy_http],
        ),
    ):
        app = create_app(settings=settings)
        async with app.router.lifespan_context(app):
            pass

    assert order.index("enforcer") < order.index("provisioner")
    assert order.index("provisioner") < order.index("etcd-close")


@pytest.mark.asyncio
async def test_gateway_shutdown_does_not_revoke_node_leases(
    test_settings: Settings,
) -> None:
    """A replacement gateway retains the full TTL window to adopt leases."""
    etcd = _etcd_client()

    with (
        patch("inference_proxy.main.EtcdClient", return_value=etcd),
        patch("inference_proxy.main.EtcdWatcher"),
        patch(
            "inference_proxy.main.run_health_checker",
            new=_bounded_health_worker,
        ),
    ):
        app = create_app(settings=_without_legacy_shutdown_delay(test_settings))
        async with app.router.lifespan_context(app):
            pass

    etcd.revoke_lease.assert_not_called()
    etcd.close.assert_called_once()


@pytest.mark.asyncio
async def test_early_startup_failure_rolls_back_without_masking_error(
    test_settings: Settings,
) -> None:
    """A3: discovery resources close and cleanup failures do not mask startup."""
    etcd = _etcd_client()
    etcd.close.side_effect = RuntimeError("cleanup also failed")
    watcher = MagicMock()

    with (
        patch("inference_proxy.main.EtcdClient", return_value=etcd),
        patch("inference_proxy.main.EtcdWatcher", return_value=watcher),
        patch(
            "inference_proxy.main.run_health_checker",
            new=_bounded_health_worker,
        ),
        patch(
            "inference_proxy.main.ModelCatalogService",
            side_effect=RuntimeError("catalog startup failed"),
        ),
    ):
        app = create_app(settings=_without_legacy_shutdown_delay(test_settings))
        with pytest.raises(RuntimeError, match="catalog startup failed"):
            async with app.router.lifespan_context(app):
                pytest.fail("lifespan unexpectedly started")

    watcher.stop.assert_called_once()
    etcd.close.assert_called_once()


@pytest.mark.asyncio
async def test_late_startup_failure_closes_constructed_resources(
    test_settings: Settings,
) -> None:
    """A3/L3: later startup failure unwinds every already-owned resource."""
    etcd = _etcd_client()
    download_service = MagicMock()
    download_service.shutdown = AsyncMock()
    provisioner = MagicMock()
    provisioner.shutdown = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    proxy_http = MagicMock()
    proxy_http.aclose = AsyncMock()

    with (
        patch("inference_proxy.main.EtcdClient", return_value=etcd),
        patch("inference_proxy.main.EtcdWatcher"),
        patch(
            "inference_proxy.main.run_health_checker",
            new=_bounded_health_worker,
        ),
        patch("inference_proxy.main.DownloadService", return_value=download_service),
        patch("inference_proxy.main.NodeProvisioner", return_value=provisioner),
        patch("inference_proxy.main.httpx.AsyncClient", return_value=proxy_http),
        patch(
            "inference_proxy.main.ProxyClient",
            side_effect=RuntimeError("proxy startup failed"),
        ),
    ):
        app = create_app(settings=_without_legacy_shutdown_delay(test_settings))
        with pytest.raises(RuntimeError, match="proxy startup failed"):
            async with app.router.lifespan_context(app):
                pytest.fail("lifespan unexpectedly started")

    proxy_http.aclose.assert_awaited_once()
    provisioner.shutdown.assert_awaited_once()
    download_service.shutdown.assert_awaited_once()
    etcd.close.assert_called_once()


async def _wait_for_server_start(server: uvicorn.Server) -> None:
    deadline = asyncio.get_running_loop().time() + 2
    while not server.started:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Uvicorn did not start before the test deadline")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_uvicorn_drains_inflight_request_before_lifespan_cleanup(
    test_settings: Settings,
) -> None:
    """T13/L2: real Uvicorn draining precedes application cleanup."""
    etcd = _etcd_client()
    request_started = asyncio.Event()
    release_request = asyncio.Event()
    cleanup_started = asyncio.Event()
    original_shutdown = DownloadService.shutdown

    async def observed_shutdown(service: DownloadService) -> None:
        cleanup_started.set()
        await original_shutdown(service)

    app = create_app(settings=_without_legacy_shutdown_delay(test_settings))

    @app.get("/_test/inflight")
    async def inflight() -> dict[str, str]:
        request_started.set()
        await release_request.wait()
        return {"status": "complete"}

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="on", log_level="critical", access_log=False)
    )
    serve_task: asyncio.Task[None] | None = None
    request_task: asyncio.Task[httpx.Response] | None = None

    try:
        with (
            patch("inference_proxy.main.EtcdClient", return_value=etcd),
            patch("inference_proxy.main.EtcdWatcher"),
            patch(
                "inference_proxy.main.run_health_checker",
                new=_bounded_health_worker,
            ),
            patch(
                "inference_proxy.main.DownloadService.shutdown",
                new=observed_shutdown,
            ),
        ):
            serve_task = asyncio.create_task(server.serve(sockets=[listener]))
            await asyncio.wait_for(_wait_for_server_start(server), timeout=2.5)
            async with httpx.AsyncClient(timeout=2) as client:
                request_task = asyncio.create_task(
                    client.get(f"http://{host}:{port}/_test/inflight")
                )
                await asyncio.wait_for(request_started.wait(), timeout=1)
                server.should_exit = True

                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(cleanup_started.wait(), timeout=0.2)

                release_request.set()
                response = await asyncio.wait_for(request_task, timeout=1)
                assert response.status_code == 200
                assert response.json() == {"status": "complete"}

            await asyncio.wait_for(serve_task, timeout=3)
            assert cleanup_started.is_set()
    finally:
        release_request.set()
        server.should_exit = True
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        if serve_task is not None and not serve_task.done():
            await asyncio.wait_for(serve_task, timeout=3)
        listener.close()
