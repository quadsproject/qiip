"""Smoke tests for the FastAPI application."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from inference_proxy.config.settings import (
    QUADSSettings,
    RedfishSettings,
    RoutingSettings,
    Settings,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.main import _initial_load
from inference_proxy.models.node import Node, NodeStatus


def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns 200 with status and nodes_registered count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["nodes_registered"] == 0


def test_health_with_nodes(client: TestClient, test_registry: NodeRegistry) -> None:
    """GET /health returns correct nodes_registered count when nodes exist."""
    test_registry.add(
        Node(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            status=NodeStatus.HEALTHY,
            model="llama-3",
        )
    )
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["nodes_registered"] == 1


def test_app_is_fastapi_instance(app: FastAPI) -> None:
    """The app fixture yields a FastAPI instance."""
    assert isinstance(app, FastAPI)


def test_importing_main_does_not_resolve_settings(tmp_path: Path) -> None:
    """Importing the factory module neither loads settings nor constructs an app."""
    env = os.environ.copy()
    env.pop("INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inference_proxy.main as module; "
                "assert not hasattr(module, 'app')"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hf_xet_disabled_before_huggingface_import(tmp_path: Path) -> None:
    """The module-level guard must run before Hugging Face freezes its value."""
    env = os.environ.copy()
    env.pop("HF_HUB_DISABLE_XET", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inference_proxy.main; "
                "from huggingface_hub import constants; "
                "assert constants.HF_HUB_DISABLE_XET is True"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_lifespan_has_no_artificial_shutdown_gate(test_settings: Settings) -> None:
    """Uvicorn owns connection draining; the app adds no dead second gate."""
    from inference_proxy.main import create_app

    app = create_app(settings=test_settings)

    assert all(
        middleware.cls.__name__ != "ShutdownMiddleware"
        for middleware in app.user_middleware
    )
    assert not (
        Path(__file__).parents[1] / "inference_proxy/resilience/shutdown.py"
    ).exists()
    assert "graceful_shutdown_timeout" not in type(test_settings.gateway).model_fields


def test_default_allowlist_rejects_lab_endpoint_from_admin_nodes(
    client: TestClient,
    test_registry: NodeRegistry,
) -> None:
    """A secure-default rejection is visible and absent from the admin view."""
    etcd_client = MagicMock()
    etcd_client.prefix = "/nodes/"
    etcd_client.get_prefix.return_value = [
        (
            json.dumps(
                {
                    "endpoint": "10.0.1.100:8000",
                    "status": "healthy",
                    "model": "llama-3",
                }
            ).encode(),
            {"key": b"/nodes/gpu01"},
        )
    ]

    with patch("inference_proxy.discovery.serializer.logger") as serializer_logger:
        routing = RoutingSettings()
        _initial_load(etcd_client, test_registry, routing.endpoint_policy())

    assert test_registry.get_all() == []
    response = client.get("/admin/nodes")

    assert response.status_code == 200
    assert response.json() == []
    serializer_logger.warning.assert_called_once()
    fields = serializer_logger.warning.call_args.kwargs
    assert fields["endpoint"] == "10.0.1.100:8000"
    assert "host is not allowed" in fields["error"]


def test_subpackages_importable() -> None:
    """All six sub-packages under inference_proxy are importable."""
    import inference_proxy.api
    import inference_proxy.config
    import inference_proxy.discovery
    import inference_proxy.models
    import inference_proxy.resilience
    import inference_proxy.routing

    # Verify they are actual modules (not None or error objects)
    assert inference_proxy.config is not None
    assert inference_proxy.models is not None
    assert inference_proxy.api is not None
    assert inference_proxy.discovery is not None
    assert inference_proxy.routing is not None
    assert inference_proxy.resilience is not None


class TestRegistryInAppState:
    """Registry is accessible in app.state after startup."""

    def test_app_state_has_registry(self, app: FastAPI) -> None:
        """app.state.registry exists and is a NodeRegistry instance."""
        assert hasattr(app.state, "registry")
        assert isinstance(app.state.registry, NodeRegistry)


class TestLifespanRegistryIntegration:
    """Lifespan creates registry and watcher on startup, cleans up on shutdown."""

    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_creates_registry(
        self,
        mock_etcd_cls: MagicMock,
        mock_watcher_cls: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Lifespan populates app.state.registry with a NodeRegistry."""
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=test_settings)
        with TestClient(app):
            assert hasattr(app.state, "registry")
            assert isinstance(app.state.registry, NodeRegistry)
        watcher = mock_watcher_cls.return_value
        watcher.run.assert_called_once()
        watcher.stop.assert_called_once()

    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_passes_canonical_nfs_export_to_provisioner(
        self,
        mock_etcd_cls: MagicMock,
        _mock_watcher_cls: MagicMock,
        test_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client
        huggingface = test_settings.huggingface.model_copy(
            update={"nfs_export": "storage.example:/exports/huggingface"}
        )
        settings = test_settings.model_copy(update={"huggingface": huggingface})

        from inference_proxy.main import create_app

        app = create_app(settings=settings)
        with TestClient(app):
            assert app.state.provisioner._nfs_export == (
                "storage.example:/exports/huggingface"
            )

    @patch("inference_proxy.main.logger")
    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_unset_endpoint_allowlist_warns_at_startup(
        self,
        mock_etcd_cls: MagicMock,
        _mock_watcher_cls: MagicMock,
        mock_logger: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Secure loopback defaults are announced loudly during startup."""
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=test_settings)
        with TestClient(app):
            pass

        warnings = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args
            and call.args[0].startswith("backend endpoint allowlist is unset")
        ]
        assert len(warnings) == 1
        assert warnings[0].kwargs == {
            "allowed_hosts": ["localhost"],
            "allowed_networks": ["127.0.0.0/8", "::1/128"],
            "allowed_ports": [8000],
        }

    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_handles_etcd_unavailability(
        self,
        mock_etcd_cls: MagicMock,
        mock_watcher_cls: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Lifespan starts with empty registry when etcd is unavailable."""
        mock_client = MagicMock()
        mock_client.get_prefix.side_effect = ConnectionError("etcd down")
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=test_settings)
        with TestClient(app):
            registry = app.state.registry
            assert isinstance(registry, NodeRegistry)
            assert registry.get_all() == []

    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_wires_breaker_cleanup_to_node_registry(
        self,
        mock_etcd_cls: MagicMock,
        _mock_watcher_cls: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Production registry removal discards the node's breaker state."""
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=test_settings)
        with TestClient(app):
            registry = app.state.registry
            breaker_registry = app.state.circuit_breaker_registry
            registry.add(
                Node(
                    node_id="node-1",
                    endpoint="10.0.1.100:8000",
                    status=NodeStatus.HEALTHY,
                    model="llama-3",
                )
            )
            stale_breaker = breaker_registry.get_or_create("node-1")
            for _ in range(3):
                stale_breaker.record_failure()
            assert stale_breaker.is_open

            registry.remove("node-1")

            assert breaker_registry.get("node-1") is None

    @patch("inference_proxy.main.RedfishClient")
    @patch("inference_proxy.main.EtcdWatcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_redfish_disabled_without_credentials(
        self,
        mock_etcd_cls: MagicMock,
        _mock_watcher_cls: MagicMock,
        mock_redfish_cls: MagicMock,
        test_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=test_settings)
        with TestClient(app):
            assert app.state.redfish_client is None
        mock_redfish_cls.assert_not_called()

    def test_redfish_http_client_has_no_ambient_auth(
        self,
        test_settings: Settings,
    ) -> None:
        mock_etcd = MagicMock()
        mock_etcd.get_prefix.return_value = []
        mock_etcd.prefix = "/nodes/"
        redfish_http = MagicMock(aclose=AsyncMock())
        proxy_http = MagicMock(aclose=AsyncMock())
        redfish = RedfishSettings(
            bmc_username="operator",
            bmc_password=SecretStr("redfish-secret"),
        )
        settings = test_settings.model_copy(update={"redfish": redfish})

        from inference_proxy.main import create_app

        with (
            patch("inference_proxy.main.EtcdClient", return_value=mock_etcd),
            patch("inference_proxy.main.EtcdWatcher"),
            patch(
                "inference_proxy.main.httpx.AsyncClient",
                side_effect=[redfish_http, proxy_http],
            ) as async_client_cls,
            patch("inference_proxy.main.RedfishClient") as redfish_client_cls,
        ):
            app = create_app(settings=settings)
            with TestClient(app):
                assert app.state.redfish_client is redfish_client_cls.return_value

        redfish_http_call = next(
            call for call in async_client_cls.call_args_list if "verify" in call.kwargs
        )
        assert "auth" not in redfish_http_call.kwargs
        redfish_client_cls.assert_called_once_with(
            redfish_http,
            bmc_host_template="mgmt-{hostname}",
            system_id="1",
            hostname_policy=ANY,
            auth=ANY,
            poll_timeout=60.0,
            poll_interval=5.0,
        )

    def test_lifespan_passes_configured_quads_server_timezone(
        self,
        test_settings: Settings,
    ) -> None:
        mock_etcd = MagicMock()
        mock_etcd.get_prefix.return_value = []
        mock_etcd.prefix = "/nodes/"
        quads = QUADSSettings(
            base_url="http://quads.example.com",
            server_timezone="America/New_York",
        )
        settings = test_settings.model_copy(update={"quads": quads})

        from inference_proxy.main import create_app

        with (
            patch("inference_proxy.main.EtcdClient", return_value=mock_etcd),
            patch("inference_proxy.main.EtcdWatcher"),
            patch("inference_proxy.main.QUADSPoller") as quads_poller_cls,
            patch("inference_proxy.main.ScheduleEnforcer") as enforcer_cls,
            patch("inference_proxy.main.QUADSClient") as quads_client_cls,
        ):
            quads_poller_cls.return_value.stop = AsyncMock()
            enforcer_cls.return_value.stop = AsyncMock()
            app = create_app(settings=settings)
            with TestClient(app):
                pass

        quads_client_cls.assert_called_once_with(
            ANY,
            "http://quads.example.com",
            server_timezone="America/New_York",
        )


class TestGetRegistryDependency:
    """get_registry dependency returns the registry from app.state."""

    def test_get_registry_returns_registry(self, app: FastAPI) -> None:
        """get_registry returns a NodeRegistry instance via dependency injection."""
        from inference_proxy.config.dependencies import get_registry

        # Simulate a request by calling get_registry with a mock request
        mock_request = MagicMock()
        mock_request.app = app
        result = get_registry(mock_request)
        assert isinstance(result, NodeRegistry)
