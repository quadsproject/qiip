"""Smoke tests for the FastAPI application."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus


def _lifespan_settings(settings: Settings) -> Settings:
    """Disable the production drain delay only for lifespan smoke tests."""
    gateway = settings.gateway.model_copy(update={"graceful_shutdown_timeout": 0})
    return settings.model_copy(update={"gateway": gateway})


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

    @patch("inference_proxy.main.run_watcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_creates_registry(
        self,
        mock_etcd_cls: MagicMock,
        mock_run_watcher: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Lifespan populates app.state.registry with a NodeRegistry."""
        mock_client = MagicMock()
        mock_client.get_prefix.return_value = []
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=_lifespan_settings(test_settings))
        with TestClient(app):
            assert hasattr(app.state, "registry")
            assert isinstance(app.state.registry, NodeRegistry)

    @patch("inference_proxy.main.run_watcher")
    @patch("inference_proxy.main.EtcdClient")
    def test_lifespan_handles_etcd_unavailability(
        self,
        mock_etcd_cls: MagicMock,
        mock_run_watcher: MagicMock,
        test_settings: Settings,
    ) -> None:
        """Lifespan starts with empty registry when etcd is unavailable."""
        mock_client = MagicMock()
        mock_client.get_prefix.side_effect = ConnectionError("etcd down")
        mock_client.prefix = "/nodes/"
        mock_etcd_cls.return_value = mock_client

        from inference_proxy.main import create_app

        app = create_app(settings=_lifespan_settings(test_settings))
        with TestClient(app):
            registry = app.state.registry
            assert isinstance(registry, NodeRegistry)
            assert registry.get_all() == []


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
