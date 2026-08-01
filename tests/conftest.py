"""Shared test fixtures for the inference proxy test suite."""

from __future__ import annotations

import base64
import os
from pathlib import Path

# Provide the required cache path for tests that explicitly load environment settings.
_TEST_HF_CACHE = Path("/tmp/test-hf-cache")
_TEST_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR",
    str(_TEST_HF_CACHE),
)
_TEST_ADMIN_USERNAME = "test-admin"
_TEST_ADMIN_PASSWORD = "test-password"
os.environ.setdefault("INFERENCE_PROXY_ADMIN__USERNAME", _TEST_ADMIN_USERNAME)
os.environ.setdefault("INFERENCE_PROXY_ADMIN__PASSWORD", _TEST_ADMIN_PASSWORD)

from collections.abc import AsyncIterator, Generator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from inference_proxy.config.dependencies import (
    get_catalog_service,
    get_circuit_breaker_registry,
    get_download_service,
    get_llmfit_runner,
    get_node_selector,
    get_provisioner,
    get_proxy_client,
    get_quads_client,
    get_quads_poller,
    get_redfish_client,
    get_request_metrics,
    get_settings,
    get_unified_node_service,
)
from inference_proxy.config.settings import (
    AdminSettings,
    EtcdSettings,
    GatewaySettings,
    HuggingFaceSettings,
    RoutingSettings,
    Settings,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.catalog import ModelCatalogResponse
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.main import create_app
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics
from inference_proxy.services.unified_nodes import UnifiedNodeService


@pytest.fixture
def test_settings() -> Settings:
    """Return a Settings instance with test-safe defaults."""
    return Settings(
        gateway=GatewaySettings(host="127.0.0.1", port=9999),
        etcd=EtcdSettings(
            endpoints=["http://localhost:2379"], node_prefix="/test-nodes/"
        ),
        routing=RoutingSettings(strategy="least_connections", max_retries=3, timeout=5),
        admin=AdminSettings(
            username=_TEST_ADMIN_USERNAME,
            password=SecretStr(_TEST_ADMIN_PASSWORD),
        ),
        huggingface=HuggingFaceSettings(cache_dir=str(_TEST_HF_CACHE)),
    )


@pytest.fixture
def test_registry() -> NodeRegistry:
    """Return a fresh empty NodeRegistry for testing."""
    return NodeRegistry()


@pytest.fixture
async def mock_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Yield a real httpx.AsyncClient for use with httpx_mock."""
    client = httpx.AsyncClient()
    yield client
    await client.aclose()


@pytest.fixture
def proxy_client(mock_http_client: httpx.AsyncClient) -> ProxyClient:
    """Return a ProxyClient wrapping the mock HTTP client."""
    return ProxyClient(mock_http_client)


@pytest.fixture
def connection_tracker() -> ConnectionTracker:
    """Return a fresh ConnectionTracker for testing."""
    return ConnectionTracker()


@pytest.fixture
def circuit_breaker_registry(
    test_registry: NodeRegistry,
) -> CircuitBreakerRegistry:
    """Return a fresh CircuitBreakerRegistry for testing."""
    registry = CircuitBreakerRegistry()
    register_listener = getattr(test_registry, "register_remove_listener", None)
    if register_listener is not None:
        register_listener(registry.remove)
    return registry


@pytest.fixture
def request_metrics() -> RequestMetrics:
    """Return a fresh RequestMetrics instance for testing."""
    return RequestMetrics()


@pytest.fixture
def node_selector(
    test_registry: NodeRegistry,
    connection_tracker: ConnectionTracker,
) -> NodeSelector:
    """Return a NodeSelector wired to the test registry and tracker."""
    return NodeSelector(test_registry, connection_tracker)


@pytest.fixture
def app(
    test_settings: Settings,
    test_registry: NodeRegistry,
    proxy_client: ProxyClient,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    request_metrics: RequestMetrics,
) -> Generator[FastAPI, None, None]:
    """Create a FastAPI app with test settings, registry, and proxy client injected."""
    application = create_app(settings=test_settings)
    application.state.registry = test_registry
    application.state.proxy_client = proxy_client
    application.state.node_selector = node_selector
    application.state.circuit_breaker_registry = circuit_breaker_registry
    application.state.request_metrics = request_metrics
    application.state.quads_poller = None
    application.state.quads_client = None
    application.state.redfish_client = None
    application.dependency_overrides[get_proxy_client] = lambda: proxy_client
    application.dependency_overrides[get_quads_client] = lambda: None
    application.dependency_overrides[get_quads_poller] = lambda: None
    application.dependency_overrides[get_redfish_client] = lambda: None
    application.dependency_overrides[get_node_selector] = lambda: node_selector
    application.dependency_overrides[get_circuit_breaker_registry] = lambda: (
        circuit_breaker_registry
    )
    application.dependency_overrides[get_request_metrics] = lambda: request_metrics
    # UnifiedNodeService with no QUADS by default
    _unified_svc = UnifiedNodeService(
        registry=test_registry,
        poller=None,
        cb_registry=circuit_breaker_registry,
        tracker=node_selector.tracker,
    )
    application.dependency_overrides[get_unified_node_service] = lambda: _unified_svc
    mock_provisioner = MagicMock()
    mock_provisioner._etcd_client = MagicMock()
    mock_provisioner.list_tasks_raw = AsyncMock(return_value=[])
    mock_provisioner.cancel_active_provision = AsyncMock(return_value=None)
    mock_provisioner.try_reserve_host = AsyncMock(
        side_effect=lambda hostname: MagicMock(hostname=hostname)
    )
    mock_provisioner.connection_count = MagicMock(return_value=0)
    mock_provisioner.cleanup_stale_node = AsyncMock()
    mock_provisioner.provision = AsyncMock()
    mock_provisioner.teardown = AsyncMock()
    application.state.provisioner = mock_provisioner
    application.dependency_overrides[get_provisioner] = lambda: mock_provisioner
    mock_runner = MagicMock(spec=LLMFitRunner)
    mock_runner.recommend = AsyncMock()
    application.state.llmfit_runner = mock_runner
    application.dependency_overrides[get_llmfit_runner] = lambda: mock_runner
    mock_catalog = MagicMock()
    mock_catalog.list_models = AsyncMock(return_value=ModelCatalogResponse(models=[]))
    application.state.catalog_service = mock_catalog
    application.dependency_overrides[get_catalog_service] = lambda: mock_catalog
    mock_download_service = MagicMock()
    mock_download_service.trigger_download = AsyncMock()
    mock_download_service.get_status = MagicMock(return_value=None)
    mock_download_service.get_all_statuses = MagicMock(return_value=[])
    application.state.download_service = mock_download_service
    application.dependency_overrides[get_download_service] = lambda: (
        mock_download_service
    )
    yield application
    application.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def mock_provisioner(app: FastAPI) -> MagicMock:
    """Return the mock provisioner from the test app."""
    return app.state.provisioner  # type: ignore[no-any-return]


@pytest.fixture
def mock_llmfit_runner(app: FastAPI) -> MagicMock:
    """Return the mock LLMFitRunner from the test app."""
    return app.state.llmfit_runner  # type: ignore[no-any-return]


@pytest.fixture
def admin_auth_headers() -> dict[str, str]:
    """Return the shared test admin credentials as an HTTP Basic header."""
    token = base64.b64encode(
        f"{_TEST_ADMIN_USERNAME}:{_TEST_ADMIN_PASSWORD}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def client(app: FastAPI, admin_auth_headers: dict[str, str]) -> TestClient:
    """Return an authenticated TestClient bound to the test app."""
    return TestClient(app, headers=admin_auth_headers)
