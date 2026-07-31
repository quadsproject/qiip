"""Behavioral coverage for non-streaming retry and failover."""

from __future__ import annotations

import httpx
import pytest
import structlog
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock, IteratorStream

import inference_proxy.api.routes as routes_module
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeSelector

_CHAT_URL = "/v1/chat/completions"
_REQUEST_BODY = {
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hi"}],
}


def _make_node(node_id: str, endpoint: str) -> Node:
    return Node(
        node_id=node_id,
        endpoint=endpoint,
        status=NodeStatus.HEALTHY,
        model="llama-3",
    )


def _add_ranked_nodes(
    registry: NodeRegistry,
    selector: NodeSelector,
    count: int,
) -> list[Node]:
    nodes = [
        _make_node(f"node-{index}", f"10.0.1.{99 + index}:8000")
        for index in range(1, count + 1)
    ]
    for rank, node in enumerate(nodes):
        registry.add(node)
        for _ in range(rank):
            selector.tracker.increment(node.node_id)
    return nodes


def _backend_url(node: Node) -> str:
    return f"http://{node.endpoint}{_CHAT_URL}"


def _success_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-success",
        "object": "chat.completion",
        "created": 1234,
        "model": "llama-3",
        "choices": [],
    }


def _set_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    attempts: int,
) -> None:
    routing = settings.routing.model_copy(update={"max_retries": attempts})
    configured = settings.model_copy(update={"routing": routing})
    monkeypatch.setattr(routes_module, "get_settings", lambda: configured)


def test_5xx_records_failure_and_fails_over(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    nodes = _add_ranked_nodes(test_registry, node_selector, 2)
    breaker = circuit_breaker_registry.get_or_create(nodes[0].node_id)
    breaker.record_failure()
    breaker.record_failure()
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        status_code=500,
        json={"error": "node 1 failed"},
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        status_code=200,
        json=_success_response(),
        is_optional=True,
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-success"
    assert breaker.is_open
    failed = test_registry.get(nodes[0].node_id)
    assert failed is not None
    assert failed.status == NodeStatus.UNHEALTHY
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]


def test_repeated_5xx_opens_breaker_at_configured_threshold(
    client: TestClient,
    test_registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    for attempt in range(1, 4):
        httpx_mock.add_response(
            url=_backend_url(node),
            status_code=500,
            json={"error": f"failure {attempt}"},
            is_optional=attempt > 1,
        )

    for attempt in range(1, 4):
        response = client.post(_CHAT_URL, json=_REQUEST_BODY)

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "failover_exhausted"
        assert breaker.is_open is (attempt == 3)
        current = test_registry.get(node.node_id)
        assert current is not None
        expected = NodeStatus.UNHEALTHY if attempt == 3 else NodeStatus.HEALTHY
        assert current.status == expected
    assert len(httpx_mock.get_requests()) == 3


def test_5xx_exhausted_preserves_upstream_status_with_failover_marker(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    nodes = _add_ranked_nodes(test_registry, node_selector, 2)
    for index, node in enumerate(nodes):
        httpx_mock.add_response(
            url=_backend_url(node),
            status_code=503,
            json={"error": f"{node.node_id} unavailable"},
            is_optional=index > 0,
        )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "2"


def test_single_node_exhaustion_is_marked_with_one_attempt(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=503,
        json={"error": "unavailable"},
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"
    assert len(httpx_mock.get_requests()) == 1


def test_non_retryable_error_carries_no_marker(
    client: TestClient,
    test_registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    breaker.record_failure()
    upstream_error = {
        "error": {
            "message": "Requested tokens exceed the model context window",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "max_tokens",
        }
    }
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=400,
        json=upstream_error,
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 400
    assert response.json() == upstream_error
    assert "X-Inference-Proxy-Failover" not in response.headers
    assert "X-Inference-Proxy-Attempts" not in response.headers
    assert len(httpx_mock.get_requests()) == 1
    assert not breaker.is_open


def test_mixed_statuses_returns_last_attempt(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    nodes = _add_ranked_nodes(test_registry, node_selector, 2)
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        status_code=503,
        text="node 1 unavailable",
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        status_code=500,
        text="node 2 failed",
        is_optional=True,
    )

    with structlog.testing.capture_logs() as captured:
        response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    statuses = [
        event["status_code"]
        for event in captured
        if event.get("event") == "backend returned error status"
    ]
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert "node 2 failed" in response.json()["error"]["message"]
    assert statuses == [503, 500]


def test_budget_exhaustion_marks_only_attempted_nodes(
    client: TestClient,
    test_settings: Settings,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_attempt_budget(monkeypatch, test_settings, attempts=2)
    nodes = _add_ranked_nodes(test_registry, node_selector, 3)
    for index, node in enumerate(nodes[:2]):
        httpx_mock.add_response(
            url=_backend_url(node),
            status_code=503,
            json={"error": f"{node.node_id} unavailable"},
            is_optional=index > 0,
        )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "2"
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadError, httpx.RemoteProtocolError],
    ids=["read_error", "remote_protocol_error"],
)
def test_transport_error_fails_over_to_next_node(
    error_type: type[httpx.TransportError],
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    nodes = _add_ranked_nodes(test_registry, node_selector, 2)
    breaker = circuit_breaker_registry.get_or_create(nodes[0].node_id)
    breaker.record_failure()
    breaker.record_failure()
    httpx_mock.add_exception(
        error_type("backend connection failed"),
        url=_backend_url(nodes[0]),
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        status_code=200,
        json=_success_response(),
        is_optional=True,
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 200
    assert breaker.is_open
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadError, httpx.RemoteProtocolError],
    ids=["read_error", "remote_protocol_error"],
)
def test_exhausted_transport_error_returns_502(
    error_type: type[httpx.TransportError],
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    httpx_mock.add_exception(
        error_type("backend connection failed"),
        url=_backend_url(node),
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"


def test_retry_budget_counts_total_attempts(
    client: TestClient,
    test_settings: Settings,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_attempt_budget(monkeypatch, test_settings, attempts=1)
    nodes = _add_ranked_nodes(test_registry, node_selector, 2)
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        status_code=503,
        json={"error": "unavailable"},
    )

    response = client.post(_CHAT_URL, json=_REQUEST_BODY)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0])
    ]


def test_streaming_5xx_mapping_remains_pr3_scope(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    """Characterize D3 so PR 2 cannot partially change its shared mapping path."""
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=500,
        headers={"content-type": "text/event-stream"},
        stream=IteratorStream([b'data: {"error":"failed"}\n\n']),
    )

    with pytest.raises(httpx.ResponseNotRead):
        client.post(
            _CHAT_URL,
            json={**_REQUEST_BODY, "stream": True},
        )
