"""Behavioral coverage for streaming handshake and error correctness."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeSelector

_CHAT_PATH = "/v1/chat/completions"
_TEST_BACKEND_TIMEOUT = 1.0
_REQUEST_BODY = {
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error_after_chunks: httpx.TransportError | None = None,
    ) -> None:
        self._chunks = chunks
        self._error_after_chunks = error_after_chunks
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error_after_chunks is not None:
            raise self._error_after_chunks

    async def aclose(self) -> None:
        self.close_calls += 1


def _make_node(node_id: str, endpoint: str, *, model: str = "llama-3") -> Node:
    return Node(
        node_id=node_id,
        endpoint=endpoint,
        status=NodeStatus.HEALTHY,
        model=model,
    )


def _add_nodes(
    registry: NodeRegistry,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> list[Node]:
    nodes = [
        _make_node(f"node-{index}", f"10.0.1.{99 + index}:8000")
        for index in range(1, count + 1)
    ]
    for node in nodes:
        registry.add(node)
    monkeypatch.setattr(
        "inference_proxy.routing.node_selector.random.choice",
        lambda tied: tied[0],
    )
    return nodes


def _backend_url(node: Node) -> str:
    return f"http://{node.endpoint}{_CHAT_PATH}"


def _payload(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1234,
        "model": "llama-3",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


def _sse_chunk(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return b"data: " + encoded + b"\n\n"


def _successful_stream(content: str = "Hello") -> _TrackedStream:
    return _TrackedStream(
        [
            _sse_chunk(_payload(content)),
            b"data: [DONE]\n\n",
        ]
    )


def _sse_data(response_body: bytes) -> list[bytes]:
    assert response_body.endswith(b"\n\n")
    frames = response_body[:-2].split(b"\n\n")
    assert all(frame.startswith(b"data: ") for frame in frames)
    assert all(b"\n" not in frame for frame in frames)
    return [frame.removeprefix(b"data: ") for frame in frames]


def _set_streaming_limits(
    app: FastAPI,
    settings: Settings,
    *,
    attempts: int,
    timeout: int | float | None = None,
) -> None:
    updates: dict[str, int | float] = {"max_attempts": attempts}
    if timeout is not None:
        updates["timeout"] = timeout
    routing = settings.routing.model_copy(update=updates)
    configured = settings.model_copy(update={"routing": routing})
    app.dependency_overrides[get_settings] = lambda: configured


def test_streaming_5xx_exhaustion_returns_status_with_failover_marker(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    breaker.record_failure()
    upstream = _TrackedStream([b'{"error":{"message":"GPU exhausted"}}'])
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=500,
        headers={"content-type": "application/json"},
        stream=upstream,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert "GPU exhausted" in response.json()["error"]["message"]
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"
    assert breaker.is_open
    assert node_selector.tracker.get(node.node_id) == 0
    assert upstream.close_calls == 1


def test_streaming_4xx_preserves_upstream_error_verbatim(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.is_open
    upstream_error = {
        "error": {
            "message": "Requested tokens exceed the model context window",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "max_tokens",
        }
    }
    upstream = _TrackedStream([json.dumps(upstream_error).encode()])
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=400,
        headers={"content-type": "application/json"},
        stream=upstream,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == 400
    assert response.json() == upstream_error
    assert "X-Inference-Proxy-Failover" not in response.headers
    assert "X-Inference-Proxy-Attempts" not in response.headers
    assert not breaker.is_open
    assert len(httpx_mock.get_requests()) == 1
    assert node_selector.tracker.get(node.node_id) == 0
    assert upstream.close_calls == 1


def test_streaming_4xx_does_not_clear_prior_failures(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    breaker.record_failure()
    client_error_stream = _TrackedStream(
        [b'{"error":{"code":"context_length_exceeded"}}']
    )
    backend_error_stream = _TrackedStream([b'{"error":"backend failed"}'])
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=400,
        headers={"content-type": "application/json"},
        stream=client_error_stream,
    )
    httpx_mock.add_response(
        url=_backend_url(node),
        status_code=500,
        headers={"content-type": "application/json"},
        stream=backend_error_stream,
        is_optional=True,
    )

    client_error = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert client_error.status_code == 400
    assert not breaker.is_open

    backend_error = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert backend_error.status_code == 500
    assert backend_error.json()["error"]["code"] == "failover_exhausted"
    assert breaker.is_open
    current = test_registry.get(node.node_id)
    assert current is not None
    assert current.status == NodeStatus.UNHEALTHY
    assert node_selector.tracker.get(node.node_id) == 0
    assert client_error_stream.close_calls == 1
    assert backend_error_stream.close_calls == 1


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError],
    ids=["connect_error", "read_error", "remote_protocol_error"],
)
def test_streaming_preconnection_transport_error_fails_over(
    error_type: type[httpx.TransportError],
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = _add_nodes(test_registry, monkeypatch, 2)
    breaker = circuit_breaker_registry.get_or_create(nodes[0].node_id)
    breaker.record_failure()
    breaker.record_failure()
    successful = _successful_stream("from node 2")
    httpx_mock.add_exception(
        error_type("upstream handshake failed"),
        url=_backend_url(nodes[0]),
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        headers={"content-type": "text/event-stream"},
        stream=successful,
        is_optional=True,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)
    data = _sse_data(response.content)
    metrics = client.get("/admin/metrics").json()

    assert response.status_code == 200
    assert json.loads(data[0]) == _payload("from node 2")
    assert data[1] == b"[DONE]"
    assert breaker.is_open
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]
    assert node_selector.tracker.get(nodes[0].node_id) == 0
    assert node_selector.tracker.get(nodes[1].node_id) == 0
    assert successful.close_calls == 1
    assert metrics == {
        "total_requests": 1,
        "per_model": {"llama-3": 1},
        "per_node": {"node-1": 1, "node-2": 1},
    }


def test_streaming_transport_exhaustion_returns_502_with_failover_marker(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    breaker.record_failure()
    httpx_mock.add_exception(
        httpx.ConnectError("connection refused"),
        url=_backend_url(node),
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"
    assert breaker.is_open
    assert len(httpx_mock.get_requests()) == 1
    assert node_selector.tracker.get(node.node_id) == 0


def test_streaming_5xx_before_first_event_fails_over(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = _add_nodes(test_registry, monkeypatch, 2)
    breaker = circuit_breaker_registry.get_or_create(nodes[0].node_id)
    breaker.record_failure()
    breaker.record_failure()
    failed = _TrackedStream([b'{"error":{"message":"overloaded"}}'])
    successful = _successful_stream("from node 2")
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        status_code=503,
        headers={"content-type": "application/json"},
        stream=failed,
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        headers={"content-type": "text/event-stream"},
        stream=successful,
        is_optional=True,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)
    data = _sse_data(response.content)

    assert response.status_code == 200
    assert json.loads(data[0]) == _payload("from node 2")
    assert data[1] == b"[DONE]"
    assert breaker.is_open
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]
    assert failed.close_calls == 1
    assert successful.close_calls == 1
    assert node_selector.tracker.get(nodes[0].node_id) == 0
    assert node_selector.tracker.get(nodes[1].node_id) == 0


def test_streaming_failover_honors_attempt_budget(
    app: FastAPI,
    client: TestClient,
    test_settings: Settings,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_streaming_limits(app, test_settings, attempts=2)
    nodes = _add_nodes(test_registry, monkeypatch, 3)
    first = _TrackedStream([b'{"error":{"message":"node 1 unavailable"}}'])
    second = _TrackedStream([b'{"error":{"message":"node 2 failed"}}'])
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        status_code=503,
        headers={"content-type": "application/json"},
        stream=first,
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[1]),
        status_code=500,
        headers={"content-type": "application/json"},
        stream=second,
        is_optional=True,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert "node 2 failed" in response.json()["error"]["message"]
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "2"
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0]),
        _backend_url(nodes[1]),
    ]
    assert first.close_calls == 1
    assert second.close_calls == 1
    assert all(node_selector.tracker.get(node.node_id) == 0 for node in nodes)


def test_streaming_handshake_uses_total_routing_timeout(
    app: FastAPI,
    client: TestClient,
    test_settings: Settings,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    _set_streaming_limits(
        app,
        test_settings,
        attempts=3,
        timeout=0.01,
    )
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)

    async def bounded_stall(_: httpx.Request) -> httpx.Response:
        try:
            await asyncio.wait_for(
                asyncio.Event().wait(),
                timeout=_TEST_BACKEND_TIMEOUT,
            )
        except TimeoutError as exc:
            raise AssertionError(
                "production did not enforce the streaming handshake timeout"
            ) from exc
        raise AssertionError("the mock backend unexpectedly resumed")

    httpx_mock.add_callback(bounded_stall, url=_backend_url(node))

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "1"
    assert len(httpx_mock.get_requests()) == 1
    assert node_selector.tracker.get(node.node_id) == 0


def test_streaming_midstream_transport_error_emits_error_and_done_without_retry(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    circuit_breaker_registry: CircuitBreakerRegistry,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = _add_nodes(test_registry, monkeypatch, 2)
    breaker = circuit_breaker_registry.get_or_create(nodes[0].node_id)
    breaker.record_failure()
    breaker.record_failure()
    first_payload = _payload("partial")
    stream = _TrackedStream(
        [_sse_chunk(first_payload)],
        error_after_chunks=httpx.ReadError("stream ended early"),
    )
    httpx_mock.add_response(
        url=_backend_url(nodes[0]),
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)
    data = _sse_data(response.content)

    assert response.status_code == 200
    assert json.loads(data[0]) == first_payload
    error = json.loads(data[1])
    assert error["error"]["type"] == "upstream_error"
    assert error["error"]["code"] == "backend_transport_error"
    assert data[2] == b"[DONE]"
    assert [str(request.url) for request in httpx_mock.get_requests()] == [
        _backend_url(nodes[0])
    ]
    assert breaker.is_open
    assert stream.close_calls == 1
    assert node_selector.tracker.get(nodes[0].node_id) == 0


@pytest.mark.parametrize(
    ("registered_model", "expected_status", "expected_code"),
    [
        (None, 503, "no_nodes"),
        ("other-model", 404, "model_not_found"),
    ],
    ids=["no_nodes", "model_not_found"],
)
def test_streaming_selection_error_has_no_failover_marker(
    registered_model: str | None,
    expected_status: int,
    expected_code: str,
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    if registered_model is not None:
        test_registry.add(
            _make_node(
                "node-1",
                "10.0.1.100:8000",
                model=registered_model,
            )
        )

    response = client.post(_CHAT_PATH, json=_REQUEST_BODY)

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert "X-Inference-Proxy-Failover" not in response.headers
    assert "X-Inference-Proxy-Attempts" not in response.headers
    assert httpx_mock.get_requests() == []
