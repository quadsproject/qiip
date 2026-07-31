"""Integration tests for OpenAI-compatible API route handlers.

Tests cover all phase 3 requirements:
- PROXY-01: Chat completion proxying (non-streaming)
- PROXY-02: Text completion proxying (non-streaming)
- PROXY-03: Model listing from registry
- PROXY-04: Health endpoint with node count
- PROXY-05: Proxy error responses (timeout, connection error)
- STRM-01: Chat completion streaming via SSE
- STRM-02: Text completion streaming via SSE
- STRM-03: SSE [DONE] termination signal
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock, IteratorStream

from inference_proxy.api.routes import _record_failure_and_trip
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
) -> Node:
    """Create a test node with sensible defaults."""
    return Node(
        node_id=node_id,
        endpoint=endpoint,
        status=status,
        model=model,
    )


# ---------------------------------------------------------------------------
# PROXY-01: Chat Completion Non-Streaming
# ---------------------------------------------------------------------------


class TestChatCompletionNonStreaming:
    """POST /v1/chat/completions proxies to vLLM and returns JSON."""

    def test_chat_completion_proxies_to_vllm(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Successful chat completion returns the vLLM response verbatim."""
        test_registry.add(_make_node())

        vllm_response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama-3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json=vllm_response,
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-abc123"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_chat_completion_no_nodes_returns_503(
        self,
        client: TestClient,
    ) -> None:
        """Empty registry returns 503 with no_nodes error code."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "no_nodes"


# ---------------------------------------------------------------------------
# PROXY-02: Text Completion Non-Streaming
# ---------------------------------------------------------------------------


class TestTextCompletionNonStreaming:
    """POST /v1/completions proxies to vLLM and returns JSON."""

    def test_text_completion_proxies_to_vllm(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Successful text completion returns the vLLM response verbatim."""
        test_registry.add(_make_node())

        vllm_response = {
            "id": "cmpl-abc123",
            "object": "text_completion",
            "created": 1700000000,
            "model": "llama-3",
            "choices": [
                {
                    "index": 0,
                    "text": "The answer is 42.",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "total_tokens": 11,
            },
        }
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/completions",
            json=vllm_response,
            status_code=200,
        )

        response = client.post(
            "/v1/completions",
            json={
                "model": "llama-3",
                "prompt": "What is the meaning of life?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "cmpl-abc123"
        assert data["choices"][0]["text"] == "The answer is 42."

    def test_text_completion_no_nodes_returns_503(
        self,
        client: TestClient,
    ) -> None:
        """Empty registry returns 503 with no_nodes error code."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "llama-3",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "no_nodes"


# ---------------------------------------------------------------------------
# STRM-01, STRM-03: Chat Completion Streaming
# ---------------------------------------------------------------------------


class TestChatCompletionStreaming:
    """POST /v1/chat/completions with stream=true returns SSE events."""

    def test_chat_streaming_returns_sse_events(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Streaming chat completion returns SSE data lines from vLLM."""
        test_registry.add(_make_node())

        sse_chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(sse_chunks),
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        body = response.text
        assert "Hello" in body
        assert " world" in body

    def test_chat_streaming_done_signal(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Streaming chat completion includes the [DONE] termination signal."""
        test_registry.add(_make_node())

        sse_chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(sse_chunks),
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "[DONE]" in response.text

    def test_chat_streaming_no_nodes_returns_503(
        self,
        client: TestClient,
    ) -> None:
        """Streaming with empty registry returns 503."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "no_nodes"


# ---------------------------------------------------------------------------
# STRM-02: Text Completion Streaming
# ---------------------------------------------------------------------------


class TestTextCompletionStreaming:
    """POST /v1/completions with stream=true returns SSE events."""

    def test_text_streaming_returns_sse_events(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Streaming text completion returns SSE data lines from vLLM."""
        test_registry.add(_make_node())

        sse_chunks = [
            b'data: {"id":"cmpl-1","object":"text_completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"text":"The answer","finish_reason":null}]}\n\n',
            b'data: {"id":"cmpl-1","object":"text_completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"text":" is 42","finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(sse_chunks),
        )

        response = client.post(
            "/v1/completions",
            json={
                "model": "llama-3",
                "prompt": "What is the meaning of life?",
                "stream": True,
            },
        )

        assert response.status_code == 200
        body = response.text
        assert "The answer" in body
        assert " is 42" in body
        assert "[DONE]" in body


# ---------------------------------------------------------------------------
# PROXY-03: List Models
# ---------------------------------------------------------------------------


class TestListModels:
    """GET /v1/models returns aggregated model list from registry."""

    def test_list_models_returns_registered_models(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Two nodes with different models return two model entries."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="mistral-7b")
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        model_ids = {m["id"] for m in data["data"]}
        assert model_ids == {"llama-3", "mistral-7b"}
        for model_entry in data["data"]:
            assert model_entry["object"] == "model"
            assert model_entry["owned_by"] == "vllm"

    def test_list_models_empty_registry(
        self,
        client: TestClient,
    ) -> None:
        """Empty registry returns empty model list."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_deduplicates(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Two nodes serving the same model produce one model entry."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "llama-3"


# ---------------------------------------------------------------------------
# PROXY-05: Proxy Errors
# ---------------------------------------------------------------------------


class TestProxyErrors:
    """Proxy failures return OpenAI-compatible error responses."""

    def test_upstream_timeout_returns_504(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Backend read timeout maps to 504 with backend_timeout code."""
        test_registry.add(_make_node())

        httpx_mock.add_exception(
            httpx.ReadTimeout("read timed out"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 504
        data = response.json()
        assert data["error"]["code"] == "backend_timeout"
        assert data["error"]["type"] == "upstream_error"

    def test_upstream_connect_error_returns_502(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Backend connection failure maps to 502 with backend_unavailable code."""
        test_registry.add(_make_node())

        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"]["code"] == "backend_unavailable"
        assert data["error"]["type"] == "upstream_error"


# ---------------------------------------------------------------------------
# PROXY-04: Health Endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health returns status and registered node count."""

    def test_health_returns_status_and_nodes(
        self,
        client: TestClient,
    ) -> None:
        """Health endpoint includes status and nodes_registered keys."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "nodes_registered" in data


# ---------------------------------------------------------------------------
# Phase 4: Model-Aware Routing
# ---------------------------------------------------------------------------


class TestLeastConnectionsRouting:
    """Requests route to the node with fewest active connections."""

    def test_routes_to_least_connections_node(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """With 2 nodes serving same model, routes to node with fewer connections."""
        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )

        # Increment connections on node-1 so node-2 has fewer
        node_selector.tracker.increment("node-1")

        httpx_mock.add_response(
            url="http://10.0.1.101:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        # Verify the request went to node-2 (least connections)
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        assert "10.0.1.101:8000" in str(requests[0].url)


class TestModelFiltering:
    """Requests route only to nodes serving the requested model."""

    def test_routes_to_matching_model_node(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """With different models, routes to node serving requested model."""
        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="mistral-7b")
        )

        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        assert "10.0.1.100:8000" in str(requests[0].url)


class TestModelNotFound:
    """404 returned when no node serves the requested model."""

    def test_model_not_found_returns_404(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Requesting a model no node serves returns 404."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "model_not_found"
        assert data["error"]["type"] == "invalid_request_error"


class TestModelUnavailable:
    """503 returned when model exists but all nodes are draining."""

    def test_model_unavailable_returns_503(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Requesting a model where all nodes are DRAINING returns 503."""
        test_registry.add(
            _make_node(node_id="node-1", model="llama-3", status=NodeStatus.DRAINING)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == "model_unavailable"


class TestDrainingExcludedFromModels:
    """DRAINING nodes do not appear in /v1/models response."""

    def test_draining_nodes_excluded(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Only healthy nodes appear in model listing."""
        test_registry.add(
            _make_node(node_id="node-1", model="llama-3", status=NodeStatus.HEALTHY)
        )
        test_registry.add(
            _make_node(
                node_id="node-2",
                endpoint="10.0.1.101:8000",
                model="mistral-7b",
                status=NodeStatus.DRAINING,
            )
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "llama-3"


class TestTextCompletionModelRouting:
    """Text completion also uses model-aware routing."""

    def test_text_completion_model_not_found(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Text completion with nonexistent model returns 404."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        response = client.post(
            "/v1/completions",
            json={"model": "nonexistent", "prompt": "Hello"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "model_not_found"


class TestStreamingModelFiltering:
    """Streaming requests also use model-aware routing."""

    def test_streaming_model_not_found(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Streaming with nonexistent model returns 404."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# Phase 4: Connection Tracking and Drain Auto-Removal
# ---------------------------------------------------------------------------


class TestConnectionTracking:
    """Connection count returns to 0 after successful non-streaming proxy call."""

    def test_connection_count_returns_to_zero(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """After non-streaming proxy call, connection count is 0."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        assert node_selector.tracker.get("node-1") == 0


class TestStreamingConnectionTracking:
    """Connection count returns to 0 after successful streaming proxy call."""

    def test_streaming_connection_count_returns_to_zero(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """After streaming proxy call, connection count is 0."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        sse_chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(sse_chunks),
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        # Consume the response to trigger the generator completion
        _ = response.text
        assert node_selector.tracker.get("node-1") == 0


class TestDrainAutoRemoval:
    """Draining node with 0 connections is auto-removed from registry (D-11, LBAL-02)."""

    def test_draining_node_removed_after_proxy_call(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """DRAINING node with 0 connections is removed after proxy call completes."""
        # Add a healthy node and a draining node both serving same model
        test_registry.add(
            _make_node(
                node_id="node-1",
                endpoint="10.0.1.100:8000",
                model="llama-3",
                status=NodeStatus.HEALTHY,
            )
        )
        test_registry.add(
            _make_node(
                node_id="node-2",
                endpoint="10.0.1.101:8000",
                model="llama-3",
                status=NodeStatus.DRAINING,
            )
        )

        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        # The draining node-2 with 0 connections should be auto-removed
        assert test_registry.get("node-2") is None


# ---------------------------------------------------------------------------
# Phase 5: Retry with Failover and Circuit Breaker Recording
# ---------------------------------------------------------------------------


class TestNonStreamingRetry:
    """Non-streaming requests retry on a different node when the first fails."""

    def test_retries_on_connect_error(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """First node fails with ConnectError, second node succeeds, client sees success."""
        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )
        # Ensure node-1 is selected first (fewer connections)
        node_selector.tracker.increment("node-2")

        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )
        httpx_mock.add_response(
            url="http://10.0.1.101:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-1"

    def test_retries_on_timeout(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        httpx_mock: HTTPXMock,
    ) -> None:
        """First node times out, second node succeeds."""
        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )
        # Ensure node-1 is selected first
        node_selector.tracker.increment("node-2")

        httpx_mock.add_exception(
            httpx.ReadTimeout("read timed out"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )
        httpx_mock.add_response(
            url="http://10.0.1.101:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200

    def test_retries_exhausted_returns_error(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """When all nodes fail, the last error is returned to the client."""
        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )

        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.101:8000/v1/chat/completions",
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"]["code"] == "backend_unavailable"


class TestCircuitBreakerStatusTransition:
    """A stale request snapshot cannot resurrect or undrain a node."""

    def test_request_failure_does_not_resurrect_removed_node(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        registry.remove(node.node_id)
        selector = NodeSelector(registry, ConnectionTracker())
        cb_registry = CircuitBreakerRegistry(threshold=1)

        _record_failure_and_trip(node, cb_registry, selector)

        assert registry.get(node.node_id) is None

    def test_request_failure_does_not_clobber_draining_node(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        selector = NodeSelector(registry, ConnectionTracker())
        cb_registry = CircuitBreakerRegistry(threshold=1)
        reservation = selector.select_and_reserve()
        assert reservation is not None
        registry.drain(node.node_id)

        _record_failure_and_trip(node, cb_registry, selector)

        draining = registry.get(node.node_id)
        assert draining is not None
        assert draining.status == NodeStatus.DRAINING

        reservation.release()
        assert registry.get(node.node_id) is None


class TestCircuitBreakerRecording:
    """Circuit breaker records success and failure for proxy calls."""

    def test_records_success_on_successful_proxy(
        self,
        test_registry: NodeRegistry,
        app: object,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Successful proxy call records success on the circuit breaker.

        Pre-seed the breaker to OPEN (3 failures with threshold=3), then
        verify that after a successful proxy call, the breaker closes.
        This proves record_success is being called during proxy success.
        """
        from fastapi import FastAPI

        from inference_proxy.config.dependencies import get_circuit_breaker_registry

        assert isinstance(app, FastAPI)
        cb_registry = CircuitBreakerRegistry(threshold=3)
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: cb_registry

        test_registry.add(_make_node(node_id="node-1", model="llama-3"))
        # Pre-seed 3 failures to trip the breaker open
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234,
                "model": "llama-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            status_code=200,
        )

        test_client = TestClient(app)
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        # After successful proxy, breaker should be closed (record_success called)
        assert breaker.is_open is False

    def test_records_failure_on_failed_proxy(
        self,
        test_registry: NodeRegistry,
        app: object,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Failed proxy call records failure on the circuit breaker."""
        from fastapi import FastAPI

        from inference_proxy.config.dependencies import get_circuit_breaker_registry

        assert isinstance(app, FastAPI)
        cb_registry = CircuitBreakerRegistry(threshold=1)
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: cb_registry

        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )

        test_client = TestClient(app)
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        # With threshold=1 and only 1 node, retries exhausted
        assert response.status_code == 502
        breaker = cb_registry.get_or_create("node-1")
        assert breaker.is_open is True

    def test_circuit_breaker_trip_marks_node_unhealthy(
        self,
        test_registry: NodeRegistry,
        app: object,
        httpx_mock: HTTPXMock,
    ) -> None:
        """When the circuit breaker trips, the node is marked UNHEALTHY in the registry."""
        from fastapi import FastAPI

        from inference_proxy.config.dependencies import get_circuit_breaker_registry

        assert isinstance(app, FastAPI)
        cb_registry = CircuitBreakerRegistry(threshold=1)
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: cb_registry

        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )

        test_client = TestClient(app)
        test_client.post(
            "/v1/chat/completions",
            json={"model": "llama-3", "messages": [{"role": "user", "content": "Hi"}]},
        )

        node = test_registry.get("node-1")
        assert node is not None
        assert node.status == NodeStatus.UNHEALTHY


class TestStreamingCircuitBreaker:
    """Streaming requests record in circuit breaker but do not retry mid-stream."""

    def test_streaming_records_failure_no_retry(
        self,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
        app: object,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Streaming error records failure but does not retry on another node."""
        from fastapi import FastAPI

        from inference_proxy.config.dependencies import get_circuit_breaker_registry

        assert isinstance(app, FastAPI)
        cb_registry = CircuitBreakerRegistry(threshold=1)
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: cb_registry

        test_registry.add(
            _make_node(node_id="node-1", endpoint="10.0.1.100:8000", model="llama-3")
        )
        test_registry.add(
            _make_node(node_id="node-2", endpoint="10.0.1.101:8000", model="llama-3")
        )
        # Ensure node-1 is selected first (fewer connections)
        node_selector.tracker.increment("node-2")

        # Simulate streaming error -- connection fails
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url="http://10.0.1.100:8000/v1/chat/completions",
        )

        test_client = TestClient(app)
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        # Should NOT have retried on node-2; response contains an error
        assert response.status_code == 200  # SSE response is always 200
        body = response.text
        assert "backend_unavailable" in body or "upstream_error" in body

        # Circuit breaker should have recorded the failure
        breaker = cb_registry.get_or_create("node-1")
        assert breaker.is_open is True

        # No request should have been sent to node-2
        requests = httpx_mock.get_requests()
        assert all("10.0.1.101" not in str(r.url) for r in requests)

    def test_streaming_records_success(
        self,
        test_registry: NodeRegistry,
        app: object,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Successful streaming records success on the circuit breaker."""
        from fastapi import FastAPI

        from inference_proxy.config.dependencies import get_circuit_breaker_registry

        assert isinstance(app, FastAPI)
        cb_registry = CircuitBreakerRegistry(threshold=3)
        app.dependency_overrides[get_circuit_breaker_registry] = lambda: cb_registry

        test_registry.add(_make_node(node_id="node-1", model="llama-3"))

        # Pre-seed breaker to OPEN to verify success closes it
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        sse_chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"llama-3","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(sse_chunks),
        )

        test_client = TestClient(app)
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        _ = response.text  # Consume the response

        # After successful streaming, breaker should be closed (record_success called)
        assert breaker.is_open is False
