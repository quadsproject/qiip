"""Integration tests for the request logging middleware.

Tests cover:
- Non-proxy routes log with target_node=None (GET /health, GET /v1/models)
- Proxy routes log with target_node set to node endpoint (POST /v1/chat/completions, POST /v1/completions)
- Streaming proxy routes log with target_node set
- Error scenarios still produce log entries with error status codes
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus


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


class TestRequestLoggingFields:
    """Non-proxy routes produce log entries with target_node=None."""

    def test_health_produces_log_entry(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        """GET /health produces a log entry with method, path, status_code, duration_ms, target_node=None."""
        with structlog.testing.capture_logs() as captured:
            response = client.get("/health")

        assert response.status_code == 200

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["method"] == "GET"
        assert log["path"] == "/health"
        assert log["status_code"] == 200
        assert log["duration_ms"] > 0
        assert log["target_node"] is None

    def test_models_produces_log_entry_with_null_target(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        """GET /v1/models produces a log entry with target_node=None (non-proxy route)."""
        with structlog.testing.capture_logs() as captured:
            response = client.get("/v1/models")

        assert response.status_code == 200

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["method"] == "GET"
        assert log["path"] == "/v1/models"
        assert log["status_code"] == 200
        assert log["target_node"] is None


class TestRequestLoggingTargetNode:
    """Proxy routes produce log entries with target_node set to the node endpoint."""

    def test_chat_completions_logs_target_node(
        self,
        app: FastAPI,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock,
    ) -> None:
        """POST /v1/chat/completions (non-streaming) logs target_node as the node endpoint string."""
        node = _make_node()
        test_registry.add(node)

        httpx_mock.add_response(
            url=f"http://{node.endpoint}/v1/chat/completions",
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            },
        )

        with structlog.testing.capture_logs() as captured:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["target_node"] == "10.0.1.100:8000"
        assert log["method"] == "POST"
        assert log["path"] == "/v1/chat/completions"
        assert log["status_code"] == 200
        assert log["duration_ms"] > 0

    def test_text_completions_logs_target_node(
        self,
        app: FastAPI,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock,
    ) -> None:
        """POST /v1/completions (non-streaming) logs target_node as the node endpoint string."""
        node = _make_node()
        test_registry.add(node)

        httpx_mock.add_response(
            url=f"http://{node.endpoint}/v1/completions",
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "choices": [{"text": "World"}],
            },
        )

        with structlog.testing.capture_logs() as captured:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "llama-3",
                    "prompt": "Hello",
                },
            )

        assert response.status_code == 200

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["target_node"] == "10.0.1.100:8000"
        assert log["method"] == "POST"
        assert log["path"] == "/v1/completions"

    def test_streaming_chat_completions_logs_target_node(
        self,
        app: FastAPI,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock,
    ) -> None:
        """POST /v1/chat/completions with stream=true logs target_node."""
        node = _make_node()
        test_registry.add(node)

        # Simulate SSE response from backend
        sse_body = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url=f"http://{node.endpoint}/v1/chat/completions",
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

        with structlog.testing.capture_logs() as captured:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        assert response.status_code == 200

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["target_node"] == "10.0.1.100:8000"


class TestRequestLoggingErrorCases:
    """Error scenarios still produce log entries."""

    def test_no_nodes_logs_error_status(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        """When no nodes are registered, middleware logs the error status code and target_node=None."""
        with structlog.testing.capture_logs() as captured:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        # Should be a 503 (no nodes available)
        assert response.status_code == 503

        request_logs = [log for log in captured if log.get("event") == "request"]
        assert len(request_logs) >= 1, (
            f"Expected a 'request' log entry, got: {captured}"
        )

        log = request_logs[0]
        assert log["status_code"] == 503
        assert log["target_node"] is None
        assert log["method"] == "POST"
        assert log["path"] == "/v1/chat/completions"
