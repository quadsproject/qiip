"""Behavioral coverage for the Anthropic Messages and OpenAI Responses routes.

Claude Code and Codex speak these APIs; qiip forwards them to backends that
implement them natively. These tests pin what the proxy itself owns: the
forwarded body, event-name-preserving streams, usage accounting, error
formats, failover and authentication.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from inference_proxy.auth.dependencies import get_sso_allowlist
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeSelector

_NODE = "10.0.1.100:8000"
_MESSAGES_URL = f"http://{_NODE}/v1/messages"
_COUNT_URL = f"http://{_NODE}/v1/messages/count_tokens"
_RESPONSES_URL = f"http://{_NODE}/v1/responses"


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


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _add_node(
    registry: NodeRegistry, node_id: str = "node-1", endpoint: str = _NODE
) -> Node:
    node = Node(
        node_id=node_id,
        endpoint=endpoint,
        status=NodeStatus.HEALTHY,
        model="qwen",
    )
    registry.add(node)
    return node


def _user_token(auth_store: AuthStore) -> tuple[int, str]:
    user = auth_store.upsert_google_user(
        google_sub="sub-1",
        email="alice@example.com",
        name="Alice",
        picture="",
    )
    return user.id, auth_store.create_token(user.id, "coding").token


def _second_user_token(auth_store: AuthStore) -> tuple[int, str]:
    user = auth_store.upsert_google_user(
        google_sub="sub-2",
        email="bob@example.com",
        name="Bob",
        picture="",
    )
    return user.id, auth_store.create_token(user.id, "other").token


class _DenyAllowlist:
    async def is_allowed(self, _email: str) -> bool:
        return False


def _break_usage_store(auth_store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def _locked(**_kwargs: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(auth_store, "record_usage", _locked)


def _enforce_tokens(app: FastAPI, test_settings: Settings) -> None:
    enforced = test_settings.model_copy(
        deep=True,
        update={
            "auth": test_settings.auth.model_copy(update={"enforce_api_tokens": True})
        },
    )
    app.dependency_overrides[get_settings] = lambda: enforced


def _forwarded_body(httpx_mock: HTTPXMock) -> Any:
    (request,) = httpx_mock.get_requests()
    return json.loads(request.content)


def _claude_code_body(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "qwen",
        "max_tokens": 32000,
        "stream": stream,
        "system": [{"type": "text", "text": "You are a coding agent."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Run the tests"}]},
            {"role": "system", "content": "# Environment\nPlatform: linux"},
        ],
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command",
                "input_schema": {"type": "object"},
            }
        ],
    }


def _anthropic_stream(*, cached: int = 0) -> list[bytes]:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "qwen",
        "usage": {
            "cache_read_input_tokens": cached,
            "input_tokens": 20,
            "output_tokens": 0,
        },
    }
    return [
        _sse("message_start", {"type": "message_start", "message": message}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 6},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _codex_body() -> dict[str, Any]:
    return {
        "model": "qwen",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Sandbox: read-only."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix the test"}],
            },
        ],
        "tools": [{"type": "namespace", "name": "multi_agent_v1", "tools": []}],
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }


def _responses_stream() -> list[bytes]:
    usage = {"input_tokens": 30, "output_tokens": 8, "total_tokens": 38}
    return [
        _sse("response.created", {"type": "response.created", "response": {}}),
        _sse(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "hi"},
        ),
        _sse(
            "response.completed",
            {"type": "response.completed", "response": {"usage": usage}},
        ),
    ]


class TestMessagesRoute:
    def test_non_streaming_request_is_forwarded_and_usage_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        backend_reply = {
            "id": "msg_1",
            "type": "message",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 90,
                "output_tokens": 4,
            },
        }
        httpx_mock.add_response(url=_MESSAGES_URL, json=backend_reply)

        response = client.post(
            "/v1/messages?beta=true",
            json=_claude_code_body(stream=False),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json() == backend_reply
        (summary,) = auth_store.get_usage_summary(user_id)
        assert summary.endpoint == "/v1/messages"
        assert (summary.prompt_tokens, summary.completion_tokens) == (100, 4)
        assert summary.total_tokens == 104

    def test_forwarded_body_keeps_client_fields_and_inlines_system_messages(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        httpx_mock.add_response(url=_MESSAGES_URL, json={"id": "msg_1"})
        body = _claude_code_body(stream=False)

        client.post("/v1/messages", json=body)

        forwarded = _forwarded_body(httpx_mock)
        assert forwarded["thinking"] == body["thinking"]
        assert forwarded["context_management"] == body["context_management"]
        assert forwarded["system"] == body["system"]
        assert forwarded["messages"][0] == body["messages"][0]
        assert forwarded["messages"][1] == {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<system-reminder>\n# Environment\nPlatform: linux\n"
                        "</system-reminder>\n"
                    ),
                }
            ],
        }

    def test_stream_is_relayed_with_event_names_and_usage_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        node = _add_node(test_registry)
        breaker = circuit_breaker_registry.get_or_create(node.node_id)
        breaker.record_failure()
        breaker.record_failure()
        chunks = _anthropic_stream(cached=80)
        upstream = _TrackedStream(chunks)
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )

        response = client.post(
            "/v1/messages",
            json=_claude_code_body(stream=True),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.content == b"".join(chunks)
        assert "[DONE]" not in response.text
        (summary,) = auth_store.get_usage_summary(user_id)
        assert (summary.prompt_tokens, summary.completion_tokens) == (100, 6)
        assert upstream.close_calls == 1
        # message_stop counted as success and reset the failure streak.
        breaker.record_failure()
        assert not breaker.is_open

    def test_x_api_key_authenticates_and_attributes_usage(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        test_settings: Settings,
        httpx_mock: HTTPXMock,
    ) -> None:
        _enforce_tokens(app, test_settings)
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            json={"usage": {"input_tokens": 1, "output_tokens": 1}},
        )

        response = TestClient(app).post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"x-api-key": token},
        )

        assert response.status_code == 200
        assert auth_store.get_usage_summary(user_id)[0].total_tokens == 2

    def test_missing_key_is_rejected_in_anthropic_format(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        _enforce_tokens(app, test_settings)

        response = TestClient(app).post(
            "/v1/messages", json=_claude_code_body(stream=False)
        )

        assert response.status_code == 401
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Authentication required for inference requests",
                "code": "invalid_api_key",
            },
        }
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_invalid_x_api_key_is_rejected_when_enforced(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        _enforce_tokens(app, test_settings)

        response = TestClient(app).post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"x-api-key": "qiip_not_a_real_token"},
        )

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Invalid API token"

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"{not json", "Request body must be valid JSON"),
            (b"[1, 2]", "Request body must be a JSON object"),
            (b'{"messages": []}', "'model' must be a non-empty string"),
            (b'{"model": "qwen", "stream": "yes"}', "'stream' must be a boolean"),
        ],
    )
    def test_malformed_body_is_rejected_in_anthropic_format(
        self,
        client: TestClient,
        content: bytes,
        message: str,
    ) -> None:
        response = client.post(
            "/v1/messages",
            content=content,
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": message,
                "code": "invalid_request",
            },
        }

    def test_no_nodes_is_reported_as_overloaded(self, client: TestClient) -> None:
        response = client.post("/v1/messages", json=_claude_code_body(stream=True))

        assert response.status_code == 503
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "overloaded_error"
        assert response.json()["error"]["code"] == "no_nodes"

    def test_unknown_model_is_not_found(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        _add_node(test_registry)
        body = {**_claude_code_body(stream=False), "model": "claude-opus"}

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found_error"
        assert "claude-opus" in response.json()["error"]["message"]

    def test_backend_failures_fail_over_then_report_exhaustion(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "inference_proxy.routing.node_selector.random.choice",
            lambda tied: tied[0],
        )
        _add_node(test_registry, "node-1", "10.0.1.100:8000")
        _add_node(test_registry, "node-2", "10.0.1.101:8000")
        for host in ("10.0.1.100", "10.0.1.101"):
            httpx_mock.add_response(
                url=f"http://{host}:8000/v1/messages",
                status_code=500,
                text="GPU exhausted",
            )

        response = client.post("/v1/messages", json=_claude_code_body(stream=False))

        assert response.status_code == 500
        assert response.json()["type"] == "error"
        assert response.json()["error"]["code"] == "failover_exhausted"
        assert response.headers["X-Inference-Proxy-Attempts"] == "2"

    def test_backend_client_error_passes_through_verbatim(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        backend_error = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad tool"},
        }
        httpx_mock.add_response(url=_MESSAGES_URL, status_code=400, json=backend_error)

        response = client.post("/v1/messages", json=_claude_code_body(stream=True))

        assert response.status_code == 400
        assert response.json() == backend_error

    def test_mid_stream_failure_is_reported_as_an_error_event(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        node = _add_node(test_registry)
        breaker = circuit_breaker_registry.get_or_create(node.node_id)
        breaker.record_failure()
        breaker.record_failure()
        chunks = _anthropic_stream()[:2]
        upstream = _TrackedStream(
            chunks, error_after_chunks=httpx.ReadError("connection reset")
        )
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )

        response = client.post("/v1/messages", json=_claude_code_body(stream=True))

        assert response.status_code == 200
        assert response.content.startswith(b"".join(chunks))
        tail = response.content[len(b"".join(chunks)) :].decode()
        head, data = tail.rstrip("\n").split("\n")
        assert head == "event: error"
        assert json.loads(data.removeprefix("data: "))["error"]["type"] == "api_error"
        assert "[DONE]" not in response.text
        assert breaker.is_open
        assert upstream.close_calls == 1


class TestMessagesRobustness:
    def test_blank_model_is_rejected(self, client: TestClient) -> None:
        body = {**_claude_code_body(stream=False), "model": "   "}

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_invalid_bearer_falls_back_to_a_valid_x_api_key(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        test_settings: Settings,
        httpx_mock: HTTPXMock,
    ) -> None:
        _enforce_tokens(app, test_settings)
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            json={"usage": {"input_tokens": 2, "output_tokens": 1}},
        )

        response = TestClient(app).post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"Authorization": "Bearer qiip_stale", "x-api-key": token},
        )

        assert response.status_code == 200
        assert auth_store.get_usage_summary(user_id)[0].total_tokens == 3

    def test_valid_bearer_wins_over_another_x_api_key(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        bearer_user, bearer = _user_token(auth_store)
        key_user, key = _second_user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            json={"usage": {"input_tokens": 2, "output_tokens": 1}},
        )

        TestClient(app).post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"Authorization": f"Bearer {bearer}", "x-api-key": key},
        )

        assert len(auth_store.get_usage_summary(bearer_user)) == 1
        assert auth_store.get_usage_summary(key_user) == []

    def test_sso_whitelist_denial_uses_the_anthropic_format(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_settings: Settings,
    ) -> None:
        auth = test_settings.auth.model_copy(
            update={
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "enforce_sso_whitelist": True,
            }
        )
        settings = test_settings.model_copy(deep=True, update={"auth": auth})
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_sso_allowlist] = lambda: _DenyAllowlist()
        _, token = _user_token(auth_store)

        response = TestClient(app).post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"x-api-key": token},
        )

        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
        assert response.json()["error"]["message"] == "SSO whitelist denied"

    def test_usage_store_failure_leaves_a_finished_stream_untouched(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
        httpx_mock: HTTPXMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, token = _user_token(auth_store)
        node = _add_node(test_registry)
        breaker = circuit_breaker_registry.get_or_create(node.node_id)
        breaker.record_failure()
        breaker.record_failure()
        _break_usage_store(auth_store, monkeypatch)
        chunks = _anthropic_stream()
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            headers={"content-type": "text/event-stream"},
            stream=_TrackedStream(chunks),
        )

        response = client.post(
            "/v1/messages",
            json=_claude_code_body(stream=True),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.content == b"".join(chunks)
        breaker.record_failure()
        assert not breaker.is_open

    def test_usage_store_failure_does_not_retry_a_served_request(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, token = _user_token(auth_store)
        _add_node(test_registry)
        _add_node(test_registry, "node-2", "10.0.1.101:8000")
        _break_usage_store(auth_store, monkeypatch)
        reply = {"id": "msg_1", "usage": {"input_tokens": 1, "output_tokens": 1}}
        httpx_mock.add_response(json=reply, is_reusable=True)

        response = client.post(
            "/v1/messages",
            json=_claude_code_body(stream=False),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json() == reply
        assert len(httpx_mock.get_requests()) == 1

    def test_stream_ending_without_message_stop_is_neutral(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        node = _add_node(test_registry)
        breaker = circuit_breaker_registry.get_or_create(node.node_id)
        breaker.record_failure()
        breaker.record_failure()
        chunks = _anthropic_stream()[:-1]
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            headers={"content-type": "text/event-stream"},
            stream=_TrackedStream(chunks),
        )

        response = client.post(
            "/v1/messages",
            json=_claude_code_body(stream=True),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.content == b"".join(chunks)
        (summary,) = auth_store.get_usage_summary(user_id)
        assert (summary.prompt_tokens, summary.completion_tokens) == (20, 6)
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open

    def test_mid_stream_failure_records_the_usage_seen_so_far(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_MESSAGES_URL,
            headers={"content-type": "text/event-stream"},
            stream=_TrackedStream(
                _anthropic_stream(cached=5)[:1],
                error_after_chunks=httpx.ReadError("connection reset"),
            ),
        )

        client.post(
            "/v1/messages",
            json=_claude_code_body(stream=True),
            headers={"Authorization": f"Bearer {token}"},
        )

        (summary,) = auth_store.get_usage_summary(user_id)
        assert (summary.prompt_tokens, summary.completion_tokens) == (25, 0)


class TestCountTokensRoute:
    def test_count_is_forwarded_without_usage_or_streaming(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(url=_COUNT_URL, json={"input_tokens": 1234})
        body = _claude_code_body(stream=True)

        response = client.post(
            "/v1/messages/count_tokens",
            json=body,
            headers={"x-api-key": token},
        )

        assert response.status_code == 200
        assert response.json() == {"input_tokens": 1234}
        assert auth_store.get_usage_summary(user_id) == []
        forwarded = _forwarded_body(httpx_mock)
        assert "stream" not in forwarded
        assert all(m["role"] != "system" for m in forwarded["messages"])

    def test_backend_failures_report_exhaustion_in_anthropic_format(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        httpx_mock.add_response(url=_COUNT_URL, status_code=502, is_reusable=True)

        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "qwen", "messages": []},
        )

        assert response.status_code == 502
        assert response.json()["type"] == "error"
        assert response.json()["error"]["code"] == "failover_exhausted"
        assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"

    def test_errors_use_the_anthropic_format(self, client: TestClient) -> None:
        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "qwen", "messages": []},
        )

        assert response.status_code == 503
        assert response.json()["error"]["type"] == "overloaded_error"


class TestResponsesRoute:
    def test_codex_stream_is_relayed_and_usage_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        chunks = _responses_stream()
        upstream = _TrackedStream(chunks)
        httpx_mock.add_response(
            url=_RESPONSES_URL,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )

        response = client.post(
            "/v1/responses",
            json=_codex_body(),
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.content == b"".join(chunks)
        (summary,) = auth_store.get_usage_summary(user_id)
        assert summary.endpoint == "/v1/responses"
        assert (summary.prompt_tokens, summary.completion_tokens) == (30, 8)
        assert summary.total_tokens == 38
        assert upstream.close_calls == 1

    def test_completed_stream_counts_as_backend_success(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        node = _add_node(test_registry)
        breaker = circuit_breaker_registry.get_or_create(node.node_id)
        breaker.record_failure()
        breaker.record_failure()
        httpx_mock.add_response(
            url=_RESPONSES_URL,
            headers={"content-type": "text/event-stream"},
            stream=_TrackedStream(_responses_stream()),
        )

        client.post("/v1/responses", json=_codex_body())

        breaker.record_failure()
        assert not breaker.is_open

    def test_developer_item_is_merged_into_instructions(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        httpx_mock.add_response(url=_RESPONSES_URL, json={"id": "resp_1"})
        body = {**_codex_body(), "stream": False}

        client.post("/v1/responses", json=body)

        forwarded = _forwarded_body(httpx_mock)
        assert forwarded["instructions"] == "You are Codex.\n\nSandbox: read-only."
        assert forwarded["input"] == body["input"][1:]
        assert forwarded["tools"] == body["tools"]
        assert forwarded["include"] == body["include"]

    def test_non_streaming_usage_is_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_RESPONSES_URL,
            json={"usage": {"input_tokens": 3, "output_tokens": 2}},
        )

        response = client.post(
            "/v1/responses",
            json={"model": "qwen", "input": "Hello"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        (summary,) = auth_store.get_usage_summary(user_id)
        assert (summary.prompt_tokens, summary.total_tokens) == (3, 5)

    def test_mid_stream_failure_is_reported_as_a_failed_response(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        chunks = _responses_stream()[:1]
        httpx_mock.add_response(
            url=_RESPONSES_URL,
            headers={"content-type": "text/event-stream"},
            stream=_TrackedStream(
                chunks, error_after_chunks=httpx.ReadError("connection reset")
            ),
        )

        response = client.post("/v1/responses", json=_codex_body())

        tail = response.content[len(b"".join(chunks)) :].decode()
        head, data = tail.rstrip("\n").split("\n")
        assert head == "event: response.failed"
        failed = json.loads(data.removeprefix("data: "))
        assert failed["response"]["status"] == "failed"
        assert failed["response"]["error"]["code"] == "backend_transport_error"

    def test_malformed_body_is_rejected_in_openai_format(
        self,
        client: TestClient,
    ) -> None:
        response = client.post("/v1/responses", json={"input": "Hello"})

        assert response.status_code == 400
        assert response.json()["error"] == {
            "message": "'model' must be a non-empty string",
            "type": "invalid_request_error",
            "param": "model",
            "code": "invalid_request",
        }

    def test_errors_use_the_openai_format(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        _add_node(test_registry)
        body = {**_codex_body(), "model": "gpt-5"}

        response = client.post("/v1/responses", json=body)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"

    def test_x_api_key_is_not_accepted(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_settings: Settings,
    ) -> None:
        _enforce_tokens(app, test_settings)
        _, token = _user_token(auth_store)

        response = TestClient(app).post(
            "/v1/responses",
            json=_codex_body(),
            headers={"x-api-key": token},
        )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"


def test_chat_stream_frames_stay_bare_data_lines(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    _add_node(test_registry)
    chunk = b'data: {"choices":[]}\n\n'
    httpx_mock.add_response(
        url=f"http://{_NODE}/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        stream=_TrackedStream([chunk, b"data: [DONE]\n\n"]),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.content == chunk + b"data: [DONE]\n\n"
    assert node_selector.tracker.get("node-1") == 0


@pytest.mark.parametrize(
    ("path", "auth_header"),
    [
        ("/v1/messages", "Authorization"),
        ("/v1/messages", "x-api-key"),
        ("/v1/messages/count_tokens", "Authorization"),
        ("/v1/messages/count_tokens", "x-api-key"),
        ("/v1/responses", "Authorization"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("permitted", [False, True])
def test_native_routes_enforce_onboarding_model_scope(
    client: TestClient,
    auth_store: AuthStore,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
    path: str,
    auth_header: str,
    stream: bool,
    permitted: bool,
) -> None:
    user_id, _ = _user_token(auth_store)
    token = auth_store.create_personal_token(
        user_id, "coding", ["qwen"] if permitted else ["other-model"], "test-secret"
    ).token
    _add_node(test_registry)
    if permitted:
        # A backend rejection proves an allowed model reached the backend,
        # including the streaming handshake, without requiring valid SSE.
        httpx_mock.add_response(
            url=f"http://{_NODE}{path}", status_code=400, json={"backend": "reached"}
        )
    response = client.post(
        path,
        json={"model": "qwen", "stream": stream, "messages": [], "input": "hi"},
        headers={
            auth_header: f"Bearer {token}" if auth_header == "Authorization" else token
        },
    )
    if permitted:
        assert response.status_code == 400
        assert response.json() == {"backend": "reached"}
        assert len(httpx_mock.get_requests()) == 1
    else:
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "model_not_permitted"
        if path.startswith("/v1/messages"):
            assert response.json()["type"] == "error"
            assert response.json()["error"]["type"] == "permission_error"
        assert httpx_mock.get_requests() == []
