"""Behavioral coverage for /v1 bearer-token auth and usage recording.

AUTH-03: tokens are always validated when presented; enforcement of
anonymous /v1 inference is config-gated (off by default).
AUTH-04: token-authenticated requests record OpenAI usage per token.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus

_CHAT_URL = "http://10.0.1.100:8000/v1/chat/completions"
_COMPLETIONS_URL = "http://10.0.1.100:8000/v1/completions"
_BODY = {
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hi"}],
}
_STREAM_BODY = {**_BODY, "stream": True}


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1


def _add_node(test_registry: NodeRegistry) -> None:
    test_registry.add(
        Node(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            status=NodeStatus.HEALTHY,
            model="llama-3",
        )
    )


def _create_user_token(auth_store: AuthStore) -> tuple[int, str]:
    user = auth_store.upsert_google_user(
        google_sub="sub-1",
        email="alice@example.com",
        name="Alice",
        picture="",
    )
    created = auth_store.create_token(user.id, "ci-job")
    return user.id, created.token


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _stream_closing_chunk(prompt: int, completion: int) -> _TrackedStream:
    usage_chunk = _sse_chunk(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1234,
            "model": "llama-3",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )
    return _TrackedStream([usage_chunk, b"data: [DONE]\n\n"])


class TestAnonymousInferenceByDefault:
    def test_anonymous_request_succeeds_when_enforcement_off(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            json={"id": "chatcmpl-1", "object": "chat.completion"},
        )

        response = client.post("/v1/chat/completions", json=_BODY)

        assert response.status_code == 200

    def test_anonymous_request_records_no_usage(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            json={
                "id": "chatcmpl-1",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

        response = client.post("/v1/chat/completions", json=_BODY)

        assert response.status_code == 200
        assert auth_store.get_usage_summary(1) == []


class TestInvalidTokenRejected:
    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Bearer qiip_totally-made-up"},
            {"Authorization": "Bearer not-even-qiip"},
        ],
    )
    def test_invalid_bearer_rejected_with_openai_401(
        self,
        client: TestClient,
        httpx_mock: HTTPXMock,
        headers: dict[str, str],
    ) -> None:
        response = client.post("/v1/chat/completions", json=_BODY, headers=headers)

        assert response.status_code == 401
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_api_key"
        assert response.headers["www-authenticate"] == "Bearer"
        # No backend was contacted.
        assert len(httpx_mock.get_requests()) == 0

    def test_basic_credentials_are_treated_as_anonymous_not_rejected(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        # With enforcement off, a non-Bearer header is simply not token auth:
        # the request continues as anonymous and hits the (empty) node pool.
        response = client.post(
            "/v1/chat/completions",
            json=_BODY,
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "no_nodes"
        assert "www-authenticate" not in response.headers


class TestTokenUsageRecording:
    def test_non_streaming_usage_is_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _create_user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            json={
                "id": "chatcmpl-1",
                "model": "llama-3",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

        response = client.post(
            "/v1/chat/completions",
            json=_BODY,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        summary = auth_store.get_usage_summary(user_id)
        assert len(summary) == 1
        assert summary[0].endpoint == "/v1/chat/completions"
        assert summary[0].prompt_tokens == 10
        assert summary[0].completion_tokens == 5
        assert summary[0].total_tokens == 15

    def test_completions_endpoint_usage_is_recorded(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _create_user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_COMPLETIONS_URL,
            status_code=200,
            json={
                "id": "cmpl-1",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            },
        )

        response = client.post(
            "/v1/completions",
            json={"model": "llama-3", "prompt": "Hello"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        summary = auth_store.get_usage_summary(user_id)
        assert summary[0].endpoint == "/v1/completions"
        assert summary[0].total_tokens == 7

    def test_streaming_usage_is_recorded_once(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _create_user_token(auth_store)
        _add_node(test_registry)
        upstream = _stream_closing_chunk(prompt=8, completion=12)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )

        response = client.post(
            "/v1/chat/completions",
            json=_STREAM_BODY,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert "data: [DONE]" in response.text
        summary = auth_store.get_usage_summary(user_id)
        assert len(summary) == 1
        assert summary[0].prompt_tokens == 8
        assert summary[0].completion_tokens == 12
        assert summary[0].total_tokens == 20
        assert upstream.close_calls == 1

    def test_response_without_usage_records_request_count_only(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        user_id, token = _create_user_token(auth_store)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            json={"id": "chatcmpl-1", "choices": []},
        )

        response = client.post(
            "/v1/chat/completions",
            json=_BODY,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        summary = auth_store.get_usage_summary(user_id)
        assert len(summary) == 0


class TestEnforcementEnabled:
    def _enable(self, app: FastAPI, test_settings: Settings) -> Settings:
        enforced = test_settings.model_copy(
            deep=True,
            update={
                "auth": test_settings.auth.model_copy(
                    update={"enforce_api_tokens": True}
                )
            },
        )
        app.dependency_overrides[get_settings] = lambda: enforced
        return enforced

    def test_anonymous_request_rejected(
        self,
        app: FastAPI,
        client: TestClient,
        test_settings: Settings,
    ) -> None:
        self._enable(app, test_settings)

        response = client.post("/v1/chat/completions", json=_BODY)

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"

    def test_valid_token_still_allowed(
        self,
        app: FastAPI,
        client: TestClient,
        auth_store: AuthStore,
        test_settings: Settings,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        _user_id, token = _create_user_token(auth_store)
        self._enable(app, test_settings)
        _add_node(test_registry)
        httpx_mock.add_response(
            url=_CHAT_URL,
            status_code=200,
            json={
                "id": "chatcmpl-1",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

        response = client.post(
            "/v1/chat/completions",
            json=_BODY,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200

    def test_models_list_stays_public_when_enforced(
        self,
        app: FastAPI,
        client: TestClient,
        test_settings: Settings,
    ) -> None:
        self._enable(app, test_settings)

        response = client.get("/v1/models")

        assert response.status_code == 200
