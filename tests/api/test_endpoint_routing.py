"""Behavioral coverage for canonical backend URLs in proxy routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from types import TracebackType
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from inference_proxy.config.dependencies import get_proxy_client
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus

_REQUEST_TIMEOUT = 2.0


class _RecordingProxy:
    def __init__(self) -> None:
        self.client = object()
        self.urls: list[str] = []
        self.url_seen = asyncio.Event()

    async def forward(
        self,
        _method: str,
        url: str,
        _json_body: dict[str, object],
    ) -> httpx.Response:
        self.urls.append(url)
        self.url_seen.set()
        return httpx.Response(
            200,
            json={"id": "chatcmpl-1"},
            request=httpx.Request("POST", "http://stub.test"),
        )


class _SSE:
    data = "[DONE]"


class _FakeEventSource:
    def __init__(self, _url: str) -> None:
        self.response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://stub.test"),
        )

    async def aiter_sse(self) -> AsyncIterator[_SSE]:
        yield _SSE()


class _FakeSSEContext:
    def __init__(self, url: str) -> None:
        self._source = _FakeEventSource(url)

    async def __aenter__(self) -> _FakeEventSource:
        return self._source

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.parametrize(
    ("endpoint", "expected_origin"),
    [
        ("10.0.1.100:8000", "http://10.0.1.100:8000"),
        ("http://10.0.1.100:8000", "http://10.0.1.100:8000"),
        ("https://gpu01.example.com:8443", "https://gpu01.example.com:8443"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_proxy_endpoint_normalization_matrix(
    endpoint: str,
    expected_origin: str,
    stream: bool,
    app: FastAPI,
    test_registry: NodeRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both proxy paths contact the normalized origin without double schemes."""
    test_registry.add(
        Node(
            node_id="node-1",
            endpoint=endpoint,
            status=NodeStatus.HEALTHY,
            model="llama-3",
        )
    )
    expected_url = f"{expected_origin}/v1/chat/completions"
    proxy = _RecordingProxy()
    app.dependency_overrides[get_proxy_client] = lambda: proxy

    def connect_sse(
        _client: object,
        _method: str,
        url: str,
        **_kwargs: Any,
    ) -> _FakeSSEContext:
        proxy.urls.append(url)
        proxy.url_seen.set()
        return _FakeSSEContext(url)

    monkeypatch.setattr("inference_proxy.api.routes.aconnect_sse", connect_sse)

    gateway = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.test",
    )
    request_task = asyncio.create_task(
        gateway.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": stream,
            },
        )
    )
    try:
        await asyncio.wait_for(
            proxy.url_seen.wait(),
            timeout=_REQUEST_TIMEOUT,
        )
        assert proxy.urls == [expected_url]
    finally:
        if not request_task.done():
            request_task.cancel()
        with suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(request_task, timeout=0.2)
        await gateway.aclose()
