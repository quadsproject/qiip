"""Behavioral tests for routing reservations, draining, and route metrics."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from pytest_httpx import HTTPXMock, IteratorStream

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.routing.node_selector import NodeSelector

_WAIT_TIMEOUT = 2.0


def _make_node(
    node_id: str,
    endpoint: str,
    *,
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
) -> Node:
    return Node(
        node_id=node_id,
        endpoint=endpoint,
        status=status,
        model=model,
    )


def _request_body(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "llama-3",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
    }


def _completion_response() -> dict[str, Any]:
    return {
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
    }


def _sse_chunks() -> list[bytes]:
    return [
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]


def _start_request(
    client: TestClient,
    *,
    stream: bool,
) -> tuple[threading.Thread, threading.Event, list[Exception], list[Any]]:
    finished = threading.Event()
    errors: list[Exception] = []
    responses: list[Any] = []

    def send_request() -> None:
        try:
            responses.append(
                client.post("/v1/chat/completions", json=_request_body(stream=stream))
            )
        except Exception as exc:  # pragma: no cover - asserted by caller
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=send_request, daemon=True)
    thread.start()
    return thread, finished, errors, responses


class _BlockingSSEStream(httpx.AsyncByteStream):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._started.set()
        released = await asyncio.to_thread(self._release.wait, _WAIT_TIMEOUT)
        assert released, "timed out waiting to release the upstream SSE stream"
        for chunk in _sse_chunks():
            yield chunk


class _PausingAfterFirstSSEStream(httpx.AsyncByteStream):
    def __init__(self, waiting_for_next_chunk: asyncio.Event) -> None:
        self._waiting_for_next_chunk = waiting_for_next_chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _sse_chunks()[0]
        self._waiting_for_next_chunk.set()
        await asyncio.Event().wait()


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_drain_between_select_and_increment_does_not_strand_request(
    stream: bool,
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    proxy_client: ProxyClient,
    httpx_mock: HTTPXMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """A drain cannot observe zero after selection but before reservation."""
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)

    increment_entered = threading.Event()
    allow_increment = threading.Event()
    backend_started = threading.Event()
    allow_backend = threading.Event()
    original_increment = node_selector.tracker.increment

    def blocking_increment(node_id: str) -> None:
        increment_entered.set()
        assert allow_increment.wait(_WAIT_TIMEOUT)
        original_increment(node_id)

    monkeypatch.setattr(node_selector.tracker, "increment", blocking_increment)

    if stream:
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=_BlockingSSEStream(backend_started, allow_backend),
        )
    else:
        original_forward = proxy_client.forward

        async def blocking_forward(
            method: str,
            url: str,
            body: dict[str, Any],
        ) -> httpx.Response:
            backend_started.set()
            released = await asyncio.to_thread(allow_backend.wait, _WAIT_TIMEOUT)
            assert released, "timed out waiting to release backend forwarding"
            return await original_forward(method, url, body)

        monkeypatch.setattr(proxy_client, "forward", blocking_forward)
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json=_completion_response(),
        )

    request_thread, request_finished, errors, responses = _start_request(
        client,
        stream=stream,
    )
    assert increment_entered.wait(_WAIT_TIMEOUT)

    drain_started = threading.Event()
    drain_finished = threading.Event()
    drain_observation: list[tuple[bool, int]] = []

    def drain_node() -> None:
        drain_started.set()
        drained = test_registry.drain(node.node_id)
        drain_observation.append((drained, node_selector.tracker.get(node.node_id)))
        drain_finished.set()

    drain_thread = threading.Thread(target=drain_node, daemon=True)
    drain_thread.start()
    try:
        assert drain_started.wait(_WAIT_TIMEOUT)
        assert not drain_finished.wait(0.1)

        allow_increment.set()
        assert backend_started.wait(_WAIT_TIMEOUT)
        assert drain_finished.wait(_WAIT_TIMEOUT)
        assert drain_observation == [(True, 1)]

        allow_backend.set()
        assert request_finished.wait(_WAIT_TIMEOUT)
    finally:
        allow_increment.set()
        allow_backend.set()
        request_finished.wait(_WAIT_TIMEOUT)
        request_thread.join(timeout=_WAIT_TIMEOUT)
        drain_thread.join(timeout=_WAIT_TIMEOUT)

    assert not request_thread.is_alive()
    assert not drain_thread.is_alive()
    assert errors == []
    assert len(responses) == 1
    assert responses[0].status_code == 200
    assert node_selector.tracker.get(node.node_id) == 0


def test_connection_is_reserved_while_backend_call_is_in_flight(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    proxy_client: ProxyClient,
    monkeypatch: MonkeyPatch,
) -> None:
    """Reservation precedes the actual non-streaming backend call."""
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    forward_started = threading.Event()
    allow_forward = threading.Event()

    async def blocking_forward(
        method: str,
        url: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        del method, url, body
        forward_started.set()
        released = await asyncio.to_thread(allow_forward.wait, _WAIT_TIMEOUT)
        assert released, "timed out waiting to release backend forwarding"
        return httpx.Response(
            200,
            json=_completion_response(),
            request=httpx.Request("POST", "http://backend/v1/chat/completions"),
        )

    monkeypatch.setattr(proxy_client, "forward", blocking_forward)
    request_thread, request_finished, errors, responses = _start_request(
        client,
        stream=False,
    )

    assert forward_started.wait(_WAIT_TIMEOUT)
    assert node_selector.tracker.get(node.node_id) == 1

    allow_forward.set()
    assert request_finished.wait(_WAIT_TIMEOUT)
    request_thread.join(timeout=_WAIT_TIMEOUT)
    assert not request_thread.is_alive()
    assert errors == []
    assert responses[0].status_code == 200
    assert node_selector.tracker.get(node.node_id) == 0


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_drain_cleanup_does_not_remove_reregistered_node(
    stream: bool,
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Both request finalizers preserve a registration replacing stale DRAINING."""
    active = _make_node("node-1", "10.0.1.100:8000")
    stale = _make_node(
        "node-2",
        "10.0.1.101:8000",
        status=NodeStatus.DRAINING,
        model="old-model",
    )
    fresh = _make_node("node-2", "10.0.1.202:9000", model="new-model")
    test_registry.add(active)
    test_registry.add(stale)

    if stream:
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(_sse_chunks()),
        )
    else:
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json=_completion_response(),
        )

    stale_observed = threading.Event()
    watcher_started = threading.Event()
    watcher_finished = threading.Event()
    original_get = node_selector.tracker.get

    def coordinated_get(node_id: str) -> int:
        if node_id == stale.node_id and not stale_observed.is_set():
            stale_observed.set()
            assert watcher_started.wait(_WAIT_TIMEOUT)
            watcher_finished.wait(0.1)
        return original_get(node_id)

    monkeypatch.setattr(node_selector.tracker, "get", coordinated_get)

    def reregister_node() -> None:
        assert stale_observed.wait(_WAIT_TIMEOUT)
        watcher_started.set()
        test_registry.add(fresh)
        watcher_finished.set()

    watcher_thread = threading.Thread(target=reregister_node, daemon=True)
    watcher_thread.start()

    response = client.post(
        "/v1/chat/completions",
        json=_request_body(stream=stream),
    )

    assert response.status_code == 200
    assert watcher_finished.wait(_WAIT_TIMEOUT)
    watcher_thread.join(timeout=_WAIT_TIMEOUT)
    assert not watcher_thread.is_alive()
    current = test_registry.get(fresh.node_id)
    assert current is not None
    assert current.status == NodeStatus.HEALTHY
    assert current.endpoint == fresh.endpoint
    assert current.model == fresh.model


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_drain_cleanup_preserves_in_flight_node(
    stream: bool,
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    """Neither request finalizer removes a DRAINING node with active work."""
    active = _make_node("node-1", "10.0.1.100:8000")
    draining = _make_node(
        "node-2",
        "10.0.1.101:8000",
        status=NodeStatus.DRAINING,
    )
    test_registry.add(active)
    test_registry.add(draining)
    node_selector.tracker.increment(draining.node_id)

    if stream:
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream(_sse_chunks()),
        )
    else:
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json=_completion_response(),
        )

    response = client.post(
        "/v1/chat/completions",
        json=_request_body(stream=stream),
    )

    assert response.status_code == 200
    current = test_registry.get(draining.node_id)
    assert current is not None
    assert current.status == NodeStatus.DRAINING
    assert node_selector.tracker.get(draining.node_id) == 1


async def _drive_disconnect(
    app: FastAPI,
    *,
    after_first_event: bool,
    generator_waiting: asyncio.Event | None = None,
) -> list[dict[str, Any]]:
    """Drive the actual FastAPI router with a deterministic disconnect."""
    body = json.dumps(_request_body(stream=True)).encode()
    request_sent = False
    response_started = asyncio.Event()
    first_event_sent = asyncio.Event()
    disconnect_returned = asyncio.Event()
    hold_send_until_cancelled = asyncio.Event()
    sent_messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        if after_first_event:
            assert generator_waiting is not None
            await first_event_sent.wait()
            await generator_waiting.wait()
        else:
            await response_started.wait()
        disconnect_returned.set()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent_messages.append(message)
        if message["type"] == "http.response.start":
            response_started.set()
            if not after_first_event:
                await disconnect_returned.wait()
                await hold_send_until_cancelled.wait()
        elif (
            after_first_event
            and message["type"] == "http.response.body"
            and message.get("body")
        ):
            first_event_sent.set()

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
        "state": {},
        "app": app,
    }
    # Bypass BaseHTTPMiddleware, which prefetches the first response chunk and
    # therefore cannot reproduce a disconnect before generator iteration.
    async with AsyncExitStack() as stack:
        scope["fastapi_middleware_astack"] = stack
        await app.router(scope, receive, send)
    return sent_messages


@pytest.mark.asyncio
async def test_stream_disconnect_before_first_event_releases_reservation(
    app: FastAPI,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
) -> None:
    """A pre-iteration disconnect cannot leak its route reservation."""
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)

    messages = await asyncio.wait_for(
        _drive_disconnect(app, after_first_event=False),
        timeout=_WAIT_TIMEOUT,
    )

    assert any(message["type"] == "http.response.start" for message in messages)
    assert node_selector.tracker.get(node.node_id) == 0


@pytest.mark.asyncio
async def test_stream_disconnect_after_first_event_balances_reservation_once(
    app: FastAPI,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """The normal started-stream disconnect path decrements exactly once."""
    node = _make_node("node-1", "10.0.1.100:8000")
    test_registry.add(node)
    generator_waiting = asyncio.Event()
    httpx_mock.add_response(
        url="http://10.0.1.100:8000/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        stream=_PausingAfterFirstSSEStream(generator_waiting),
    )

    decrement_calls = 0
    decremented = asyncio.Event()
    original_decrement = node_selector.tracker.decrement

    def counting_decrement(node_id: str) -> None:
        nonlocal decrement_calls
        decrement_calls += 1
        original_decrement(node_id)
        decremented.set()

    monkeypatch.setattr(node_selector.tracker, "decrement", counting_decrement)

    messages = await asyncio.wait_for(
        _drive_disconnect(
            app,
            after_first_event=True,
            generator_waiting=generator_waiting,
        ),
        timeout=_WAIT_TIMEOUT,
    )

    assert any(
        message["type"] == "http.response.body" and message.get("body")
        for message in messages
    )
    await asyncio.wait_for(decremented.wait(), timeout=_WAIT_TIMEOUT)
    assert decrement_calls == 1
    assert node_selector.tracker.get(node.node_id) == 0


def test_non_streaming_route_updates_metrics(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    """The non-streaming route records its request before returning."""
    test_registry.add(_make_node("node-1", "10.0.1.100:8000"))
    httpx_mock.add_response(
        url="http://10.0.1.100:8000/v1/chat/completions",
        json=_completion_response(),
    )

    response = client.post("/v1/chat/completions", json=_request_body())
    metrics = client.get("/admin/metrics").json()

    assert response.status_code == 200
    assert metrics == {
        "total_requests": 1,
        "per_model": {"llama-3": 1},
        "per_node": {"node-1": 1},
    }


def test_streaming_route_updates_metrics(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
) -> None:
    """The streaming route records through its distinct metrics call site."""
    test_registry.add(_make_node("node-1", "10.0.1.100:8000"))
    httpx_mock.add_response(
        url="http://10.0.1.100:8000/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        stream=IteratorStream(_sse_chunks()),
    )

    response = client.post(
        "/v1/chat/completions",
        json=_request_body(stream=True),
    )
    metrics = client.get("/admin/metrics").json()

    assert response.status_code == 200
    assert metrics == {
        "total_requests": 1,
        "per_model": {"llama-3": 1},
        "per_node": {"node-1": 1},
    }


def test_retry_attempts_update_per_node_metrics(
    client: TestClient,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
) -> None:
    """A logical request is counted once while every node attempt is visible."""
    first = _make_node("node-1", "10.0.1.100:8000")
    second = _make_node("node-2", "10.0.1.101:8000")
    test_registry.add(first)
    test_registry.add(second)
    node_selector.tracker.increment(second.node_id)
    httpx_mock.add_exception(
        httpx.ConnectError("connection refused"),
        url="http://10.0.1.100:8000/v1/chat/completions",
    )
    httpx_mock.add_response(
        url="http://10.0.1.101:8000/v1/chat/completions",
        json=_completion_response(),
    )

    response = client.post("/v1/chat/completions", json=_request_body())
    metrics = client.get("/admin/metrics").json()

    assert response.status_code == 200
    assert metrics == {
        "total_requests": 1,
        "per_model": {"llama-3": 1},
        "per_node": {"node-1": 1, "node-2": 1},
    }
