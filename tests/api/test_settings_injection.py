"""Regression coverage for application-owned settings injection."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from pytest_httpx import HTTPXMock

import inference_proxy.api.routes as routes_module
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.routing.node_selector import NodeSelector


@pytest.mark.parametrize(
    ("path", "request_body"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "llama-3",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        ),
        (
            "/v1/completions",
            {"model": "llama-3", "prompt": "Hi"},
        ),
    ],
    ids=["chat", "text"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_injected_settings_govern_routes(
    path: str,
    request_body: dict[str, object],
    stream: bool,
    app: FastAPI,
    test_settings: Settings,
    test_registry: NodeRegistry,
    node_selector: NodeSelector,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four completion paths use the app's injected attempt budget."""
    routing = test_settings.routing.model_copy(update={"max_attempts": 2})
    injected_settings = test_settings.model_copy(update={"routing": routing})
    app.dependency_overrides[get_settings] = lambda: injected_settings

    # A restored direct call would see the process-global three-attempt budget,
    # while FastAPI keeps resolving the original dependency through the override.
    monkeypatch.setattr(routes_module, "get_settings", lambda: test_settings)

    nodes = [
        Node(
            node_id=f"node-{index}",
            endpoint=f"10.0.1.{99 + index}:8000",
            status=NodeStatus.HEALTHY,
            model="llama-3",
        )
        for index in range(1, 4)
    ]
    for rank, node in enumerate(nodes):
        test_registry.add(node)
        for _ in range(rank):
            node_selector.tracker.increment(node.node_id)

    attempted_urls = [f"http://{node.endpoint}{path}" for node in nodes[:2]]
    for url in attempted_urls:
        httpx_mock.add_response(
            url=url,
            status_code=503,
            json={"error": {"message": "backend unavailable"}},
            headers={"content-type": "application/json"},
            is_optional=True,
        )
    third_url = f"http://{nodes[2].endpoint}{path}"
    if stream:
        httpx_mock.add_response(
            url=third_url,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
            is_optional=True,
        )
    else:
        httpx_mock.add_response(
            url=third_url,
            json={"id": "unexpected-third-attempt"},
            is_optional=True,
        )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await asyncio.wait_for(
            client.post(path, json={**request_body, "stream": stream}),
            timeout=2,
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "failover_exhausted"
    assert response.headers["X-Inference-Proxy-Failover"] == "exhausted"
    assert response.headers["X-Inference-Proxy-Attempts"] == "2"
    contacted_urls = [str(request.url) for request in httpx_mock.get_requests()]
    assert contacted_urls == attempted_urls
    assert third_url not in contacted_urls
