"""Request-fidelity regressions for OpenAI-compatible payloads."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock, IteratorStream

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus

pytestmark = pytest.mark.httpx_mock(assert_all_responses_were_requested=False)

_MODEL = "llama-3"
_ENDPOINT = "10.0.1.100:8000"


def _register_node(registry: NodeRegistry) -> None:
    registry.add(
        Node(
            node_id="node-1",
            endpoint=_ENDPOINT,
            status=NodeStatus.HEALTHY,
            model=_MODEL,
        )
    )


def _mock_backend(
    httpx_mock: HTTPXMock,
    endpoint_path: str,
    *,
    stream: bool,
) -> None:
    url = f"http://{_ENDPOINT}{endpoint_path}"
    if stream:
        httpx_mock.add_response(
            url=url,
            headers={"content-type": "text/event-stream"},
            stream=IteratorStream([b"data: [DONE]\n\n"]),
        )
        return
    httpx_mock.add_response(url=url, json={"id": "response-1"})


def _forwarded_body(httpx_mock: HTTPXMock) -> dict[str, object]:
    [request] = httpx_mock.get_requests()
    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body


@pytest.mark.parametrize("stream", [False, True])
def test_tool_call_history_reaches_backend_unchanged(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
    stream: bool,
) -> None:
    _register_node(test_registry)
    _mock_backend(httpx_mock, "/v1/chat/completions", stream=stream)
    messages = [
        {
            "role": "user",
            "content": "What is the weather?",
            "name": "operator",
            "vendor_extension": {"trace_id": "trace-123"},
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Raleigh"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "72 F",
            "tool_call_id": "call-weather",
            "name": "weather",
        },
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": messages, "stream": stream},
    )

    assert response.status_code == 200
    forwarded = _forwarded_body(httpx_mock)
    assert forwarded["messages"] == messages
    assistant = forwarded["messages"][1]
    assert "content" in assistant
    assert assistant["content"] is None


@pytest.mark.parametrize("stream", [False, True])
def test_multimodal_content_parts_reach_backend_unchanged(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
    stream: bool,
) -> None:
    _register_node(test_registry)
    _mock_backend(httpx_mock, "/v1/chat/completions", stream=stream)
    content = [
        {"type": "text", "text": "What is shown?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,AA==",
                "detail": "low",
            },
        },
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        },
    )

    assert response.status_code == 200
    forwarded = _forwarded_body(httpx_mock)
    assert forwarded["messages"] == [{"role": "user", "content": content}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prompt",
    [
        [101, 202, 303],
        [[101, 202], [303, 404]],
    ],
)
def test_completion_token_prompt_forms_reach_backend_unchanged(
    client: TestClient,
    test_registry: NodeRegistry,
    httpx_mock: HTTPXMock,
    prompt: list[int] | list[list[int]],
    stream: bool,
) -> None:
    _register_node(test_registry)
    _mock_backend(httpx_mock, "/v1/completions", stream=stream)

    response = client.post(
        "/v1/completions",
        json={"model": _MODEL, "prompt": prompt, "stream": stream},
    )

    assert response.status_code == 200
    forwarded_prompt = _forwarded_body(httpx_mock)["prompt"]
    assert forwarded_prompt == prompt
    if prompt and isinstance(prompt[0], list):
        assert all(
            type(token) is int for sequence in forwarded_prompt for token in sequence
        )
    else:
        assert all(type(token) is int for token in forwarded_prompt)
