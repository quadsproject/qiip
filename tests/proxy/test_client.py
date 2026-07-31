"""Unit tests for the ProxyClient wrapper around httpx.AsyncClient.

Tests cover forward() request delegation, response pass-through,
timeout propagation, and the client property accessor.
"""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from inference_proxy.proxy.client import ProxyClient


class TestProxyClientForward:
    """forward() delegates requests to the underlying httpx.AsyncClient."""

    async def test_forward_sends_json_body(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url="http://node1:8000/v1/chat/completions",
            json={"id": "cmpl-1", "choices": []},
        )
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            await proxy.forward(
                "POST",
                "http://node1:8000/v1/chat/completions",
                {"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
            )

        request = httpx_mock.get_request()
        assert request is not None
        assert request.method == "POST"
        assert str(request.url) == "http://node1:8000/v1/chat/completions"
        body = request.read()
        assert b'"model"' in body
        assert b'"llama"' in body

    async def test_forward_returns_response(self, httpx_mock: HTTPXMock) -> None:
        expected_json = {"id": "cmpl-1", "object": "chat.completion", "choices": []}
        httpx_mock.add_response(
            url="http://node1:8000/v1/chat/completions",
            json=expected_json,
            status_code=200,
        )
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            response = await proxy.forward(
                "POST",
                "http://node1:8000/v1/chat/completions",
                {"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 200
        assert response.json() == expected_json

    async def test_forward_propagates_timeout(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_exception(
            httpx.ReadTimeout("read timed out"),
            url="http://node1:8000/v1/chat/completions",
        )
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            with pytest.raises(httpx.TimeoutException):
                await proxy.forward(
                    "POST",
                    "http://node1:8000/v1/chat/completions",
                    {"model": "llama", "messages": []},
                )

    async def test_forward_raises_for_error_status(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            url="http://node1:8000/v1/chat/completions",
            status_code=500,
            text="backend failed",
        )
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            with pytest.raises(httpx.HTTPStatusError) as caught:
                await proxy.forward(
                    "POST",
                    "http://node1:8000/v1/chat/completions",
                    {"model": "llama", "messages": []},
                )

        assert caught.value.response.status_code == 500

    async def test_forward_returns_client_error_response(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        expected = {
            "error": {
                "message": "context window exceeded",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        httpx_mock.add_response(
            url="http://node1:8000/v1/chat/completions",
            status_code=400,
            json=expected,
        )
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            response = await proxy.forward(
                "POST",
                "http://node1:8000/v1/chat/completions",
                {"model": "llama", "messages": []},
            )

        assert response.status_code == 400
        assert response.json() == expected


class TestProxyClientProperty:
    """client property returns the underlying httpx.AsyncClient."""

    async def test_client_property_returns_underlying_client(self) -> None:
        async with httpx.AsyncClient() as http_client:
            proxy = ProxyClient(http_client)

            assert proxy.client is http_client
