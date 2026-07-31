"""Unit tests for the proxy error mapping functions.

Tests cover mapping of httpx exception types to OpenAI-compatible error
responses and the no-nodes-available helper.
"""

from __future__ import annotations

import httpx
import pytest

from inference_proxy.api.errors import (
    map_proxy_error,
    model_not_found_error,
    model_unavailable_error,
    no_nodes_error,
)


class TestMapProxyError:
    """map_proxy_error maps exceptions to (status_code, ErrorResponse) tuples."""

    def test_connect_error_returns_502(self) -> None:
        exc = httpx.ConnectError("Connection refused")

        status, response = map_proxy_error(exc)

        assert status == 502
        assert response.error.type == "upstream_error"
        assert response.error.code == "backend_unavailable"
        assert "connect" in response.error.message.lower()

    def test_timeout_returns_504(self) -> None:
        exc = httpx.ReadTimeout("read timed out")

        status, response = map_proxy_error(exc)

        assert status == 504
        assert response.error.type == "upstream_error"
        assert response.error.code == "backend_timeout"
        assert "timed out" in response.error.message.lower()

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ReadError("response ended early"),
            httpx.RemoteProtocolError("malformed response"),
        ],
        ids=["read_error", "remote_protocol_error"],
    )
    def test_transport_error_returns_502(self, exc: httpx.TransportError) -> None:
        status, response = map_proxy_error(exc)

        assert status == 502
        assert response.error.type == "upstream_error"
        assert response.error.code == "backend_transport_error"

    def test_http_status_error_returns_upstream_status(self) -> None:
        mock_request = httpx.Request("POST", "http://node1:8000/v1/chat/completions")
        mock_response = httpx.Response(
            status_code=422,
            text="Unprocessable Entity",
            request=mock_request,
        )
        exc = httpx.HTTPStatusError(
            "422 Unprocessable Entity",
            request=mock_request,
            response=mock_response,
        )

        status, response = map_proxy_error(exc)

        assert status == 422
        assert response.error.type == "upstream_error"
        assert response.error.code == "422"
        assert "Unprocessable Entity" in response.error.message

    def test_generic_exception_returns_500(self) -> None:
        exc = RuntimeError("something went wrong")

        status, response = map_proxy_error(exc)

        assert status == 500
        assert response.error.type == "server_error"
        assert response.error.code == "internal_error"
        assert "internal" in response.error.message.lower()


class TestNoNodesError:
    """no_nodes_error returns a 503 with no_nodes code."""

    def test_returns_503(self) -> None:
        status, response = no_nodes_error()

        assert status == 503
        assert response.error.code == "no_nodes"
        assert "No inference nodes available" in response.error.message
        assert response.error.type == "server_error"


class TestModelNotFoundError:
    """model_not_found_error returns a 404 with model_not_found code."""

    def test_returns_404(self) -> None:
        status, response = model_not_found_error("llama-3")

        assert status == 404
        assert response.error.code == "model_not_found"
        assert response.error.type == "invalid_request_error"
        assert "llama-3" in response.error.message


class TestModelUnavailableError:
    """model_unavailable_error returns a 503 with model_unavailable code."""

    def test_returns_503(self) -> None:
        status, response = model_unavailable_error("llama-3")

        assert status == 503
        assert response.error.code == "model_unavailable"
        assert response.error.type == "server_error"
        assert "llama-3" in response.error.message
