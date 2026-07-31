"""Thin wrapper around httpx.AsyncClient for proxying to vLLM backends.

This module is the **sole consumer** of ``httpx`` for proxy operations in
the codebase, following the Dependency Inversion Principle (DIP): all other
modules depend on this wrapper rather than importing ``httpx`` directly for
request forwarding.

The ``ProxyClient`` receives a pre-built ``httpx.AsyncClient`` via constructor
injection.  Lifecycle (creation and cleanup) is managed by the application
lifespan, not by this class.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class ProxyClient:
    """Wrapper around ``httpx.AsyncClient`` for proxying to vLLM backends.

    Exposes ``forward()`` for non-streaming requests and a ``client``
    property for direct access (needed by ``httpx-sse``'s
    ``aconnect_sse`` for streaming operations).

    Attributes:
        client: The underlying ``httpx.AsyncClient`` instance.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the underlying httpx.AsyncClient for SSE streaming."""
        return self._client

    async def forward(
        self,
        method: str,
        url: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        """Forward a non-streaming request to a vLLM backend.

        Args:
            method: HTTP method (e.g., ``"POST"``, ``"GET"``).
            url: Full URL of the target vLLM endpoint.
            body: JSON-serialisable request body.

        Returns:
            The raw ``httpx.Response`` from the backend. Client-error
            responses are returned unchanged.

        Raises:
            httpx.HTTPStatusError: The backend returned a server-error status.
        """
        logger.debug(
            "forwarding request to backend",
            method=method,
            url=url,
        )
        response = await self._client.request(
            method=method,
            url=url,
            json=body,
        )
        if response.status_code >= 500:
            response.raise_for_status()
        return response
