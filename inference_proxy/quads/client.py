"""QUADS REST API client.

Thin async wrapper over httpx for fetching GPU host inventory and
availability from a QUADS server.  The httpx.AsyncClient is injected
via the constructor for testability (DIP).

Per D-11: no connectivity check at construction time.
Per D-09: all API errors surface as QUADSConnectionError.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import structlog

from inference_proxy.models.quads import QUADSHost

logger = structlog.get_logger()


class QUADSConnectionError(Exception):
    """Raised when the QUADS API is unreachable or returns an error."""


def canonical_hostname(raw: str) -> str:
    """Normalize a hostname to canonical form for merge-key matching (D-02)."""
    return raw.strip().lower().rstrip(".")


class QUADSClient:
    """Async client for the QUADS REST API.

    Args:
        http_client: A pre-built httpx.AsyncClient (lifecycle managed externally).
        base_url: QUADS server base URL (e.g. ``https://quads.example.com``).
    """

    def __init__(self, http_client: httpx.AsyncClient, base_url: str) -> None:
        self._client = http_client
        self._base_url = base_url.rstrip("/")

    async def get_hosts(self) -> list[QUADSHost]:
        """Fetch GPU hosts from QUADS, filtering out broken/retired (D-06).

        Only hosts with at least one ``processor_type == "GPU"`` entry
        are returned (QUADS-03).  GPU vendor and model are taken from
        the first GPU processor entry.
        """
        data = await self._get("/api/v3/hosts")

        hosts: list[QUADSHost] = []
        for raw in data:
            if raw.get("broken") or raw.get("retired"):
                continue
            gpus = [
                p for p in raw.get("processors", []) if p.get("processor_type") == "GPU"
            ]
            if not gpus:
                continue
            hosts.append(
                QUADSHost(
                    hostname=canonical_hostname(raw["name"]),
                    gpu_vendor=gpus[0].get("vendor", ""),
                    gpu_model=gpus[0].get("product", ""),
                    gpu_count=len(gpus),
                )
            )
        logger.debug("fetched QUADS hosts", count=len(hosts))
        return hosts

    async def get_available(self, *, end: datetime | None = None) -> list[str]:
        """Fetch available hostnames from QUADS, normalized (D-07/D-08).

        When *end* is given, passes it as a query param so QUADS returns
        only hosts available for the entire window ``[now, end)``.
        """
        params: dict[str, str] = {}
        if end is not None:
            params["end"] = end.strftime("%Y-%m-%dT%H:%M")
        data = await self._get("/api/v3/available", params=params)
        return [canonical_hostname(h) for h in data]

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET a JSON endpoint, wrapping errors in QUADSConnectionError."""
        url = f"{self._base_url}{path}"
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise QUADSConnectionError(str(exc)) from exc
        return resp.json()
