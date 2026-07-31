"""QUADS REST API client.

Thin async wrapper over httpx for fetching GPU host inventory and
availability from a QUADS server.  The httpx.AsyncClient is injected
via the constructor for testability (DIP).

Per D-11: no connectivity check at construction time.
Per D-09: all API errors surface as QUADSConnectionError.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import structlog
from pydantic import ValidationError

from inference_proxy.models.quads import QUADSHost

logger = structlog.get_logger()


class QUADSConnectionError(Exception):
    """Raised when the QUADS API is unreachable or returns an error."""


class _InvalidQUADSResponseError(ValueError):
    """Raised only for response-shape faults detected at the API boundary."""


def canonical_hostname(raw: str) -> str:
    """Normalize a hostname to canonical form for merge-key matching (D-02)."""
    return raw.strip().lower().rstrip(".")


def availability_window_end(
    lookahead_hours: int,
    *,
    now: datetime | None = None,
) -> datetime:
    """Return an absolute UTC deadline for a QUADS availability window."""
    current = datetime.now(tz=UTC) if now is None else now
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("availability window start must be timezone-aware")
    return current.astimezone(UTC) + timedelta(hours=lookahead_hours)


class QUADSClient:
    """Async client for the QUADS REST API.

    Args:
        http_client: A pre-built httpx.AsyncClient (lifecycle managed externally).
        base_url: QUADS server base URL (e.g. ``https://quads.example.com``).
        server_timezone: IANA timezone used by the QUADS server's local clock.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        *,
        server_timezone: str = "UTC",
    ) -> None:
        self._client = http_client
        self._base_url = base_url.rstrip("/")
        self._server_timezone = ZoneInfo(server_timezone)

    async def get_hosts(self) -> list[QUADSHost]:
        """Fetch GPU hosts from QUADS, filtering out broken/retired (D-06).

        Only hosts with at least one ``processor_type == "GPU"`` entry
        are returned (QUADS-03).  GPU vendor and model are taken from
        the first GPU processor entry.
        """
        path = "/api/v3/hosts"
        data = await self._get(path)

        try:
            if not isinstance(data, list):
                raise _InvalidQUADSResponseError("host inventory must be a list")

            hosts: list[QUADSHost] = []
            for raw in data:
                if not isinstance(raw, dict):
                    raise _InvalidQUADSResponseError(
                        "host inventory entries must be objects"
                    )
                if raw.get("broken") or raw.get("retired"):
                    continue

                processors = raw.get("processors", [])
                if not isinstance(processors, list) or not all(
                    isinstance(processor, dict) for processor in processors
                ):
                    raise _InvalidQUADSResponseError(
                        "host processors must be a list of objects"
                    )
                gpus = [
                    processor
                    for processor in processors
                    if processor.get("processor_type") == "GPU"
                ]
                if not gpus:
                    continue

                name = raw.get("name")
                if not isinstance(name, str):
                    raise _InvalidQUADSResponseError("GPU host name must be a string")
                vendor = gpus[0].get("vendor", "")
                product = gpus[0].get("product", "")
                if not isinstance(vendor, str) or not isinstance(product, str):
                    raise _InvalidQUADSResponseError(
                        "GPU vendor and product must be strings"
                    )
                hosts.append(
                    QUADSHost(
                        hostname=canonical_hostname(name),
                        gpu_vendor=vendor,
                        gpu_model=product,
                        gpu_count=len(gpus),
                    )
                )
        except (_InvalidQUADSResponseError, ValidationError) as exc:
            raise QUADSConnectionError(
                f"invalid QUADS response from {path}: {exc}"
            ) from exc

        logger.debug("fetched QUADS hosts", count=len(hosts))
        return hosts

    async def get_available(self, *, end: datetime | None = None) -> list[str]:
        """Fetch available hostnames from QUADS, normalized (D-07/D-08).

        When *end* is given, passes it as a query param so QUADS returns
        only hosts available for the entire window ``[now, end)``.
        """
        params: dict[str, str] = {}
        if end is not None:
            if end.tzinfo is None or end.utcoffset() is None:
                raise ValueError("QUADS availability end must be timezone-aware")
            # QUADS commit bbada78 parses this with a strict, timezone-naive
            # strptime. Convert the absolute deadline into the server's local
            # timezone before intentionally omitting the offset. The protocol
            # cannot disambiguate the repeated hour during a DST fall-back, so
            # deadlines in that hour may differ from the intended instant by
            # one hour until QUADS accepts offset-aware timestamps.
            server_end = end.astimezone(self._server_timezone)
            params["end"] = server_end.strftime("%Y-%m-%dT%H:%M")
        path = "/api/v3/available"
        data = await self._get(path, params=params)
        try:
            if not isinstance(data, list) or not all(
                isinstance(hostname, str) for hostname in data
            ):
                raise _InvalidQUADSResponseError(
                    "available hosts must be a list of strings"
                )
            return [canonical_hostname(hostname) for hostname in data]
        except _InvalidQUADSResponseError as exc:
            raise QUADSConnectionError(
                f"invalid QUADS response from {path}: {exc}"
            ) from exc

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET a JSON endpoint, wrapping errors in QUADSConnectionError."""
        url = f"{self._base_url}{path}"
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QUADSConnectionError(str(exc)) from exc
