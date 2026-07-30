"""QUADS background poller with caching.

Periodically fetches host inventory and availability from QUADS,
keeping a cached snapshot for request-path consumers.  On failure
the last good data is retained (D-08).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime

import structlog

from inference_proxy.models.quads import QUADSHost
from inference_proxy.quads.client import QUADSClient

logger = structlog.get_logger()


class QUADSPoller:
    """Polls QUADSClient on an interval, caching results.

    Args:
        client: The QUADS API client (injected, DIP).
        poll_interval: Seconds between polls (default 300).
    """

    def __init__(self, client: QUADSClient, poll_interval: int = 300) -> None:
        self._client = client
        self._interval = poll_interval
        self._hosts: list[QUADSHost] = []
        self._available: list[str] = []
        self._last_sync: datetime | None = None
        self._consecutive_failures: int = 0
        self._task: asyncio.Task[None] | None = None

    # -- read-only properties --

    @property
    def hosts(self) -> list[QUADSHost]:
        return list(self._hosts)

    @property
    def available_hostnames(self) -> list[str]:
        return list(self._available)

    @property
    def last_sync(self) -> datetime | None:
        return self._last_sync

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # -- lifecycle --

    def start(self) -> None:
        """Kick off the background poll loop."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to finish."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    # -- internals --

    async def _poll_loop(self) -> None:
        """Initial poll, then sleep-poll cycle."""
        await self._poll_once()
        while True:
            await asyncio.sleep(self._interval)
            await self._poll_once()

    async def _poll_once(self) -> None:
        """Fetch hosts and availability; on error retain cache."""
        try:
            hosts = await self._client.get_hosts()
            available = await self._client.get_available()
        except Exception:
            self._consecutive_failures += 1
            logger.warning(
                "quads_poll_failed",
                consecutive_failures=self._consecutive_failures,
                exc_info=True,
            )
            return

        self._hosts = hosts
        self._available = available
        self._last_sync = datetime.now(tz=UTC)
        self._consecutive_failures = 0
        logger.debug(
            "quads_poll_success",
            host_count=len(hosts),
            available_count=len(available),
        )
