"""Per-hostname log buffer for provisioning live-streaming.

Stores log entries in memory and broadcasts to SSE consumers via
asyncio.Condition. Host count is bounded (dozens), so memory is not
a concern.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TypedDict


class LogEntry(TypedDict):
    ts: str
    level: str
    msg: str
    stream: str | None


class _HostLog:
    __slots__ = ("entries", "condition", "complete")

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self.condition = asyncio.Condition()
        self.complete = False


class ProvisioningLogBuffer:
    """Per-hostname log buffer with async broadcast for SSE consumers."""

    def __init__(self) -> None:
        self._hosts: dict[str, _HostLog] = {}

    def create(self, hostname: str) -> None:
        self._hosts[hostname] = _HostLog()

    def append(
        self,
        hostname: str,
        level: str,
        msg: str,
        *,
        stream: str | None = None,
    ) -> None:
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return
        entry: LogEntry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "msg": msg,
            "stream": stream,
        }
        host_log.entries.append(entry)
        # ponytail: fire-and-forget notify; condition.notify_all needs the lock
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._notify(host_log))
            )
        except RuntimeError:
            pass

    async def _notify(self, host_log: _HostLog) -> None:
        async with host_log.condition:
            host_log.condition.notify_all()

    def mark_complete(self, hostname: str) -> None:
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return
        host_log.complete = True
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._notify(host_log))
            )
        except RuntimeError:
            pass

    def has(self, hostname: str) -> bool:
        return hostname in self._hosts

    def get_entries(self, hostname: str) -> list[LogEntry]:
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return []
        return list(host_log.entries)

    async def iter_from(
        self, hostname: str, pos: int = 0
    ) -> AsyncIterator[tuple[int, LogEntry]]:
        """Yield ``(position, entry)`` starting from *pos*.

        Blocks when caught up until new entries arrive or the log is
        marked complete.
        """
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return

        idx = pos
        while True:
            while idx < len(host_log.entries):
                yield idx, host_log.entries[idx]
                idx += 1

            if host_log.complete:
                return

            async with host_log.condition:
                if idx >= len(host_log.entries) and not host_log.complete:
                    await host_log.condition.wait()
