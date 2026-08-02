"""Bounded per-host log buffers for provisioning live-streaming.

Each host owns one operation generation. Replacing or evicting a generation
closes and notifies it before removing the mapping, so consumers that already
captured the object always terminate cleanly. Retention uses absolute entry
positions so a slow SSE consumer can detect a discarded prefix and resume at
the oldest retained entry without hanging.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

_TRUNCATION_MARKER = b" ... [truncated]"


class LogEntry(TypedDict):
    ts: str
    level: str
    msg: str
    stream: str | None


@dataclass(frozen=True, slots=True)
class _StoredEntry:
    position: int
    entry: LogEntry
    message_bytes: int


class _HostLog:
    __slots__ = (
        "complete",
        "completed_sequence",
        "condition",
        "entries",
        "message_bytes",
        "next_position",
    )

    def __init__(self) -> None:
        self.entries: deque[_StoredEntry] = deque()
        self.condition = asyncio.Condition()
        self.complete = False
        self.completed_sequence: int | None = None
        self.message_bytes = 0
        self.next_position = 0


class ProvisioningLogBuffer:
    """Bounded per-host log storage with async broadcast for SSE consumers."""

    def __init__(
        self,
        *,
        max_entries_per_host: int = 1_000,
        max_bytes_per_host: int = 1_048_576,
        max_entry_bytes: int = 16_384,
        max_completed_hosts: int = 64,
    ) -> None:
        if max_entries_per_host < 1:
            raise ValueError("max_entries_per_host must be at least 1")
        if max_bytes_per_host < 1:
            raise ValueError("max_bytes_per_host must be at least 1")
        if (
            max_entry_bytes < len(_TRUNCATION_MARKER)
            or max_entry_bytes > max_bytes_per_host
        ):
            raise ValueError(
                "max_entry_bytes must fit the truncation marker and not exceed "
                "max_bytes_per_host"
            )
        if max_completed_hosts < 1:
            raise ValueError("max_completed_hosts must be at least 1")

        self._max_entries_per_host = max_entries_per_host
        self._max_bytes_per_host = max_bytes_per_host
        self._max_entry_bytes = max_entry_bytes
        self._max_completed_hosts = max_completed_hosts
        self._hosts: dict[str, _HostLog] = {}
        self._completion_sequence = 0

    def create(self, hostname: str) -> None:
        """Start a fresh operation generation for *hostname*.

        Host lifecycle leases serialize production operations. Closing an
        unfinished displaced generation is a defensive component-level
        guarantee for direct callers and future refactors.
        """
        previous = self._hosts.pop(hostname, None)
        if previous is not None:
            self._close(previous)
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

        bounded_msg = self._truncate_message(msg)
        message_bytes = len(bounded_msg.encode("utf-8"))
        entry: LogEntry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "msg": bounded_msg,
            "stream": stream,
        }
        host_log.entries.append(
            _StoredEntry(
                position=host_log.next_position,
                entry=entry,
                message_bytes=message_bytes,
            )
        )
        host_log.next_position += 1
        host_log.message_bytes += message_bytes
        self._trim(host_log)
        self._schedule_notify(host_log)

    def _truncate_message(self, msg: str) -> str:
        encoded = msg.encode("utf-8")
        if len(encoded) <= self._max_entry_bytes:
            return msg

        available = self._max_entry_bytes - len(_TRUNCATION_MARKER)
        prefix = encoded[:available].decode("utf-8", errors="ignore")
        return f"{prefix}{_TRUNCATION_MARKER.decode()}"

    def _trim(self, host_log: _HostLog) -> None:
        while host_log.entries and (
            len(host_log.entries) > self._max_entries_per_host
            or host_log.message_bytes > self._max_bytes_per_host
        ):
            removed = host_log.entries.popleft()
            host_log.message_bytes -= removed.message_bytes

    def _schedule_notify(self, host_log: _HostLog) -> None:
        # condition.notify_all() needs the lock. Capture the generation object,
        # not its hostname, so replacement and eviction wake existing readers.
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

    def _close(self, host_log: _HostLog) -> None:
        if not host_log.complete:
            host_log.complete = True
            self._completion_sequence += 1
            host_log.completed_sequence = self._completion_sequence
        self._schedule_notify(host_log)

    def mark_complete(self, hostname: str) -> None:
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return
        self._close(host_log)
        self._evict_completed()

    def _evict_completed(self) -> None:
        completed = [
            (host_log.completed_sequence, hostname, host_log)
            for hostname, host_log in self._hosts.items()
            if host_log.completed_sequence is not None
        ]
        completed.sort(key=lambda item: item[0])
        for _sequence, hostname, host_log in completed[: -self._max_completed_hosts]:
            # Existing iterators retain the object. Reusing the same close and
            # notify path as replacement guarantees those readers terminate.
            self._close(host_log)
            if self._hosts.get(hostname) is host_log:
                self._hosts.pop(hostname)

    def has(self, hostname: str) -> bool:
        return hostname in self._hosts

    def get_entries(self, hostname: str) -> list[LogEntry]:
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return []
        return [stored.entry for stored in host_log.entries]

    @staticmethod
    def _gap_entry(count: int) -> LogEntry:
        noun = "entry" if count == 1 else "entries"
        return {
            "ts": datetime.now(UTC).isoformat(),
            "level": "warning",
            "msg": f"{count} earlier log {noun} evicted by retention limits",
            "stream": None,
        }

    async def iter_from(
        self, hostname: str, pos: int = 0
    ) -> AsyncIterator[tuple[int, LogEntry]]:
        """Yield retained ``(position, entry)`` pairs starting from *pos*.

        A synthetic warning makes any retention gap visible before replay
        resumes. Positions are absolute within an operation generation, so
        prefix eviction cannot strand a live iterator on a shifted list index.
        """
        host_log = self._hosts.get(hostname)
        if host_log is None:
            return

        cursor = max(0, pos)
        while True:
            if host_log.entries:
                first_position = host_log.entries[0].position
                if cursor < first_position:
                    dropped = first_position - cursor
                    yield first_position - 1, self._gap_entry(dropped)
                    cursor = first_position

                offset = cursor - first_position
                if 0 <= offset < len(host_log.entries):
                    stored = host_log.entries[offset]
                    yield stored.position, stored.entry
                    cursor = stored.position + 1
                    continue

            if host_log.complete:
                return

            async with host_log.condition:
                first_position = (
                    host_log.entries[0].position
                    if host_log.entries
                    else host_log.next_position
                )
                caught_up = cursor >= host_log.next_position
                fell_behind = cursor < first_position
                if caught_up and not fell_behind and not host_log.complete:
                    await host_log.condition.wait()
