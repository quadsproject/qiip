"""Host-scoped coordination for provisioning and teardown operations."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass


@dataclass
class _HostLockEntry:
    lock: asyncio.Lock
    users: int = 0


class HostLifecycleLease:
    """Exclusive ownership of one host's lifecycle operation."""

    def __init__(
        self,
        coordinator: HostLifecycleCoordinator,
        hostname: str,
        entry: _HostLockEntry,
    ) -> None:
        self._coordinator = coordinator
        self._hostname = hostname
        self._entry = entry
        self._released = False
        self._release_lock = threading.Lock()

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    def release(self) -> None:
        """Release the host exactly once."""
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._coordinator._release(self._hostname, self._entry)

    def belongs_to(
        self,
        coordinator: HostLifecycleCoordinator,
        hostname: str,
    ) -> bool:
        """Return whether this live lease owns *hostname* for *coordinator*."""
        with self._release_lock:
            return (
                not self._released
                and self._coordinator is coordinator
                and self._hostname == hostname
            )


class HostLifecycleCoordinator:
    """Serialize lifecycle work per host on one asyncio event loop.

    The bookkeeping lock protects synchronous entry/refcount updates, but the
    underlying ``asyncio.Lock`` instances are not safe to use across threads
    or event loops.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _HostLockEntry] = {}
        self._entries_lock = threading.Lock()

    async def acquire(self, hostname: str) -> HostLifecycleLease:
        """Wait for exclusive lifecycle ownership of *hostname*."""
        with self._entries_lock:
            entry = self._entries.get(hostname)
            if entry is None:
                entry = _HostLockEntry(asyncio.Lock())
                self._entries[hostname] = entry
            entry.users += 1

        try:
            await entry.lock.acquire()
        except BaseException:
            self._drop_user(hostname, entry)
            raise
        return HostLifecycleLease(self, hostname, entry)

    async def try_acquire(self, hostname: str) -> HostLifecycleLease | None:
        """Acquire *hostname* immediately, or return ``None`` when reserved."""
        with self._entries_lock:
            if hostname in self._entries:
                return None
            entry = _HostLockEntry(asyncio.Lock(), users=1)
            self._entries[hostname] = entry

        # This cannot block: the entry was created privately above and is
        # published as busy before another caller can observe it.
        try:
            await entry.lock.acquire()
        except BaseException:
            self._drop_user(hostname, entry)
            raise
        return HostLifecycleLease(self, hostname, entry)

    def is_busy(self, hostname: str) -> bool:
        """Return whether a holder or waiter exists for *hostname*."""
        with self._entries_lock:
            return hostname in self._entries

    def _release(self, hostname: str, entry: _HostLockEntry) -> None:
        entry.lock.release()
        self._drop_user(hostname, entry)

    def _drop_user(self, hostname: str, entry: _HostLockEntry) -> None:
        with self._entries_lock:
            current = self._entries.get(hostname)
            if current is not entry:
                return
            entry.users -= 1
            if entry.users == 0:
                self._entries.pop(hostname, None)
