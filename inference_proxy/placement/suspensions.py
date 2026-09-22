"""Persistent operator opt-outs, independent of nodes and placement claims.

Presence of a key blocks automatic placement, even if its value is unreadable.
No lease is attached: restarting a gateway or deleting a node cannot erase
operator intent. Claims and their reset action never change this namespace.
"""

from __future__ import annotations

import asyncio

from inference_proxy.placement.claims import ClaimEtcd
from inference_proxy.quads.client import canonical_hostname

SUSPENSION_PREFIX = "/placement/suspensions/"


class SuspensionStore:
    def __init__(self, etcd: ClaimEtcd) -> None:
        self._etcd = etcd

    async def list(self) -> set[str]:
        snapshot = await asyncio.to_thread(self._etcd.get_snapshot, SUSPENSION_PREFIX)
        return {
            record.key.decode("utf-8").removeprefix(SUSPENSION_PREFIX)
            for record in snapshot.records
        }

    async def suspend(self, hostname: str) -> None:
        key = SUSPENSION_PREFIX + canonical_hostname(hostname)
        snapshot = await asyncio.to_thread(self._etcd.get_snapshot, key)
        record = next(
            (r for r in snapshot.records if r.key.decode("utf-8") == key), None
        )
        # Renew the revision even when already suspended: a concurrent resume
        # must not erase a newer teardown's intent.
        revision = await asyncio.to_thread(
            self._etcd.replace_if_revision,
            key,
            '{"reason":"manual_teardown"}',
            expected_mod_revision=record.mod_revision if record else 0,
            lease_id=0,
        )
        if revision is None:
            raise ValueError("Suspension changed concurrently; retry teardown")

    async def resume(self, hostname: str) -> None:
        key = SUSPENSION_PREFIX + canonical_hostname(hostname)
        snapshot = await asyncio.to_thread(self._etcd.get_snapshot, key)
        record = next(
            (r for r in snapshot.records if r.key.decode("utf-8") == key), None
        )
        if record is None:
            return  # Repeated resume is harmless.
        removed = await asyncio.to_thread(
            self._etcd.delete_if_revision,
            key,
            expected_mod_revision=record.mod_revision,
        )
        if not removed:
            raise ValueError("Suspension changed concurrently; refresh and retry")
