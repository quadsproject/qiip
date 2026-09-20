"""Persistent, compare-and-swap placement claims.

One claim per host, at ``/placement/claims/<hostname>``, records that automatic
placement owns that host and which profile it chose. Claims are deliberately
**not** bound to an etcd lease: a managed node's own key is leased and can
expire while its server keeps running, and a claim that vanished with it would
let the next reconcile pass assign the host a second time.

Every write is conditional on the revision that was read. A writer that loses
the race learns it immediately and must stop acting on the host.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from inference_proxy.discovery.etcd_client import EtcdSnapshot

CLAIM_PREFIX = "/placement/claims/"


class ClaimState(StrEnum):
    # A gateway is provisioning the host and refreshing ``heartbeat_at``.
    PROVISIONING = "provisioning"
    # The node registered healthy with this claim's id.
    ACTIVE = "active"
    # The last attempt failed; retried after ``retry_at`` while attempts remain.
    FAILED = "failed"
    # Attempts are used up. An operator resets the claim to try again.
    EXHAUSTED = "exhausted"


class PlacementClaim(BaseModel):
    """Automation's durable record of one host it placed a profile on."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    claim_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    hostname: str = Field(min_length=1)
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    profile_version: int = Field(ge=1)
    gpu_class: str = Field(pattern=r"^[a-z0-9]+$")
    state: ClaimState
    attempts: int = Field(ge=0)
    holder: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    updated_at: datetime
    heartbeat_at: datetime
    retry_at: datetime | None = None
    last_error: str = Field(default="", max_length=2000)

    @property
    def counts_toward_ratio(self) -> bool:
        """Whether this claim occupies a host for ratio purposes."""
        return self.state is not ClaimState.EXHAUSTED


@dataclass(frozen=True)
class StoredClaim:
    claim: PlacementClaim
    mod_revision: int


class ClaimEtcd(Protocol):
    """The slice of ``EtcdClient`` the claim store depends on."""

    def get_snapshot(self, prefix: str | None = None) -> EtcdSnapshot: ...

    def replace_if_revision(
        self,
        key: str,
        value: str | bytes,
        *,
        expected_mod_revision: int,
        lease_id: int,
    ) -> int | None: ...

    def delete_if_revision(self, key: str, *, expected_mod_revision: int) -> bool: ...


class ClaimStore:
    """Async facade over the synchronous etcd client."""

    def __init__(self, etcd: ClaimEtcd) -> None:
        self._etcd = etcd

    @staticmethod
    def key(hostname: str) -> str:
        return CLAIM_PREFIX + hostname

    async def list(self) -> tuple[list[StoredClaim], list[str]]:
        """Return every readable claim plus the keys that failed to parse.

        An unreadable claim still blocks its host: placing onto a host whose
        ownership record cannot be read would risk a duplicate provision.
        """
        snapshot = await asyncio.to_thread(self._etcd.get_snapshot, CLAIM_PREFIX)
        claims: list[StoredClaim] = []
        unreadable: list[str] = []
        for record in snapshot.records:
            key = record.key.decode("utf-8")
            hostname = key.removeprefix(CLAIM_PREFIX)
            try:
                claim = PlacementClaim.model_validate_json(record.value)
                if claim.hostname != hostname:
                    raise ValueError("claim hostname differs from its key")
            except (ValidationError, ValueError):
                unreadable.append(hostname)
                continue
            claims.append(StoredClaim(claim, record.mod_revision))
        return claims, unreadable

    async def create(self, claim: PlacementClaim) -> int | None:
        """Create the host's claim only if none exists. ``None``: lost the race."""
        return await self._write(claim, expected_mod_revision=0)

    async def update(self, stored: StoredClaim, claim: PlacementClaim) -> int | None:
        """Replace exactly the revision read. ``None``: someone else wrote."""
        return await self._write(claim, expected_mod_revision=stored.mod_revision)

    async def delete(self, stored: StoredClaim) -> bool:
        return await asyncio.to_thread(
            self._etcd.delete_if_revision,
            self.key(stored.claim.hostname),
            expected_mod_revision=stored.mod_revision,
        )

    async def _write(
        self, claim: PlacementClaim, *, expected_mod_revision: int
    ) -> int | None:
        return await asyncio.to_thread(
            self._etcd.replace_if_revision,
            self.key(claim.hostname),
            claim.model_dump_json(),
            expected_mod_revision=expected_mod_revision,
            lease_id=0,
        )


def utcnow() -> datetime:
    return datetime.now(UTC)
