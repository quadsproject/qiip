"""Claims are persistent and every write is a real compare-and-swap."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from inference_proxy.placement.claims import (
    CLAIM_PREFIX,
    ClaimState,
    ClaimStore,
    PlacementClaim,
)
from tests.placement.fakes import FakeEtcd

NOW = datetime(2026, 9, 19, tzinfo=UTC)


def _claim(hostname: str = "l4-01", **changes: object) -> PlacementClaim:
    values: dict[str, object] = {
        "claim_id": "1" * 32,
        "hostname": hostname,
        "profile_id": "qwen3.8-27b-24g",
        "profile_version": 1,
        "gpu_class": "l4",
        "state": ClaimState.PROVISIONING,
        "attempts": 1,
        "holder": "a" * 32,
        "created_at": NOW,
        "updated_at": NOW,
        "heartbeat_at": NOW,
    }
    return PlacementClaim.model_validate({**values, **changes})


@pytest.mark.asyncio
async def test_only_one_of_two_gateways_can_create_a_hosts_claim() -> None:
    etcd = FakeEtcd()
    first, second = ClaimStore(etcd), ClaimStore(etcd)

    won = await first.create(_claim(holder="a" * 32))
    lost = await second.create(_claim(holder="b" * 32, claim_id="2" * 32))

    assert won is not None and lost is None
    claims, unreadable = await first.list()
    assert [item.claim.holder for item in claims] == ["a" * 32]
    assert unreadable == []


@pytest.mark.asyncio
async def test_a_stale_writer_cannot_overwrite_a_newer_claim() -> None:
    etcd = FakeEtcd()
    store = ClaimStore(etcd)
    await store.create(_claim())
    (seen_by_a,), _ = await store.list()
    (seen_by_b,), _ = await store.list()

    assert await store.update(seen_by_b, _claim(holder="b" * 32)) is not None
    # A still holds the old revision: its heartbeat must fail, not clobber B.
    assert await store.update(seen_by_a, _claim(state=ClaimState.ACTIVE)) is None
    assert await store.delete(seen_by_a) is False
    (current,), _ = await store.list()
    assert current.claim.holder == "b" * 32


@pytest.mark.asyncio
async def test_an_unreadable_claim_still_blocks_its_host() -> None:
    etcd = FakeEtcd()
    etcd.data[CLAIM_PREFIX + "l4-02"] = (b"{not json", 7)
    etcd.data[CLAIM_PREFIX + "l4-03"] = (
        _claim("some-other-host").model_dump_json().encode(),
        8,
    )

    claims, unreadable = await ClaimStore(etcd).list()

    assert claims == []
    assert unreadable == ["l4-02", "l4-03"]


def test_only_an_exhausted_claim_stops_counting_toward_the_ratio() -> None:
    assert _claim(state=ClaimState.FAILED).counts_toward_ratio
    assert _claim(state=ClaimState.ACTIVE).counts_toward_ratio
    assert not _claim(state=ClaimState.EXHAUSTED).counts_toward_ratio
