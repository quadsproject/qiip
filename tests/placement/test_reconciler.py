"""End-to-end placement passes against in-memory etcd, QUADS and provisioner."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from inference_proxy.config.settings import PlacementSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.artifacts import GGUFArtifactIndex
from inference_proxy.models.node import NodePlacement, NodeStatus
from inference_proxy.models.quads import QUADSHost
from inference_proxy.placement.catalog import BUILTIN_PROFILES
from inference_proxy.placement.claims import ClaimState, ClaimStore, PlacementClaim
from inference_proxy.placement.reconciler import (
    PlacementReconciler,
    PlacementStatus,
)
from inference_proxy.provisioning.host_lifecycle import HostLifecycleLease
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import QUADSClient
from inference_proxy.quads.poller import QUADSPoller
from tests.placement.fakes import (
    FakeArtifactIndex,
    FakeClock,
    FakeEtcd,
    FakePoller,
    FakeProvisioner,
    FakeQuads,
    a30_host,
    catalog_artifacts,
    l4_host,
    node,
)

QUALIFIED = tuple(
    profile.model_copy(update={"qualified_gpus": ("l4", "a30")})
    for profile in BUILTIN_PROFILES
)


class Rig:
    def __init__(
        self,
        hosts: list[QUADSHost],
        *,
        etcd: FakeEtcd | None = None,
        registry: NodeRegistry | None = None,
        provisioner: FakeProvisioner | None = None,
        holder: str = "a" * 32,
        clock: FakeClock | None = None,
        extra_settings: dict[str, object] | None = None,
        **settings: object,
    ) -> None:
        self.etcd = etcd or FakeEtcd()
        self.registry = registry or NodeRegistry()
        self.provisioner = provisioner or FakeProvisioner(self.registry)
        self.quads = FakeQuads([host.hostname for host in hosts])
        self.poller = FakePoller(hosts)
        self.index = FakeArtifactIndex(catalog_artifacts())
        self.clock = clock or FakeClock()
        values: dict[str, object] = {"max_concurrent": 32}
        self.settings = PlacementSettings.model_validate(
            {**values, **settings, **(extra_settings or {})}
        )
        self.reconciler = PlacementReconciler(
            settings=self.settings,
            quads_client=cast(QUADSClient, self.quads),
            quads_poller=cast(QUADSPoller, self.poller),
            registry=self.registry,
            provisioner=cast(NodeProvisioner, self.provisioner),
            artifact_index=cast(GGUFArtifactIndex, self.index),
            claims=ClaimStore(self.etcd),
            lookahead_hours=24,
            profiles=QUALIFIED,
            clock=self.clock,
            holder=holder,
        )

    async def run(self) -> None:
        await self.reconciler.reconcile_once()
        await self.provisioner.drain()

    async def claims(self) -> dict[str, PlacementClaim]:
        stored, _ = await ClaimStore(self.etcd).list()
        return {item.claim.hostname: item.claim for item in stored}


def _l4s(count: int) -> list[QUADSHost]:
    return [l4_host(f"l4-{index:02d}") for index in range(count)]


def _placed(rig: Rig) -> dict[str, str]:
    return {
        call["hostname"]: call["placement"].profile_id for call in rig.provisioner.calls
    }


@pytest.mark.asyncio
async def test_eight_free_l4_hosts_are_placed_five_one_one_one() -> None:
    rig = Rig(_l4s(8))

    await rig.run()

    placed = _placed(rig)
    assert sorted(placed.values()).count("qwen3.8-27b-24g") == 5
    assert sorted(placed.values()).count("qwen3.6-35b-a3b-24g") == 1
    assert sorted(placed.values()).count("muse-glimmer-30b-24g") == 1
    assert sorted(placed.values()).count("gemma-4-31b-24g") == 1
    claims = await rig.claims()
    assert {claim.state for claim in claims.values()} == {ClaimState.ACTIVE}
    for call in rig.provisioner.calls:
        request = call["request"]
        assert call["managed"] is True
        assert request.profile.profile_id == call["placement"].profile_id
        assert request.fit_target_mib == rig.settings.reserve_mib
        node_record = rig.registry.get(call["hostname"])
        assert node_record is not None and node_record.placement is not None
        assert node_record.placement.claim_id == claims[call["hostname"]].claim_id
    muse = next(c for c in rig.provisioner.calls if "muse" in c["placement"].profile_id)
    assert muse["request"].profile.draft_artifact_id == "d" * 64
    gemma = next(
        c for c in rig.provisioner.calls if "gemma" in c["placement"].profile_id
    )
    assert gemma["request"].profile.draft_artifact_id == "f" * 64
    assert gemma["request"].profile.draft_cache_type is None


@pytest.mark.asyncio
async def test_a_second_pass_changes_nothing() -> None:
    rig = Rig(_l4s(8))
    await rig.run()
    before = dict(rig.etcd.data)

    await rig.run()

    assert len(rig.provisioner.calls) == 8
    assert rig.etcd.data == before
    assert rig.reconciler.status.held == {
        "qwen3.8-27b-24g": 5,
        "qwen3.6-35b-a3b-24g": 1,
        "muse-glimmer-30b-24g": 1,
        "gemma-4-31b-24g": 1,
    }


@pytest.mark.asyncio
async def test_hosts_that_join_later_fill_the_remaining_deficit() -> None:
    rig = Rig(_l4s(3))
    await rig.run()
    # Three hosts for four profiles: catalog order, so Gemma waits.
    assert sorted(_placed(rig).values()) == sorted(
        item.profile_id for item in BUILTIN_PROFILES[:3]
    )

    joined = [*_l4s(3), a30_host("a30-00"), l4_host("l4-90")]
    rig.poller.hosts = joined
    rig.quads.available = [host.hostname for host in joined]
    await rig.run()

    placed = _placed(rig)
    # Five hosts: targets 2/1/1/1. Qwen3.8 takes the A30; Gemma, which is
    # only validated on the L4, takes the new L4.
    assert placed["a30-00"] == "qwen3.8-27b-24g"
    assert placed["l4-90"] == "gemma-4-31b-24g"
    # The original three placements were not touched.
    assert len(rig.provisioner.calls) == 5


@pytest.mark.asyncio
async def test_concurrency_is_bounded_and_in_flight_work_is_counted() -> None:
    rig = Rig(_l4s(8), max_concurrent=2)
    rig.provisioner.hold = True

    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)

    # Two provisions in flight; the second pass started nothing new.
    assert len(rig.provisioner.calls) == 2
    claims = await rig.claims()
    assert [c.state for c in claims.values()] == [ClaimState.PROVISIONING] * 2
    for gate in rig.provisioner.gates.values():
        gate.set()
    await rig.provisioner.drain()


@pytest.mark.asyncio
async def test_manual_owned_and_self_setup_nodes_are_left_alone() -> None:
    registry = NodeRegistry()
    registry.add(node("l4-00", NodeStatus.HEALTHY))  # set up by an operator
    registry.add(node("l4-01", NodeStatus.AVAILABLE, managed=False, owner="a@b.c"))
    registry.add(node("l4-02", NodeStatus.UNHEALTHY, self_setup=True))
    registry.add(node("l4-03", NodeStatus.FAILED))  # a person's failed setup
    rig = Rig(_l4s(6), registry=registry)

    await rig.run()

    assert set(_placed(rig)) == {"l4-04", "l4-05"}
    assert rig.provisioner.cleaned == []
    skipped = {item.hostname: item.reason for item in rig.reconciler.status.skipped}
    assert set(skipped) == {"l4-00", "l4-01", "l4-02", "l4-03"}
    # The denominator is the two free hosts only: Qwen3.8, then Qwen3.6.
    assert sorted(_placed(rig).values()) == [
        "qwen3.6-35b-a3b-24g",
        "qwen3.8-27b-24g",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(NodeStatus))
async def test_self_setup_node_is_never_placed_even_when_quads_reports_it_free(
    status: NodeStatus,
) -> None:
    rig = Rig(_l4s(1))
    adopted = node("l4-00", status, self_setup=True)
    rig.registry.add(adopted)

    await rig.run()
    await rig.run()

    assert rig.provisioner.calls == []
    assert rig.provisioner.cleaned == []
    assert rig.registry.get("l4-00") == adopted
    assert await rig.claims() == {}
    assert all(target == 0 for target in rig.reconciler.status.targets.values())


@pytest.mark.asyncio
async def test_an_unowned_pool_host_is_a_candidate() -> None:
    registry = NodeRegistry()
    registry.add(node("l4-00", NodeStatus.AVAILABLE, managed=False))
    rig = Rig(_l4s(1), registry=registry)

    await rig.run()

    assert set(_placed(rig)) == {"l4-00"}


@pytest.mark.asyncio
async def test_multi_gpu_busy_excluded_and_unscheduled_hosts_are_refused() -> None:
    hosts = [l4_host("l4-multi", gpus=4), *_l4s(3)]
    rig = Rig(hosts, exclude_hosts=["L4-01."])
    rig.quads.available = ["l4-multi", "l4-00", "l4-01"]  # l4-02 is assigned
    lease = await rig.provisioner.try_reserve_host("l4-00")  # an operator is busy

    await rig.run()

    assert _placed(rig) == {}
    reasons = {item.hostname: item.reason for item in rig.reconciler.status.skipped}
    assert "exactly one GPU per host" in reasons["l4-multi"]
    assert "in progress" in reasons["l4-00"]
    assert "exclude_hosts" in reasons["l4-01"]
    assert "QUADS scheduling window" in reasons["l4-02"]
    assert lease is not None
    lease.release()


@pytest.mark.asyncio
async def test_a_t4_is_never_a_candidate() -> None:
    t4 = QUADSHost(
        hostname="t4-00",
        gpu_vendor="NVIDIA",
        gpu_model="TU104GL [Tesla T4]",
        gpu_count=1,
    )
    rig = Rig([t4])

    await rig.run()

    assert _placed(rig) == {}
    assert rig.reconciler.status.skipped == ()


@pytest.mark.asyncio
async def test_unqualified_gpu_classes_are_not_placed_by_default() -> None:
    rig = Rig([*_l4s(2), a30_host("a30-00")])
    rig.reconciler._profiles = tuple(  # only the L4 has been validated
        profile.model_copy(update={"qualified_gpus": ("l4",)})
        for profile in BUILTIN_PROFILES
    )

    await rig.run()

    assert set(_placed(rig)) == {"l4-00", "l4-01"}
    reasons = {item.hostname: item.reason for item in rig.reconciler.status.skipped}
    assert reasons == {"a30-00": "no placeable profile is qualified for a30"}


@pytest.mark.asyncio
async def test_missing_files_are_reported_and_only_block_their_profile() -> None:
    rig = Rig(_l4s(8))
    rig.index.artifacts = [
        item for item in catalog_artifacts() if "Muse" not in item.repo_id
    ]

    await rig.run()

    status = rig.reconciler.status
    assert "muse-glimmer-30b-24g" not in _placed(rig).values()
    assert len(_placed(rig)) == 7  # the eighth host waits for Muse's files
    assert status.unfilled == {"muse-glimmer-30b-24g": 1}
    assert {(m.profile_id, m.role.value) for m in status.missing_artifacts} == {
        ("muse-glimmer-30b-24g", "target"),
        ("muse-glimmer-30b-24g", "draft"),
    }

    rig.index.artifacts = catalog_artifacts()
    await rig.run()
    assert sorted(_placed(rig).values()).count("muse-glimmer-30b-24g") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", ["quads", "cache", "etcd", "unsynced"])
async def test_an_outage_places_nothing_and_is_reported(outage: str) -> None:
    rig = Rig(_l4s(3))
    if outage == "quads":
        rig.quads.fail = True
    elif outage == "cache":
        rig.index.fail = True
    elif outage == "etcd":
        rig.etcd.fail = True
    else:
        rig.poller.last_sync = None

    await rig.run()  # must not raise

    assert _placed(rig) == {}
    assert rig.reconciler.status.error


@pytest.mark.asyncio
async def test_failures_retry_with_backoff_then_stop() -> None:
    rig = Rig(_l4s(1), max_attempts=3, retry_backoff_seconds=60)
    rig.provisioner.failures["l4-00"] = ["boom 1", "boom 2", "boom 3"]

    await rig.run()
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.attempts, claim.last_error) == (
        ClaimState.FAILED,
        1,
        "boom 1",
    )

    await rig.run()  # still inside the backoff: nothing happens
    assert len(rig.provisioner.calls) == 1

    rig.clock.advance(61)
    await rig.run()
    assert rig.provisioner.cleaned == ["l4-00"]  # its own failed record, cleared
    rig.clock.advance(121)
    await rig.run()

    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.attempts) == (ClaimState.EXHAUSTED, 3)
    assert len({call["placement"].claim_id for call in rig.provisioner.calls}) == 1

    rig.clock.advance(100_000)
    await rig.run()
    assert len(rig.provisioner.calls) == 3  # exhausted: never retried again
    # An exhausted host no longer counts, and is not handed to anyone either.
    assert not any(rig.reconciler.status.held.values())


@pytest.mark.asyncio
async def test_an_operator_can_reset_an_exhausted_claim() -> None:
    rig = Rig(_l4s(1), max_attempts=1)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()

    assert await rig.reconciler.reset_claim("l4-00") is True
    await rig.run()

    assert (await rig.claims())["l4-00"].state is ClaimState.ACTIVE


@pytest.mark.asyncio
async def test_reset_refuses_a_healthy_placement() -> None:
    rig = Rig(_l4s(1))
    await rig.run()

    with pytest.raises(ValueError, match="only a failed or exhausted"):
        await rig.reconciler.reset_claim("l4-00")
    assert await rig.reconciler.reset_claim("unknown-host") is False


@pytest.mark.asyncio
async def test_a_retry_keeps_the_profile_it_first_chose() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    # More hosts join, which would shift a fresh computation.
    hosts = _l4s(8)
    rig.poller.hosts = hosts
    rig.quads.available = [host.hostname for host in hosts]
    rig.clock.advance(61)

    await rig.run()

    first, *_ = rig.provisioner.calls
    retries = [c for c in rig.provisioner.calls if c["hostname"] == "l4-00"]
    assert len(retries) == 2
    assert retries[1]["placement"] == first["placement"]


@pytest.mark.asyncio
async def test_a_restarted_gateway_waits_for_the_claim_to_go_stale() -> None:
    etcd, registry = FakeEtcd(), NodeRegistry()
    clock = FakeClock()
    old = Rig(_l4s(1), etcd=etcd, registry=registry, clock=clock, holder="a" * 32)
    old.provisioner.hold = True
    await old.reconciler.reconcile_once()
    await asyncio.sleep(0)
    for task in old.provisioner.tasks.values():  # the gateway dies mid-provision
        task.cancel()
    await old.provisioner.drain()
    # A cancelled run records its failure; simulate a hard crash instead.
    stored, _ = await ClaimStore(etcd).list()
    crashed = stored[0].claim.model_copy(update={"state": ClaimState.PROVISIONING})
    await ClaimStore(etcd).update(stored[0], crashed)
    registry.add(
        node(
            "l4-00",
            NodeStatus.PROVISIONING,
            placement=NodePlacement(
                profile_id=crashed.profile_id,
                profile_version=1,
                claim_id=crashed.claim_id,
            ),
        )
    )

    new = Rig(_l4s(1), etcd=etcd, registry=registry, clock=clock, holder="b" * 32)
    await new.run()
    assert new.provisioner.calls == []  # the old holder may still be alive

    clock.advance(new.settings.claim_stale_seconds + 1)
    await new.run()  # marks the claim abandoned
    await new.run()  # and retries it

    assert [call["hostname"] for call in new.provisioner.calls] == ["l4-00"]
    assert new.provisioner.cleaned == ["l4-00"]
    claim = (await new.claims())["l4-00"]
    assert (claim.state, claim.holder, claim.claim_id) == (
        ClaimState.ACTIVE,
        "b" * 32,
        crashed.claim_id,
    )


@pytest.mark.asyncio
async def test_two_gateways_never_provision_the_same_host() -> None:
    etcd = FakeEtcd()
    first = Rig(_l4s(3), etcd=etcd, holder="a" * 32)
    second = Rig(_l4s(3), etcd=etcd, holder="b" * 32)

    await asyncio.gather(
        first.reconciler.reconcile_once(), second.reconciler.reconcile_once()
    )
    await first.provisioner.drain()
    await second.provisioner.drain()

    hosts = [c["hostname"] for c in first.provisioner.calls + second.provisioner.calls]
    assert sorted(hosts) == ["l4-00", "l4-01", "l4-02"]


@pytest.mark.asyncio
async def test_a_holder_that_loses_its_claim_cancels_its_own_provision() -> None:
    rig = Rig(_l4s(1))
    rig.provisioner.hold = True
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    stored, _ = await ClaimStore(rig.etcd).list()
    taken = stored[0].claim.model_copy(update={"holder": "b" * 32})
    await ClaimStore(rig.etcd).update(stored[0], taken)  # another gateway took over

    task = rig.provisioner.tasks["l4-00"]
    owned = rig.reconciler._owned["l4-00"]
    await rig.reconciler._heartbeat_once(owned, task)
    await rig.provisioner.drain()

    assert task.cancelled()
    # The loser wrote nothing: the new holder's claim is intact.
    assert (await rig.claims())["l4-00"] == taken
    assert not rig.provisioner.lifecycle.is_busy("l4-00")


@pytest.mark.asyncio
async def test_an_expired_node_lease_does_not_cause_a_second_placement() -> None:
    rig = Rig(_l4s(2))
    await rig.run()
    rig.registry.remove("l4-00")  # its leased etcd key expired; the claim did not

    await rig.run()

    # Not re-planned as a fresh host, and not retried before the grace period.
    assert len(rig.provisioner.calls) == 2
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.last_error) == (
        ClaimState.FAILED,
        "the node record disappeared",
    )

    rig.clock.advance(rig.settings.claim_stale_seconds + 1)
    await rig.run()
    retried = [c for c in rig.provisioner.calls if c["hostname"] == "l4-00"]
    assert len(retried) == 2 and retried[0]["placement"] == retried[1]["placement"]


@pytest.mark.asyncio
async def test_a_host_an_operator_takes_over_is_released() -> None:
    rig = Rig(_l4s(1))
    await rig.run()
    rig.registry.add(node("l4-00", NodeStatus.HEALTHY, owner="a@b.c"))  # manual setup

    await rig.run()

    assert await rig.claims() == {}
    assert rig.provisioner.cleaned == []


@pytest.mark.asyncio
async def test_a_host_quads_takes_back_releases_its_claim() -> None:
    rig = Rig(_l4s(2), retry_backoff_seconds=60)
    rig.provisioner.failures["l4-01"] = ["boom"]
    await rig.run()
    rig.quads.available = []

    await rig.run()
    # The healthy node is the schedule enforcer's to tear down; the failed
    # record is only a record, so placement clears it with its claim.
    assert set(await rig.claims()) == {"l4-00"}
    assert rig.provisioner.cleaned == ["l4-01"]

    rig.registry.remove("l4-00")  # the enforcer's teardown finished
    await rig.run()
    assert await rig.claims() == {}


@pytest.mark.asyncio
async def test_capacity_refusal_does_not_use_up_an_attempt() -> None:
    rig = Rig(_l4s(1))
    rig.provisioner.capacity = 0

    await rig.run()

    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.attempts) == (ClaimState.FAILED, 0)
    assert not rig.provisioner.lifecycle.is_busy("l4-00")


@pytest.mark.asyncio
async def test_disabled_placement_never_starts_its_loop() -> None:
    rig = Rig(_l4s(1), enabled=False)

    rig.reconciler.start()

    assert rig.reconciler._task is None
    await rig.reconciler.stop()


@pytest.mark.asyncio
async def test_the_loop_runs_a_pass_survives_an_error_and_stops() -> None:
    rig = Rig(_l4s(1), interval_seconds=30)
    calls = 0
    real = rig.reconciler.reconcile_once

    async def flaky() -> PlacementStatus:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("unexpected")
        return await real()

    rig.reconciler.reconcile_once = flaky  # type: ignore[method-assign]
    rig.reconciler._settings = rig.settings.model_copy(update={"interval_seconds": 0})

    rig.reconciler.start()
    rig.reconciler.start()  # idempotent
    # The pass hops through worker threads, so wait on the outcome, not on a
    # fixed number of event-loop turns.
    deadline = asyncio.get_running_loop().time() + 10
    while not rig.provisioner.calls:
        assert asyncio.get_running_loop().time() < deadline, "no pass placed the host"
        await asyncio.sleep(0.01)
    await rig.reconciler.stop()
    await rig.provisioner.drain()

    assert calls >= 2 and set(_placed(rig)) == {"l4-00"}
    assert rig.reconciler._task is None


@pytest.mark.asyncio
async def test_a_heartbeat_refreshes_the_claim_and_tolerates_an_etcd_outage() -> None:
    rig = Rig(_l4s(1))
    rig.provisioner.hold = True
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    owned = rig.reconciler._owned["l4-00"]
    task = rig.provisioner.tasks["l4-00"]
    before = (await rig.claims())["l4-00"].heartbeat_at

    rig.clock.advance(30)
    await rig.reconciler._heartbeat_once(owned, task)
    assert (await rig.claims())["l4-00"].heartbeat_at > before

    rig.etcd.fail = True
    await rig.reconciler._heartbeat_once(owned, task)  # logged, not fatal
    rig.etcd.fail = False
    assert not owned.lost and not task.done()

    rig.provisioner.gates["l4-00"].set()
    await rig.provisioner.drain()
    assert (await rig.claims())["l4-00"].state is ClaimState.ACTIVE


@pytest.mark.asyncio
async def test_a_final_write_lost_to_etcd_is_repaired_on_the_next_pass() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.hold = True
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    rig.etcd.fail = True
    rig.provisioner.gates["l4-00"].set()
    await rig.provisioner.drain()
    rig.etcd.fail = False
    assert (await rig.claims())["l4-00"].state is ClaimState.PROVISIONING

    await rig.run()

    # Nothing is in flight for a claim this gateway holds: it is settled, and
    # the healthy node it produced is recognized instead of being rebuilt.
    claim = (await rig.claims())["l4-00"]
    assert claim.state is ClaimState.ACTIVE
    assert len(rig.provisioner.calls) == 1


@pytest.mark.asyncio
async def test_a_retry_waits_while_an_operator_holds_the_host() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    rig.clock.advance(61)
    lease = await rig.provisioner.try_reserve_host("l4-00")

    await rig.run()

    assert len(rig.provisioner.calls) == 1
    assert lease is not None
    lease.release()
    with pytest.raises(ValueError, match="busy; try again"):
        held = await rig.provisioner.try_reserve_host("l4-00")
        try:
            await rig.reconciler.reset_claim("l4-00")
        finally:
            assert held is not None
            held.release()


@pytest.mark.asyncio
async def test_a_claim_for_a_superseded_profile_version_is_not_retried() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    rig.reconciler._profiles = tuple(
        profile.model_copy(update={"version": 2}) for profile in QUALIFIED
    )
    rig.clock.advance(61)

    await rig.run()

    assert len(rig.provisioner.calls) == 1


# --- Review checkpoint 1: a retry is a placement decision like any other.


async def _failed_once() -> Rig:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    assert (await rig.claims())["l4-00"].state is ClaimState.FAILED
    rig.clock.advance(61)
    return rig


def _restarted(old: Rig, settings: dict[str, object] | None = None) -> Rig:
    """A new gateway process over the same etcd and registry, new settings."""
    new = Rig(
        _l4s(1),
        etcd=old.etcd,
        registry=old.registry,
        clock=old.clock,
        holder="b" * 32,
        retry_backoff_seconds=60,
        extra_settings=settings,
    )
    new.provisioner.calls = old.provisioner.calls
    new.provisioner.cleaned = old.provisioner.cleaned
    return new


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(
    "change",
    ["exclude", "zero_weight", "unqualified", "multi_gpu", "gone", "reclassed"],
)
async def test_a_retry_obeys_the_policy_as_it_stands_now(
    change: str, restart: bool
) -> None:
    rig = await _failed_once()
    settings: dict[str, object] = {}
    if change == "exclude":
        settings["exclude_hosts"] = ["l4-00"]
    elif change == "zero_weight":
        settings["ratios"] = {"qwen3.8-27b-24g": 0, "qwen3.6-35b-a3b-24g": 1}
    if restart:
        rig = _restarted(rig, settings)
    elif settings:
        rig.reconciler._settings = rig.settings.model_copy(update=settings)
    if change == "unqualified":
        rig.reconciler._profiles = tuple(  # the validation was withdrawn
            profile.model_copy(update={"qualified_gpus": ()})
            for profile in BUILTIN_PROFILES
        )
    elif change == "multi_gpu":
        rig.poller.hosts = [l4_host("l4-00", gpus=2)]
    elif change == "gone":
        rig.poller.hosts = []
    elif change == "reclassed":
        rig.poller.hosts = [a30_host("l4-00")]

    await rig.run()

    assert len(rig.provisioner.calls) == 1, "the refused host was provisioned again"
    # The claim is given up, with its own failed node record, and the host is
    # reported instead of silently holding a slot in the ratio.
    assert await rig.claims() == {}
    assert rig.provisioner.cleaned == ["l4-00"]
    if change not in ("gone", "zero_weight"):
        assert "l4-00" in {item.hostname for item in rig.reconciler.status.skipped}


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize(
    "takeover", ["owner", "self_setup", "unmanaged", "set_up_again"]
)
async def test_a_retry_never_touches_a_node_someone_took_over(
    takeover: str, restart: bool
) -> None:
    rig = await _failed_once()
    failed = rig.registry.get("l4-00")
    assert failed is not None and failed.placement is not None
    if takeover == "owner":
        # As update_node_owner leaves it, and also with a stale marker still on.
        replacement = failed.model_copy(update={"owner": "new-owner@example.com"})
    elif takeover == "self_setup":
        replacement = node("l4-00", NodeStatus.FAILED, self_setup=True).model_copy(
            update={"placement": failed.placement}
        )
    elif takeover == "unmanaged":
        replacement = failed.model_copy(update={"managed": False})
    else:
        replacement = node("l4-00", NodeStatus.FAILED)
    rig.registry.add(replacement)
    if restart:
        rig = _restarted(rig)

    await rig.run()
    await rig.run()

    assert len(rig.provisioner.calls) == 1
    assert rig.provisioner.cleaned == []
    assert rig.registry.get("l4-00") == replacement  # owner and record intact
    assert await rig.claims() == {}


@pytest.mark.asyncio
async def test_reset_does_not_clear_a_node_someone_took_over() -> None:
    rig = await _failed_once()
    failed = rig.registry.get("l4-00")
    assert failed is not None
    owned = failed.model_copy(update={"owner": "new-owner@example.com"})
    rig.registry.add(owned)

    assert await rig.reconciler.reset_claim("l4-00") is True

    assert rig.provisioner.cleaned == []
    assert rig.registry.get("l4-00") == owned


@pytest.mark.asyncio
async def test_missing_files_pause_a_retry_without_giving_the_claim_up() -> None:
    rig = await _failed_once()
    rig.index.artifacts = []

    await rig.run()

    assert len(rig.provisioner.calls) == 1
    assert (await rig.claims())["l4-00"].state is ClaimState.FAILED


# --- Review checkpoint 3: fencing.


@pytest.mark.asyncio
async def test_a_partitioned_holder_gives_up_before_anyone_may_take_over() -> None:
    rig = Rig(_l4s(1))
    rig.provisioner.hold = True
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    owned = rig.reconciler._owned["l4-00"]
    task = rig.provisioner.tasks["l4-00"]
    rig.etcd.fail = True  # only this gateway lost etcd

    rig.clock.advance(rig.settings.claim_stale_seconds / 2 - 1)
    await rig.reconciler._heartbeat_once(owned, task)
    assert not owned.lost and rig.provisioner.fenced == []

    rig.clock.advance(2)  # past half the stale period, well before takeover
    await rig.reconciler._heartbeat_once(owned, task)
    await rig.provisioner.drain()

    assert owned.lost and task.cancelled()
    # Fenced with the explicit cancel that also stops the remote worker.
    assert rig.provisioner.fenced == ["l4-00"]
    assert not rig.provisioner.lifecycle.is_busy("l4-00")


@pytest.mark.asyncio
async def test_a_lost_claim_is_fenced_not_merely_cancelled() -> None:
    rig = Rig(_l4s(1))
    rig.provisioner.hold = True
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    stored, _ = await ClaimStore(rig.etcd).list()
    await ClaimStore(rig.etcd).update(
        stored[0], stored[0].claim.model_copy(update={"holder": "b" * 32})
    )

    await rig.reconciler._heartbeat_once(
        rig.reconciler._owned["l4-00"], rig.provisioner.tasks["l4-00"]
    )
    await rig.provisioner.drain()

    assert rig.provisioner.fenced == ["l4-00"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["still_running", "cannot_check"])
async def test_a_retry_is_blocked_while_earlier_remote_work_may_be_running(
    case: str,
) -> None:
    """A cancelled gateway task can leave setup.sh running on the host."""
    rig = await _failed_once()
    if case == "still_running":
        rig.provisioner.remote_processes["l4-00"] = ["202 bash auto-llamacpp/setup.sh"]
    else:
        rig.provisioner.remote_check_error = ConnectionError("ssh: no route to host")

    await rig.run()

    claim = (await rig.claims())["l4-00"]
    assert len(rig.provisioner.calls) == 1, "a second provision was started on top"
    assert rig.provisioner.cleaned == []
    assert (claim.state, claim.attempts) == (ClaimState.FAILED, 1)  # no attempt used
    assert claim.last_error.startswith("blocked: ")
    assert claim.retry_at is not None and claim.retry_at > rig.clock()
    assert not rig.provisioner.lifecycle.is_busy("l4-00")

    rig.provisioner.remote_processes.clear()
    rig.provisioner.remote_check_error = None
    rig.clock.advance(61)
    await rig.run()
    assert (await rig.claims())["l4-00"].state is ClaimState.ACTIVE


@pytest.mark.asyncio
async def test_stale_takeover_does_not_overlap_the_old_holders_remote_command() -> None:
    etcd, registry, clock = FakeEtcd(), NodeRegistry(), FakeClock()
    old = Rig(_l4s(1), etcd=etcd, registry=registry, clock=clock, holder="a" * 32)
    old.provisioner.hold = True
    await old.reconciler.reconcile_once()
    await asyncio.sleep(0)
    # The old gateway is partitioned and wedged: it neither refreshes nor stops.
    new = Rig(_l4s(1), etcd=etcd, registry=registry, clock=clock, holder="b" * 32)
    new.provisioner.remote_processes["l4-00"] = ["202 bash auto-llamacpp/setup.sh"]
    clock.advance(new.settings.claim_stale_seconds + 1)

    await new.run()  # marks the claim abandoned
    await new.run()  # would retry, but the host is still busy with the old run

    assert new.provisioner.calls == []
    assert (await new.claims())["l4-00"].last_error.startswith("blocked: ")
    old.provisioner.gates["l4-00"].set()
    await old.provisioner.drain()


@pytest.mark.asyncio
async def test_only_hosts_limits_placement_to_the_listed_hosts() -> None:
    rig = Rig(_l4s(3), only_hosts=["L4-01."])

    await rig.run()

    assert set(_placed(rig)) == {"l4-01"}
    reasons = {item.hostname: item.reason for item in rig.reconciler.status.skipped}
    assert reasons == {
        "l4-00": "not listed in placement.only_hosts",
        "l4-02": "not listed in placement.only_hosts",
    }
    # A host that joins later is kept out too, which an exclusion list cannot do.
    rig.poller.hosts = [*_l4s(3), l4_host("l4-99")]
    rig.quads.available.append("l4-99")
    await rig.run()
    assert set(_placed(rig)) == {"l4-01"}


# --- Review checkpoint 5: every launch asks the host, the attempt budget holds
# on every path into a launch, and a takeover is never undone. ---


_SETUP_RUNNING = ["202 bash auto-llamacpp/setup.sh"]


def _skips(rig: Rig) -> dict[str, str]:
    return {item.hostname: item.reason for item in rig.reconciler.status.skipped}


@pytest.mark.asyncio
async def test_a_reset_does_not_bypass_the_remote_work_guard() -> None:
    """Reset deletes the claim; the running setup.sh is still on the host."""
    rig = await _failed_once()
    rig.provisioner.remote_processes["l4-00"] = list(_SETUP_RUNNING)
    await rig.run()
    assert (await rig.claims())["l4-00"].last_error.startswith("blocked: ")

    assert await rig.reconciler.reset_claim("l4-00")
    await rig.run()

    assert len(rig.provisioner.calls) == 1, "a second provision was started on top"
    assert await rig.claims() == {}, "a blocked first launch writes no claim"
    assert _skips(rig)["l4-00"] == (
        "blocked: an earlier provisioning command is still running on the host"
    )
    assert not rig.provisioner.lifecycle.is_busy("l4-00")

    # The host is asked again only after the backoff, and then launches.
    rig.provisioner.remote_processes.clear()
    asked = len(rig.provisioner.remote_checks)
    await rig.run()
    assert len(rig.provisioner.remote_checks) == asked, "asked again inside the backoff"
    assert _skips(rig)["l4-00"].startswith("blocked: ")
    rig.clock.advance(61)
    await rig.run()
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, len(rig.provisioner.calls)) == (ClaimState.ACTIVE, 2)
    assert "l4-00" not in _skips(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["still_running", "cannot_check"])
async def test_a_first_launch_is_blocked_and_its_share_goes_to_another_host(
    case: str,
) -> None:
    """The planner is deterministic: left in, l4-00 would be picked every pass.

    A short fleet is served in catalog order, so the one host is owed Qwen3.8.
    When a second host joins, Qwen3.8 must go to it rather than stay pinned to
    the host that cannot be launched on while the newcomer takes Qwen3.6.
    """
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    if case == "still_running":
        rig.provisioner.remote_processes["l4-00"] = list(_SETUP_RUNNING)
    else:
        rig.provisioner.remote_check_errors["l4-00"] = ConnectionError("no route")
    await rig.run()
    assert rig.provisioner.calls == [] and await rig.claims() == {}
    assert _skips(rig)["l4-00"].startswith("blocked: ")
    if case == "cannot_check":
        assert "no route" in _skips(rig)["l4-00"]
    assert not rig.provisioner.lifecycle.is_busy("l4-00")

    rig.poller.hosts = _l4s(2)
    rig.quads.available = ["l4-00", "l4-01"]
    await rig.run()

    assert _placed(rig) == {"l4-01": "qwen3.8-27b-24g"}
    assert rig.provisioner.remote_checks == ["l4-00", "l4-01"]
    assert _skips(rig)["l4-00"].startswith("blocked: ")


@pytest.mark.asyncio
async def test_a_deferred_host_that_leaves_the_inventory_is_forgotten() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60)
    rig.provisioner.remote_processes["l4-00"] = list(_SETUP_RUNNING)
    await rig.run()
    assert "l4-00" in rig.reconciler._deferred

    rig.poller.hosts = []
    await rig.run()
    assert rig.reconciler._deferred == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "state", "calls"),
    [(1, ClaimState.EXHAUSTED, 1), (2, ClaimState.ACTIVE, 2)],
)
async def test_an_abandoned_attempt_spends_the_attempt_budget(
    limit: int, state: ClaimState, calls: int
) -> None:
    """A gateway restart mid-provision must not grant an attempt past the limit."""
    old = await _failed_once()
    store = ClaimStore(old.etcd)
    stored = (await store.list())[0][0]
    await store.update(
        stored, stored.claim.model_copy(update={"state": ClaimState.PROVISIONING})
    )
    new = _restarted(old, {"max_attempts": limit})
    new.clock.advance(new.settings.claim_stale_seconds + 1)

    await new.run()
    await new.run()

    claim = (await new.claims())["l4-00"]
    assert (claim.state, len(new.provisioner.calls)) == (state, calls)
    if state is ClaimState.EXHAUSTED:
        assert claim.attempts == 1 and claim.retry_at is None
        assert claim.last_error == (
            "the provisioning gateway stopped refreshing this claim"
        )
        assert new.provisioner.remote_checks == [], "asked a host it may not retry"


@pytest.mark.asyncio
async def test_an_abandoned_attempt_that_is_serving_is_adopted_at_the_limit() -> None:
    old = await _failed_once()
    store = ClaimStore(old.etcd)
    stored = (await store.list())[0][0]
    await store.update(
        stored, stored.claim.model_copy(update={"state": ClaimState.PROVISIONING})
    )
    old.registry.add(
        node(
            "l4-00",
            NodeStatus.HEALTHY,
            placement=NodePlacement(
                profile_id=stored.claim.profile_id,
                profile_version=stored.claim.profile_version,
                claim_id=stored.claim.claim_id,
            ),
        )
    )
    new = _restarted(old, {"max_attempts": 1})
    new.clock.advance(new.settings.claim_stale_seconds + 1)

    await new.run()

    claim = (await new.claims())["l4-00"]
    assert (claim.state, claim.attempts) == (ClaimState.ACTIVE, 0)
    assert len(new.provisioner.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_claim_over_a_lowered_limit_is_exhausted_untouched() -> None:
    """The limit is enforced before the host is asked or its record cleared."""
    old = await _failed_once()
    new = _restarted(old, {"max_attempts": 1})

    await new.run()

    claim = (await new.claims())["l4-00"]
    assert (claim.state, claim.attempts, claim.retry_at) == (
        ClaimState.EXHAUSTED,
        1,
        None,
    )
    assert claim.last_error == "boom"
    assert len(new.provisioner.calls) == 1
    assert new.provisioner.remote_checks == [] and new.provisioner.cleaned == []
    assert not new.provisioner.lifecycle.is_busy("l4-00")

    # An operator reset starts a new budget.
    assert await new.reconciler.reset_claim("l4-00")
    await new.run()
    claim = (await new.claims())["l4-00"]
    assert (claim.state, len(new.provisioner.calls)) == (ClaimState.ACTIVE, 2)


@pytest.mark.asyncio
async def test_the_budget_counts_attempts_since_the_last_success() -> None:
    """A node that needed every attempt must still survive losing its record."""
    rig = Rig(_l4s(1), retry_backoff_seconds=60, max_attempts=2)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    rig.clock.advance(61)
    await rig.run()
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.attempts) == (ClaimState.ACTIVE, 0)

    for expected_calls in (3, 4):  # Twice: a lifetime count would exhaust here.
        rig.registry.remove("l4-00")
        await rig.run()
        lost = (await rig.claims())["l4-00"]
        assert (lost.state, lost.attempts) == (ClaimState.FAILED, 0)
        assert lost.last_error == "the node record disappeared"
        rig.clock.advance(rig.settings.claim_stale_seconds + 1)
        await rig.run()
        claim = (await rig.claims())["l4-00"]
        assert (claim.state, claim.attempts) == (ClaimState.ACTIVE, 0)
        assert len(rig.provisioner.calls) == expected_calls


@pytest.mark.asyncio
async def test_an_active_claim_with_an_old_lifetime_count_restarts_its_budget() -> None:
    """Claims written before successes reset the count can carry attempts > 0."""
    rig = Rig(_l4s(1), retry_backoff_seconds=60, max_attempts=2)
    await rig.run()
    store = ClaimStore(rig.etcd)
    stored = (await store.list())[0][0]
    await store.update(stored, stored.claim.model_copy(update={"attempts": 2}))

    rig.registry.remove("l4-00")
    await rig.run()
    rig.clock.advance(rig.settings.claim_stale_seconds + 1)
    await rig.run()

    claim = (await rig.claims())["l4-00"]
    assert (claim.state, len(rig.provisioner.calls)) == (ClaimState.ACTIVE, 2)


@pytest.mark.asyncio
async def test_a_capacity_refusal_on_the_last_attempt_does_not_exhaust() -> None:
    rig = Rig(_l4s(1), retry_backoff_seconds=60, max_attempts=2)
    rig.provisioner.failures["l4-00"] = ["boom"]
    await rig.run()
    rig.clock.advance(61)

    rig.provisioner.capacity = 0
    await rig.run()
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, claim.attempts) == (ClaimState.FAILED, 1)
    assert claim.last_error == "provisioning capacity reached"

    rig.provisioner.capacity = 32
    rig.clock.advance(61)
    await rig.run()
    claim = (await rig.claims())["l4-00"]
    assert (claim.state, len(rig.provisioner.calls)) == (ClaimState.ACTIVE, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_claim", [True, False])
async def test_a_record_that_changes_during_the_probe_is_not_replaced(
    existing_claim: bool,
) -> None:
    """Health checks and the etcd watch write without the lifecycle lease."""
    if existing_claim:
        rig = await _failed_once()
    else:
        rig = Rig(_l4s(1), retry_backoff_seconds=60)
    calls = len(rig.provisioner.calls)

    async def adopt(hostname: str) -> None:
        rig.registry.add(node(hostname, NodeStatus.HEALTHY, self_setup=True))

    rig.provisioner.during_remote_check = adopt
    await rig.run()

    assert len(rig.provisioner.calls) == calls
    assert rig.provisioner.cleaned == []
    record = rig.registry.get("l4-00")
    assert record is not None and record.self_setup
    assert not rig.provisioner.lifecycle.is_busy("l4-00")


@pytest.mark.asyncio
async def test_a_record_that_changes_before_the_lease_is_not_probed_or_replaced() -> (
    None
):
    """The pass planned from a snapshot; an operator adopted the host since."""
    rig = Rig(_l4s(1))
    reserve = rig.provisioner.try_reserve_host

    async def adopt_then_reserve(hostname: str) -> HostLifecycleLease | None:
        rig.registry.add(node(hostname, NodeStatus.HEALTHY, self_setup=True))
        return await reserve(hostname)

    rig.provisioner.try_reserve_host = adopt_then_reserve  # type: ignore[method-assign]
    await rig.run()

    assert rig.provisioner.calls == [] and rig.provisioner.remote_checks == []
    assert await rig.claims() == {}
    assert not rig.provisioner.lifecycle.is_busy("l4-00")
