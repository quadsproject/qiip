"""Operator intent survives node deletion and gateway restarts."""

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from inference_proxy.placement.suspensions import SuspensionStore
from tests.placement.fakes import FakeEtcd, l4_host
from tests.placement.test_reconciler import Rig


@pytest.mark.asyncio
async def test_suspension_survives_restart_and_resume_restores_placement() -> None:
    rig = Rig([l4_host("l4-00")])
    await rig.run()
    store = SuspensionStore(rig.etcd)
    await store.suspend("l4-00")
    rig.registry.remove("l4-00")  # Successful teardown removes the leased node.
    restarted = Rig([l4_host("l4-00")], etcd=rig.etcd, registry=rig.registry)
    restarted.clock.advance(10000)
    await restarted.run()
    await restarted.run()
    assert restarted.provisioner.calls == []
    assert not any(restarted.reconciler.status.held.values())
    assert await SuspensionStore(rig.etcd).list() == {"l4-00"}
    await store.resume("l4-00")
    await restarted.run()
    assert len(restarted.provisioner.calls) == 1


@pytest.mark.asyncio
async def test_suspension_blocks_unclaimed_host_and_fails_closed() -> None:
    rig = Rig([l4_host("l4-00")])
    store = SuspensionStore(rig.etcd)
    await store.suspend("l4-00")
    await store.suspend("l4-00")  # Idempotent.
    await rig.run()
    assert rig.provisioner.calls == []
    assert any("suspended" in item.reason for item in rig.reconciler.status.skipped)
    rig.etcd.fail = True
    with pytest.raises(ConnectionError):
        await store.resume("l4-00")
    rig.etcd.fail = False
    assert await store.list() == {"l4-00"}


@pytest.mark.asyncio
async def test_suspension_during_readiness_blocks_stale_plan() -> None:
    rig = Rig([l4_host("l4-00")])
    rig.provisioner.during_remote_check = SuspensionStore(rig.etcd).suspend
    await rig.run()
    assert rig.provisioner.calls == []
    assert await rig.claims() == {}


@pytest.mark.asyncio
async def test_suspended_failed_claim_is_not_retried_or_counted() -> None:
    rig = Rig([l4_host("l4-00")])
    rig.provisioner.failures["l4-00"] = ["failure"]
    await rig.run()
    await SuspensionStore(rig.etcd).suspend("l4-00")
    rig.clock.advance(10000)
    await rig.run()
    assert len(rig.provisioner.calls) == 1
    assert not any(rig.reconciler.status.held.values())
    assert rig.provisioner.cleaned == []


@pytest.mark.asyncio
async def test_unreadable_suspension_still_blocks_host() -> None:
    rig = Rig([l4_host("l4-00")])
    rig.etcd.data["/placement/suspensions/l4-00"] = (b"broken", 1)
    await rig.run()
    assert rig.provisioner.calls == []
    await SuspensionStore(rig.etcd).resume("l4-00")
    await rig.run()
    assert len(rig.provisioner.calls) == 1


@pytest.mark.asyncio
async def test_real_manual_teardown_stays_stopped_until_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from inference_proxy.api.admin import teardown_node
    from tests.provisioning.test_provisioner import _make_teardown_provisioner

    rig = Rig([l4_host("host1")])
    await rig.run()
    provisioner, _, _, _, _ = _make_teardown_provisioner()
    provisioner._registry = rig.registry
    monkeypatch.setattr(provisioner, "_reconcile_host", AsyncMock(return_value=False))
    monkeypatch.setattr(provisioner, "_upload_scripts", AsyncMock())
    tasks: list[asyncio.Task[None]] = []

    def background(
        coro: Coroutine[Any, Any, None], **kwargs: Any
    ) -> asyncio.Task[None]:

        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(provisioner, "fire_background", background)
    store = SuspensionStore(rig.etcd)
    await teardown_node(
        "host1",
        force=False,
        registry=rig.registry,
        provisioner=provisioner,
        suspensions=store,
    )
    await tasks[0]
    assert rig.registry.get("host1") is None
    rig.clock.advance(10000)
    await rig.run()
    assert len(rig.provisioner.calls) == 1
    await store.resume("host1")
    await rig.run()
    assert len(rig.provisioner.calls) == 2


@pytest.mark.asyncio
async def test_stale_resume_cannot_erase_new_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etcd = FakeEtcd()
    store = SuspensionStore(etcd)
    await store.suspend("l4-00")
    original_delete = etcd.delete_if_revision

    def concurrent_suspend(key: str, *, expected_mod_revision: int) -> bool:
        etcd.replace_if_revision(
            key,
            "new suspension",
            expected_mod_revision=expected_mod_revision,
            lease_id=0,
        )
        return original_delete(key, expected_mod_revision=expected_mod_revision)

    monkeypatch.setattr(etcd, "delete_if_revision", concurrent_suspend)
    with pytest.raises(ValueError, match="concurrently"):
        await store.resume("l4-00")
    assert await store.list() == {"l4-00"}


@pytest.mark.asyncio
async def test_resume_only_changes_exact_canonical_host() -> None:
    etcd = FakeEtcd()
    store = SuspensionStore(etcd)
    await store.suspend(" L4-00. ")
    await store.suspend("l4-001")
    await store.resume("L4-00.")
    await store.resume("l4-00")
    assert await store.list() == {"l4-001"}


@pytest.mark.asyncio
async def test_conflicting_suspend_does_not_report_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etcd = FakeEtcd()
    monkeypatch.setattr(etcd, "replace_if_revision", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="retry teardown"):
        await SuspensionStore(etcd).suspend("l4-00")


@pytest.mark.asyncio
async def test_claim_reset_does_not_resume_host() -> None:
    rig = Rig([l4_host("l4-00")])
    rig.provisioner.failures["l4-00"] = ["failure"]
    await rig.run()
    await SuspensionStore(rig.etcd).suspend("l4-00")
    assert await rig.reconciler.reset_claim("l4-00")
    await rig.run()
    assert len(rig.provisioner.calls) == 1
    assert await SuspensionStore(rig.etcd).list() == {"l4-00"}


@pytest.mark.asyncio
async def test_suspension_store_outage_blocks_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    rig = Rig([l4_host("l4-00")])
    monkeypatch.setattr(
        rig.reconciler._suspensions,
        "list",
        AsyncMock(side_effect=ConnectionError("unavailable")),
    )
    await rig.run()
    assert rig.provisioner.calls == []
    assert rig.reconciler.status.error
