"""A manual takeover and automatic placement contend for one host.

These run the real ``NodeProvisioner.update_node_owner`` against the real
reconciler. The two share one host lifecycle coordinator, as they do in the
gateway, so what is tested is the coordination and not a pre-owned record.
"""

from __future__ import annotations

import asyncio
import threading
from typing import cast
from unittest.mock import MagicMock

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdClient, EtcdRecord
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import NodeStatus
from inference_proxy.placement.claims import ClaimState
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
)
from tests.placement.fakes import FakeEtcd, node
from tests.placement.test_reconciler import Rig, _failed_once, _l4s

HOST = "l4-00"
OWNER = "admin@example.com"
BUSY = "lifecycle operation in progress"


class NodeEtcd:
    """The slice of ``EtcdClient`` an owner update uses, over the rig's etcd."""

    prefix = "/nodes/"

    def __init__(self, etcd: FakeEtcd) -> None:
        self.etcd = etcd
        self.writing = threading.Event()
        self.may_write: threading.Event | None = None

    def get_record(self, key: str) -> EtcdRecord | None:
        data = self.etcd.data.get(key)
        if data is None:
            return None
        return EtcdRecord(
            key=key.encode(), value=data[0], mod_revision=data[1], lease_id=0
        )

    def replace_if_revision(
        self,
        key: str,
        value: str | bytes,
        *,
        expected_mod_revision: int,
        lease_id: int,
    ) -> int | None:
        self.writing.set()
        if self.may_write is not None:
            assert self.may_write.wait(5), "the test never released the write"
        return self.etcd.replace_if_revision(
            key, value, expected_mod_revision=expected_mod_revision, lease_id=lease_id
        )


def _owner_api(rig: Rig) -> tuple[NodeProvisioner, NodeEtcd]:
    """A normally constructed provisioner sharing the rig's coordinator."""
    etcd = NodeEtcd(rig.etcd)
    provisioner = NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=cast(EtcdClient, etcd),
        settings=ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=EndpointPolicy.from_values(
            allowed_hosts=[HOST], allowed_networks=[], allowed_ports=[8000]
        ),
        registry=rig.registry,
        lifecycle_coordinator=rig.provisioner.lifecycle,
        nfs_export="nfs.example:/exports/huggingface",
    )
    return provisioner, etcd


def _publish(rig: Rig) -> None:
    """Write the registry's record for HOST to etcd, as the provisioner would."""
    record = rig.registry.get(HOST)
    assert record is not None
    key, value = node_to_etcd(record, "/nodes/")
    current = rig.etcd.data.get(key, (b"", 0))[1]
    assert rig.etcd.replace_if_revision(
        key, value, expected_mod_revision=current, lease_id=0
    )


def _etcd_owner(etcd: NodeEtcd) -> str:
    record = etcd.get_record(f"/nodes/{HOST}")
    assert record is not None
    return record.value.decode()


@pytest.mark.asyncio
async def test_a_takeover_before_the_retry_is_respected() -> None:
    rig = await _failed_once()
    _publish(rig)
    owner_api, etcd = _owner_api(rig)

    owned = await owner_api.update_node_owner(HOST, OWNER)
    assert (owned.owner, owned.placement) == (OWNER, None)
    assert not rig.provisioner.lifecycle.is_busy(HOST)

    await rig.run()

    record = rig.registry.get(HOST)
    assert record is not None and record.owner == OWNER
    assert OWNER in _etcd_owner(etcd)
    assert len(rig.provisioner.calls) == 1 and rig.provisioner.cleaned == []
    assert await rig.claims() == {}, "the claim yields to the new owner"


@pytest.mark.asyncio
async def test_a_takeover_during_a_retry_is_refused_not_erased() -> None:
    """The retry decided to clear the failed record before it asked the host."""
    rig = await _failed_once()
    _publish(rig)
    owner_api, etcd = _owner_api(rig)
    before = _etcd_owner(etcd)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def hold(hostname: str) -> None:
        entered.set()
        await resume.wait()

    rig.provisioner.during_remote_check = hold
    retry = asyncio.create_task(rig.run())
    await asyncio.wait_for(entered.wait(), 1)
    assert rig.provisioner.lifecycle.is_busy(HOST)
    try:
        with pytest.raises(ProvisioningError, match=BUSY):
            await owner_api.update_node_owner(HOST, OWNER)
        assert _etcd_owner(etcd) == before, "a refused takeover wrote nothing"
        record = rig.registry.get(HOST)
        assert record is not None and record.owner == ""
    finally:
        resume.set()
        await retry

    # The retry ran to completion as an automatic placement...
    assert len(rig.provisioner.calls) == 2
    assert (await rig.claims())[HOST].state is ClaimState.ACTIVE
    # ...and the host is free again, so the same request now succeeds.
    _publish(rig)
    owned = await owner_api.update_node_owner(HOST, OWNER)
    assert (owned.owner, owned.placement) == (OWNER, None)
    record = rig.registry.get(HOST)
    assert record is not None and record.owner == OWNER
    assert not rig.provisioner.lifecycle.is_busy(HOST)


@pytest.mark.asyncio
async def test_a_takeover_during_a_running_provision_is_refused() -> None:
    """A provision writes its final record from its own arguments: no owner."""
    rig = Rig(_l4s(1))
    rig.provisioner.hold = True
    owner_api, _ = _owner_api(rig)
    await rig.reconciler.reconcile_once()
    await asyncio.sleep(0)
    assert rig.provisioner.lifecycle.is_busy(HOST)

    with pytest.raises(ProvisioningError, match=BUSY):
        await owner_api.update_node_owner(HOST, OWNER)

    rig.provisioner.gates[HOST].set()
    await rig.provisioner.drain()
    _publish(rig)
    assert (await owner_api.update_node_owner(HOST, OWNER)).owner == OWNER
    assert not rig.provisioner.lifecycle.is_busy(HOST)


@pytest.mark.asyncio
async def test_an_owner_can_be_set_and_cleared_on_a_node_placement_never_touched() -> (
    None
):
    rig = Rig(_l4s(1), enabled=False)
    rig.registry.add(node(HOST, NodeStatus.HEALTHY))
    _publish(rig)
    owner_api, _ = _owner_api(rig)

    assert (await owner_api.update_node_owner(HOST, OWNER)).owner == OWNER
    assert (await owner_api.update_node_owner(HOST, "")).owner == ""
    assert not rig.provisioner.lifecycle.is_busy(HOST)


@pytest.mark.asyncio
async def test_a_cancelled_owner_update_keeps_the_host_until_its_write_lands() -> None:
    """A client that disconnects must not free the host under a live etcd write."""
    rig = await _failed_once()
    _publish(rig)
    owner_api, etcd = _owner_api(rig)
    etcd.may_write = threading.Event()

    update = asyncio.create_task(owner_api.update_node_owner(HOST, OWNER))
    assert await asyncio.to_thread(etcd.writing.wait, 5)
    update.cancel()
    with pytest.raises(asyncio.CancelledError):
        await update

    # The caller is gone; the write is still in flight and still holds the host.
    assert rig.provisioner.lifecycle.is_busy(HOST)
    await rig.run()
    assert len(rig.provisioner.calls) == 1, "placement launched under a live write"

    etcd.may_write.set()
    for _ in range(200):
        if not rig.provisioner.lifecycle.is_busy(HOST):
            break
        await asyncio.sleep(0.01)
    assert not rig.provisioner.lifecycle.is_busy(HOST)
    record = rig.registry.get(HOST)
    assert record is not None and (record.owner, record.placement) == (OWNER, None)
    assert OWNER in _etcd_owner(etcd)


@pytest.mark.asyncio
async def test_an_owner_write_cancelled_at_shutdown_still_frees_the_host() -> None:
    rig = await _failed_once()
    _publish(rig)
    owner_api, etcd = _owner_api(rig)
    etcd.may_write = threading.Event()

    update = asyncio.create_task(owner_api.update_node_owner(HOST, OWNER))
    assert await asyncio.to_thread(etcd.writing.wait, 5)
    (write,) = owner_api._owner_updates
    write.cancel()
    with pytest.raises(asyncio.CancelledError):
        await update
    etcd.may_write.set()

    assert not rig.provisioner.lifecycle.is_busy(HOST)
    assert owner_api._owner_updates == set()
