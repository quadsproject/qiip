"""Managed node lease adoption and keepalive behavior (R6)."""

from __future__ import annotations

from unittest.mock import MagicMock

from structlog.testing import capture_logs

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.node_leases import (
    NodeLeaseManager,
    NodeLeaseObservation,
)
from inference_proxy.models.node import Node, NodeStatus


def _node(*, managed: bool = True) -> Node:
    return Node(
        node_id="gpu01",
        endpoint="gpu01:8000",
        status=NodeStatus.HEALTHY,
        model="model-a",
        managed=managed,
    )


def _manager(
    observation: NodeLeaseObservation,
) -> tuple[NodeLeaseManager, MagicMock]:
    client = MagicMock(spec=EtcdClient)
    client.prefix = "/nodes/"
    manager = NodeLeaseManager(client)
    manager.reconcile_snapshot({"gpu01": observation})
    return manager, client


def test_restart_recovers_and_refreshes_existing_lease() -> None:
    """A snapshot-recovered lease is refreshed without a replacement grant."""
    observation = NodeLeaseObservation(b"node-json", 41, 7001)
    manager, client = _manager(observation)
    client.refresh_lease.return_value = 600

    manager.maintain_after_success(_node())

    client.refresh_lease.assert_called_once_with(7001)
    client.grant_node_lease.assert_not_called()
    assert manager.get("gpu01") == observation


def test_expired_lease_return_is_not_treated_as_refresh_success() -> None:
    """etcd3gw reports an expired lease as -1 instead of raising."""
    manager, client = _manager(NodeLeaseObservation(b"node-json", 41, 7001))
    client.refresh_lease.return_value = -1

    with capture_logs() as logs:
        manager.maintain_after_success(_node())

    assert manager.get("gpu01") is None
    assert any(event["event"] == "node_lease_expired" for event in logs)
    assert not any(event["event"] == "node_lease_refreshed" for event in logs)


def test_unleased_managed_node_is_adopted_after_successful_probe() -> None:
    """An existing managed key converges without requiring reprovisioning."""
    observation = NodeLeaseObservation(b"node-json", 41, 0)
    manager, client = _manager(observation)
    client.grant_node_lease.return_value = 7002
    client.attach_lease_if_current.return_value = True

    manager.maintain_after_success(_node())

    client.attach_lease_if_current.assert_called_once_with(
        "/nodes/gpu01",
        b"node-json",
        expected_mod_revision=41,
        expected_lease_id=0,
        lease_id=7002,
    )
    assert manager.get("gpu01") == NodeLeaseObservation(b"node-json", 41, 7002)


def test_lease_adoption_cas_preserves_concurrent_registration() -> None:
    """A changed key makes adoption fail rather than accepting a stale value."""
    observation = NodeLeaseObservation(b"old-node-json", 41, 0)
    manager, client = _manager(observation)
    client.grant_node_lease.return_value = 7002
    client.attach_lease_if_current.return_value = False

    manager.maintain_after_success(_node())

    client.attach_lease_if_current.assert_called_once_with(
        "/nodes/gpu01",
        b"old-node-json",
        expected_mod_revision=41,
        expected_lease_id=0,
        lease_id=7002,
    )
    client.revoke_lease.assert_called_once_with(7002)
    assert manager.get("gpu01") == observation


def test_unmanaged_node_lease_is_never_refreshed() -> None:
    """The external registrant remains the sole owner of its lease."""
    manager, client = _manager(NodeLeaseObservation(b"node-json", 41, 7001))

    manager.maintain_after_success(_node(managed=False))

    client.refresh_lease.assert_not_called()
    client.grant_node_lease.assert_not_called()


def test_lease_maintenance_waits_for_authoritative_snapshot() -> None:
    """Startup cannot mistake missing metadata for an unleased registration."""
    client = MagicMock(spec=EtcdClient)
    manager = NodeLeaseManager(client)

    manager.maintain_after_success(_node())

    client.refresh_lease.assert_not_called()
    client.grant_node_lease.assert_not_called()
