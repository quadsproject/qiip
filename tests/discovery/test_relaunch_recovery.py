"""Restart reconciliation tests for interrupted llama.cpp relaunches."""

from __future__ import annotations

from unittest.mock import MagicMock

from inference_proxy.discovery.etcd_client import EtcdRecord
from inference_proxy.discovery.relaunch_recovery import (
    reconcile_interrupted_relaunch,
)
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus

_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["gpu01"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _client() -> MagicMock:
    client = MagicMock()
    client.prefix = "/nodes/"
    return client


def _record(node: Node, revision: int = 41) -> EtcdRecord:
    key, value = node_to_etcd(node, "/nodes/")
    return EtcdRecord(key.encode(), value, revision, 7001)


def test_non_relaunch_and_malformed_records_pass_to_normal_discovery() -> None:
    client = _client()
    healthy = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.HEALTHY,
        )
    )
    malformed = EtcdRecord(b"/nodes/gpu01", b"not-json", 42, 7001)

    assert reconcile_interrupted_relaunch(client, healthy, _POLICY) is healthy
    assert reconcile_interrupted_relaunch(client, malformed, _POLICY) is malformed
    client.replace_if_revision.assert_not_called()


def test_invalid_relaunch_record_is_dropped_fail_closed() -> None:
    client = _client()
    invalid = EtcdRecord(
        b"/nodes/gpu01",
        b'{"status":"relaunching","endpoint":"forbidden:8000"}',
        41,
        7001,
    )

    assert reconcile_interrupted_relaunch(client, invalid, _POLICY) is None
    client.replace_if_revision.assert_not_called()


def test_repeated_revision_conflicts_drop_stale_relaunch_state() -> None:
    client = _client()
    interrupted = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCHING,
            managed=True,
        )
    )
    client.replace_if_revision.return_value = None
    client.get_record.return_value = interrupted

    assert reconcile_interrupted_relaunch(client, interrupted, _POLICY) is None
    assert client.replace_if_revision.call_count == 3
    assert client.get_record.call_count == 3


def test_missing_record_after_cas_conflict_is_not_resurrected() -> None:
    client = _client()
    interrupted = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCHING,
            managed=True,
        )
    )
    client.replace_if_revision.return_value = None
    client.get_record.return_value = None

    assert reconcile_interrupted_relaunch(client, interrupted, _POLICY) is None
    client.replace_if_revision.assert_called_once()
