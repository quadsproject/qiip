"""Restart reconciliation tests for interrupted llama.cpp relaunches."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from inference_proxy.discovery.etcd_client import EtcdRecord
from inference_proxy.discovery.relaunch_recovery import (
    reconcile_interrupted_relaunch,
)
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.state import ProvisioningState, ProvisioningStep

_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["gpu01"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _client() -> MagicMock:
    client = MagicMock()
    client.prefix = "/nodes/"
    client.get_record.return_value = None
    return client


def _record(node: Node, revision: int = 41) -> EtcdRecord:
    key, value = node_to_etcd(node, "/nodes/")
    return EtcdRecord(key.encode(), value, revision, 7001)


def _task_record(
    step: ProvisioningStep = ProvisioningStep.STARTING_LLAMACPP,
    *,
    hostname: str = "gpu01",
    revision: int = 51,
    extra: dict[str, object] | None = None,
) -> EtcdRecord:
    state = ProvisioningState(
        hostname=hostname,
        current_step=step,
        started_at=datetime(2026, 8, 6, 1, 54, tzinfo=UTC),
        updated_at=datetime(2026, 8, 6, 1, 55, tzinfo=UTC),
    )
    document = state.model_dump(mode="json")
    document.update(extra or {})
    return EtcdRecord(
        key=f"/provisioning/{hostname}".encode(),
        value=json.dumps(document).encode(),
        mod_revision=revision,
        lease_id=0,
    )


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
    non_object = EtcdRecord(b"/nodes/gpu01", b"[]", 43, 7001)

    assert reconcile_interrupted_relaunch(client, healthy, _POLICY) is healthy
    assert reconcile_interrupted_relaunch(client, malformed, _POLICY) is malformed
    assert reconcile_interrupted_relaunch(client, non_object, _POLICY) is non_object
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


def test_relaunch_recovery_marks_matching_task_failed_with_revision_cas() -> None:
    client = _client()
    interrupted = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCHING,
            managed=True,
        )
    )
    task = _task_record(extra={"future_field": "preserved"})
    client.get_record.side_effect = lambda key: (
        task if key == "/provisioning/gpu01" else None
    )
    client.replace_if_revision.side_effect = [42, 52]

    recovered = reconcile_interrupted_relaunch(client, interrupted, _POLICY)

    assert recovered is not None
    assert recovered.mod_revision == 42
    assert client.replace_if_revision.call_count == 2
    task_call = client.replace_if_revision.call_args_list[1]
    assert task_call.args[0] == "/provisioning/gpu01"
    assert task_call.kwargs == {"expected_mod_revision": 51, "lease_id": 0}
    document = json.loads(task_call.args[1])
    state = ProvisioningState.model_validate(document)
    assert state.current_step is ProvisioningStep.FAILED
    assert state.failed_step == ProvisioningStep.STARTING_LLAMACPP.value
    assert state.error == (
        "Gateway restarted during llama.cpp relaunch; teardown required"
    )
    assert state.started_at == datetime(2026, 8, 6, 1, 54, tzinfo=UTC)
    assert document["future_field"] == "preserved"


def test_existing_terminal_node_repairs_stale_relaunch_task() -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.return_value = _task_record(ProvisioningStep.ROLLING_BACK)
    client.replace_if_revision.return_value = 52

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal

    task_call = client.replace_if_revision.call_args
    assert task_call.args[0] == "/provisioning/gpu01"
    state = ProvisioningState.model_validate_json(task_call.args[1])
    assert state.current_step is ProvisioningStep.FAILED
    assert state.failed_step == ProvisioningStep.ROLLING_BACK.value


def test_task_revision_conflict_never_overwrites_or_retries() -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.return_value = _task_record()
    client.replace_if_revision.return_value = None

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal
    client.replace_if_revision.assert_called_once()
    client.get_record.assert_called_once_with("/provisioning/gpu01")


@pytest.mark.parametrize(
    "step",
    [
        ProvisioningStep.PENDING,
        ProvisioningStep.COMPLETE,
        ProvisioningStep.FAILED,
        ProvisioningStep.TEARDOWN_COMPLETE,
    ],
)
def test_unrelated_or_finished_task_is_not_rewritten(step: ProvisioningStep) -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.return_value = _task_record(step)

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal
    client.replace_if_revision.assert_not_called()


def test_task_read_failure_does_not_drop_terminal_node() -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.side_effect = RuntimeError("etcd unavailable")

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal
    client.replace_if_revision.assert_not_called()


def test_malformed_task_does_not_drop_terminal_node() -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.return_value = EtcdRecord(
        b"/provisioning/gpu01",
        b"not-json",
        51,
        0,
    )

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal
    client.replace_if_revision.assert_not_called()


def test_task_write_failure_does_not_drop_terminal_node() -> None:
    client = _client()
    terminal = _record(
        Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            status=NodeStatus.RELAUNCH_FAILED,
            managed=True,
        )
    )
    client.get_record.return_value = _task_record()
    client.replace_if_revision.side_effect = RuntimeError("etcd unavailable")

    assert reconcile_interrupted_relaunch(client, terminal, _POLICY) is terminal
    client.replace_if_revision.assert_called_once()
