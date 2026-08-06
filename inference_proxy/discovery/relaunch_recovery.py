"""Fail-closed recovery for llama.cpp relaunches interrupted by a restart."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog
from pydantic import ValidationError

from inference_proxy.discovery.etcd_client import EtcdClient, EtcdRecord
from inference_proxy.discovery.registry import node_with_status
from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import NodeStatus
from inference_proxy.provisioning.state import ProvisioningState, ProvisioningStep

logger = structlog.get_logger()

_RECONCILE_ATTEMPTS = 3
_INTERRUPTED_RELAUNCH_ERROR = (
    "Gateway restarted during llama.cpp relaunch; teardown required"
)
_RELAUNCH_TASK_STEPS = frozenset(
    {
        ProvisioningStep.RELAUNCH_VALIDATING,
        ProvisioningStep.DRAINING,
        ProvisioningStep.STOPPING_LLAMACPP,
        ProvisioningStep.STARTING_LLAMACPP,
        ProvisioningStep.HEALTH_POLL,
        ProvisioningStep.REGISTERING,
        ProvisioningStep.ROLLING_BACK,
    }
)


def _reconcile_interrupted_relaunch_task(
    etcd_client: EtcdClient,
    hostname: str,
) -> None:
    """Mark only the exact stale relaunch task revision as failed."""
    key = f"/provisioning/{hostname}"
    try:
        record = etcd_client.get_record(key)
    except Exception:
        logger.warning(
            "interrupted llama.cpp relaunch task read failed",
            hostname=hostname,
            exc_info=True,
        )
        return
    if record is None:
        return

    try:
        document = json.loads(record.value)
        state = ProvisioningState.model_validate(document)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
        logger.warning(
            "interrupted llama.cpp relaunch task is malformed",
            hostname=hostname,
        )
        return
    if (
        not isinstance(document, dict)
        or state.hostname != hostname
        or state.current_step not in _RELAUNCH_TASK_STEPS
    ):
        return

    failed = state.model_copy(
        update={
            "current_step": ProvisioningStep.FAILED,
            "updated_at": datetime.now(UTC),
            "failed_step": state.current_step.value,
            "error": _INTERRUPTED_RELAUNCH_ERROR,
        }
    )
    # Preserve unknown fields written by a newer gateway version while this
    # version updates only the task fields it owns.
    document.update(failed.model_dump(mode="json"))
    value = json.dumps(document).encode("utf-8")
    try:
        committed_revision = etcd_client.replace_if_revision(
            key,
            value,
            expected_mod_revision=record.mod_revision,
            lease_id=0,
        )
    except Exception:
        logger.warning(
            "interrupted llama.cpp relaunch task write failed",
            hostname=hostname,
            exc_info=True,
        )
        return
    if committed_revision is None:
        logger.warning(
            "interrupted llama.cpp relaunch task changed during reconciliation",
            hostname=hostname,
        )
        return
    logger.warning(
        "interrupted llama.cpp relaunch task marked failed",
        hostname=hostname,
        failed_step=state.current_step.value,
    )


def reconcile_interrupted_relaunch(
    etcd_client: EtcdClient,
    initial_record: EtcdRecord,
    endpoint_policy: EndpointPolicy,
) -> EtcdRecord | None:
    """Replace an orphaned relaunch with a persistent teardown-only record.

    The returned record is the exact value and revision committed by the CAS,
    or the latest concurrently-written record when another writer wins.  A
    repeated conflict returns ``None`` so stale state is never routed.
    """
    record: EtcdRecord | None = initial_record
    for _attempt in range(_RECONCILE_ATTEMPTS):
        if record is None:
            return None
        try:
            document = json.loads(record.value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # The normal discovery parser owns diagnostics for malformed
            # records. Avoid parsing every healthy record twice at startup.
            return record
        if not isinstance(document, dict):
            return record
        status = document.get("status")
        if status not in {
            NodeStatus.RELAUNCHING.value,
            NodeStatus.RELAUNCH_FAILED.value,
        }:
            return record
        node = node_from_etcd(
            record.key,
            record.value,
            etcd_client.prefix,
            endpoint_policy=endpoint_policy,
        )
        if node is None:
            return None
        if status == NodeStatus.RELAUNCH_FAILED.value:
            _reconcile_interrupted_relaunch_task(etcd_client, node.node_id)
            return record

        failed = node_with_status(
            node,
            NodeStatus.RELAUNCH_FAILED,
            llamacpp_runtime=None,
        )
        key, value = node_to_etcd(failed, etcd_client.prefix)
        committed_revision = etcd_client.replace_if_revision(
            key,
            value,
            expected_mod_revision=record.mod_revision,
            # Failure records intentionally outlive the serving lease, like
            # provisioning FAILED records, until an operator tears them down.
            lease_id=0,
        )
        if committed_revision is not None:
            logger.warning(
                "interrupted llama.cpp relaunch requires teardown",
                hostname=node.node_id,
            )
            _reconcile_interrupted_relaunch_task(etcd_client, node.node_id)
            return EtcdRecord(
                key=key.encode(),
                value=value,
                mod_revision=committed_revision,
                lease_id=0,
            )
        record = etcd_client.get_record(key)

    logger.warning(
        "interrupted llama.cpp relaunch reconciliation raced repeatedly",
        key=initial_record.key,
    )
    return None
