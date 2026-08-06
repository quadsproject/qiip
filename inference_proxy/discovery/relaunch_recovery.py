"""Fail-closed recovery for llama.cpp relaunches interrupted by a restart."""

from __future__ import annotations

import json

import structlog

from inference_proxy.discovery.etcd_client import EtcdClient, EtcdRecord
from inference_proxy.discovery.registry import node_with_status
from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import NodeStatus

logger = structlog.get_logger()

_RECONCILE_ATTEMPTS = 3


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
        if (
            not isinstance(document, dict)
            or document.get("status") != NodeStatus.RELAUNCHING.value
        ):
            return record
        node = node_from_etcd(
            record.key,
            record.value,
            etcd_client.prefix,
            endpoint_policy=endpoint_policy,
        )
        if node is None:
            return None

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
