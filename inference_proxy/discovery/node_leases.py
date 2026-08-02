"""Per-node etcd lease observation, adoption, and keepalive.

The watcher owns authoritative lease metadata from snapshot and watch KVs.
The health checker supplies the evidence that permits refreshing or adopting
one proxy-managed registration.  Keeping those responsibilities separate
prevents a gateway process from blindly extending every lease it can see.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import structlog

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.models.node import Node, NodeStatus

logger = structlog.get_logger()


@dataclass(frozen=True)
class NodeLeaseObservation:
    """Lease-bearing etcd metadata for one exact node-key revision."""

    value: bytes
    mod_revision: int
    lease_id: int


class NodeLeaseManager:
    """Maintain leases only after watcher metadata and health evidence agree."""

    def __init__(self, etcd_client: EtcdClient) -> None:
        self._etcd_client = etcd_client
        self._lock = threading.Lock()
        self._records: dict[str, NodeLeaseObservation] = {}
        self._snapshot_ready = False

    @property
    def snapshot_ready(self) -> bool:
        """Return whether an authoritative snapshot seeded lease metadata."""
        with self._lock:
            return self._snapshot_ready

    def reconcile_snapshot(
        self,
        records: dict[str, NodeLeaseObservation],
    ) -> None:
        """Replace observed metadata from one authoritative range snapshot."""
        with self._lock:
            self._records = dict(records)
            self._snapshot_ready = True

    def observe_put(
        self,
        node_id: str,
        *,
        value: bytes,
        mod_revision: int,
        lease_id: int,
    ) -> None:
        """Record a revision-gated PUT delivered by the watcher."""
        observation = NodeLeaseObservation(value, mod_revision, lease_id)
        with self._lock:
            current = self._records.get(node_id)
            if current is None or mod_revision >= current.mod_revision:
                self._records[node_id] = observation

    def observe_delete(self, node_id: str, *, mod_revision: int) -> None:
        """Forget metadata after a revision-gated DELETE event."""
        with self._lock:
            current = self._records.get(node_id)
            if current is None or mod_revision >= current.mod_revision:
                self._records.pop(node_id, None)

    def get(self, node_id: str) -> NodeLeaseObservation | None:
        """Return the latest immutable observation for tests and diagnostics."""
        with self._lock:
            return self._records.get(node_id)

    def maintain_after_success(self, node: Node) -> None:
        """Refresh or adopt one managed node after valid health evidence.

        Every etcd operation is isolated here so a keepalive failure cannot
        terminate the health-check thread or prevent later nodes from being
        probed and drained.
        """
        if not node.managed or node.status != NodeStatus.HEALTHY:
            return
        with self._lock:
            if not self._snapshot_ready:
                return
            observation = self._records.get(node.node_id)
        if observation is None:
            return
        if observation.lease_id:
            self._refresh(node.node_id, observation)
            return
        self._adopt(node.node_id, observation)

    def _refresh(
        self,
        node_id: str,
        observation: NodeLeaseObservation,
    ) -> None:
        try:
            remaining_ttl = self._etcd_client.refresh_lease(observation.lease_id)
        except Exception:
            logger.warning(
                "node_lease_refresh_failed",
                node_id=node_id,
                lease_id=observation.lease_id,
                exc_info=True,
            )
            return
        if remaining_ttl == -1:
            self._discard_if_current(node_id, observation)
            logger.warning(
                "node_lease_expired",
                node_id=node_id,
                lease_id=observation.lease_id,
            )
            return
        logger.debug(
            "node_lease_refreshed",
            node_id=node_id,
            lease_id=observation.lease_id,
            remaining_ttl=remaining_ttl,
        )

    def _adopt(
        self,
        node_id: str,
        observation: NodeLeaseObservation,
    ) -> None:
        try:
            lease_id = self._etcd_client.grant_node_lease()
        except Exception:
            logger.warning(
                "node_lease_grant_failed",
                node_id=node_id,
                exc_info=True,
            )
            return

        key = f"{self._etcd_client.prefix}{node_id}"
        try:
            attached = self._etcd_client.attach_lease_if_current(
                key,
                observation.value,
                expected_mod_revision=observation.mod_revision,
                expected_lease_id=observation.lease_id,
                lease_id=lease_id,
            )
        except Exception:
            logger.warning(
                "node_lease_adoption_failed",
                node_id=node_id,
                lease_id=lease_id,
                exc_info=True,
            )
            self._revoke_unused(node_id, lease_id)
            return

        if not attached:
            logger.debug(
                "node_lease_adoption_skipped_after_registration_changed",
                node_id=node_id,
                lease_id=lease_id,
            )
            self._revoke_unused(node_id, lease_id)
            return

        with self._lock:
            if self._records.get(node_id) == observation:
                self._records[node_id] = replace(observation, lease_id=lease_id)
        logger.info("node_lease_adopted", node_id=node_id, lease_id=lease_id)

    def _discard_if_current(
        self,
        node_id: str,
        observation: NodeLeaseObservation,
    ) -> None:
        with self._lock:
            if self._records.get(node_id) == observation:
                self._records.pop(node_id, None)

    def _revoke_unused(self, node_id: str, lease_id: int) -> None:
        try:
            self._etcd_client.revoke_lease(lease_id)
        except Exception:
            logger.warning(
                "unused_node_lease_revoke_failed",
                node_id=node_id,
                lease_id=lease_id,
                exc_info=True,
            )
