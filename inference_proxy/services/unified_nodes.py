"""Unified node list service — merges QUADS hosts with etcd nodes (D-01).

Produces a single list of AdminNodeResponse with computed state and
actions by joining QUADS GPU inventory with etcd-registered nodes by
hostname.  Etcd status wins when a host appears in both sources (D-05).
Every registered node remains visible; QUADS enriches matching nodes and
adds unregistered hosts that are currently available.
"""

from __future__ import annotations

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.admin import AdminNodeResponse, TaskStatusResponse
from inference_proxy.models.node import Node
from inference_proxy.models.quads import QUADSHost
from inference_proxy.quads.client import canonical_hostname
from inference_proxy.quads.poller import QUADSPoller
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker

# D-07: state -> available actions
_STATE_ACTIONS: dict[str, list[str]] = {
    "available": ["setup"],
    "healthy": ["teardown"],
    "unhealthy": ["teardown"],
    "provisioning": ["cancel"],
    "failed": ["setup", "teardown"],
    "draining": ["force_teardown"],
    "relaunching": [],
    "relaunch_failed": ["teardown"],
}


class UnifiedNodeService:
    """Merges QUADS hosts with etcd nodes into a unified view (D-01)."""

    def __init__(
        self,
        registry: NodeRegistry,
        poller: QUADSPoller | None,
        cb_registry: CircuitBreakerRegistry,
        tracker: ConnectionTracker,
    ) -> None:
        self._registry = registry
        self._poller = poller
        self._cb_registry = cb_registry
        self._tracker = tracker

    def get_unified_nodes(
        self,
        task_map: dict[str, TaskStatusResponse] | None = None,
        *,
        viewer_admin: bool = True,
        viewer_email: str | None = None,
    ) -> list[AdminNodeResponse]:
        """Return merged QUADS + etcd node list sorted by node_id.

        ``viewer_admin`` controls the fleet-visibility contract: a non-admin
        viewer never sees admin-only nodes, never sees QUADS-only available
        hosts, and receives no operational actions. ``viewer_email`` is the
        signed-in Google user's email (lowercase): nodes owned by someone
        else are excluded from the non-admin view, matching ``/v1/models``
        and the endpoint picker (RFE-107 ownership privacy).
        """
        etcd_map = {canonical_hostname(n.node_id): n for n in self._registry.get_all()}

        # Graceful degradation: no QUADS -> etcd-only
        if self._poller is None:
            etcd_only = [
                self._from_etcd(n, task_map=task_map) for n in etcd_map.values()
            ]
            return self._finalize(etcd_only, viewer_admin, viewer_email)

        quads_map: dict[str, QUADSHost] = {h.hostname: h for h in self._poller.hosts}
        available_set = set(self._poller.available_hostnames)
        result: list[AdminNodeResponse] = []

        for hostname, host in quads_map.items():
            etcd_node = etcd_map.pop(hostname, None)
            if etcd_node is not None:
                # D-05: etcd status wins
                result.append(self._from_etcd(etcd_node, host, task_map=task_map))
            elif hostname in available_set and viewer_admin:
                result.append(self._from_available(host))
            # else: not available and not in etcd -> skip

        # Registry membership is authoritative for operational visibility.
        # QUADS may be empty, stale, or omit broken/retired hosts, but none of
        # those conditions should hide a node the proxy is still routing to.
        for node in etcd_map.values():
            result.append(self._from_etcd(node, task_map=task_map))

        return self._finalize(result, viewer_admin, viewer_email)

    @staticmethod
    def _finalize(
        result: list[AdminNodeResponse],
        viewer_admin: bool,
        viewer_email: str | None = None,
    ) -> list[AdminNodeResponse]:
        """Apply the fleet-visibility contract to a completed node list."""
        if viewer_admin:
            return sorted(result, key=lambda r: r.node_id)
        filtered = [
            # Non-admin callers never see admin-only nodes, operational
            # actions, or any node's owner email (ownership is private per
            # RFE-107: owned nodes are already hidden from /v1/models and the
            # endpoint picker, so the fleet must not disclose who owns them).
            # Nodes owned by another user are excluded entirely so their
            # endpoint/model/engine/artifact/GPU identity is not disclosed.
            item.model_copy(update={"actions": [], "owner": ""})
            for item in result
            if not item.admin_only
            and not (
                viewer_email
                and item.owner
                and item.owner.lower() != viewer_email
            )
        ]
        return sorted(filtered, key=lambda r: r.node_id)

    def _from_etcd(
        self,
        node: Node,
        host: QUADSHost | None = None,
        *,
        task_map: dict[str, TaskStatusResponse] | None = None,
    ) -> AdminNodeResponse:
        state = node.status.value
        breaker = self._cb_registry.get(node.node_id)
        task = task_map.get(node.node_id) if task_map else None
        actions = list(_STATE_ACTIONS.get(state, []))
        if node.self_setup:
            actions = ["remove"]
        elif state == "available" and not node.managed:
            actions.append("remove")
        return AdminNodeResponse(
            node_id=node.node_id,
            name=node.name,
            endpoint=node.endpoint,
            model=node.model,
            status=node.status.value,
            active_connections=self._tracker.get(node.node_id),
            circuit_breaker_state=breaker.state if breaker else "closed",
            engine=node.engine,
            artifact_id=node.artifact_id,
            llamacpp_runtime=node.llamacpp_runtime,
            state=state,
            actions=actions,
            gpu_vendor=host.gpu_vendor if host else None,
            gpu_model=host.gpu_model if host else None,
            gpu_count=host.gpu_count if host else None,
            managed=node.managed,
            self_setup=node.self_setup,
            admin_only=node.admin_only,
            owner=node.owner,
            failed_step=task.failed_step if task else None,
            error=task.error if task else None,
        )

    @staticmethod
    def _from_available(host: QUADSHost) -> AdminNodeResponse:
        return AdminNodeResponse(
            node_id=host.hostname,
            endpoint="",
            model="",
            status="",
            active_connections=0,
            circuit_breaker_state="closed",
            state="available",
            actions=list(_STATE_ACTIONS["available"]),
            gpu_vendor=host.gpu_vendor,
            gpu_model=host.gpu_model,
            gpu_count=host.gpu_count,
        )
