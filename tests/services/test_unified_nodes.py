"""Unit tests for UnifiedNodeService merge logic, state computation, filtering."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.admin import TaskStatusResponse
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.models.quads import QUADSHost
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.services.unified_nodes import UnifiedNodeService


def _host(
    hostname: str = "gpu01",
    gpu_vendor: str = "NVIDIA",
    gpu_model: str = "A100",
    gpu_count: int = 4,
) -> QUADSHost:
    return QUADSHost(
        hostname=hostname,
        gpu_vendor=gpu_vendor,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
    )


def _node(
    node_id: str = "gpu01",
    endpoint: str = "10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
) -> Node:
    return Node(node_id=node_id, endpoint=endpoint, status=status, model=model)


def _poller(
    hosts: list[QUADSHost] | None = None,
    available: list[str] | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.hosts = hosts or []
    mock.available_hostnames = available or []
    return mock


def _service(
    registry: NodeRegistry | None = None,
    poller: MagicMock | None = None,
    cb_registry: CircuitBreakerRegistry | None = None,
    tracker: ConnectionTracker | None = None,
    *,
    include_poller: bool = True,
) -> UnifiedNodeService:
    resolved_poller = poller
    if resolved_poller is None and include_poller:
        resolved_poller = MagicMock()
    return UnifiedNodeService(
        registry=registry or NodeRegistry(),
        poller=resolved_poller,
        cb_registry=cb_registry or CircuitBreakerRegistry(),
        tracker=tracker or ConnectionTracker(),
    )


class TestAvailableOnly:
    """QUADS-only host (available, not in etcd) returns state=available."""

    def test_available_host_state_and_actions(self) -> None:
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(poller=poller)
        nodes = svc.get_unified_nodes()

        assert len(nodes) == 1
        n = nodes[0]
        assert n.node_id == "gpu01"
        assert n.state == "available"
        assert n.actions == ["setup"]
        assert n.endpoint == ""
        assert n.model == ""
        assert n.active_connections == 0
        assert n.circuit_breaker_state == "closed"

    def test_available_host_gpu_fields(self) -> None:
        poller = _poller(
            hosts=[_host("gpu01", gpu_vendor="NVIDIA", gpu_model="A100", gpu_count=4)],
            available=["gpu01"],
        )
        svc = _service(poller=poller)
        n = svc.get_unified_nodes()[0]

        assert n.gpu_vendor == "NVIDIA"
        assert n.gpu_model == "A100"
        assert n.gpu_count == 4


class TestEtcdNodeStates:
    """QUADS host in etcd returns etcd status as state with correct actions."""

    def test_healthy_state_and_actions(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.HEALTHY))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.state == "healthy"
        assert n.actions == ["teardown"]
        assert n.gpu_vendor == "NVIDIA"
        assert n.gpu_model == "A100"

    def test_unhealthy_state_and_actions(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.UNHEALTHY))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.state == "unhealthy"
        assert n.actions == ["teardown", "retry"]

    def test_provisioning_state_and_actions(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.PROVISIONING))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.state == "provisioning"
        assert n.actions == ["cancel"]

    def test_draining_state_and_actions(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.DRAINING))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.state == "draining"
        assert n.actions == ["force_teardown"]


class TestFiltering:
    """Exclusion rules for the unified list."""

    def test_etcd_node_not_in_quads_excluded(self) -> None:
        """D-03: etcd node without matching QUADS host is excluded."""
        registry = NodeRegistry()
        registry.add(_node("orphan-node"))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        nodes = svc.get_unified_nodes()
        ids = [n.node_id for n in nodes]
        assert "orphan-node" not in ids

    def test_quads_host_not_available_not_in_etcd_excluded(self) -> None:
        """QUADS host that is neither available nor in etcd is skipped."""
        poller = _poller(hosts=[_host("gpu01")], available=[])
        svc = _service(poller=poller)

        assert svc.get_unified_nodes() == []

    def test_quads_host_not_available_but_in_etcd_included(self) -> None:
        """QUADS host not in available list but registered in etcd should appear (provisioned)."""
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.HEALTHY))
        poller = _poller(hosts=[_host("gpu01")], available=[])
        svc = _service(registry=registry, poller=poller)

        nodes = svc.get_unified_nodes()
        assert len(nodes) == 1
        assert nodes[0].state == "healthy"


class TestGracefulDegradation:
    """When QUADS poller is None, return etcd-only nodes."""

    def test_none_poller_returns_etcd_nodes(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.HEALTHY))
        svc = _service(registry=registry, include_poller=False)

        nodes = svc.get_unified_nodes()
        assert len(nodes) == 1
        assert nodes[0].state == "healthy"
        assert nodes[0].actions == ["teardown"]
        assert nodes[0].gpu_vendor is None

    def test_none_poller_empty_registry(self) -> None:
        svc = _service(include_poller=False)
        assert svc.get_unified_nodes() == []


class TestEnrichedFields:
    """active_connections and circuit_breaker_state populated from tracker/breaker."""

    def test_active_connections_from_tracker(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01"))
        tracker = ConnectionTracker()
        tracker.increment("gpu01")
        tracker.increment("gpu01")
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller, tracker=tracker)

        n = svc.get_unified_nodes()[0]
        assert n.active_connections == 2

    def test_circuit_breaker_state_from_registry(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01"))
        cb = CircuitBreakerRegistry()
        breaker = cb.get_or_create("gpu01")
        for _ in range(3):
            breaker.record_failure()
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller, cb_registry=cb)

        n = svc.get_unified_nodes()[0]
        assert n.circuit_breaker_state == "open"


class TestSorting:
    """Output is sorted by node_id for stable ordering."""

    def test_sorted_by_node_id(self) -> None:
        poller = _poller(
            hosts=[_host("gpu03"), _host("gpu01"), _host("gpu02")],
            available=["gpu03", "gpu01", "gpu02"],
        )
        svc = _service(poller=poller)

        nodes = svc.get_unified_nodes()
        ids = [n.node_id for n in nodes]
        assert ids == ["gpu01", "gpu02", "gpu03"]


class TestFailedState:
    """D-02: Failed nodes get setup + teardown actions and error fields from task_map."""

    def test_failed_state_and_actions(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.FAILED))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.state == "failed"
        assert n.actions == ["setup", "teardown"]

    def test_failed_node_error_fields_from_task_map(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.FAILED))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        now = datetime.now(UTC)
        task_map = {
            "gpu01": TaskStatusResponse(
                hostname="gpu01",
                current_step="failed",
                started_at=now,
                updated_at=now,
                failed_step="uploading_scripts",
                error="connection refused",
            )
        }
        nodes = svc.get_unified_nodes(task_map=task_map)
        n = nodes[0]
        assert n.failed_step == "uploading_scripts"
        assert n.error == "connection refused"

    def test_no_task_map_error_fields_none(self) -> None:
        registry = NodeRegistry()
        registry.add(_node("gpu01", status=NodeStatus.FAILED))
        poller = _poller(hosts=[_host("gpu01")], available=["gpu01"])
        svc = _service(registry=registry, poller=poller)

        n = svc.get_unified_nodes()[0]
        assert n.failed_step is None
        assert n.error is None
