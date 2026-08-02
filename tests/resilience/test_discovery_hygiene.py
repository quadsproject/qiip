"""Health-driven leases and traffic-independent drain cleanup (R6, R7)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, call

import httpx

from inference_proxy.discovery.etcd_client import (
    EtcdEvent,
    EtcdRecord,
    EtcdSnapshot,
    EtcdWatchBatch,
)
from inference_proxy.discovery.node_leases import (
    NodeLeaseManager,
    NodeLeaseObservation,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.discovery.watcher import _apply_batch, _reconcile_snapshot
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.resilience.health_checker import _probe_all_nodes
from inference_proxy.routing import drain_cleanup
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector


class _FailureCounts(dict[str, int]):
    def reset(self, node_id: str) -> None:
        self[node_id] = 0

    def increment(self, node_id: str) -> int:
        count = self.get(node_id, 0) + 1
        self[node_id] = count
        return count

    def close(self) -> None:
        pass


def _node(
    node_id: str,
    *,
    status: NodeStatus = NodeStatus.HEALTHY,
    managed: bool = True,
    endpoint: str | None = None,
) -> Node:
    return Node(
        node_id=node_id,
        endpoint=endpoint or f"{node_id}:8000",
        status=status,
        model="model-a",
        managed=managed,
    )


def _lease_manager(
    records: dict[str, tuple[bytes, int, int]],
) -> tuple[NodeLeaseManager, MagicMock]:
    client = MagicMock()
    client.prefix = "/nodes/"
    manager = NodeLeaseManager(client)
    manager.reconcile_snapshot(
        {
            node_id: NodeLeaseObservation(value, revision, lease_id)
            for node_id, (value, revision, lease_id) in records.items()
        }
    )
    return manager, client


def test_refresh_exception_does_not_abort_health_cycle() -> None:
    """One etcd failure cannot stop later probes or the checker thread."""
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    registry.add(_node("gpu02"))
    manager, etcd = _lease_manager(
        {
            "gpu01": (b"one", 1, 7001),
            "gpu02": (b"two", 2, 7002),
        }
    )
    etcd.refresh_lease.side_effect = [RuntimeError("etcd unavailable"), 600]
    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=200)

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        3,
        lease_manager=manager,
    )

    assert http.get.call_count == 2
    assert etcd.refresh_lease.call_args_list == [call(7001), call(7002)]


def test_failed_probe_does_not_adopt_or_refresh_lease() -> None:
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    manager, etcd = _lease_manager({"gpu01": (b"one", 1, 0)})
    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=503)

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        3,
        lease_manager=manager,
    )

    etcd.grant_node_lease.assert_not_called()
    etcd.refresh_lease.assert_not_called()


def test_probe_of_replaced_endpoint_does_not_refresh_new_registration() -> None:
    """Health evidence is valid only for the exact endpoint that was probed."""
    registry = NodeRegistry()
    registry.add(_node("gpu01", endpoint="old-gpu01:8000"))
    manager, etcd = _lease_manager({"gpu01": (b"new", 2, 7002)})
    http = MagicMock(spec=httpx.Client)

    def replace_during_probe(_url: str) -> MagicMock:
        registry.add(_node("gpu01", endpoint="new-gpu01:8000"))
        return MagicMock(status_code=200)

    http.get.side_effect = replace_during_probe

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        3,
        lease_manager=manager,
    )

    etcd.refresh_lease.assert_not_called()


def test_malformed_put_does_not_replace_lease_observation() -> None:
    """Only a PUT that becomes a registry node may become a lease target."""
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    original = NodeLeaseObservation(b"old", 1, 7001)
    manager, _etcd = _lease_manager({"gpu01": (b"old", 1, 7001)})
    revisions = {"gpu01": 1}

    _apply_batch(
        EtcdWatchBatch(
            (
                EtcdEvent(
                    key=b"/nodes/gpu01",
                    value=b"not-json",
                    mod_revision=2,
                    is_delete=False,
                    lease_id=7002,
                ),
            ),
            revision=2,
        ),
        registry,
        "/nodes/",
        revisions,
        EndpointPolicy.from_values(
            allowed_hosts=["gpu01"],
            allowed_networks=[],
            allowed_ports=[8000],
        ),
        manager,
    )

    assert manager.get("gpu01") == original


def test_malformed_snapshot_record_is_not_a_lease_target() -> None:
    """A present-but-invalid key is retained locally but never refreshed."""
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    manager, _etcd = _lease_manager({"gpu01": (b"old", 1, 7001)})

    _reconcile_snapshot(
        EtcdSnapshot(
            (
                EtcdRecord(
                    key=b"/nodes/gpu01",
                    value=b"not-json",
                    mod_revision=2,
                    lease_id=7002,
                ),
            ),
            revision=2,
        ),
        registry,
        "/nodes/",
        EndpointPolicy.from_values(
            allowed_hosts=["gpu01"],
            allowed_networks=[],
            allowed_ports=[8000],
        ),
        manager,
    )

    assert registry.get("gpu01") is not None
    assert manager.get("gpu01") is None


def test_half_open_failure_withholds_refresh_until_inference_succeeds() -> None:
    registry = NodeRegistry()
    registry.add(_node("gpu01", status=NodeStatus.UNHEALTHY))
    breakers = CircuitBreakerRegistry(threshold=1)
    breaker = breakers.get_or_create("gpu01")
    breaker.record_failure()
    manager, etcd = _lease_manager({"gpu01": (b"one", 1, 7001)})
    etcd.refresh_lease.return_value = 600
    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=200)
    http.post.return_value = httpx.Response(
        503,
        request=httpx.Request("POST", "http://gpu01:8000/v1/completions"),
    )

    _probe_all_nodes(
        registry,
        breakers,
        http,
        _FailureCounts(),
        3,
        lease_manager=manager,
    )
    etcd.refresh_lease.assert_not_called()

    http.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "http://gpu01:8000/v1/completions"),
    )
    _probe_all_nodes(
        registry,
        breakers,
        http,
        _FailureCounts(),
        3,
        lease_manager=manager,
    )
    etcd.refresh_lease.assert_called_once_with(7001)


def test_health_cycle_preserves_draining_node_with_active_connection() -> None:
    registry = NodeRegistry()
    registry.add(_node("gpu01", status=NodeStatus.DRAINING))
    tracker = ConnectionTracker()
    tracker.increment("gpu01")

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        MagicMock(spec=httpx.Client),
        _FailureCounts(),
        3,
        connection_tracker=tracker,
    )

    assert registry.get("gpu01") is not None
    assert tracker.get("gpu01") == 1


def test_refresh_failure_does_not_prevent_draining_sweep() -> None:
    registry = NodeRegistry()
    registry.add(_node("draining", status=NodeStatus.DRAINING))
    registry.add(_node("healthy"))
    manager, etcd = _lease_manager({"healthy": (b"healthy", 2, 7002)})
    etcd.refresh_lease.side_effect = RuntimeError("etcd unavailable")
    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=200)

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        3,
        connection_tracker=ConnectionTracker(),
        lease_manager=manager,
    )

    assert registry.get("draining") is None
    assert registry.get("healthy") is not None
    http.get.assert_called_once_with("http://healthy:8000/health")


def test_drain_sweep_does_not_remove_concurrent_reregistration() -> None:
    registry = NodeRegistry()
    registry.add(_node("gpu01", status=NodeStatus.DRAINING))
    tracker = ConnectionTracker()
    started = threading.Event()
    finished = threading.Event()

    def sweep() -> None:
        started.set()
        drain_cleanup.sweep_drained_nodes(registry, tracker)
        finished.set()

    with registry.locked():
        thread = threading.Thread(target=sweep, daemon=True)
        thread.start()
        assert started.wait(timeout=1)
        registry.add(_node("gpu01", status=NodeStatus.HEALTHY))
        assert not finished.wait(timeout=0.05)
    thread.join(timeout=1)
    assert not thread.is_alive()

    current = registry.get("gpu01")
    assert current is not None
    assert current.status == NodeStatus.HEALTHY


def test_request_release_and_health_cycle_share_atomic_drain_cleanup(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def record_call(
        _registry: NodeRegistry,
        _tracker: ConnectionTracker,
    ) -> tuple[str, ...]:
        calls.append("sweep")
        return ()

    monkeypatch.setattr(drain_cleanup, "sweep_drained_nodes", record_call)
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    tracker = ConnectionTracker()
    selector = NodeSelector(registry, tracker)
    reservation = selector.select_and_reserve("model-a")
    assert reservation is not None
    reservation.release()

    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=500)
    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        3,
        connection_tracker=tracker,
    )

    assert calls == ["sweep", "sweep"]


@dataclass
class _LeaseClock:
    ttl: int
    now: int = 0
    last_refresh: int = 0

    def advance_to_expiry(self, *, key: bytes, revision: int) -> EtcdWatchBatch:
        self.now = self.last_refresh + self.ttl
        assert self.now >= self.last_refresh + self.ttl
        return EtcdWatchBatch(
            (
                EtcdEvent(
                    key=key,
                    value=None,
                    mod_revision=revision,
                    is_delete=True,
                    lease_id=0,
                ),
            ),
            revision,
        )


def test_dead_node_lease_expiry_drains_and_sweeps_registration() -> None:
    """R6 and R7 compose from restart snapshot to final ghost removal."""
    node = _node("gpu01")
    key, value = node_to_etcd(node, "/nodes/")
    snapshot = EtcdSnapshot(
        (
            EtcdRecord(
                key=key.encode(),
                value=value,
                mod_revision=10,
                lease_id=7001,
            ),
        ),
        revision=10,
    )
    registry = NodeRegistry()
    manager, etcd = _lease_manager({})
    revisions = _reconcile_snapshot(
        snapshot,
        registry,
        "/nodes/",
        EndpointPolicy.from_values(
            allowed_hosts=["gpu01"],
            allowed_networks=[],
            allowed_ports=[8000],
        ),
        manager,
    )
    http = MagicMock(spec=httpx.Client)
    http.get.return_value = MagicMock(status_code=503)

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        1,
        lease_manager=manager,
    )
    etcd.refresh_lease.assert_not_called()

    deletion = _LeaseClock(ttl=1).advance_to_expiry(
        key=key.encode(),
        revision=11,
    )
    _apply_batch(
        deletion,
        registry,
        "/nodes/",
        revisions,
        EndpointPolicy.from_values(
            allowed_hosts=["gpu01"],
            allowed_networks=[],
            allowed_ports=[8000],
        ),
        manager,
    )
    drained = registry.get("gpu01")
    assert drained is not None
    assert drained.status == NodeStatus.DRAINING

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        http,
        _FailureCounts(),
        1,
        connection_tracker=ConnectionTracker(),
        lease_manager=manager,
    )
    assert registry.get("gpu01") is None
