"""Behavioral tests for host-scoped provisioning lifecycle coordination."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker

_TIMEOUT = 1.0
_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["gpu01"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _provisioner(
    *,
    etcd: MagicMock | None = None,
    registry: NodeRegistry | None = None,
    tracker: ConnectionTracker | None = None,
    breakers: CircuitBreakerRegistry | None = None,
) -> NodeProvisioner:
    etcd = etcd or MagicMock()
    etcd.prefix = "/nodes/"
    return NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=etcd,
        settings=ProvisioningSettings(),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_ENDPOINT_POLICY,
        registry=registry,
        connection_tracker=tracker,
        circuit_breaker_registry=breakers,
        nfs_export="nfs.example:/exports/huggingface",
    )


@pytest.mark.parametrize("first_operation", ["provision", "teardown"])
async def test_same_host_provision_and_teardown_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
    first_operation: str,
) -> None:
    """Neither lifecycle body can overlap the other for one hostname."""
    provisioner = _provisioner()
    entered = {
        "provision": asyncio.Event(),
        "teardown": asyncio.Event(),
    }
    release_first = asyncio.Event()
    active = 0
    max_active = 0

    async def run_body(operation: str) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered[operation].set()
        try:
            if operation == first_operation:
                await release_first.wait()
        finally:
            active -= 1

    async def provision_body(
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        engine: str = "vllm",
        artifact: object | None = None,
    ) -> None:
        assert hostname == "gpu01"
        assert managed is True
        assert model is None
        await run_body("provision")

    async def teardown_body(hostname: str, *, force: bool = False) -> None:
        assert hostname == "gpu01"
        assert force is False
        await run_body("teardown")

    monkeypatch.setattr(provisioner, "_provision", provision_body)
    monkeypatch.setattr(provisioner, "_teardown", teardown_body)

    operations = {
        "provision": lambda: provisioner.provision("gpu01"),
        "teardown": lambda: provisioner.teardown("gpu01"),
    }
    second_operation = "teardown" if first_operation == "provision" else "provision"

    first_task = asyncio.create_task(operations[first_operation]())
    assert await asyncio.wait_for(
        entered[first_operation].wait(),
        timeout=_TIMEOUT,
    )

    second_started = asyncio.Event()

    async def start_second() -> None:
        second_started.set()
        await operations[second_operation]()

    second_task = asyncio.create_task(start_second())
    assert await asyncio.wait_for(second_started.wait(), timeout=_TIMEOUT)
    assert not entered[second_operation].is_set()
    assert max_active == 1

    release_first.set()
    assert await asyncio.wait_for(
        entered[second_operation].wait(),
        timeout=_TIMEOUT,
    )
    await asyncio.wait_for(
        asyncio.gather(first_task, second_task),
        timeout=_TIMEOUT,
    )

    assert max_active == 1
    assert not provisioner.host_operation_in_progress("gpu01")


async def test_different_hosts_can_run_lifecycle_operations_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator is host-scoped, not a global bottleneck."""
    provisioner = _provisioner()
    provision_entered = asyncio.Event()
    teardown_entered = asyncio.Event()
    release = asyncio.Event()

    async def provision_body(
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        engine: str = "vllm",
        artifact: object | None = None,
    ) -> None:
        assert hostname == "gpu01"
        provision_entered.set()
        await release.wait()

    async def teardown_body(hostname: str, *, force: bool = False) -> None:
        assert hostname == "gpu02"
        teardown_entered.set()
        await release.wait()

    monkeypatch.setattr(provisioner, "_provision", provision_body)
    monkeypatch.setattr(provisioner, "_teardown", teardown_body)

    provision_task = asyncio.create_task(provisioner.provision("gpu01"))
    teardown_task = asyncio.create_task(provisioner.teardown("gpu02"))
    both_entered = asyncio.gather(
        provision_entered.wait(),
        teardown_entered.wait(),
    )

    assert all(await asyncio.wait_for(both_entered, timeout=_TIMEOUT))
    release.set()
    await asyncio.wait_for(
        asyncio.gather(provision_task, teardown_task),
        timeout=_TIMEOUT,
    )


async def test_reserved_lease_remains_busy_through_background_provision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setup reservation transfers into provision without an unlocked gap."""
    provisioner = _provisioner()
    provision_entered = asyncio.Event()
    release = asyncio.Event()

    async def provision_body(
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        engine: str = "vllm",
        artifact: object | None = None,
    ) -> None:
        provision_entered.set()
        await release.wait()

    monkeypatch.setattr(provisioner, "_provision", provision_body)
    lease = await provisioner.try_reserve_host("gpu01")
    assert lease is not None
    assert provisioner.host_operation_in_progress("gpu01")

    task = asyncio.create_task(provisioner.provision("gpu01", lifecycle_lease=lease))
    assert await asyncio.wait_for(provision_entered.wait(), timeout=_TIMEOUT)
    assert await provisioner.try_reserve_host("gpu01") is None

    release.set()
    await asyncio.wait_for(task, timeout=_TIMEOUT)
    assert not provisioner.host_operation_in_progress("gpu01")
    replacement = await provisioner.try_reserve_host("gpu01")
    assert replacement is not None
    replacement.release()


async def test_cleanup_stale_node_removes_all_routing_state() -> None:
    """Retry cleanup deletes etcd before removing each local state owner."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    etcd.delete.return_value = True
    registry = NodeRegistry()
    registry.add(
        Node(
            node_id="gpu01",
            endpoint="gpu01:8000",
            status=NodeStatus.FAILED,
            model="model-a",
        )
    )
    tracker = ConnectionTracker()
    tracker.increment("gpu01")
    breakers = CircuitBreakerRegistry()
    breaker = breakers.get_or_create("gpu01")
    for _ in range(3):
        breaker.record_failure()
    provisioner = _provisioner(
        etcd=etcd,
        registry=registry,
        tracker=tracker,
        breakers=breakers,
    )

    await provisioner.cleanup_stale_node("gpu01")

    etcd.delete.assert_called_once_with("/nodes/gpu01")
    assert registry.get("gpu01") is None
    assert tracker.get("gpu01") == 0
    assert breakers.get("gpu01") is None


async def test_cleanup_failure_preserves_local_state() -> None:
    """An etcd failure aborts cleanup before provisioning can lose its guard."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    etcd.delete.side_effect = RuntimeError("etcd down")
    registry = NodeRegistry()
    node = Node(
        node_id="gpu01",
        endpoint="gpu01:8000",
        status=NodeStatus.FAILED,
        model="model-a",
    )
    registry.add(node)
    tracker = ConnectionTracker()
    tracker.increment("gpu01")
    breakers = CircuitBreakerRegistry()
    breaker = breakers.get_or_create("gpu01")
    breaker.record_failure()
    provisioner = _provisioner(
        etcd=etcd,
        registry=registry,
        tracker=tracker,
        breakers=breakers,
    )

    with pytest.raises(RuntimeError, match="etcd down"):
        await provisioner.cleanup_stale_node("gpu01")

    assert registry.get("gpu01") == node
    assert tracker.get("gpu01") == 1
    assert breakers.get("gpu01") is breaker
