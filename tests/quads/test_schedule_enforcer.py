"""Unit tests for ScheduleEnforcer — auto-teardown of scheduled nodes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import QUADSClient, QUADSConnectionError
from inference_proxy.quads.schedule_enforcer import ScheduleEnforcer

_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["gpu01"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _node(hostname: str, status: NodeStatus = NodeStatus.HEALTHY) -> Node:
    return Node(
        node_id=hostname,
        endpoint=f"{hostname}:8000",
        status=status,
        model="meta-llama/Llama-3",
        last_heartbeat=datetime.now(tz=UTC),
    )


def _enforcer(
    available: list[str] | None = None,
    nodes: list[Node] | None = None,
) -> tuple[ScheduleEnforcer, NodeRegistry, MagicMock]:
    client = AsyncMock(spec=QUADSClient)
    client.get_available.return_value = available or []

    registry = NodeRegistry()
    for n in nodes or []:
        registry.add(n)

    provisioner = MagicMock(spec=NodeProvisioner)
    provisioner.try_reserve_host = AsyncMock(
        side_effect=lambda hostname: MagicMock(hostname=hostname)
    )
    provisioner.teardown = AsyncMock()

    def close_background(coro):
        coro.close()
        return MagicMock()

    provisioner.fire_background = MagicMock(side_effect=close_background)

    enforcer = ScheduleEnforcer(
        client=client,
        registry=registry,
        provisioner=provisioner,
        lookahead_hours=24,
        check_interval=300,
    )
    return enforcer, registry, provisioner


class TestEnforceOnce:
    async def test_tears_down_unavailable_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_called_once()

    async def test_no_action_for_available_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=["gpu01"],
            nodes=[_node("gpu01")],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_not_called()

    async def test_skips_provisioning_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", NodeStatus.PROVISIONING)],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_not_called()

    async def test_skips_draining_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", NodeStatus.DRAINING)],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_not_called()

    async def test_tears_down_unhealthy_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", NodeStatus.UNHEALTHY)],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_called_once()

    async def test_skips_host_with_lifecycle_operation_in_progress(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        provisioner.try_reserve_host.side_effect = None
        provisioner.try_reserve_host.return_value = None

        await enforcer._enforce_once()

        provisioner.fire_background.assert_not_called()
        assert "gpu01" not in enforcer.teardown_initiated

    async def test_multiple_nodes_mixed(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=["gpu02"],
            nodes=[_node("gpu01"), _node("gpu02"), _node("gpu03")],
        )
        await enforcer._enforce_once()

        assert provisioner.fire_background.call_count == 2


class TestDedup:
    async def test_no_duplicate_teardown(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        await enforcer._enforce_once()
        await enforcer._enforce_once()

        provisioner.fire_background.assert_called_once()

    async def test_teardown_initiated_property(self) -> None:
        enforcer, _, _ = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        await enforcer._enforce_once()

        assert "gpu01" in enforcer.teardown_initiated


class TestPruning:
    async def test_prunes_when_node_removed(self) -> None:
        enforcer, registry, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        await enforcer._enforce_once()
        assert "gpu01" in enforcer.teardown_initiated

        registry.remove("gpu01")
        # ponytail: prune happens at start of next _enforce_once
        enforcer._prune_completed()

        assert "gpu01" not in enforcer.teardown_initiated


class TestPollerFailure:
    async def test_continues_on_quads_error(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        enforcer._client.get_available.side_effect = QUADSConnectionError("down")

        await enforcer._enforce_once()

        provisioner.fire_background.assert_not_called()


async def test_schedule_enforcer_uses_same_host_lifecycle_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforcement cannot enter teardown while provisioning owns the host."""
    registry = NodeRegistry()
    registry.add(_node("gpu01"))
    client = AsyncMock(spec=QUADSClient)
    client.get_available.return_value = []
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    provisioner = NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=etcd,
        settings=ProvisioningSettings(),
        endpoint_policy=_ENDPOINT_POLICY,
        registry=registry,
    )
    provision_entered = asyncio.Event()
    release_provision = asyncio.Event()
    teardown_entered = asyncio.Event()

    async def provision_body(
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
    ) -> None:
        assert hostname == "gpu01"
        provision_entered.set()
        await release_provision.wait()

    async def teardown_body(hostname: str, *, force: bool = False) -> None:
        assert hostname == "gpu01"
        teardown_entered.set()

    monkeypatch.setattr(provisioner, "_provision", provision_body)
    monkeypatch.setattr(provisioner, "_teardown", teardown_body)
    fire_background = MagicMock()
    monkeypatch.setattr(provisioner, "fire_background", fire_background)
    enforcer = ScheduleEnforcer(
        client=client,
        registry=registry,
        provisioner=provisioner,
    )

    provision_task = asyncio.create_task(provisioner.provision("gpu01"))
    assert await asyncio.wait_for(provision_entered.wait(), timeout=1)

    await enforcer._enforce_once()
    fire_background.assert_not_called()
    assert not teardown_entered.is_set()
    assert "gpu01" not in enforcer.teardown_initiated

    release_provision.set()
    await asyncio.wait_for(provision_task, timeout=1)
    await enforcer._enforce_once()
    teardown_coro = fire_background.call_args.args[0]
    await asyncio.wait_for(teardown_coro, timeout=1)

    assert teardown_entered.is_set()
    assert "gpu01" in enforcer.teardown_initiated
