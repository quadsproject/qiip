"""Unit tests for ScheduleEnforcer — auto-teardown of scheduled nodes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import QUADSClient, QUADSConnectionError
from inference_proxy.quads.schedule_enforcer import ScheduleEnforcer


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
    provisioner.teardown = MagicMock(return_value=AsyncMock()())
    provisioner.fire_background = MagicMock()

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
