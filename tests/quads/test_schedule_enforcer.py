"""Unit tests for ScheduleEnforcer — auto-teardown of scheduled nodes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_from_etcd
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


def _node(
    hostname: str,
    status: NodeStatus = NodeStatus.HEALTHY,
    *,
    managed: bool = True,
) -> Node:
    return Node(
        node_id=hostname,
        endpoint=f"{hostname}:8000",
        status=status,
        model="meta-llama/Llama-3",
        last_heartbeat=datetime.now(tz=UTC),
        managed=managed,
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

    def close_background(
        coro: Coroutine[Any, Any, None],
        *,
        task_name: str | None = None,
    ) -> MagicMock:
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


def _get_available_mock(enforcer: ScheduleEnforcer) -> AsyncMock:
    """Return the runtime AsyncMock behind the typed QUADS client method."""
    get_available = enforcer._client.get_available
    assert isinstance(get_available, AsyncMock)
    return get_available


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

    async def test_active_draining_node_is_not_duplicated(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", NodeStatus.DRAINING)],
        )
        provisioner.try_reserve_host.side_effect = None
        provisioner.try_reserve_host.return_value = None

        await enforcer._enforce_once()

        provisioner.try_reserve_host.assert_awaited_once_with("gpu01")
        provisioner.fire_background.assert_not_called()

    async def test_reconciled_draining_managed_node_is_torn_down(self) -> None:
        enforcer, registry, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        registry.drain("gpu01")

        await enforcer._enforce_once()

        provisioner.try_reserve_host.assert_awaited_once_with("gpu01")
        provisioner.fire_background.assert_called_once()

    async def test_tears_down_unhealthy_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", NodeStatus.UNHEALTHY)],
        )
        await enforcer._enforce_once()

        provisioner.fire_background.assert_called_once()

    async def test_schedule_enforcer_skips_explicitly_unmanaged_node(self) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01", managed=False)],
        )

        await enforcer._enforce_once()

        provisioner.try_reserve_host.assert_not_awaited()
        provisioner.fire_background.assert_not_called()
        provisioner.teardown.assert_not_awaited()

    async def test_etcd_node_without_managed_is_never_enforced(self) -> None:
        payload = json.dumps(
            {
                "endpoint": "gpu01:8000",
                "status": "healthy",
                "model": "meta-llama/Llama-3",
            }
        ).encode()
        node = node_from_etcd(
            "/nodes/gpu01",
            payload,
            "/nodes/",
            endpoint_policy=_ENDPOINT_POLICY,
        )
        assert node is not None

        enforcer, registry, provisioner = _enforcer(available=[])
        registry.add(node)

        await enforcer._enforce_once()

        assert registry.get("gpu01") is node
        assert node.managed is False
        provisioner.try_reserve_host.assert_not_awaited()
        provisioner.fire_background.assert_not_called()
        provisioner.teardown.assert_not_awaited()

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

    @pytest.mark.parametrize(
        ("node_id", "available"),
        [("GPU01", "gpu01"), ("gpu01.", "GPU01")],
    )
    async def test_enforcer_matches_canonical_node_ids(
        self,
        node_id: str,
        available: str,
    ) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[available],
            nodes=[_node(node_id)],
        )

        await enforcer._enforce_once()

        provisioner.try_reserve_host.assert_not_awaited()
        provisioner.fire_background.assert_not_called()

    async def test_enforcer_requests_exact_configured_lookahead_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected_end = datetime(2026, 8, 1, 16, 45, tzinfo=UTC)

        def fixed_window(lookahead_hours: int) -> datetime:
            assert lookahead_hours == 24
            return expected_end

        monkeypatch.setattr(
            "inference_proxy.quads.schedule_enforcer.availability_window_end",
            fixed_window,
        )
        enforcer, _, _ = _enforcer(available=["gpu01"])

        await enforcer._enforce_once()

        _get_available_mock(enforcer).assert_awaited_once_with(end=expected_end)


class TestTeardownRetry:
    async def test_failed_teardown_is_observed_and_retried_after_backoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = [100.0]
        monkeypatch.setattr(
            "inference_proxy.quads.schedule_enforcer.monotonic",
            lambda: now[0],
        )
        enforcer, registry, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        enforcer._interval = 10
        tasks: list[asyncio.Task[None]] = []
        attempts = 0

        def fire_background(
            coro: Coroutine[Any, Any, None],
            *,
            task_name: str | None = None,
        ) -> asyncio.Task[None]:
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        async def teardown(
            hostname: str,
            *,
            lifecycle_lease: MagicMock,
        ) -> None:
            nonlocal attempts
            attempts += 1
            registry.drain(hostname)
            if attempts == 1:
                raise RuntimeError("ssh unavailable")
            registry.remove(hostname)

        provisioner.fire_background.side_effect = fire_background
        provisioner.teardown.side_effect = teardown

        with capture_logs() as logs:
            await enforcer._enforce_once()
            await asyncio.wait_for(tasks[-1], timeout=1)

        node = registry.get("gpu01")
        assert node is not None
        assert node.status == NodeStatus.DRAINING
        assert "gpu01" not in enforcer.teardown_initiated
        assert enforcer.teardown_retry_attempts == {"gpu01": 1}
        assert any(
            log.get("event") == "schedule_enforcer_teardown_retry_scheduled"
            and log.get("attempt") == 1
            and log.get("retry_delay_seconds") == 10.0
            for log in logs
        )

        _get_available_mock(enforcer).return_value = ["gpu01"]
        await enforcer._enforce_once()
        assert len(tasks) == 1

        now[0] = 110.0
        await enforcer._enforce_once()
        await asyncio.wait_for(tasks[-1], timeout=1)

        assert len(tasks) == 2
        assert registry.get("gpu01") is None
        assert enforcer.teardown_retry_attempts == {}

    async def test_repeated_failures_back_off_and_escalate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = [0.0]
        monkeypatch.setattr(
            "inference_proxy.quads.schedule_enforcer.monotonic",
            lambda: now[0],
        )
        enforcer, registry, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01")],
        )
        tasks: list[asyncio.Task[None]] = []

        def fire_background(
            coro: Coroutine[Any, Any, None],
            *,
            task_name: str | None = None,
        ) -> asyncio.Task[None]:
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        async def teardown(
            hostname: str,
            *,
            lifecycle_lease: MagicMock,
        ) -> None:
            registry.drain(hostname)
            raise RuntimeError("host unreachable")

        provisioner.fire_background.side_effect = fire_background
        provisioner.teardown.side_effect = teardown
        expected_delays = [300.0, 600.0, 1200.0, 2400.0, 3600.0]

        with capture_logs() as logs:
            for expected_attempt, expected_delay in enumerate(
                expected_delays,
                start=1,
            ):
                await enforcer._enforce_once()
                await asyncio.wait_for(tasks[-1], timeout=1)
                assert enforcer.teardown_retry_attempts == {"gpu01": expected_attempt}
                now[0] += expected_delay

        retry_events = [
            log
            for log in logs
            if log.get("event")
            in {
                "schedule_enforcer_teardown_retry_scheduled",
                "schedule_enforcer_teardown_requires_operator",
            }
        ]
        assert [log["retry_delay_seconds"] for log in retry_events] == expected_delays
        escalation = [
            log
            for log in logs
            if log.get("event") == "schedule_enforcer_teardown_requires_operator"
        ]
        assert len(escalation) == 1
        assert escalation[0]["attempt"] == 5
        assert escalation[0]["retry_delay_seconds"] == 3600.0

    async def test_teardown_schedule_failure_backs_off_and_continues(
        self,
    ) -> None:
        enforcer, _, provisioner = _enforcer(
            available=[],
            nodes=[_node("gpu01"), _node("gpu02")],
        )
        scheduled = 0

        def fire_background(
            coro: Coroutine[Any, Any, None],
            *,
            task_name: str | None = None,
        ) -> MagicMock:
            nonlocal scheduled
            scheduled += 1
            if scheduled == 1:
                raise RuntimeError("scheduler unavailable")
            coro.close()
            return MagicMock()

        provisioner.fire_background.side_effect = fire_background

        await enforcer._enforce_once()

        assert scheduled == 2
        assert enforcer.teardown_retry_attempts == {"gpu01": 1}
        assert "gpu02" in enforcer.teardown_initiated


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
        _get_available_mock(enforcer).side_effect = QUADSConnectionError("down")

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
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_ENDPOINT_POLICY,
        registry=registry,
        nfs_export="nfs.example:/exports/huggingface",
    )
    provision_entered = asyncio.Event()
    release_provision = asyncio.Event()
    teardown_entered = asyncio.Event()

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
