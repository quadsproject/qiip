"""Resource-bound and observation tests for provisioner background tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import InferenceEngine, Node, NodeStatus
from inference_proxy.provisioning.provisioner import (
    BackgroundOperation,
    NodeProvisioner,
    ProvisioningCapacityError,
    ProvisioningIdentity,
)
from inference_proxy.quads.client import QUADSClient
from inference_proxy.quads.schedule_enforcer import ScheduleEnforcer


def _make_provisioner(*, limit: int = 32) -> NodeProvisioner:
    return NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=MagicMock(),
        settings=ProvisioningSettings(max_concurrent_provisions=limit),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=EndpointPolicy.from_values(
            allowed_hosts=["gpu01", "gpu02", "gpu03"],
            allowed_networks=[],
            allowed_ports=[8000],
        ),
    )


def _fire_background(
    provisioner: NodeProvisioner,
    coro: Coroutine[Any, Any, None],
    *,
    provisioning_hostname: str | None = None,
    task_name: str | None = None,
) -> asyncio.Task[None]:
    return provisioner.fire_background(
        coro,
        provisioning_hostname=provisioning_hostname,
        provisioning_identity=(
            ProvisioningIdentity(InferenceEngine.VLLM)
            if provisioning_hostname is not None
            else None
        ),
        task_name=task_name,
    )


@pytest.mark.parametrize(
    ("hostname", "identity"),
    [
        ("gpu01", None),
        (None, ProvisioningIdentity(InferenceEngine.VLLM)),
    ],
)
def test_provisioning_hostname_and_identity_are_required_together(
    hostname: str | None,
    identity: ProvisioningIdentity | None,
) -> None:
    """Host task ownership can never exist without exact serving identity."""
    provisioner = _make_provisioner()

    async def pending() -> None:
        return None

    background = pending()
    try:
        with pytest.raises(ValueError, match="must be supplied together"):
            provisioner.fire_background(
                background,
                provisioning_hostname=hostname,
                provisioning_identity=identity,
            )
    finally:
        background.close()


@pytest.mark.asyncio
async def test_provisioning_capacity_admits_exact_limit_then_rejects() -> None:
    """S9: exactly N active provisions fit; N+1 is rejected, not queued."""
    provisioner = _make_provisioner(limit=2)
    release = asyncio.Event()

    async def blocked() -> None:
        await release.wait()

    tasks = [
        provisioner.fire_background(
            blocked(),
            provisioning_hostname=f"gpu0{number}",
            provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
        )
        for number in (1, 2)
    ]
    rejected = blocked()
    rejected_task: asyncio.Task[None] | None = None
    try:
        try:
            rejected_task = provisioner.fire_background(
                rejected,
                provisioning_hostname="gpu03",
                provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
            )
        except ProvisioningCapacityError as error:
            assert error.active == 2
            assert error.limit == 2
        else:
            pytest.fail("provisioning task above the configured limit was admitted")
        assert not tasks[0].done()
        assert not tasks[1].done()
    finally:
        if rejected_task is None:
            rejected.close()
        else:
            tasks.append(rejected_task)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)


@pytest.mark.asyncio
async def test_capacity_is_released_when_provision_task_finishes() -> None:
    provisioner = _make_provisioner(limit=1)

    async def finish() -> None:
        return None

    first = provisioner.fire_background(
        finish(),
        provisioning_hostname="gpu01",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
    )
    await asyncio.wait_for(first, timeout=1)
    await asyncio.sleep(0)  # let the ownership-removal callback run

    second = provisioner.fire_background(
        finish(),
        provisioning_hostname="gpu02",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
    )
    await asyncio.wait_for(second, timeout=1)


@pytest.mark.asyncio
async def test_relaunch_counts_capacity_but_is_not_cancelled_as_provisioning() -> None:
    provisioner = _make_provisioner(limit=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def relaunch() -> None:
        started.set()
        await release.wait()

    task = provisioner.fire_background(
        relaunch(),
        provisioning_hostname="gpu01",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.LLAMA_CPP),
        operation=BackgroundOperation.RELAUNCH,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    rejected = asyncio.sleep(0)
    try:
        with pytest.raises(ProvisioningCapacityError):
            provisioner.fire_background(
                rejected,
                provisioning_hostname="gpu02",
                provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
            )
        assert await provisioner.cancel_active_provision("gpu01") is None
        assert not task.done()
    finally:
        rejected.close()
        release.set()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_scheduling_failure_does_not_consume_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A create_task failure happens before the capacity slot is attached."""
    provisioner = _make_provisioner(limit=1)
    real_create_task = asyncio.create_task
    calls = 0

    def fail_once(
        coro: Coroutine[Any, Any, None],
        *,
        name: str | None = None,
    ) -> asyncio.Task[None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("scheduler unavailable")
        if name is None:
            return real_create_task(coro)
        return real_create_task(coro, name=name)

    monkeypatch.setattr(
        "inference_proxy.provisioning.provisioner.asyncio.create_task",
        fail_once,
    )

    async def finish() -> None:
        return None

    rejected = finish()
    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        provisioner.fire_background(
            rejected,
            provisioning_hostname="gpu01",
            provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
        )
    rejected.close()

    admitted = provisioner.fire_background(
        finish(),
        provisioning_hostname="gpu02",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
    )
    await asyncio.wait_for(admitted, timeout=1)


@pytest.mark.asyncio
async def test_full_provisioning_capacity_does_not_block_teardown() -> None:
    """Cleanup tasks are intentionally outside the provisioning admission cap."""
    provisioner = _make_provisioner(limit=1)
    release = asyncio.Event()
    teardown_ran = asyncio.Event()

    async def blocked_provision() -> None:
        await release.wait()

    async def teardown() -> None:
        teardown_ran.set()

    provision = provisioner.fire_background(
        blocked_provision(),
        provisioning_hostname="gpu01",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
    )
    teardown_task = _fire_background(
        provisioner,
        teardown(),
        task_name="teardown:gpu02",
    )
    try:
        await asyncio.wait_for(teardown_task, timeout=1)
        assert teardown_ran.is_set()
        assert not provision.done()
    finally:
        release.set()
        await asyncio.wait_for(provision, timeout=1)


@pytest.mark.asyncio
async def test_failed_background_task_is_observed_once_with_context() -> None:
    """C4: escaped failures are retrieved and emitted as one structured event."""
    provisioner = _make_provisioner()
    error = RuntimeError("registration exploded")

    async def fail() -> None:
        raise error

    with capture_logs() as logs:
        task = provisioner.fire_background(
            fail(),
            provisioning_hostname="gpu01",
            provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
        )
        with pytest.raises(RuntimeError, match="registration exploded"):
            await asyncio.wait_for(task, timeout=1)
        await asyncio.sleep(0)

    failures = [log for log in logs if log.get("event") == "background_task_failed"]
    assert len(failures) == 1
    assert failures[0]["task_name"] == "provision:gpu01"
    assert failures[0]["hostname"] == "gpu01"
    assert failures[0]["exception_type"] == "RuntimeError"
    assert failures[0]["error"] == "registration exploded"
    assert failures[0]["exc_info"][1] is error


@pytest.mark.asyncio
async def test_successful_background_task_emits_no_failure() -> None:
    provisioner = _make_provisioner()

    async def succeed() -> None:
        return None

    with capture_logs() as logs:
        task = _fire_background(
            provisioner,
            succeed(),
            task_name="maintenance",
        )
        await asyncio.wait_for(task, timeout=1)
        await asyncio.sleep(0)

    assert not any(log.get("event") == "background_task_failed" for log in logs)


@pytest.mark.asyncio
async def test_shutdown_cancellation_emits_no_failure_or_callback_error() -> None:
    """PR 20 cancellation is lifecycle, not an observed task failure."""
    provisioner = _make_provisioner()
    started = asyncio.Event()
    loop_errors: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def blocked() -> None:
        started.set()
        await asyncio.Event().wait()

    try:
        with capture_logs() as logs:
            task = provisioner.fire_background(
                blocked(),
                provisioning_hostname="gpu01",
                provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            await asyncio.wait_for(provisioner.shutdown(), timeout=1)
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert task.cancelled()
    assert loop_errors == []
    assert not any(log.get("event") == "background_task_failed" for log in logs)


@pytest.mark.asyncio
async def test_schedule_enforcer_consumed_failure_is_not_double_logged() -> None:
    """PR 16 owns retry logging; the generic observer sees a successful wrapper."""
    provisioner = _make_provisioner()
    provisioner.teardown = AsyncMock(side_effect=RuntimeError("host unreachable"))  # type: ignore[method-assign]
    registry = NodeRegistry()
    registry.add(
        Node(
            node_id="gpu01",
            endpoint="gpu01:8000",
            status=NodeStatus.HEALTHY,
            model="org/model",
            last_heartbeat=datetime.now(UTC),
            managed=True,
        )
    )
    client = AsyncMock(spec=QUADSClient)
    client.get_available.return_value = []
    enforcer = ScheduleEnforcer(
        client=client,
        registry=registry,
        provisioner=provisioner,
        check_interval=10,
    )

    with capture_logs() as logs:
        await enforcer._enforce_once()
        tasks = tuple(provisioner._background_tasks)
        assert len(tasks) == 1
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        await asyncio.sleep(0)

    assert (
        sum(
            log.get("event") == "schedule_enforcer_teardown_retry_scheduled"
            for log in logs
        )
        == 1
    )
    assert not any(log.get("event") == "background_task_failed" for log in logs)
