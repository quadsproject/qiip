"""Concurrency regressions for provisioning cancellation and teardown."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_proxy.api.admin import setup_node, teardown_node
from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.admin import SetupRequest
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.provisioning.ssh_client import RemoteCommandError

_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["gpu01"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _provisioner(
    *,
    etcd: MagicMock | None = None,
    registry: NodeRegistry | MagicMock | None = None,
) -> NodeProvisioner:
    etcd_client = etcd if etcd is not None else MagicMock()
    etcd_client.prefix = "/nodes/"
    if etcd is None:
        etcd_client.put = MagicMock(return_value=True)
        etcd_client.delete = MagicMock(return_value=True)
    ssh_client = MagicMock()
    ssh_client.upload = AsyncMock()
    return NodeProvisioner(
        ssh_client=ssh_client,
        etcd_client=etcd_client,
        settings=ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            drain_timeout=0,
        ),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_ENDPOINT_POLICY,
        registry=registry,
        connection_tracker=MagicMock(get=MagicMock(return_value=0)),
        log_buffer=MagicMock(),
    )


@pytest.mark.asyncio
async def test_teardown_cancels_active_provision(
    test_registry: NodeRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DELETE handler awaits real task cancellation before teardown."""
    etcd = MagicMock()
    provisioner = _provisioner(etcd=etcd, registry=test_registry)
    poll_started = asyncio.Event()
    teardown_started = asyncio.Event()
    order: list[str] = []
    scheduled: list[asyncio.Task[None]] = []

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    async def block_health(_hostname: str) -> None:
        poll_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            order.append("provision_cancelled")

    async def teardown(_hostname: str, *, force: bool = False) -> None:
        assert force is False
        order.append("teardown_started")
        teardown_started.set()

    monkeypatch.setattr(
        provisioner, "_power_on_if_needed", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(provisioner, "preflight", AsyncMock(return_value=None))
    monkeypatch.setattr(provisioner, "_upload_scripts", AsyncMock(return_value=None))
    monkeypatch.setattr(provisioner, "_run_setup", AsyncMock(return_value=None))
    monkeypatch.setattr(provisioner, "_verify_gpu", AsyncMock(return_value=None))
    monkeypatch.setattr(
        provisioner,
        "_run_start_vllm",
        AsyncMock(return_value="org/model"),
    )
    monkeypatch.setattr(provisioner, "_poll_health", block_health)
    register = AsyncMock()
    monkeypatch.setattr(provisioner, "_register_node", register)
    monkeypatch.setattr(provisioner, "_teardown", teardown)

    original_fire = provisioner.fire_background

    def capture_task(
        coro: Coroutine[object, object, None],
        *,
        provisioning_hostname: str | None = None,
    ) -> asyncio.Task[None]:
        if provisioning_hostname is None:
            # Keep the regression runnable against the pre-fix method
            # signature so it fails on behavior, not a new keyword.
            task = original_fire(coro)
        else:
            task = original_fire(
                coro,
                provisioning_hostname=provisioning_hostname,
            )
        scheduled.append(task)
        return task

    monkeypatch.setattr(provisioner, "fire_background", capture_task)

    try:
        setup = await setup_node(
            SetupRequest(hostname="gpu01", model="org/model"),
            registry=test_registry,
            provisioner=provisioner,
            quads_client=None,
        )
        assert setup.task_id == "gpu01"
        await asyncio.wait_for(poll_started.wait(), timeout=1)
        provision_task = scheduled[0]

        response = await asyncio.wait_for(
            teardown_node(
                "gpu01",
                force=False,
                registry=test_registry,
                provisioner=provisioner,
            ),
            timeout=1,
        )

        assert response.task_id == "gpu01"
        await asyncio.wait_for(teardown_started.wait(), timeout=1)
        await asyncio.wait_for(scheduled[1], timeout=1)

        assert provision_task.cancelled() is True
        assert order == ["provision_cancelled", "teardown_started"]
        register.assert_not_awaited()

        state_payloads = [
            json.loads(call.args[1])
            for call in etcd.put.call_args_list
            if str(call.args[0]).startswith("/provisioning/")
        ]
        assert state_payloads[-1]["current_step"] == "failed"
        assert state_payloads[-1]["failed_step"] == "cancelled"
        assert state_payloads[-1]["error"] == "Provisioning cancelled by teardown"
    finally:
        for task in scheduled:
            if task.done():
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_immediate_teardown_cancels_task_before_coroutine_entry() -> None:
    """Provision ownership is visible before the scheduled coroutine can run."""
    etcd = MagicMock()
    provisioner = _provisioner(etcd=etcd)
    entered = False

    async def provision_body() -> None:
        nonlocal entered
        entered = True

    task = provisioner.fire_background(
        provision_body(),
        provisioning_hostname="gpu01",
    )
    cancelled = await asyncio.wait_for(
        provisioner.cancel_active_provision("gpu01"),
        timeout=1,
    )

    assert cancelled is task
    assert task.cancelled() is True
    assert entered is False
    payload = json.loads(etcd.put.call_args.args[1])
    assert payload["current_step"] == "failed"
    assert payload["failed_step"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_active_provision_rejects_swallowed_cancellation() -> None:
    """A task that suppresses CancelledError is not reported as cancelled."""
    provisioner = _provisioner()
    started = asyncio.Event()

    async def swallow_cancellation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    task = provisioner.fire_background(
        swallow_cancellation(),
        provisioning_hostname="gpu01",
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="swallowed cancellation"):
        await asyncio.wait_for(
            provisioner.cancel_active_provision("gpu01"),
            timeout=1,
        )

    assert task.done() is True
    assert task.cancelled() is False


@pytest.mark.asyncio
async def test_failed_provision_does_not_resurrect_after_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host lease orders a failure PUT before teardown's final DELETE."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    node_state: dict[str, bytes] = {}
    node_operations: list[str] = []

    def put(key: str, value: bytes) -> bool:
        if key.startswith("/nodes/"):
            node_operations.append("put")
            node_state[key] = value
        return True

    def delete(key: str) -> bool:
        if key.startswith("/nodes/"):
            node_operations.append("delete")
            node_state.pop(key, None)
        return True

    etcd.put = MagicMock(side_effect=put)
    etcd.delete = MagicMock(side_effect=delete)

    def replace(key: str, expected: bytes, new: bytes) -> bool:
        if node_state.get(key) != expected:
            return False
        node_operations.append("replace")
        node_state[key] = new
        return True

    etcd.replace = MagicMock(side_effect=replace)
    registry = MagicMock()
    provisioner = _provisioner(etcd=etcd, registry=registry)
    failure_reached = asyncio.Event()
    release_failure = asyncio.Event()
    upload_calls = 0

    async def fail_upload(_hostname: str) -> None:
        nonlocal upload_calls
        upload_calls += 1
        if upload_calls > 1:
            return
        failure_reached.set()
        await release_failure.wait()
        raise RemoteCommandError("gpu01", "upload", 1)

    monkeypatch.setattr(
        provisioner, "_power_on_if_needed", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(provisioner, "preflight", AsyncMock(return_value=None))
    monkeypatch.setattr(provisioner, "_upload_scripts", fail_upload)
    monkeypatch.setattr(provisioner, "_ssh_run_command", AsyncMock(return_value=""))
    monkeypatch.setattr(provisioner, "_drain_wait", AsyncMock(return_value=None))

    provision_task = asyncio.create_task(provisioner.provision("gpu01"))
    await asyncio.wait_for(failure_reached.wait(), timeout=1)
    teardown_task = asyncio.create_task(provisioner.teardown("gpu01", force=True))
    release_failure.set()

    results = await asyncio.wait_for(
        asyncio.gather(provision_task, teardown_task, return_exceptions=True),
        timeout=1,
    )

    assert "Command 'upload'" in str(results[0])
    assert results[1] is None
    assert node_operations[-1] == "delete"
    assert "/nodes/gpu01" not in node_state
    registry.remove.assert_called_once_with("gpu01")


@pytest.mark.asyncio
async def test_failed_provision_does_not_resurrect_deleted_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure CAS cannot recreate a node removed during provisioning."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    node_state: dict[str, bytes] = {}

    def put(key: str, value: bytes) -> bool:
        if key.startswith("/nodes/"):
            node_state[key] = value
        return True

    def delete(key: str) -> bool:
        node_state.pop(key, None)
        return True

    def replace(key: str, expected: bytes, new: bytes) -> bool:
        if node_state.get(key) != expected:
            return False
        node_state[key] = new
        return True

    etcd.put = MagicMock(side_effect=put)
    etcd.delete = MagicMock(side_effect=delete)
    etcd.replace = MagicMock(side_effect=replace)
    provisioner = _provisioner(etcd=etcd)
    failure_reached = asyncio.Event()
    release_failure = asyncio.Event()

    async def fail_upload(_hostname: str) -> None:
        failure_reached.set()
        await release_failure.wait()
        raise RemoteCommandError("gpu01", "upload", 1)

    monkeypatch.setattr(
        provisioner, "_power_on_if_needed", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(provisioner, "preflight", AsyncMock(return_value=None))
    monkeypatch.setattr(provisioner, "_upload_scripts", fail_upload)

    provision_task = asyncio.create_task(provisioner.provision("gpu01"))
    await asyncio.wait_for(failure_reached.wait(), timeout=1)
    assert "/nodes/gpu01" in node_state

    # This represents teardown or another authoritative owner deleting the
    # key while the provision task is still unwinding.
    etcd.delete("/nodes/gpu01")
    release_failure.set()

    result = await asyncio.wait_for(
        asyncio.gather(provision_task, return_exceptions=True),
        timeout=1,
    )

    assert "Command 'upload'" in str(result[0])
    assert "/nodes/gpu01" not in node_state
    etcd.replace.assert_called_once()
