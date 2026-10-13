"""Provisioner integration tests for the memory-aware vLLM launch path.

These cover the orchestration around ``vllm_fit.plan_vllm_launch``: device
memory is read from the node, effective plan values reach the start script,
explicit settings are never silently changed, recognized OOMs are retried
within bounds with a clean stop between attempts, and the requested/effective
state is persisted on the registered node.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.models.node import (
    InferenceEngine,
    VllmModelSelection,
    VllmParams,
    VllmRuntimeState,
)
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
)
from inference_proxy.provisioning.ssh_client import RemoteCommandError
from inference_proxy.provisioning.vllm_fit import (
    VllmDeviceMemory,
    VllmRequested,
    plan_vllm_launch,
)
from tests.provisioning.test_provisioner import _make_provisioner, _recording_etcd

_AUTO_MODEL = "Qwen/Qwen2.5-32B-Instruct"
_MEMORY_ROW = "0, 81920, 40000, 41920"


def _provisioner(
    streaming: Any = None,
    *,
    settings: ProvisioningSettings | None = None,
    etcd_client: MagicMock | None = None,
    hf_cache_dir: str | None = None,
) -> tuple[NodeProvisioner, MagicMock]:
    """Build a provisioner whose SSH client streams ``streaming``."""
    ssh = MagicMock()
    if streaming is not None:
        ssh.run_streaming = streaming
    provisioner = _make_provisioner(
        ssh_client=ssh,
        settings=settings,
        etcd_client=etcd_client,
        hf_cache_dir=hf_cache_dir,
    )
    return provisioner, ssh


def _hopper(free_mib: int = 80_000, *, total_mib: int = 81_920) -> VllmDeviceMemory:
    return VllmDeviceMemory(
        index=0, total_mib=total_mib, free_mib=free_mib, compute_capability="9.0"
    )


def _streaming(
    commands: list[str],
    *,
    model: str = _AUTO_MODEL,
    fail: str = "never",
) -> Any:
    """Build an SSH streaming double.

    ``fail="never"`` succeeds on every start; ``fail="first"`` raises a
    ``RemoteCommandError`` on the first start and then succeeds;
    ``fail="always"`` raises on every start.  ``nvidia-smi`` probes always
    return a post-load memory row so telemetry parsing is exercised.
    """
    calls = {"count": 0}

    async def streaming(host: str, command: str) -> AsyncIterator[tuple[str, str]]:
        if "nvidia-smi" in command:
            yield "stdout", _MEMORY_ROW
            return
        commands.append(command)
        calls["count"] += 1
        first = calls["count"] == 1
        if fail == "always" or (fail == "first" and first):
            raise RemoteCommandError(host, command, 1)
        yield "stdout", "# Model:              " + model

    return streaming


@pytest.mark.asyncio
async def test_fitting_plan_reaches_the_start_script_and_is_persisted() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands))

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert model == _AUTO_MODEL
    assert isinstance(runtime, VllmRuntimeState)
    assert runtime.effective.model_selection is VllmModelSelection.AUTO
    assert runtime.effective.model == _AUTO_MODEL
    assert "automatic policy: profile hopper" in runtime.effective.model_reason
    assert runtime.approximate is False
    assert runtime.effective.max_model_len == 8192
    assert runtime.estimate.usable_budget_mib == 73_728
    assert runtime.estimate.free_per_gpu_mib == 80_000
    assert runtime.attempt == 0
    assert runtime.gpus[0].used_mib == 40_000
    # The incumbent engine is stopped once before the first measurement.
    stop.assert_awaited_once()
    assert len(commands) == 1
    command = commands[0]
    assert "AUTOVLLM_MODEL=Qwen/Qwen2.5-32B-Instruct" in command
    assert "AUTOVLLM_MAX_MODEL_LEN=8192" in command
    assert "AUTOVLLM_MAX_BATCHED_TOKENS=8192" in command
    assert "AUTOVLLM_GPU_DEVICES=0" in command
    assert "AUTOVLLM_ATTENTION_BACKEND=FLASHINFER" in command
    assert "AUTOVLLM_FIT_BUDGET_MIB=73728" in command
    assert "AUTOVLLM_FIT_RESERVE_MIB=1475" in command


@pytest.mark.asyncio
async def test_initial_non_fitting_plan_refuses_before_starting_anything() -> None:
    provisioner, ssh = _provisioner(AsyncMock())
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            # Only 5 GiB free: no catalog model fits the usable budget.
            return_value=(_hopper(5_120),),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        pytest.raises(ProvisioningError, match="does not fit host1"),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )
    ssh.run_streaming.assert_not_awaited()


@pytest.mark.asyncio
async def test_incumbent_engine_is_stopped_before_the_first_measurement() -> None:
    provisioner, _ssh = _provisioner(_streaming([]))
    order: list[str] = []

    async def record_stop(_host: str) -> None:
        order.append("stop")

    async def record_read(_host: str) -> tuple[VllmDeviceMemory, ...]:
        order.append("read")
        return ()

    with (
        patch.object(provisioner, "_stop_failed_vllm_start", new=record_stop),
        patch.object(provisioner, "_read_vllm_device_memory", new=record_read),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert order == ["stop", "read"]


@pytest.mark.asyncio
async def test_busy_gpu_that_still_fits_shrinks_the_context() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands))

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(73_500),),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        _model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert runtime is not None
    assert runtime.estimate.usable_budget_mib == 72_908
    assert runtime.effective.gpu_memory_utilization == 0.89
    assert runtime.effective.max_model_len == 4096
    assert [(item.field, item.reason) for item in runtime.adjustments] == [
        ("gpu_memory_utilization", "free_memory")
    ]
    assert "AUTOVLLM_MAX_MODEL_LEN=4096" in commands[0]


@pytest.mark.asyncio
async def test_explicit_configuration_is_never_adjusted_and_never_retried() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands, fail="always"))
    explicit = VllmParams(
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=8192,
        dtype="float16",
        gpu_devices=(0,),
    )

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner, "_vllm_oom_in_engine_log", new_callable=AsyncMock
        ) as oom_probe,
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model="Qwen/Qwen2.5-7B-Instruct",
            vllm_params=explicit,
            started_at=datetime.now(UTC),
        )

    # A fully explicit configuration leaves nothing to adjust, so the OOM is
    # not even classified as retryable. One stop is the pre-planning incumbent
    # teardown, one is the failure cleanup.
    oom_probe.assert_not_awaited()
    assert stop.await_count == 2
    assert len(commands) == 1
    assert "AUTOVLLM_MODEL=Qwen/Qwen2.5-7B-Instruct" in commands[0]
    assert "AUTOVLLM_DTYPE=float16" in commands[0]


@pytest.mark.asyncio
async def test_recognized_oom_is_retried_once_with_a_smaller_context() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands, fail="first"))
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        _model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert runtime is not None
    assert runtime.attempt == 1
    assert len(commands) == 2
    assert len(runtime.adjustments) == 1
    adjustment = runtime.adjustments[0]
    assert adjustment.field == "max_model_len"
    assert adjustment.reason == "oom_retry"
    assert "AUTOVLLM_MAX_MODEL_LEN=8192" in commands[0]
    assert "AUTOVLLM_MAX_MODEL_LEN=4096" in commands[1]
    # Once before planning and once before the retry.
    assert stop.await_count == 2


@pytest.mark.asyncio
async def test_exhausted_retry_budget_stops_cleanly_and_raises() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(
        _streaming(commands, fail="always"),
        settings=ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            vllm_memory_retry_max_attempts=2,
        ),
    )
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    # Attempt zero retries once (attempt 1); attempt 1 exhausts the budget.
    assert len(commands) == 2
    # Pre-planning teardown, the retry teardown, and the terminal cleanup.
    assert stop.await_count == 3
    assert "AUTOVLLM_MAX_MODEL_LEN=8192" in commands[0]
    assert "AUTOVLLM_MAX_MODEL_LEN=4096" in commands[1]


@pytest.mark.asyncio
async def test_unmeasured_host_falls_back_to_the_legacy_start_path() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands))
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock) as poll,
    ):
        _model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert runtime is None
    poll.assert_awaited_once()
    assert len(commands) == 1
    assert "AUTOVLLM_FIT_BUDGET_MIB" not in commands[0]


@pytest.mark.asyncio
async def test_read_vllm_device_memory_parses_free_and_compute_capability() -> None:
    provisioner, _ssh = _provisioner()
    rows = "0, 81920, 80000, 9.0\n\n1, 81920, 79000, 9.0\n"
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value=rows,
    ):
        devices = await provisioner._read_vllm_device_memory("host1")

    assert devices == (
        VllmDeviceMemory(0, 81_920, 80_000, "9.0"),
        VllmDeviceMemory(1, 81_920, 79_000, "9.0"),
    )


@pytest.mark.asyncio
async def test_read_vllm_device_memory_refuses_partial_readings() -> None:
    provisioner, _ssh = _provisioner()
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value="0, 81920, not-a-number, 9.0\n",
    ):
        assert await provisioner._read_vllm_device_memory("host1") == ()
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ssh down"),
    ):
        assert await provisioner._read_vllm_device_memory("host1") == ()
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value="",
    ):
        assert await provisioner._read_vllm_device_memory("host1") == ()


@pytest.mark.asyncio
async def test_registration_persists_the_requested_and_effective_plan() -> None:
    etcd, values, _state_payloads = _recording_etcd()
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands), etcd_client=etcd)

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        _model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )
        assert runtime is not None
        await provisioner._register_node(
            "host1",
            _AUTO_MODEL,
            engine=InferenceEngine.VLLM,
            managed=False,
            vllm_runtime=runtime,
        )

    stored = json.loads(values["/nodes/host1"])
    assert stored["vllm_runtime"]["effective"]["model"] == _AUTO_MODEL
    assert stored["vllm_runtime"]["effective"]["model_selection"] == "automatic"
    assert stored["vllm_runtime"]["effective"]["model_reason"].startswith(
        "automatic policy"
    )
    assert stored["vllm_runtime"]["approximate"] is False
    assert stored["vllm_runtime"]["estimate"]["usable_budget_mib"] == 73_728
    # The requested model is absent because the operator did not name one.
    assert "model" not in stored["vllm_runtime"]["requested"]


@pytest.mark.asyncio
async def test_explicit_device_subset_selects_only_those_devices() -> None:
    provisioner, _ssh = _provisioner()
    devices = (
        VllmDeviceMemory(0, 81_920, 80_000, "9.0"),
        VllmDeviceMemory(1, 81_920, 80_000, "9.0"),
        VllmDeviceMemory(2, 81_920, 80_000, "9.0"),
    )
    selected = provisioner._select_vllm_devices(devices, VllmParams(gpu_devices=(2, 0)))
    assert tuple(device.index for device in selected) == (2, 0)

    with pytest.raises(ProvisioningError, match=r"\[7\]"):
        provisioner._select_vllm_devices(devices, VllmParams(gpu_devices=(7,)))


@pytest.mark.asyncio
async def test_oom_observed_during_health_poll_is_retried() -> None:
    commands: list[str] = []
    # The start script itself succeeds; the engine OOMs while loading.
    provisioner, _ssh = _provisioner(_streaming(commands))
    poll = AsyncMock(side_effect=[TimeoutError("health"), None])
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(provisioner, "_poll_health", new=poll),
    ):
        _model, runtime = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert runtime is not None
    assert runtime.attempt == 1
    assert len(commands) == 2
    assert poll.await_count == 2
    # Pre-planning teardown and the retry teardown.
    assert stop.await_count == 2


@pytest.mark.asyncio
async def test_unrecognized_start_failure_is_not_retried() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands, fail="always"))
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert len(commands) == 1
    # Pre-planning teardown and the terminal failure cleanup.
    assert stop.await_count == 2


@pytest.mark.asyncio
async def test_vllm_oom_probe_reads_the_engine_log() -> None:
    provisioner, _ssh = _provisioner()
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value="torch.cuda.OutOfMemoryError: CUDA out of memory",
    ):
        assert await provisioner._vllm_oom_in_engine_log("host1") is True
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value="INFO started server",
    ):
        assert await provisioner._vllm_oom_in_engine_log("host1") is False
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ssh down"),
    ):
        assert await provisioner._vllm_oom_in_engine_log("host1") is False


@pytest.mark.asyncio
async def test_planning_falls_back_when_the_gpu_is_unclassifiable() -> None:
    provisioner, _ssh = _provisioner()
    unknown = VllmDeviceMemory(0, 81_920, 80_000, None)
    with patch.object(
        provisioner,
        "_read_vllm_device_memory",
        new_callable=AsyncMock,
        return_value=(unknown,),
    ):
        assert (
            await provisioner._plan_vllm_launch(
                "host1", model=None, params=None, attempt=0
            )
            is None
        )

    unparseable = VllmDeviceMemory(0, 81_920, 80_000, "mystery")
    with patch.object(
        provisioner,
        "_read_vllm_device_memory",
        new_callable=AsyncMock,
        return_value=(unparseable,),
    ):
        assert (
            await provisioner._plan_vllm_launch(
                "host1", model=None, params=None, attempt=0
            )
            is None
        )


@pytest.mark.asyncio
async def test_verify_vllm_runtime_skips_blank_and_unselected_rows() -> None:
    provisioner, _ssh = _provisioner()
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=(_hopper(),), requested=VllmRequested()
    )
    telemetry = "\n1, 81920, 40000, 41920\n0, 81920, 1000, 80920\n"
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value=telemetry,
    ):
        runtime = await provisioner._verify_vllm_runtime(
            "host1", plan=plan, vllm_params=None, selected=(_hopper(),)
        )

    assert tuple(row.index for row in runtime.gpus) == (0,)


@pytest.mark.asyncio
async def test_verify_vllm_runtime_rejects_unreadable_or_empty_telemetry() -> None:
    provisioner, _ssh = _provisioner()
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=(_hopper(),), requested=VllmRequested()
    )
    with (
        patch.object(
            provisioner,
            "_ssh_run_command",
            new_callable=AsyncMock,
            return_value="0, not-a-number, 1, 2\n",
        ),
        pytest.raises(ProvisioningError, match="could not parse"),
    ):
        await provisioner._verify_vllm_runtime(
            "host1", plan=plan, vllm_params=None, selected=(_hopper(),)
        )
    with (
        patch.object(
            provisioner,
            "_ssh_run_command",
            new_callable=AsyncMock,
            return_value="5, 81920, 1, 81919\n",
        ),
        pytest.raises(ProvisioningError, match="no selected GPUs"),
    ):
        await provisioner._verify_vllm_runtime(
            "host1", plan=plan, vllm_params=None, selected=(_hopper(),)
        )


@pytest.mark.asyncio
async def test_retry_stops_cleanly_when_replanning_loses_the_host() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands, fail="always"))
    reads = AsyncMock(side_effect=[(_hopper(),), ()])
    with (
        patch.object(provisioner, "_read_vllm_device_memory", new=reads),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert len(commands) == 1
    # Pre-planning teardown, the retry teardown, and the terminal cleanup.
    assert stop.await_count == 3


@pytest.mark.asyncio
async def test_unrecognized_health_failure_stops_cleanly_and_raises() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands))
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(
            provisioner,
            "_poll_health",
            new=AsyncMock(side_effect=TimeoutError("health")),
        ),
        pytest.raises(TimeoutError, match="health"),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    assert len(commands) == 1
    # Pre-planning teardown and the terminal failure cleanup.
    assert stop.await_count == 2


@pytest.mark.asyncio
async def test_verification_failure_stops_the_engine_and_raises() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands))
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(),),
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
        patch.object(
            provisioner,
            "_verify_vllm_runtime",
            new_callable=AsyncMock,
            side_effect=ProvisioningError("unverifiable runtime"),
        ),
        pytest.raises(ProvisioningError, match="unverifiable runtime"),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    # A running engine we cannot verify must not stay registered as healthy:
    # one stop before planning, one after the failed verification.
    assert stop.await_count == 2


@pytest.mark.asyncio
async def test_non_fitting_retry_preserves_the_original_oom_failure() -> None:
    commands: list[str] = []
    provisioner, _ssh = _provisioner(_streaming(commands, fail="always"))
    # The host loses most of its free VRAM between the first attempt and the
    # re-plan, so the agreed adjustment can no longer fit.
    reads = AsyncMock(side_effect=[(_hopper(80_000),), (_hopper(5_120),)])
    with (
        patch.object(provisioner, "_read_vllm_device_memory", new=reads),
        patch.object(
            provisioner,
            "_vllm_oom_in_engine_log",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC),
        )

    # No second launch, and the raised failure is the OOM, not a misleading
    # "does not fit" plan error.
    assert len(commands) == 1
    assert stop.await_count == 3


@pytest.mark.asyncio
async def test_effective_device_subset_matches_the_planned_tensor_parallel() -> None:
    provisioner, _ssh = _provisioner()
    devices = (
        VllmDeviceMemory(0, 81_920, 80_000, "9.0"),
        VllmDeviceMemory(1, 81_920, 80_000, "9.0"),
        VllmDeviceMemory(2, 81_920, 80_000, "9.0"),
    )
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested()
    )
    assert plan.effective.tensor_parallel_size == 2

    effective = provisioner._plan_vllm_effective(None, plan, devices)
    assert effective.gpu_devices == (0, 1)


@pytest.mark.asyncio
async def test_oom_probe_falls_back_to_the_retained_preflight_output() -> None:
    provisioner, _ssh = _provisioner()
    provisioner._log_buffer.create("host1")
    provisioner._log(
        "host1",
        "error",
        "GPU 0 has 1000 MiB free but the memory plan reserved 4096 MiB",
    )
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value="INFO started server",
    ):
        assert await provisioner._vllm_oom_in_engine_log("host1") is True


@pytest.mark.asyncio
async def test_cached_snapshot_is_used_for_an_unknown_explicit_model(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "models--acme--fits"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"0" * 8192)

    provisioner, _ssh = _provisioner(hf_cache_dir=str(tmp_path))
    with patch.object(
        provisioner,
        "_read_vllm_device_memory",
        new_callable=AsyncMock,
        return_value=(_hopper(),),
    ):
        planned = await provisioner._plan_vllm_launch(
            "host1",
            model="acme/fits",
            params=VllmParams(),
            attempt=0,
        )
    assert planned is not None
    plan, _selected = planned
    assert plan.approximate is False
    assert plan.estimate.weights_mib > 0


def test_vllm_failure_is_memory_classifies_a_preflight_refusal() -> None:
    refusal = RemoteCommandError(
        "host1",
        "bash auto-vllm/start-vllm.sh",
        1,
        stderr=(
            "[FAIL] GPU 0 has 1000 MiB free but the memory plan reserved "
            "4096 MiB; another process is using the device"
        ),
    )

    assert NodeProvisioner._vllm_failure_is_memory(refusal) is True
    assert NodeProvisioner._vllm_failure_is_memory(None) is False
    assert NodeProvisioner._vllm_failure_is_memory(RuntimeError("boom")) is False


@pytest.mark.asyncio
async def test_stop_failed_vllm_start_uses_the_uploaded_bundle() -> None:
    provisioner, _ssh = _provisioner()
    provisioner._remote_bundles[("host1", InferenceEngine.VLLM)] = (
        ".qiip/bundles/abc123"
    )
    commands: list[str] = []

    async def run_command(host: str, command: str) -> str:
        commands.append(command)
        return ""

    with patch.object(provisioner, "_ssh_run_command", new=run_command):
        assert await provisioner._stop_failed_vllm_start("host1") is True

    assert len(commands) == 1
    assert ".qiip/bundles/abc123/common/generations.py" in commands[0]
    assert commands[0].endswith("stop-vllm.sh")
    assert "auto-vllm/stop-vllm.sh" not in commands[0]


@pytest.mark.asyncio
async def test_stop_failed_vllm_start_reports_a_failed_stop() -> None:
    provisioner, _ssh = _provisioner()

    async def fail_command(host: str, command: str) -> str:
        raise RemoteCommandError(host, command, 1)

    with patch.object(provisioner, "_ssh_run_command", new=fail_command):
        assert await provisioner._stop_failed_vllm_start("host1") is False


@pytest.mark.asyncio
async def test_preflight_memory_refusal_retries_without_an_engine_log() -> None:
    commands: list[str] = []
    starts = {"count": 0}

    async def streaming(host: str, command: str) -> AsyncIterator[tuple[str, str]]:
        if "nvidia-smi" in command:
            yield "stdout", _MEMORY_ROW
            return
        starts["count"] += 1
        commands.append(command)
        if starts["count"] == 1:
            raise RemoteCommandError(
                host,
                command,
                1,
                stderr=(
                    "[FAIL] GPU 0 has 1000 MiB free but the memory plan "
                    "reserved 4096 MiB; another process is using the device"
                ),
            )
        yield "stdout", "# Model:              " + _AUTO_MODEL

    provisioner, _ssh = _provisioner(streaming)

    async def run_command(host: str, command: str) -> str:
        if "nvidia-smi" in command:
            return _MEMORY_ROW
        if "vllm-serve.log" in command:
            # The engine never started, so there is no OOM text to observe.
            return "INFO starting engine"
        return ""

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(73_500),),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        patch.object(provisioner, "_ssh_run_command", new=run_command),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
    ):
        model, runtime = await provisioner._start_vllm_with_retries(
            "host1", model=None, vllm_params=None, started_at=datetime.now(UTC)
        )

    assert model == _AUTO_MODEL
    assert runtime is not None
    assert runtime.attempt == 1
    assert len(commands) == 2


@pytest.mark.asyncio
async def test_retry_is_skipped_when_the_engine_cannot_be_stopped() -> None:
    commands: list[str] = []

    async def streaming(host: str, command: str) -> AsyncIterator[tuple[str, str]]:
        if "nvidia-smi" in command:
            yield "stdout", _MEMORY_ROW
            return
        commands.append(command)
        raise RemoteCommandError(
            host,
            command,
            1,
            stderr=(
                "[FAIL] GPU 0 has 1000 MiB free but the memory plan reserved "
                "4096 MiB; another process is using the device"
            ),
        )

    provisioner, _ssh = _provisioner(streaming)

    async def run_command(host: str, command: str) -> str:
        return ""

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(_hopper(73_500),),
        ),
        patch.object(
            provisioner,
            "_stop_failed_vllm_start",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(provisioner, "_ssh_run_command", new=run_command),
        pytest.raises(RemoteCommandError),
    ):
        await provisioner._start_vllm_with_retries(
            "host1", model=None, vllm_params=None, started_at=datetime.now(UTC)
        )

    # One start, no re-plan: the incumbent is still holding the VRAM.
    assert len(commands) == 1
