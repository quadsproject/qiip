"""Verification gates for freshly provisioned vLLM servers (issue #134).

The gateway must not route to a node whose server merely answers ``/health``.
These tests cover the evidence boundary directly: process ownership, model and
runtime identity, a bounded inference probe, and the refusal paths for a wrong
model, a failing inference request, a stale listener, and a timeout.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from inference_proxy.models.node import (
    VllmGPUState,
    VllmMemoryEstimate,
    VllmModelSelection,
    VllmRuntimeEffective,
    VllmRuntimeRequest,
    VllmRuntimeState,
    VllmServingVerification,
)
from inference_proxy.provisioning import vllm_serving
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
)
from inference_proxy.provisioning.vllm_fit import VllmDeviceMemory
from inference_proxy.provisioning.vllm_serving import (
    VllmServingError,
    model_identity_matches,
    parse_vllm_processes,
    probe_inference,
    revision_from_root,
    select_model,
    select_process,
)
from tests.provisioning.test_provisioner import _make_provisioner
from tests.provisioning.test_vllm_fit_provisioner import _streaming

pytestmark = pytest.mark.real_vllm_serving

_MODEL = "Qwen/Qwen2.5-32B-Instruct"
_RUNTIME = "/opt/qiip/vllm/runtimes/" + "a" * 64
_COMMIT = "b" * 40


def _process_line(
    *,
    pid: int = 987,
    age: int = 30,
    model: str = _MODEL,
    port: int = 8000,
    runtime: str = _RUNTIME,
) -> str:
    return (
        f"{pid} {age} {runtime}/bin/python {runtime}/bin/vllm serve {model} "
        f"--host 0.0.0.0 --port {port} --tensor-parallel-size 1"
    )


def _models_payload(
    *,
    model: str = _MODEL,
    root: str | None = None,
    max_model_len: int = 8192,
) -> dict[str, Any]:
    card: dict[str, Any] = {"id": model, "max_model_len": max_model_len}
    if root is not None:
        card["root"] = root
    return {"object": "list", "data": [card]}


def _version_payload(version: str = "0.26.0") -> dict[str, str]:
    return {"version": version}


def _provisioner(
    *, process_output: str | None = None, revision: str = _COMMIT
) -> tuple[NodeProvisioner, AsyncMock, AsyncMock]:
    provisioner = _make_provisioner()
    ssh_run = AsyncMock()

    async def run(_hostname: str, command: str) -> str:
        if command.startswith("ps -eo"):
            return process_output if process_output is not None else _process_line()
        if command.startswith("cat --"):
            return revision
        raise AssertionError(f"unexpected command: {command}")

    ssh_run.side_effect = run
    patcher = patch.object(provisioner, "_ssh_run_command", new=ssh_run)
    stop = patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock)
    patcher.start()
    stopper = stop.start()
    return provisioner, ssh_run, stopper


# ── process ownership ──────────────────────────────────────────────────


def test_parse_vllm_processes_identifies_the_serving_process() -> None:
    processes = parse_vllm_processes(_process_line())

    assert len(processes) == 1
    process = processes[0]
    assert (process.pid, process.age_seconds) == (987, 30)
    assert process.model == _MODEL
    assert process.port == 8000
    assert process.executable == f"{_RUNTIME}/bin/vllm"
    assert process.runs_from(_RUNTIME)


def test_parse_vllm_processes_ignores_unrelated_commands_and_defaults_port() -> None:
    output = "\n".join(
        (
            "1 9 /usr/bin/python /opt/vllm/bin/vllm serve model-a",
            "2 9 /usr/bin/other serve model-b --port 9000",
            "3 9 python -c import multiprocessing",
            "",
        )
    )

    processes = parse_vllm_processes(output)

    assert [(process.pid, process.port) for process in processes] == [(1, 8000)]


def test_select_process_accepts_this_attempts_process() -> None:
    process = select_process(
        parse_vllm_processes(_process_line(age=40)),
        port=8000,
        expected_model=_MODEL,
        runtime_path=_RUNTIME,
        attempt_elapsed_seconds=50,
        slack_seconds=10,
    )

    assert process.pid == 987


def test_select_process_refuses_a_stale_listener() -> None:
    with pytest.raises(VllmServingError, match="predates this provisioning attempt"):
        select_process(
            parse_vllm_processes(_process_line(age=900)),
            port=8000,
            expected_model=_MODEL,
            runtime_path=_RUNTIME,
            attempt_elapsed_seconds=60,
            slack_seconds=10,
        )


def test_select_process_refuses_no_process_on_the_port() -> None:
    with pytest.raises(VllmServingError, match="no running vLLM process"):
        select_process(
            parse_vllm_processes(_process_line(port=9000)),
            port=8000,
            expected_model=_MODEL,
            runtime_path=None,
            attempt_elapsed_seconds=60,
            slack_seconds=10,
        )


def test_select_process_refuses_multiple_processes_on_the_port() -> None:
    output = "\n".join((_process_line(pid=1), _process_line(pid=2, age=20)))
    with pytest.raises(VllmServingError, match="2 vLLM processes"):
        select_process(
            parse_vllm_processes(output),
            port=8000,
            expected_model=_MODEL,
            runtime_path=None,
            attempt_elapsed_seconds=60,
            slack_seconds=10,
        )


def test_select_process_refuses_a_different_model() -> None:
    with pytest.raises(VllmServingError, match="serves model"):
        select_process(
            parse_vllm_processes(_process_line(model="other/model")),
            port=8000,
            expected_model=_MODEL,
            runtime_path=None,
            attempt_elapsed_seconds=60,
            slack_seconds=10,
        )


def test_select_process_refuses_another_generation() -> None:
    with pytest.raises(VllmServingError, match="selected generation"):
        select_process(
            parse_vllm_processes(_process_line()),
            port=8000,
            expected_model=_MODEL,
            runtime_path="/opt/qiip/vllm/runtimes/" + "c" * 64,
            attempt_elapsed_seconds=60,
            slack_seconds=10,
        )


def test_model_identity_matches_normalizes_paths() -> None:
    assert model_identity_matches("Qwen/Qwen2.5-7B/", "Qwen/Qwen2.5-7B")
    assert model_identity_matches("/models/qwen/../qwen", "/models/qwen")
    assert not model_identity_matches("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B")


# ── server identity ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_server_identity_reads_version_and_model_cards(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models",
        json=_models_payload(
            root=f"/srv/hf-cache/hub/models--Qwen--Qwen2.5-32B/snapshots/{_COMMIT}"
        ),
    )

    async with httpx.AsyncClient() as client:
        identity = await vllm_serving.fetch_server_identity(
            client, "http://host1:8000", timeout=5
        )

    assert identity.version == "0.26.0"
    card = select_model(identity.models, _MODEL)
    assert card.max_model_len == 8192
    assert revision_from_root(card.root) == _COMMIT


@pytest.mark.asyncio
async def test_fetch_server_identity_requires_a_runtime_version(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json={})

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="did not report a version"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


def test_select_model_refuses_a_healthy_server_for_another_model() -> None:
    with pytest.raises(VllmServingError, match="expected"):
        select_model((vllm_serving.VllmModelCard(model_id="other/model"),), _MODEL)


def test_revision_from_root_requires_a_snapshot_path() -> None:
    assert revision_from_root(f"/cache/snapshots/{_COMMIT}") == _COMMIT
    assert revision_from_root("/cache/model") is None
    assert revision_from_root(None) is None


# ── bounded inference probe ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_inference_returns_latency_without_requiring_text(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "anything"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )

    async with httpx.AsyncClient() as client:
        probe = await probe_inference(
            client, "http://host1:8000", model=_MODEL, timeout=5
        )

    assert probe.task == "chat"
    assert probe.endpoint == "/v1/chat/completions"
    assert probe.latency_ms >= 0
    assert (probe.prompt_tokens, probe.completion_tokens) == (3, 1)
    assert probe.finish_reason == "length"


@pytest.mark.asyncio
async def test_probe_inference_falls_back_when_the_task_is_unsupported(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        status_code=400,
        text="chat template not found",
    )
    httpx_mock.add_response(
        url="http://host1:8000/v1/completions",
        json={"choices": [{"index": 0, "text": "x", "finish_reason": "length"}]},
    )

    async with httpx.AsyncClient() as client:
        probe = await probe_inference(
            client, "http://host1:8000", model=_MODEL, timeout=5
        )

    assert probe.task == "completion"
    assert probe.endpoint == "/v1/completions"


@pytest.mark.asyncio
async def test_probe_inference_refuses_a_failing_request(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        status_code=500,
        text="engine crashed",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="HTTP 500"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_probe_inference_refuses_a_response_without_choices(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions", json={"choices": []}
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="no choices"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_probe_inference_times_out(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(
        httpx.ReadTimeout("slow"),
        url="http://host1:8000/v1/chat/completions",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="timed out"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


# ── provisioner gate ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serving_verification_collects_identity_and_probe(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, ssh_run, _stop = _provisioner()
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json=_models_payload(max_model_len=8192)
    )
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        json={
            "choices": [{"index": 0, "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )

    with patch.object(
        provisioner,
        "_selected_generation",
        return_value={"generation_id": "g" * 64, "runtime_path": _RUNTIME},
    ):
        evidence = await provisioner._verify_vllm_serving(
            "host1",
            expected_model=_MODEL,
            started_at=datetime.now(UTC) - timedelta(seconds=60),
            expected_max_model_len=8192,
        )

    assert evidence.served_model == _MODEL
    assert evidence.runtime_version == "0.26.0"
    assert evidence.generation_id == "g" * 64
    assert evidence.runtime_path == _RUNTIME
    assert evidence.model_revision == _COMMIT
    assert evidence.model_max_model_len == 8192
    assert evidence.process_id == 987
    assert evidence.task == "chat"
    assert evidence.completion_tokens == 1
    assert evidence.verified_at.tzinfo is UTC
    assert any("cat --" in str(call.args[1]) for call in ssh_run.await_args_list)


@pytest.mark.asyncio
async def test_serving_verification_refuses_a_healthy_wrong_model(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, _ssh, _stop = _provisioner()
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json=_models_payload(model="other/model")
    )

    with pytest.raises(ProvisioningError, match="expected"):
        await provisioner._verify_vllm_serving(
            "host1", expected_model=_MODEL, started_at=datetime.now(UTC)
        )


@pytest.mark.asyncio
async def test_serving_verification_refuses_a_healthy_but_failing_server(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, _ssh, _stop = _provisioner()
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", json=_models_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        status_code=500,
        text="engine crashed",
    )

    with pytest.raises(ProvisioningError, match="HTTP 500"):
        await provisioner._verify_vllm_serving(
            "host1", expected_model=_MODEL, started_at=datetime.now(UTC)
        )


@pytest.mark.asyncio
async def test_serving_verification_refuses_a_stale_listener(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, _ssh, _stop = _provisioner(process_output=_process_line(age=900))

    with pytest.raises(ProvisioningError, match="stale listener"):
        await provisioner._verify_vllm_serving(
            "host1",
            expected_model=_MODEL,
            started_at=datetime.now(UTC) - timedelta(seconds=30),
        )

    # A stale listener must be rejected before any HTTP response is trusted.
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_serving_verification_refuses_a_probe_timeout(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, _ssh, _stop = _provisioner()
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", json=_models_payload())
    httpx_mock.add_exception(
        httpx.ReadTimeout("slow"),
        url="http://host1:8000/v1/chat/completions",
    )

    with pytest.raises(ProvisioningError, match="timed out"):
        await provisioner._verify_vllm_serving(
            "host1", expected_model=_MODEL, started_at=datetime.now(UTC)
        )


@pytest.mark.asyncio
async def test_serving_verification_refuses_a_max_model_len_mismatch(
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, _ssh, _stop = _provisioner()
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json=_models_payload(max_model_len=4096)
    )

    with pytest.raises(ProvisioningError, match="max_model_len 4096"):
        await provisioner._verify_vllm_serving(
            "host1",
            expected_model=_MODEL,
            started_at=datetime.now(UTC),
            expected_max_model_len=8192,
        )


@pytest.mark.asyncio
async def test_start_with_retries_persists_serving_evidence(
    httpx_mock: HTTPXMock,
) -> None:
    ssh_client = MagicMock()
    commands: list[str] = []
    ssh_client.run_streaming = _streaming(commands)
    provisioner = _make_provisioner(ssh_client=ssh_client)
    runtime = _runtime_state(None)
    steps: list[str] = []

    async def ssh_run(_hostname: str, command: str) -> str:
        if command.startswith("ps -eo"):
            return _process_line()
        if command.startswith("cat --"):
            return _COMMIT
        raise AssertionError(command)

    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json=_models_payload(max_model_len=8192)
    )
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        json={"choices": [{"index": 0, "message": {"content": "x"}}]},
    )
    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(VllmDeviceMemory(0, 81_920, 80_000, "9.0"),),
        ),
        patch.object(provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
        patch.object(
            provisioner, "_ssh_run_command", new=AsyncMock(side_effect=ssh_run)
        ),
        patch.object(
            provisioner,
            "_verify_vllm_runtime",
            new_callable=AsyncMock,
            return_value=runtime,
        ) as verify_runtime,
    ):
        model, result = await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC) - timedelta(seconds=60),
            on_step=steps.append,
        )

    assert steps == [
        "starting_engine",
        "health_poll",
        "vllm_serving_verify",
        "vllm_runtime_verify",
    ]
    assert model == _MODEL
    assert result is runtime
    assert verify_runtime.await_args is not None
    serving = verify_runtime.await_args.kwargs["serving"]
    assert isinstance(serving, VllmServingVerification)
    assert serving.served_model == _MODEL


@pytest.mark.asyncio
async def test_start_with_retries_stops_the_engine_on_serving_failure() -> None:
    ssh_client = MagicMock()
    commands: list[str] = []
    ssh_client.run_streaming = _streaming(commands)
    provisioner = _make_provisioner(ssh_client=ssh_client)

    async def ssh_run(_hostname: str, command: str) -> str:
        if command.startswith("ps -eo"):
            return _process_line(age=900)
        raise AssertionError(command)

    with (
        patch.object(
            provisioner,
            "_read_vllm_device_memory",
            new_callable=AsyncMock,
            return_value=(VllmDeviceMemory(0, 81_920, 80_000, "9.0"),),
        ),
        patch.object(
            provisioner, "_stop_failed_vllm_start", new_callable=AsyncMock
        ) as stop,
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
        patch.object(
            provisioner, "_ssh_run_command", new=AsyncMock(side_effect=ssh_run)
        ),
        pytest.raises(ProvisioningError, match="stale listener"),
    ):
        await provisioner._start_vllm_with_retries(
            "host1",
            model=None,
            vllm_params=None,
            started_at=datetime.now(UTC) - timedelta(seconds=30),
        )

    # Pre-planning incumbent teardown plus the post-verification failure stop.
    assert stop.await_count == 2


# ── malformed and unavailable evidence ────────────────────────────────


def test_parse_vllm_processes_skips_malformed_lines() -> None:
    output = "\n".join(
        (
            "not-a-pid 9 /opt/vllm/bin/vllm serve model",
            "4 not-an-age /opt/vllm/bin/vllm serve model",
            "5 9 serve model --port 8000",
            "6 9 /opt/vllm/bin/python /opt/other serve model --port 8000",
            "7 9 /opt/vllm/bin/vllm serve",
            "8 9 /opt/vllm/bin/vllm serve model --port not-a-number",
            "9 9 /opt/vllm/bin/vllm serve model --port=8001",
            "short",
        )
    )

    processes = parse_vllm_processes(output)

    assert [(process.pid, process.port, process.model) for process in processes] == [
        (7, 8000, ""),
        (9, 8001, "model"),
    ]


def test_select_model_matches_a_server_that_reports_the_root_path() -> None:
    card = select_model(
        (vllm_serving.VllmModelCard(model_id="served-name", root="/models/qwen"),),
        "/models/qwen",
    )

    assert card.model_id == "served-name"


@pytest.mark.asyncio
async def test_probe_inference_uses_completions_for_a_base_model(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/completions",
        json={"choices": [{"index": 0, "text": "x"}]},
    )

    async with httpx.AsyncClient() as client:
        probe = await probe_inference(
            client, "http://host1:8000/", model="org/model-base", timeout=5
        )

    assert (probe.task, probe.endpoint) == ("completion", "/v1/completions")


@pytest.mark.asyncio
async def test_probe_inference_reports_an_invalid_json_response(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        content=b"not-json",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="invalid JSON"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_probe_inference_reports_a_transport_error(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("refused"),
        url="http://host1:8000/v1/chat/completions",
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="failed: refused"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_probe_inference_rejects_a_non_object_payload(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/v1/chat/completions", json=[])

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="not a JSON object"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_fetch_server_identity_reports_a_version_timeout(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("slow"), url="http://host1:8000/version")

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="version probe timed out"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_reports_a_version_transport_error(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("refused"), url="http://host1:8000/version"
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="version probe failed"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_a_version_error_status(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", status_code=503)

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="HTTP 503"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_invalid_version_json(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", content=b"not-json")

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="not valid JSON"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_a_non_object_version(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=[])

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="not an object"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_reports_a_model_discovery_timeout(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_exception(
        httpx.ReadTimeout("slow"), url="http://host1:8000/v1/models"
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="model discovery timed out"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_reports_a_model_discovery_transport_error(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_exception(
        httpx.ConnectError("refused"), url="http://host1:8000/v1/models"
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="model discovery failed"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_a_model_error_status(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", status_code=500)

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="HTTP 500"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_invalid_model_json(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", content=b"not-json")

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="not valid JSON"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_a_non_object_model_list(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", json=[])

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="not an object"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_an_empty_model_list(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(url="http://host1:8000/v1/models", json={"data": []})

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="no models"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


@pytest.mark.asyncio
async def test_fetch_server_identity_skips_unusable_model_entries(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models",
        json={
            "data": [
                "not-a-card",
                {"id": ""},
                {"id": _MODEL, "root": "", "max_model_len": True},
            ]
        },
    )

    async with httpx.AsyncClient() as client:
        identity = await vllm_serving.fetch_server_identity(
            client, "http://host1:8000", timeout=5
        )

    assert identity.models == (
        vllm_serving.VllmModelCard(model_id=_MODEL, root=None, max_model_len=None),
    )


@pytest.mark.asyncio
async def test_fetch_server_identity_rejects_a_list_of_unusable_entries(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json={"data": [{"id": ""}]}
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="no usable model ids"):
            await vllm_serving.fetch_server_identity(
                client, "http://host1:8000", timeout=5
            )


def test_selected_generation_reads_the_attempt_store() -> None:
    provisioner = _make_provisioner()
    store = MagicMock()
    store.get.return_value = {
        "selected_generation": {"generation_id": "g" * 64, "runtime_path": _RUNTIME}
    }
    provisioner._log_buffer.store = store
    provisioner._log_buffer.attempts["host1"] = "attempt-1"

    selected = provisioner._selected_generation("host1")

    assert selected == {"generation_id": "g" * 64, "runtime_path": _RUNTIME}
    store.get.assert_called_once_with("attempt-1")


def test_selected_generation_is_absent_without_an_attempt() -> None:
    provisioner = _make_provisioner()
    provisioner._log_buffer.store = MagicMock()

    assert provisioner._selected_generation("host1") is None


def test_selected_generation_ignores_unusable_or_failing_store_rows() -> None:
    provisioner = _make_provisioner()
    store = MagicMock()
    provisioner._log_buffer.store = store
    provisioner._log_buffer.attempts["host1"] = "attempt-1"
    store.get.return_value = {"selected_generation": "not-a-dict"}
    assert provisioner._selected_generation("host1") is None

    store.get.side_effect = OSError("database is locked")
    assert provisioner._selected_generation("host1") is None


@pytest.mark.asyncio
async def test_probe_inference_reports_both_tasks_unsupported(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions", status_code=400
    )
    httpx_mock.add_response(url="http://host1:8000/v1/completions", status_code=405)

    async with httpx.AsyncClient() as client:
        with pytest.raises(VllmServingError, match="does not support"):
            await probe_inference(client, "http://host1:8000", model=_MODEL, timeout=5)


@pytest.mark.asyncio
async def test_serving_verification_uses_the_snapshot_revision_without_ssh_read(
    httpx_mock: HTTPXMock,
) -> None:
    root = f"/srv/hf-cache/hub/models--Qwen--Qwen2.5-32B/snapshots/{_COMMIT}"
    provisioner, ssh_run, _stop = _provisioner(revision="")
    httpx_mock.add_response(url="http://host1:8000/version", json=_version_payload())
    httpx_mock.add_response(
        url="http://host1:8000/v1/models", json=_models_payload(root=root)
    )
    httpx_mock.add_response(
        url="http://host1:8000/v1/chat/completions",
        json={"choices": [{"index": 0, "message": {"content": "x"}}]},
    )

    evidence = await provisioner._verify_vllm_serving(
        "host1", expected_model=_MODEL, started_at=datetime.now(UTC)
    )

    assert evidence.model_revision == _COMMIT
    assert not any("cat --" in str(call.args[1]) for call in ssh_run.await_args_list)


def _runtime_state(
    serving: VllmServingVerification | None,
) -> VllmRuntimeState:
    return VllmRuntimeState(
        requested=VllmRuntimeRequest(),
        effective=VllmRuntimeEffective(
            model=_MODEL,
            model_selection=VllmModelSelection.AUTO,
            model_reason="fits",
            tensor_parallel_size=1,
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.9,
        ),
        estimate=VllmMemoryEstimate(
            weights_mib=1,
            weights_per_gpu_mib=1,
            kv_cache_mib=1,
            runtime_overhead_mib=1,
            reserve_mib=1,
            usable_budget_mib=1,
            free_per_gpu_mib=1,
        ),
        reason="fits",
        gpus=(VllmGPUState(index=0, total_mib=1, used_mib=0, free_mib=1),),
        serving=serving,
        observed_at=datetime.now(UTC),
    )


def test_runtime_state_omits_absent_serving_evidence_from_the_wire() -> None:
    assert "serving" not in _runtime_state(None).model_dump(mode="json")


def test_runtime_state_serializes_serving_evidence() -> None:
    evidence = VllmServingVerification(
        served_model=_MODEL,
        runtime_version="0.26.0",
        process_id=1,
        process_age_seconds=1,
        task="chat",
        endpoint="/v1/chat/completions",
        latency_ms=1.0,
        verified_at=datetime.now(UTC),
    )

    dumped = _runtime_state(evidence).model_dump(mode="json")

    assert dumped["serving"]["served_model"] == _MODEL
    assert dumped["serving"]["runtime_version"] == "0.26.0"


def test_serving_evidence_requires_a_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        VllmServingVerification(
            served_model=_MODEL,
            runtime_version="0.26.0",
            process_id=1,
            process_age_seconds=1,
            task="chat",
            endpoint="/v1/chat/completions",
            latency_ms=1.0,
            verified_at=datetime(2026, 1, 1),
        )


@pytest.mark.asyncio
async def test_model_revision_is_absent_for_paths_without_a_hub_slug() -> None:
    provisioner, ssh_run, _stop = _provisioner()

    assert await provisioner._read_vllm_model_revision("host1", "/models/x") is None
    assert await provisioner._read_vllm_model_revision("host1", "single") is None
    ssh_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_revision_rejects_a_non_commit_ref() -> None:
    provisioner, ssh_run, _stop = _provisioner(revision="main")

    assert await provisioner._read_vllm_model_revision("host1", "org/model") is None
    assert ssh_run.await_args is not None
    assert "models--org--model/refs/main" in ssh_run.await_args.args[1]


@pytest.mark.asyncio
async def test_model_revision_is_absent_when_the_cache_is_unreadable() -> None:
    provisioner, ssh_run, _stop = _provisioner()
    ssh_run.side_effect = RuntimeError("ssh failed")

    assert await provisioner._read_vllm_model_revision("host1", "org/model") is None
