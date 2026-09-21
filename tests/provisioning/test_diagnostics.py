"""Failure evidence through the shipped setup/launch and node recorder boundary."""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning import diagnostics
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    PreflightError,
    ProvisioningError,
)
from inference_proxy.provisioning.remote_logs import RemoteLogCollector
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHConnectionError,
)
from tests.provisioning.test_attempt_logs import LocalNodeSSH
from tests.provisioning.test_attempt_logs import harness as harness
from tests.provisioning.test_vllm_scripts import _write_executable


def _prepare(
    provisioner: NodeProvisioner, ssh: LocalNodeSSH, message: str | None = None
) -> None:
    setup = ssh.root / "auto-vllm/setup.sh"
    text, count = re.subn(
        r"(?m)^    step system_update run_system_update$",
        """    run_system_update() { :; }
    install_nvidia_driver() { :; }
    install_cuda_toolkit() { :; }
    ensure_fabric_manager() { :; }
    install_vllm() { :; }
    install_vllm_unit() { :; }
    mount_nfs_cache() { :; }
    configure_firewall() { :; }
    install_llmfit() { :; }
    step system_update run_system_update""",
        setup.read_text(),
    )
    assert count == 1, "Refuse to run real installation functions"
    if message:
        text = text.replace(
            "install_cuda_toolkit() { :; }",
            f"install_cuda_toolkit() {{ echo '{message}' >&2; return 7; }}",
        )
    setup.write_text(text)
    # Exercise the actual setup/start/recorder. Replace only hardware probes,
    # network reachability and uploads which would overwrite controlled fixtures.
    provisioner.preflight = AsyncMock()  # type: ignore[method-assign]
    provisioner._reconcile_host = AsyncMock(return_value=False)  # type: ignore[method-assign]
    provisioner._upload_scripts = AsyncMock()  # type: ignore[method-assign]
    provisioner._verify_gpu = AsyncMock()  # type: ignore[method-assign]
    provisioner._read_gpu_inventory = AsyncMock(return_value=[])  # type: ignore[method-assign]
    _write_executable(
        Path(ssh.environment["PATH"].split(":")[0]) / "journalctl",
        """#!/bin/bash
case "$*" in
  *--kernel*) echo 'Out of memory: Killed process 123 (vllm)'; echo 'NVRM: Xid 79 GPU lost' ;;
  *) echo 'NVIDIA Fabric Manager failed initialization' ;;
esac
""",
    )


@pytest.mark.parametrize("kind", ["cuda", "oom", "health"])
async def test_failures_have_durable_correlated_bundle(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    kind: str,
    client: TestClient,
    mock_provisioner: MagicMock,
    httpx_mock: HTTPXMock,
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh, "CUDA installation failed" if kind == "cuda" else None)
    if kind == "oom":
        _write_executable(
            Path(ssh.environment["AUTOVLLM_BIN"]),
            "#!/bin/bash\necho 'CUDA out of memory' >&2\nexit 1\n",
        )
    if kind == "health":
        provisioner._settings.health_poll_timeout = 0
        httpx_mock.add_response(url="http://host1:8000/health", status_code=503)
    with pytest.raises((RemoteCommandError, ProvisioningError)) as caught:
        await provisioner._provision("host1", model="org/model")
    attempt = store.history("host1")["attempts"][0]
    failure = attempt["failure"]
    assert failure["original_error"] == str(caught.value)
    assert (
        failure["failed_stage"]
        == {"cuda": "cuda_toolkit", "oom": "starting_engine", "health": "health_poll"}[
            kind
        ]
    )
    assert failure["started_at"] <= failure["failed_at"] <= attempt["finished_at"]
    assert failure["duration_seconds"] >= 0
    assert failure["command"]["phase_id"] in attempt["phases"]
    assert failure["command"]["duration_seconds"] >= 0
    assert failure["exit_code"] == (None if kind == "health" else 1)
    sources = attempt["diagnostics"]["sources"]
    assert set(sources) == set(diagnostics.SOURCE_NAMES)
    assert sources["kernel_gpu_oom"]["status"] == "collected"
    assert sources["nvidia_services"]["status"] == "collected"
    assert sources["os"]["status"] == "collected"
    assert (
        sources["setup.stderr" if kind == "cuda" else "engine"]["status"] == "collected"
    )
    # The existing admin API and downloadable bundle carry the same evidence,
    # including after reopening the gateway store.
    mock_provisioner.log_buffer = provisioner.log_buffer
    base = f"/admin/provisioning/host1/attempts/{attempt['attempt_id']}"
    response = client.get(base + "/logs")
    assert response.status_code == 200
    assert response.json()["attempt"]["failure"] == failure
    bundle = [
        json.loads(line)
        for line in gzip.decompress(client.get(base + "/bundle").content).splitlines()
    ]
    assert bundle[0]["manifest"]["failure"] == failure
    assert any("Killed process" in record.get("msg", "") for record in bundle)
    assert AttemptLogStore(store.path).get(attempt["attempt_id"])["failure"] == failure
    assert ssh.launches == (1 if kind == "cuda" else 2)


async def test_ssh_loss_defers_sources_and_recovers_without_relaunch(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh)
    original_run = ssh.run

    async def disconnect(
        host: str, command: str, timeout: float = 60, *, log_label: str | None = None
    ) -> tuple[str, str, int]:
        if log_label == "provisioning log recorder (diagnose)":
            raise SSHConnectionError(host, "node disconnected")
        return await original_run(host, command, timeout, log_label=log_label)

    provisioner._poll_health = AsyncMock(  # type: ignore[method-assign]
        side_effect=SSHConnectionError("host1", "original SSH loss")
    )
    with (
        patch.object(ssh, "run", side_effect=disconnect),
        pytest.raises(SSHConnectionError, match="original SSH loss"),
    ):
        await provisioner._provision("host1", model="org/model")
    attempt = store.history("host1")["attempts"][0]
    failure = attempt["failure"]
    assert all(
        source["deferred"] for source in attempt["diagnostics"]["sources"].values()
    )
    # Recovery is keyed to the failed attempt even when a new operation exists.
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    await provisioner._recover_pending_evidence("host1")
    recovered = store.get(attempt["attempt_id"])
    assert recovered["failure"] == failure
    assert recovered["failure_summary"] == attempt["failure_summary"]
    assert recovered["diagnostics"]["sources"]["engine"]["status"] == "collected"
    assert (
        recovered["diagnostics"]["sources"]["os"]["collected_at"]
        >= failure["failed_at"]
    )
    assert ssh.launches == 2
    before = store.read(attempt["attempt_id"], source="diagnostics.engine")["records"]
    await provisioner.collect_logs("host1", attempt["attempt_id"])
    assert (
        store.read(attempt["attempt_id"], source="diagnostics.engine")["records"]
        == before
    )


async def test_disconnect_at_setup_boundary_retains_remote_failure(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh, "CUDA initialization failed while disconnected")
    provisioner._settings.log_reconnect_attempts = 0
    original_run = ssh.run
    disconnected = False

    async def lose_connection(
        host: str, command: str, timeout: float = 60, *, log_label: str | None = None
    ) -> tuple[str, str, int]:
        nonlocal disconnected
        if disconnected:
            raise SSHConnectionError(host, "original lost connection")
        result = await original_run(host, command, timeout, log_label=log_label)
        if log_label == "provisioning log recorder (launch)":
            disconnected = True
            raise SSHConnectionError(host, "original lost connection")
        return result

    with (
        patch.object(ssh, "run", side_effect=lose_connection),
        pytest.raises(SSHConnectionError),
    ):
        await provisioner._provision("host1", model="org/model")
    attempt = store.history("host1")["attempts"][0]
    assert attempt["failure"]["failed_stage"] == "setup"
    assert attempt["failure"]["command"]["phase_id"] in attempt["phases"]
    assert all(
        source["deferred"] for source in attempt["diagnostics"]["sources"].values()
    )
    await asyncio.sleep(0.5)
    await provisioner.collect_logs("host1", attempt["attempt_id"])
    recovered = store.get(attempt["attempt_id"])
    assert (
        recovered["failure"]["original_error"] == attempt["failure"]["original_error"]
    )
    assert recovered["failure"]["command"]["exit_status"] == 1
    assert store.read(
        attempt["attempt_id"],
        source="diagnostics.setup.stderr",
        query="CUDA initialization",
    )["records"]
    assert ssh.launches == 1


@pytest.mark.parametrize(
    "collector_error", [RuntimeError("collector broken"), TimeoutError()]
)
async def test_preflight_error_survives_collector_failure(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    collector_error: Exception,
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh)
    error = PreflightError("host1", ["original disk failure"])
    provisioner.preflight = AsyncMock(side_effect=error)  # type: ignore[method-assign]
    with (
        patch.object(ssh, "run", side_effect=collector_error),
        pytest.raises(PreflightError) as caught,
    ):
        await provisioner._provision("host1")
    assert caught.value is error
    attempt = store.history("host1")["attempts"][0]
    assert attempt["failure"]["command"] is None
    assert attempt["failure"]["original_error"] == str(error)
    assert "original disk failure" in attempt["failure_summary"]
    assert all(
        s["status"]
        == ("timed_out" if isinstance(collector_error, TimeoutError) else "unavailable")
        for s in attempt["diagnostics"]["sources"].values()
    )


async def test_overall_gateway_deadline(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._settings.diagnostics_timeout = 0.05
    provisioner._begin_log("host1", InferenceEngine.VLLM)

    async def hang(*args: Any, **kwargs: Any) -> None:
        await asyncio.Event().wait()

    started = time.monotonic()
    with patch.object(ssh, "run", side_effect=hang):
        await provisioner._capture_failure(
            "host1", "health_poll", TimeoutError("original timeout")
        )
    assert time.monotonic() - started < 1
    assert all(
        s["status"] == "timed_out"
        for s in store.history("host1")["attempts"][0]["diagnostics"][
            "sources"
        ].values()
    )


def test_source_deadlines_truncation_signals_and_missing_tools() -> None:
    output = diagnostics.capture(
        [sys.executable, "-c", "print('x' * 10000); print('retained tail')"], 2, 512
    )
    assert output["status"] == "truncated"
    assert output["output"].endswith("retained tail")
    assert len(output["output"].encode()) <= 512
    assert (
        diagnostics.capture(["/nonexistent-diagnostic-tool"], 1, 512)["status"]
        == "unavailable"
    )
    hung = diagnostics.capture(
        [
            sys.executable,
            "-c",
            "import time; print('before timeout', flush=True); time.sleep(30)",
        ],
        0.1,
        512,
    )
    assert hung["status"] == "timed_out"
    assert hung["duration_seconds"] < 1
    assert "before timeout" in hung["output"]
    killed = diagnostics.capture(["bash", "-c", "kill -KILL $$"], 2, 512)
    assert killed["signal"] == 9


def test_node_deadline_and_independent_collector_failure(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "node.sqlite3")
    attempt = store.create("host1")
    config: dict[str, Any] = dict(
        attempt_id=attempt,
        engine="vllm",
        mount_point="/tmp",
        diagnostics_timeout=0.1,
        diagnostics_source_timeout=0.04,
        diagnostics_source_max_bytes=512,
        failure=dict(
            started_at=store.get(attempt)["started_at"],
            failed_at=store.get(attempt)["started_at"],
        ),
    )
    calls = 0

    def broken(*args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("collector failure")
        time.sleep(0.04)
        return dict(status="collected", output="evidence")

    with patch.object(diagnostics, "capture", side_effect=broken):
        diagnostics.collect(config, store)
    sources = store.get(attempt)["diagnostics"]["sources"]
    assert sources["nvidia_services"]["status"] == "unavailable"
    assert "ValueError" in sources["nvidia_services"]["reason"]
    assert sources["kernel_gpu_oom"]["status"] == "collected"
    assert sources["mounts"]["status"] == "timed_out"


def test_failure_signal_and_command_secrets(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, _, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    collector = provisioner._remote_logs
    assert isinstance(collector, RemoteLogCollector)
    attempt = provisioner.log_buffer.attempts["host1"]
    collector.record_failure(attempt, "setup", RemoteCommandError("host1", "setup", -9))
    assert store.get(attempt)["failure"]["signal"] == 9
    assert store.get(attempt)["failure"]["exit_code"] is None
    collector.record_failure(attempt, "collector", RuntimeError("later error"))
    assert store.get(attempt)["failure"]["failed_stage"] == "setup"


def test_paginated_diagnostics_do_not_claim_missing_evidence_is_collected(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    tmp_path: Path,
) -> None:
    provisioner, _, gateway = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    gateway.append(attempt, "gateway record changes sequence offsets")
    node = AttemptLogStore(tmp_path / "remote.sqlite3")
    node.create("host1", attempt_id=attempt)
    node.append(attempt, "GPU inventory", source="diagnostics.gpu")
    node.append(attempt, "free VRAM", source="diagnostics.gpu")
    node.update(
        attempt,
        diagnostics=dict(
            sources=dict(
                gpu=dict(status="collected", deferred=False, first_seq=0, last_seq=1)
            )
        ),
    )
    collector = provisioner._remote_logs
    assert collector is not None
    collector._ingest_page(attempt, node.read(attempt, limit=1))
    partial = gateway.get(attempt)["diagnostics"]["sources"]["gpu"]
    assert partial["status"] == "unavailable"
    assert partial["deferred"]
    assert partial["first_seq"] == 1
    collector._ingest_page(attempt, node.read(attempt, after=1))
    complete = gateway.get(attempt)["diagnostics"]["sources"]["gpu"]
    assert complete["status"] == "collected"
    assert not complete["deferred"]
    assert complete["last_seq"] == 2
    gateway.attempt_max_bytes = 800
    gateway.append(attempt, "x" * 3000)
    assert (
        gateway.get(attempt)["diagnostics"]["sources"]["gpu"]["status"] == "truncated"
    )


async def test_lost_diagnostic_response_recovers_node_snapshot(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    run = ssh.run

    async def lose_response(
        host: str, command: str, timeout: float = 60, *, log_label: str | None = None
    ) -> tuple[str, str, int]:
        result = await run(host, command, timeout, log_label=log_label)
        if log_label == "provisioning log recorder (diagnose)":
            raise SSHConnectionError(host, "response lost after collection")
        return result

    with patch.object(ssh, "run", side_effect=lose_response):
        await provisioner._capture_failure(
            "host1", "preflight", RuntimeError("original failure")
        )
    assert store.get(attempt)["diagnostics"]["sources"]["os"]["status"] == "unavailable"
    node = AttemptLogStore(ssh.root / "logs/attempts.sqlite3")
    captured = node.get(attempt)["diagnostics"]["sources"]["os"]["collected_at"]
    await provisioner.collect_logs("host1", attempt)
    recovered = store.get(attempt)
    assert recovered["failure"]["original_error"] == "original failure"
    assert recovered["diagnostics"]["sources"]["os"]["collected_at"] == captured
    assert recovered["diagnostics"]["sources"]["os"]["status"] == "collected"
    assert node.get(attempt)["status"] == "failed"
    assert ssh.launches == 0
