"""Reconciliation behavior: remote liveness, blocking, fencing, and recovery."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
    RelaunchPreconditionError,
)
from inference_proxy.provisioning.ssh_client import RemoteCommandError
from tests.provisioning.test_attempt_logs import LocalNodeSSH
from tests.provisioning.test_vllm_scripts import _script_environment, _write_executable

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def harness(tmp_path: Path) -> tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore]:
    node = tmp_path / "node"
    node.mkdir()
    for bundle in ("auto-vllm", "auto-llamacpp", "common"):
        shutil.copytree(ROOT / bundle, node / bundle)
    shutil.copy(
        ROOT / "inference_proxy/provisioning/log_store.py", node / "common/log_store.py"
    )
    fake_engine = tmp_path / "fake-vllm"
    _write_executable(
        fake_engine,
        "#!/bin/bash\necho 'engine boot evidence'\necho 'engine stderr evidence' >&2\nsleep 0.5\n",
    )
    environment = _script_environment(
        tmp_path, vllm_bin=fake_engine, process_log=tmp_path / "engine-events"
    )
    environment["AUTOVLLM_SCRIPT_DIR"] = str(node / "auto-vllm")
    _write_executable(
        Path(environment["PATH"].split(":")[0]) / "journalctl",
        '#!/bin/bash\necho \'{"__CURSOR":"cursor-1","MESSAGE":"service evidence"}\'\nsleep 0.2\n',
    )
    ssh = LocalNodeSSH(node, environment)
    store = AttemptLogStore(tmp_path / "gateway.sqlite3")
    buffer = ProvisioningLogBuffer(store=store)
    settings = ProvisioningSettings(
        scripts_dir=node / "auto-vllm",
        nfs_mount_point=environment["AUTOVLLM_NFS_MOUNT_POINT"],
        log_remote_root=str(node / "logs"),
        log_poll_interval=0.02,
        health_poll_timeout=1,
    )
    provisioner = NodeProvisioner(
        ssh_client=ssh,
        etcd_client=MagicMock(prefix="/nodes/"),
        settings=settings,
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=EndpointPolicy.from_values(
            allowed_hosts=["host1"], allowed_networks=[], allowed_ports=[8000]
        ),
        log_buffer=buffer,
        nfs_export="fixture:/cache",
    )
    return provisioner, ssh, store


def _recorder_config(
    attempt_id: str,
    hostname: str,
    root: Path,
    *,
    command: str = "echo done",
    phase: str = "phase-1",
    stage: str = "setup",
    engine: str = "vllm",
) -> dict[str, object]:
    return dict(
        attempt_id=attempt_id,
        hostname=hostname,
        engine=engine,
        model=None,
        bundle_version="sha256:abc",
        root=str(root),
        max_bytes=64 * 1024 * 1024,
        attempt_max_bytes=16 * 1024 * 1024,
        max_attempts=32,
        retention_days=7,
        max_record_bytes=16384,
        health_timeout=2,
        inactivity_timeout=30,
        after=0,
        phase=phase,
        stage=stage,
        command=command,
        timeout=30,
    )


def _recorder_command(config: dict[str, object], action: str) -> str:
    payload = shlex.quote(json.dumps(config))
    return f"printf %s {payload} | python3 common/provision-logs.py {action}"


async def _launch(ssh: LocalNodeSSH, config: dict[str, object]) -> None:
    await ssh.run("host1", _recorder_command(config, "launch"))


async def _node_manifest(
    ssh: LocalNodeSSH, config: dict[str, object]
) -> dict[str, Any]:
    stdout, _stderr, _status = await ssh.run("host1", _recorder_command(config, "read"))
    return cast(dict[str, Any], json.loads(stdout))


def test_pending_hosts_and_latest_running(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    finished = store.create("host1")
    store.update(finished, status="complete")
    interrupted = store.create("host1")
    store.update(interrupted, status="interrupted")
    running = store.create("host1")
    store.update(running, status="running")
    relaunch = store.create("host2", operation="relaunch")
    store.update(relaunch, status="interrupted")
    assert set(store.pending_hosts()) == {"host1", "host2"}
    assert store.latest_running("host1") == running
    assert store.latest_running("host1", exclude=running) is None


@pytest.mark.asyncio
async def test_reconcile_blocks_running_remote_operation(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    config = _recorder_config(attempt, "host1", ssh.root / "logs", command="sleep 20")
    await _launch(ssh, config)

    blocked = await provisioner._reconcile_host("host1")
    assert blocked is True
    attempt_state = store.get(attempt)
    assert attempt_state["status"] == "interrupted"
    assert attempt_state["survivor"] is False
    assert "still running on the node" in attempt_state["failure_summary"]


@pytest.mark.asyncio
async def test_reconcile_allows_completed_remote_operation(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    config = _recorder_config(attempt, "host1", ssh.root / "logs", command="echo done")
    await _launch(ssh, config)
    for _ in range(50):
        manifest = await _node_manifest(ssh, config)
        phases = manifest.get("attempt", {}).get("phases", {})
        if phases and list(phases.values())[-1].get("status") == "complete":
            break
        await asyncio.sleep(0.05)

    assert await provisioner._reconcile_host("host1") is False


@pytest.mark.asyncio
async def test_reconcile_ssh_loss_gateway_failed_remote_running(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    """SSH loss marks the gateway attempt failed while the node keeps running."""
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    config = _recorder_config(attempt, "host1", ssh.root / "logs", command="sleep 20")
    await _launch(ssh, config)
    store.update(attempt, status="failed", failure_summary="setup: ssh lost")

    blocked = await provisioner._reconcile_host("host1")
    assert blocked is True


@pytest.mark.asyncio
async def test_reconcile_evicted_remote_is_safe(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    store.update(attempt, status="failed", failure_summary="setup: ssh lost")
    # No node-side attempt was ever created: the node returns unavailable.
    assert await provisioner._reconcile_host("host1") is False


@pytest.mark.asyncio
async def test_reconcile_unreachable_host_blocks(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, _ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    store.update(attempt, status="failed", failure_summary="setup: ssh lost")
    monkeypatch.setattr(
        provisioner._remote_logs,
        "host_active",
        AsyncMock(return_value={"unreachable": True, "error": "ssh lost"}),
    )

    assert await provisioner._reconcile_host("host1") is True


@pytest.mark.asyncio
async def test_node_flock_blocks_second_worker(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    first = store.create("host1", operation="provision", bundle_version="sha256:abc")
    second = store.create("host1", operation="provision", bundle_version="sha256:abc")
    config_a = _recorder_config(first, "host1", ssh.root / "logs", command="sleep 20")
    await _launch(ssh, config_a)
    # Wait until worker A holds the lock (phase running) before contending,
    # otherwise worker B can win the race and A becomes the busy one.
    for _ in range(50):
        manifest = await _node_manifest(ssh, config_a)
        phases = manifest["attempt"].get("phases", {})
        if phases and list(phases.values())[-1].get("status") == "running":
            break
        await asyncio.sleep(0.05)
    config_b = _recorder_config(
        second,
        "host1",
        ssh.root / "logs",
        command="echo never",
        phase="phase-b",
        engine="llamacpp",
    )
    await _launch(ssh, config_b)
    manifest_b: dict[str, Any] = {}
    for _ in range(50):
        manifest_b = await _node_manifest(ssh, config_b)
        if manifest_b["attempt"]["status"] == "failed":
            break
        await asyncio.sleep(0.05)
    assert manifest_b["attempt"]["status"] == "failed"
    assert "Host busy" in json.dumps(manifest_b["attempt"]["issues"])
    phase = list(manifest_b["attempt"]["phases"].values())[-1]
    assert phase["exit_status"] == 126


@pytest.mark.asyncio
async def test_provision_blocks_when_host_active(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, _ssh, store = harness
    monkeypatch.setattr(
        provisioner._remote_logs,
        "host_active",
        AsyncMock(return_value={"active": True, "status": "running"}),
    )
    with pytest.raises(ProvisioningError, match="still active on the node"):
        await provisioner.provision("host1", model="org/model")
    assert provisioner.log_buffer.attempts.get("host1") is None
    assert store.history("host1")["total"] == 0


@pytest.mark.asyncio
async def test_relaunch_blocks_when_host_active(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, _ssh, _store = harness
    monkeypatch.setattr(
        provisioner._remote_logs,
        "host_active",
        AsyncMock(return_value={"active": True, "status": "survivor"}),
    )
    monkeypatch.setattr(provisioner, "_relaunch_llamacpp", AsyncMock())
    request = provisioner._default_llamacpp_runtime_request()
    with pytest.raises(RelaunchPreconditionError, match="still active on the node"):
        await provisioner.relaunch_llamacpp("host1", request)
    assert provisioner.log_buffer.attempts.get("host1") is None


@pytest.mark.asyncio
async def test_host_active_normalizes_recorder_failure(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    store.update(attempt, status="failed", failure_summary="setup: ssh lost")

    async def recorder_crash(*_args: object, **_kwargs: object) -> tuple[str, str, int]:
        raise RemoteCommandError("host1", "active", 1, "recorder boom")

    monkeypatch.setattr(ssh, "run", recorder_crash)
    collector = provisioner._remote_logs
    assert collector is not None
    probe = await collector.host_active("host1")
    assert probe.get("unreachable") is True
    assert await provisioner._reconcile_host("host1") is True


@pytest.mark.asyncio
async def test_survivor_phase_blocks_reconcile(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner.log_buffer.create(
        "host1", engine="vllm", operation="provision", bundle_version="sha256:abc"
    )
    attempt = provisioner.log_buffer.attempts["host1"]
    config = _recorder_config(attempt, "host1", ssh.root / "logs", command="echo done")
    await _launch(ssh, config)
    phase = None
    for _ in range(100):
        manifest = await _node_manifest(ssh, config)
        phases = manifest.get("attempt", {}).get("phases", {})
        if phases:
            phase = list(phases.keys())[0]
            status = list(phases.values())[0].get("status")
            recording = list(phases.values())[0].get("recording")
            # Wait for the worker to exit; a late command_done overwrite would
            # clear the survivor status we are about to force.
            if status == "complete" and recording is False:
                break
        await asyncio.sleep(0.05)
    assert phase is not None
    node_store = AttemptLogStore(ssh.root / "logs" / "attempts.sqlite3")
    node_store.update_phase(
        attempt, phase, status="survivor", exit_status=130, pid=os.getpgrp()
    )

    blocked = await provisioner._reconcile_host("host1")
    assert blocked is True
    assert store.get(attempt)["survivor"] is True
    history = store.history("host1")
    assert history["attempts"][0]["survivor"] is True
