"""Controlled node/gateway harness for durable provisioning evidence.

The transport runs the uploaded recorder in a temporary node directory. It can
lose an acknowledgement or reads while the real detached worker keeps running.
No SSH/GPU/root services or external network is required.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from inference_proxy.config.settings import (
    LLMFitSettings,
    ProvisioningSettings,
    SSHSettings,
)
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningIdentity,
)
from inference_proxy.provisioning.remote_logs import RemoteLogCollector
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)
from tests.provisioning.test_vllm_scripts import _script_environment, _write_executable

ROOT = Path(__file__).resolve().parents[2]


class LocalNodeSSH(SSHClient):
    def __init__(self, root: Path, environment: dict[str, str]) -> None:
        super().__init__(SSHSettings(streaming_command_timeout=5))
        self.root = root
        self.environment = environment
        self.lose_launch = False
        self.fail_reads = 0
        self.launches = 0
        self.read_faults_injected = 0
        self.launch_faults_injected = 0
        self.connections = 0

    @asynccontextmanager
    async def connection(self, host: str) -> AsyncIterator[None]:
        self.connections += 1
        yield

    async def run(
        self,
        host: str,
        command: str,
        timeout: float = 60,
        *,
        log_label: str | None = None,
    ) -> tuple[str, str, int]:
        if log_label == "provisioning log recorder (read)" and self.fail_reads:
            self.fail_reads -= 1
            self.read_faults_injected += 1
            raise SSHConnectionError(host, "controlled stream interruption")
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=self.root,
            env=self.environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        if proc.returncode:
            raise RemoteCommandError(
                host, "controlled recorder", proc.returncode, stderr.decode()
            )
        if log_label == "provisioning log recorder (launch)":
            self.launches += 1
            if self.lose_launch:
                self.lose_launch = False
                self.launch_faults_injected += 1
                raise SSHConnectionError(host, "controlled lost launch acknowledgement")
        return stdout.decode(), stderr.decode(), 0


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
    # The node bundle runs without a real NFS mount; a mounts-file fixture gives
    # the strict source/options verification the same evidence a mounted host
    # provides without touching the host's /proc/mounts.
    mounts_file = tmp_path / "mounts"
    mount_point = environment["AUTOVLLM_NFS_MOUNT_POINT"]
    mounts_file.write_text(
        f"fixture:/cache {mount_point} nfs "
        "rw,vers=3,hard,proto=tcp,timeo=600,retrans=3,sec=sys 0 0\n"
    )
    environment["AUTOVLLM_MOUNTS_FILE"] = str(mounts_file)
    # A deterministic relevant service-journal source, independent of systemd.
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
        min_disk_gb=1,
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


def test_store_restart_rotation_retries_and_atomic_cursor(tmp_path: Path) -> None:
    path = tmp_path / "logs.sqlite3"
    store = AttemptLogStore(path, attempt_max_bytes=1500, max_record_bytes=100)
    first = store.create(
        "host1", engine="vllm", model="org/model", bundle_version="sha256:abc"
    )
    for seq in range(10):
        store.append(first, f"line {seq}", source="setup.stderr", remote_seq=seq)
    store.update(first, status="failed", failure_summary="setup: driver install failed")
    reopened = AttemptLogStore(path, attempt_max_bytes=1500, max_record_bytes=100)
    assert reopened.get(first)["remote_cursor"] == 10
    assert reopened.append(first, "duplicate", remote_seq=0) is None
    result = reopened.read(first)
    assert result["attempt"]["incomplete"]
    assert result["attempt"]["dropped_records"] > 0
    assert all(r["bundle_version"] == "sha256:abc" for r in result["records"])
    second = reopened.create("host1")
    assert second != first
    assert len(reopened.history("host1")["attempts"]) == 2
    reopened.append(second, "x" * 500)
    assert reopened.read(second)["records"][0]["truncated"]
    reopened.interrupt_running()
    assert reopened.get(second)["status"] == "interrupted"
    assert reopened.get(first)["failure_summary"] == "setup: driver install failed"


def test_global_budget_manifest_eviction_and_literal_search(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3", max_bytes=1600, max_attempts=2)
    attempts = []
    for _ in range(3):
        attempt = store.create("host1")
        attempts.append(attempt)
        for seq in range(4):
            store.append(attempt, f"line {seq} 100%_literal")
        store.update(attempt, status="complete")
    result = store.history("host1")
    assert result["evicted_attempts"] == 1
    assert result["total"] == 2
    assert sum(a["retained_bytes"] for a in result["attempts"]) <= 1600
    assert store.read(attempts[-1], query="%_LITERAL")["records"]
    assert not store.read(attempts[-1], query="missing")["records"]
    store.retention_days = 0
    assert store.history("host1")["total"] == 0


@pytest.mark.asyncio
async def test_setup_boundary_recovers_lost_ack_and_interruption(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    # Use the shipped setup entrypoint and real step() boundary, replacing only
    # privileged installation functions with deterministic fixture operations.
    setup = ssh.root / "auto-vllm/setup.sh"
    text = setup.read_text()
    text, injected = re.subn(
        r"(?m)^    step system_update run_system_update$",
        """    run_system_update() { echo "setup stdout"; echo "setup stderr" >&2; sleep 0.1; }
    install_nvidia_driver() { :; }
    install_cuda_toolkit() { :; }
    ensure_fabric_manager() { :; }
    install_vllm() { :; }
    install_vllm_unit() { :; }
    mount_nfs_cache() { :; }
    configure_firewall() { :; }
    install_llmfit() { :; }
    step system_update run_system_update""",
        text,
    )
    assert injected == 1, "setup fixture boundary changed; refusing to run installers"
    assert "install_nvidia_driver() { :; }" in text
    setup.write_text(text)
    provisioner._begin_log("host1", InferenceEngine.VLLM, model="org/model")
    attempt = provisioner.log_buffer.attempts["host1"]
    ssh.lose_launch = True
    ssh.fail_reads = 1
    steps: list[str] = []
    await provisioner._run_setup(
        "host1", started_at=datetime.now(UTC), on_step=steps.append
    )
    await provisioner._finish_remote_logs("host1")
    page = store.read(attempt)
    messages = [r["msg"] for r in page["records"]]
    assert messages.count("setup stdout") == 1
    assert messages.count("setup stderr") == 1
    assert "system_update" in steps
    assert any(r["source"] == "journal" for r in page["records"])
    assert ssh.launches == 1
    assert ssh.launch_faults_injected == 1
    assert ssh.read_faults_injected == 1
    before = len(page["records"])
    await provisioner.collect_logs("host1", attempt)
    assert len(store.read(attempt)["records"]) == before


@pytest.mark.asyncio
async def test_real_engine_launch_output_survives_gateway_restart(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM, model="org/model")
    attempt = provisioner.log_buffer.attempts["host1"]
    model = await provisioner._run_start_vllm("host1", model="org/model")
    assert model == "org/model"
    reopened = AttemptLogStore(store.path)
    reopened.interrupt_running()
    collector = RemoteLogCollector(
        ssh, reopened, ProvisioningLogBuffer(store=reopened), provisioner._settings
    )
    await collector.collect(attempt, finish=True)
    page = reopened.read(attempt)
    engine = [r["msg"] for r in page["records"] if r["source"] == "engine"]
    assert "engine boot evidence" in engine
    assert "engine stderr evidence" in engine
    assert page["attempt"]["status"] == "interrupted"
    assert ssh.launches == 1


@pytest.mark.asyncio
async def test_restart_during_command_recovers_unseen_output(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    collector = provisioner._remote_logs
    assert collector is not None
    output = collector.run(
        "host1", "echo before; sleep .2; echo after >&2", stage="setup"
    )
    assert await anext(output) == ("stdout", "before")
    await output.aclose()  # gateway disappears without telling the node to stop
    reopened = AttemptLogStore(store.path)
    reopened.interrupt_running()
    replacement = RemoteLogCollector(
        ssh, reopened, ProvisioningLogBuffer(store=reopened), provisioner._settings
    )
    await asyncio.sleep(0.3)
    await replacement.collect(attempt, finish=True)
    records = reopened.read(attempt)["records"]
    assert len([r for r in records if r["msg"] == "before"]) == 1
    assert len([r for r in records if r["msg"] == "after"]) == 1


@pytest.mark.asyncio
async def test_missing_remote_log_is_explicit(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, _ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    result = await provisioner.collect_logs("host1", attempt)
    assert result["incomplete"] is True
    assert store.get(attempt)["sources"]["remote"] == "unavailable"


def test_admin_history_search_bundle_resume_and_host_isolation(
    client: TestClient, mock_provisioner: MagicMock, tmp_path: Path
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    buffer = ProvisioningLogBuffer(store=store)
    buffer.create("host1", engine="vllm", model="org/model")
    buffer.append("host1", "error", "driver FAILURE")
    buffer.append("host1", "info", "next")
    buffer.mark_complete("host1")
    attempt = buffer.attempts["host1"]
    mock_provisioner.log_buffer = buffer
    mock_provisioner.collect_logs = AsyncMock(return_value=store.get(attempt))
    base = f"/admin/provisioning/host1/attempts/{attempt}"
    assert client.get("/admin/provisioning/host1/attempts").json()["total"] == 1
    assert len(client.get(base + "/logs?q=failure").json()["records"]) == 1
    assert client.get(base + "/logs?after=-1").status_code == 422
    assert client.post(base + "/collect", json={}).status_code == 200
    bundle = client.get(base + "/bundle")
    lines = gzip.decompress(bundle.content).decode().splitlines()
    assert json.loads(lines[0])["manifest"]["attempt_id"] == attempt
    assert json.loads(lines[1])["msg"] == "driver FAILURE"
    resumed = client.get(
        "/admin/provisioning/host1/logs", headers={"Last-Event-ID": f"{attempt}:0"}
    )
    assert "driver FAILURE" not in resumed.text
    assert "next" in resumed.text
    assert "event: complete" in resumed.text
    assert client.get(base.replace("host1", "host2") + "/logs").status_code == 404
    assert (
        client.get(
            "/admin/provisioning/host1/logs", headers={"Last-Event-ID": "invalid"}
        ).status_code
        == 400
    )


@pytest.mark.asyncio
async def test_remote_rotation_and_repeated_attempts_are_explicit(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, _ssh, store = harness
    provisioner._settings.log_remote_attempt_max_bytes = 16_384
    collector = provisioner._remote_logs
    assert collector is not None
    ids = []
    for _ in range(2):
        provisioner._begin_log("host1", InferenceEngine.VLLM)
        attempt = provisioner.log_buffer.attempts["host1"]
        ids.append(attempt)
        # Deliberately do not retrieve while this command exceeds node retention.
        await collector._request(
            attempt,
            "launch",
            phase="rotation",
            stage="setup",
            command="for i in {1..100}; do printf '%s %0900d\\n' \"$i\" 0; done",
            timeout=5,
        )
        await asyncio.sleep(0.4)
        await collector.collect(attempt, finish=True)
        manifest = store.get(attempt)
        assert manifest["remote_dropped_records"] > 0
        assert manifest["incomplete"] is True
        assert any("Remote records 0.." in issue for issue in manifest["issues"])
        before = store.read(attempt)["records"]
        await collector.collect(attempt)
        assert store.read(attempt)["records"] == before
        provisioner.log_buffer.mark_complete("host1")
    assert ids[0] != ids[1]
    assert store.history("host1")["total"] == 2


@pytest.mark.asyncio
async def test_failed_command_summary_and_unavailable_journal(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    _write_executable(
        Path(ssh.environment["PATH"].split(":")[0]) / "journalctl",
        "#!/bin/bash\necho 'journal access denied' >&2\ntouch journal-failed\nexit 3\n",
    )
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    collector = provisioner._remote_logs
    assert collector is not None
    with pytest.raises(RemoteCommandError, match="driver rejected"):
        async for _ in collector.run(
            "host1",
            "while [ ! -f journal-failed ]; do sleep .01; done; echo 'driver rejected' >&2; exit 7",
            stage="setup",
        ):
            pass
    assert store.get(attempt)["remote_sources"]["journal"] == "unavailable"
    assert store.get(attempt)["incomplete"]


@pytest.mark.asyncio
async def test_engine_sink_rotation_bounds_raw_and_durable_output(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._settings.log_remote_max_bytes = 65536
    provisioner._settings.log_remote_max_attempts = 1
    provisioner._settings.log_remote_attempt_max_bytes = 16384
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    collector = provisioner._remote_logs
    assert collector is not None
    raw_path = ssh.root / "logs" / f"{attempt}.engine.log"
    # Exercise exactly the pipe consumer invoked by the shipped launch scripts.
    async for _ in collector.run(
        "host1",
        "for i in {1..80}; do printf '%s %0900d\\n' \"$i\" 0; done | python3 common/provision-logs.py engine",
        stage="start",
        timeout=5,
        engine_log=str(raw_path),
    ):
        pass  # Wait for the pipe consumer to commit and the command to exit.
    await collector.collect(attempt, finish=True)
    assert raw_path.stat().st_size <= 16384
    assert store.get(attempt)["remote_dropped_records"] > 0
    assert any(
        "Raw engine log rotated" in issue for issue in store.get(attempt)["issues"]
    )


@pytest.mark.asyncio
async def test_gateway_parser_sees_records_collected_by_another_reader(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, _ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    collector = provisioner._remote_logs
    assert collector is not None
    attempt = provisioner.log_buffer.attempts["host1"]
    output = collector.run("host1", "echo first; sleep .15; echo second", stage="setup")
    assert await anext(output) == ("stdout", "first")
    await asyncio.sleep(0.2)
    await collector.collect(attempt)
    remaining = [entry async for entry in output]
    assert ("stdout", "second") in remaining
    assert len([r for r in store.read(attempt)["records"] if r["msg"] == "second"]) == 1


@pytest.mark.asyncio
async def test_full_attempt_failure_persists_stage_and_summary(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, ssh, store = harness
    monkeypatch.setattr(provisioner, "preflight", AsyncMock())
    monkeypatch.setattr(provisioner, "_upload_scripts", AsyncMock())
    # The transport and recorder remain real; no privileged host mutation.
    (ssh.root / "auto-vllm/setup.sh").write_text(
        "echo '[STEP:nvidia_driver:START]'\n"
        "echo 'driver rejected by kernel' >&2\n"
        "echo '[STEP:nvidia_driver:FAIL]'\nexit 7\n"
    )
    with pytest.raises(RemoteCommandError, match="driver rejected by kernel"):
        await provisioner.provision("host1", model="org/model")
    attempt = store.history("host1")["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["finished_at"]
    assert "nvidia_driver" in attempt["failure_summary"]
    assert "driver rejected by kernel" in attempt["failure_summary"]
    records = store.read(attempt["attempt_id"])["records"]
    assert any(
        r["stage"] == "nvidia_driver" and r["source"] == "setup.stdout" for r in records
    )


def test_admin_paging_rotation_export_and_authentication(
    client: TestClient,
    mock_provisioner: MagicMock,
    tmp_path: Path,
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3", attempt_max_bytes=2000)
    buffer = ProvisioningLogBuffer(store=store)
    buffer.create("host1")
    for i in range(12):
        buffer.append("host1", "info", f"line {i}")
    buffer.mark_complete("host1")
    mock_provisioner.log_buffer = buffer
    attempt = buffer.attempts["host1"]
    base = f"/admin/provisioning/host1/attempts/{attempt}"
    first = client.get(base + "/logs?limit=1").json()
    assert first["has_more"]
    second = client.get(base + f"/logs?limit=1&after={first['next_offset']}").json()
    assert second["records"][0]["seq"] > first["records"][0]["seq"]
    stream = client.get("/admin/provisioning/host1/logs")
    assert "earlier log" in stream.text
    bundle = client.get(base + "/bundle")
    footer = json.loads(gzip.decompress(bundle.content).decode().splitlines()[-1])
    assert footer["export"]["incomplete"] is False
    for suffix in ("/logs", "/bundle"):
        assert (
            client.get(base + suffix, headers={"Authorization": ""}).status_code == 401
        )
    assert client.get(base + "/logs?source=missing").json()["records"] == []
    assert client.get(base.replace(attempt, "missing") + "/logs").status_code == 404
    mock_provisioner.log_buffer = ProvisioningLogBuffer()
    assert client.get("/admin/provisioning/host1/attempts").status_code == 503


@pytest.mark.asyncio
async def test_exhausted_reconnect_and_empty_rotated_remote_suffix(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    collector = provisioner._remote_logs
    assert collector is not None
    ssh.fail_reads = 20
    with pytest.raises(SSHConnectionError, match="controlled stream interruption"):
        async for _ in collector.run("host1", "sleep .2; echo retained", stage="setup"):
            pass
    assert store.get(attempt)["incomplete"]
    ssh.fail_reads = 0
    await asyncio.sleep(0.3)
    # All payloads can expire while the durable remote high-water mark survives.
    node_store = AttemptLogStore(ssh.root / "logs/attempts.sqlite3")
    with node_store._db() as db:
        node_store._rotate(db, attempt, 0)
    await collector.collect(attempt, finish=True)
    assert store.get(attempt)["remote_cursor"] == node_store.get(attempt)["next_seq"]
    assert any("unavailable suffix" in issue for issue in store.get(attempt)["issues"])


@pytest.mark.asyncio
async def test_explicit_gateway_cancellation_stops_detached_setup(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, ssh, store = harness
    monkeypatch.setattr(provisioner, "preflight", AsyncMock())
    monkeypatch.setattr(provisioner, "_upload_scripts", AsyncMock())
    (ssh.root / "auto-vllm/setup.sh").write_text(
        "echo '[STEP:nvidia_driver:START]'\n"
        "echo before > before-cancel\nsleep 1\necho unsafe > after-cancel\n"
    )
    task = provisioner.fire_background(
        provisioner.provision("host1", model="org/model"),
        provisioning_hostname="host1",
        provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
    )
    async with asyncio.timeout(3):
        while not (ssh.root / "before-cancel").exists():
            await asyncio.sleep(0.02)
    await provisioner.cancel_active_provision("host1")
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.1)
    assert not (ssh.root / "after-cancel").exists()
    attempt = store.history("host1")["attempts"][0]
    assert attempt["status"] == "failed"
    assert "cancelled" in attempt["failure_summary"]
    assert any("Remote command cancelled" in issue for issue in attempt["issues"])
