"""Failure boundaries found during the second provisioning-log review."""

from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_proxy.config.settings import ProvisioningSettings, SSHSettings
from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.provisioning.ssh_client import SSHClient, SSHConnectionError
from tests.provisioning.test_attempt_logs import LocalNodeSSH
from tests.provisioning.test_attempt_logs import harness as harness


@pytest.mark.asyncio
async def test_shutdown_leaves_detached_command_running(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    collector = provisioner._remote_logs
    assert collector is not None

    async def provision() -> None:
        provisioner._begin_log("host1", InferenceEngine.VLLM)
        try:
            async for _ in collector.run(
                "host1", "touch started; sleep .5; touch survived", stage="start"
            ):
                pass
        finally:
            await provisioner._finish_remote_logs("host1")

    task = provisioner.fire_background(provision())
    async with asyncio.timeout(3):
        while not (ssh.root / "started").exists():
            await asyncio.sleep(0.01)
    await provisioner.shutdown()
    assert task.cancelled()
    await asyncio.sleep(0.7)
    assert (ssh.root / "survived").exists()
    attempt = store.history("host1")["attempts"][0]["attempt_id"]
    node_store = AttemptLogStore(ssh.root / "logs/attempts.sqlite3")
    assert not node_store.get(attempt).get("cancel_command")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [sqlite3.OperationalError("locked"), KeyError("evicted"), ValueError("invalid")],
)
async def test_finish_logging_does_not_raise_storage_errors(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    provisioner, _ssh, store = harness
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    monkeypatch.setattr(store, "get", MagicMock(side_effect=error))
    await provisioner._finish_remote_logs("host1")
    await provisioner._finish_remote_logs("missing")


@pytest.mark.parametrize("suffix", ["logs", "collect", "bundle"])
def test_attempt_eviction_at_ownership_boundary(
    client: TestClient,
    mock_provisioner: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    store.append(attempt, "evidence")
    store.update(attempt, status="complete")
    peer = AttemptLogStore(store.path)
    peer.retention_days = -1
    original_get = store.get

    def get_then_evict(attempt_id: str) -> dict[str, object]:
        result = original_get(attempt_id)
        peer.history("host1")
        return result

    monkeypatch.setattr(store, "get", get_then_evict)
    mock_provisioner.log_buffer = ProvisioningLogBuffer(store=store)
    mock_provisioner.collect_logs = AsyncMock(
        side_effect=lambda *args: store.get(attempt)
    )
    url = f"/admin/provisioning/host1/attempts/{attempt}/{suffix}"
    result = client.post(url, json={}) if suffix == "collect" else client.get(url)
    if suffix == "bundle" and result.status_code == 200:
        assert json.loads(gzip.decompress(result.content).splitlines()[-1])["export"][
            "incomplete"
        ]
    else:
        assert result.status_code == 404


def test_collect_passes_canonical_hostname(
    client: TestClient,
    mock_provisioner: MagicMock,
    tmp_path: Path,
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("node.example.com")
    mock_provisioner.log_buffer = ProvisioningLogBuffer(store=store)

    async def collect(hostname: str, attempt_id: str) -> dict[str, object]:
        if hostname != "node.example.com":
            raise KeyError(hostname)
        return store.get(attempt_id)

    mock_provisioner.collect_logs = AsyncMock(side_effect=collect)
    response = client.post(
        f"/admin/provisioning/NODE.EXAMPLE.COM./attempts/{attempt}/collect", json={}
    )
    assert response.status_code == 200
    mock_provisioner.collect_logs.assert_awaited_once_with("node.example.com", attempt)


def test_batch_commit_rolls_back_records_and_cursor_on_failure(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    with pytest.raises(AttributeError):
        store.append_many(attempt, [dict(msg="first", remote_seq=0), dict(msg=None)])
    assert store.read(attempt)["records"] == []
    assert store.get(attempt)["remote_cursor"] == 0
    assert store.get(attempt)["next_seq"] == 0


def test_batch_replay_and_rotation_use_one_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3", attempt_max_bytes=1500)
    attempt = store.create("host1")
    original_db = store._db
    statements: list[str] = []

    @contextmanager
    def traced_db() -> Iterator[sqlite3.Connection]:
        with original_db() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "_db", traced_db)
    batch = [dict(msg=f"line {i}", remote_seq=i) for i in range(100)]
    assert len(store.append_many(attempt, batch)) == 100
    assert statements.count("COMMIT") == 1
    assert store.append_many(attempt, batch) == []
    manifest = store.get(attempt)
    assert manifest["remote_cursor"] == manifest["next_seq"] == 100
    assert 0 < manifest["retained_bytes"] <= 1500
    assert manifest["dropped_records"] > 0


def test_wal_reader_does_not_block_writer(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    with store._db() as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM records").fetchone()[0] == 0
        store.append(attempt, "committed during reader snapshot")
        assert reader.execute("SELECT count(*) FROM records").fetchone()[0] == 0
    assert len(store.read(attempt)["records"]) == 1


def test_history_without_expiry_does_not_take_writer_lock(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    store.create("host1")
    with store._db() as writer:
        writer.execute("BEGIN IMMEDIATE")
        assert store.history("host1")["total"] == 1


def test_remote_metadata_preserves_concurrent_gateway_warning(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    stale = store.get(attempt)["issues"]
    store.issue(attempt, "gateway warning")
    store.update(attempt, issues=stale + ["Node: remote warning"])
    assert store.get(attempt)["issues"] == ["gateway warning", "Node: remote warning"]


@pytest.mark.asyncio
async def test_page_ingestion_keeps_event_loop_responsive(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, _ssh, store = harness
    collector = provisioner._remote_logs
    assert collector is not None
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    attempt = provisioner.log_buffer.attempts["host1"]
    page = dict(records=[], attempt=store.get(attempt), has_more=False)
    entered = threading.Event()
    released = threading.Event()
    original = store.append_many
    completed_without_blocking = False

    def slow_append(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert released.wait(2), "ingestion blocked the event loop"
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "append_many", slow_append)
    task = asyncio.create_task(collector._ingest(attempt, page))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        completed_without_blocking = not released.is_set()
    finally:
        released.set()
        await task
    assert completed_without_blocking


@pytest.mark.asyncio
async def test_scoped_ssh_connection_reuses_and_resets_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    connection.run = AsyncMock(
        return_value=MagicMock(stdout="ok", stderr="", exit_status=0)
    )
    context = AsyncMock()
    context.__aenter__.return_value = connection
    connect = MagicMock(return_value=context)
    monkeypatch.setattr(
        "inference_proxy.provisioning.ssh_client.asyncssh.connect", connect
    )
    ssh = SSHClient(SSHSettings())
    with pytest.raises(SSHConnectionError):
        async with ssh.connection("host1"):
            await ssh.run("host1", "read one")
            await ssh.run("host1", "read two")
            assert connect.call_count == 1
            raise OSError("disconnected")
    await ssh.run("host1", "read after reconnect")
    assert connect.call_count == 2
    assert context.__aexit__.await_count == 2


@pytest.mark.asyncio
async def test_follower_reconnects_and_reuses_connection_between_polls(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, ssh, _store = harness
    collector = provisioner._remote_logs
    assert collector is not None
    provisioner._begin_log("host1", InferenceEngine.VLLM)
    collect = AsyncMock(
        side_effect=[
            [],
            SSHConnectionError("host1", "lost"),
            [],
            asyncio.CancelledError(),
        ]
    )
    monkeypatch.setattr(collector, "collect", collect)
    with pytest.raises(asyncio.CancelledError):
        await collector.follow(provisioner.log_buffer.attempts["host1"])
    assert collect.await_count == 4
    assert ssh.connections == 2


@pytest.mark.parametrize("root", ["/", "//", "///", "relative", "/var/../"])
def test_log_root_requires_dedicated_directory(root: str) -> None:
    with pytest.raises(ValidationError):
        ProvisioningSettings(log_remote_root=root)


def test_log_budgets_cover_one_attempt() -> None:
    with pytest.raises(ValidationError, match="log_storage_max_bytes"):
        ProvisioningSettings(log_storage_max_bytes=65536, log_attempt_max_bytes=65537)
    with pytest.raises(ValidationError, match="log_remote_max_bytes"):
        ProvisioningSettings(
            log_remote_max_bytes=65536, log_remote_attempt_max_bytes=65536
        )
    assert ProvisioningSettings(
        log_storage_max_bytes=65536,
        log_attempt_max_bytes=65536,
        log_remote_max_bytes=65536,
        log_remote_attempt_max_bytes=32768,
    )
