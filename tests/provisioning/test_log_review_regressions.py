"""Regression coverage for recorder faults and concurrent log retention."""

from __future__ import annotations

import gzip
import json
import runpy
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from inference_proxy.api.admin import download_attempt_logs, stream_provisioning_logs
from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.provisioning import log_store
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.remote_logs import RemoteLogCollector

ROOT = Path(__file__).resolve().parents[2]


def test_engine_sink_batches_and_flushes_before_eof(tmp_path: Path) -> None:
    consumer_script = """
import json, os, runpy, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/inference_proxy/provisioning')
recorder = runpy.run_path(sys.argv[1] + '/common/provision-logs.py')
root = Path(sys.argv[2])
config = dict(root=str(root), max_bytes=4194304, attempt_max_bytes=2097152,
              max_attempts=2, retention_days=7, max_record_bytes=1024,
              attempt_id='b' * 32, engine='vllm', engine_log=str(root / 'engine.log'))
store = recorder['open_store'](config)
store.create('host1', attempt_id=config['attempt_id'])
counts = dict(get=0, commits=0)
original_get, original_append = store.get, store.append_many
def get(*args, **kwargs):
    counts['get'] += 1
    return original_get(*args, **kwargs)
def append(*args, **kwargs):
    result = original_append(*args, **kwargs)
    counts['commits'] += 1
    (root / 'committed').touch()
    return result
store.get, store.append_many = get, append
recorder['engine_sink'].__globals__['open_store'] = lambda config: store
os.environ['QIIP_LOG_CONFIG'] = json.dumps(config)
recorder['engine_sink']()
print(json.dumps(counts))
"""
    with subprocess.Popen(
        [sys.executable, "-c", consumer_script, str(ROOT), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as consumer:
        try:
            assert consumer.stdin is not None
            consumer.stdin.write(b"first line\n")
            consumer.stdin.flush()
            deadline = time.monotonic() + 3
            while not (tmp_path / "committed").exists():
                assert time.monotonic() < deadline, (
                    "no timed flush while pipe stayed open"
                )
                time.sleep(0.01)
            output, errors = consumer.communicate(b"engine output\n" * 2000, timeout=10)
            assert consumer.returncode == 0, errors.decode()
        finally:
            if consumer.poll() is None:
                consumer.kill()
    counts = json.loads(output)
    assert 2 <= counts["commits"] < 20
    assert counts["get"] == counts["commits"]
    store = AttemptLogStore(tmp_path / "attempts.sqlite3")
    assert len(store.read("b" * 32, limit=3000)["records"]) == 2001


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setitem(sys.modules, "log_store", log_store)
    return runpy.run_path(str(ROOT / "common/provision-logs.py"))


@pytest.mark.parametrize(
    "fault",
    ["open", "get", "append_many", "raw", "interrupt", "exit", "sigint", "sigterm"],
)
def test_recorder_fault_keeps_engine_pipe_draining(tmp_path: Path, fault: str) -> None:
    """A real producer with default SIGPIPE must survive its recorder failing."""
    consumer_script = """
import json, os, runpy, signal, sqlite3, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/inference_proxy/provisioning')
recorder = runpy.run_path(sys.argv[1] + '/common/provision-logs.py')
root, fault = Path(sys.argv[2]), sys.argv[3]
config = dict(root=str(root), max_bytes=65536, attempt_max_bytes=16384,
              max_attempts=2, retention_days=7, max_record_bytes=1024,
              attempt_id='a' * 32, engine='vllm', engine_log=str(root / 'engine.log'))
store = recorder['open_store'](config)
store.create('host1', attempt_id=config['attempt_id'])
def fail(*args, **kwargs):
    if fault in ('sigint', 'sigterm'):
        os.kill(os.getpid(), signal.SIGINT if fault == 'sigint' else signal.SIGTERM)
    if fault == 'interrupt':
        raise KeyboardInterrupt()
    if fault == 'exit':
        raise SystemExit('injected exit')
    raise sqlite3.OperationalError('injected recorder storage fault')
if fault == 'open':
    recorder['engine_sink'].__globals__['open_store'] = fail
else:
    if fault in ('get', 'append_many'):
        setattr(store, fault, fail)
    elif fault in ('interrupt', 'exit', 'sigint', 'sigterm'):
        store.append_many = fail
    else:
        (root / 'blocked').write_text('not a directory')
        config['engine_log'] = str(root / 'blocked' / 'engine.log')
    recorder['engine_sink'].__globals__['open_store'] = lambda config: store
os.environ['QIIP_LOG_CONFIG'] = json.dumps(config)
recorder['engine_sink']()
"""
    with subprocess.Popen(
        [sys.executable, "-c", consumer_script, str(ROOT), str(tmp_path), fault],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as consumer:
        try:
            producer = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os, signal; signal.signal(signal.SIGPIPE, signal.SIG_DFL); "
                    "[os.write(1, b'engine output\\n' * 1024) for _ in range(128)]",
                ],
                stdout=consumer.stdin,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            assert consumer.stdin is not None
            consumer.stdin.close()
            consumer.wait(timeout=10)
            assert producer.returncode == 0, "logging failure killed the engine"
            assert consumer.returncode == 0
        finally:
            if consumer.poll() is None:
                consumer.kill()
    if fault != "open":
        manifest = AttemptLogStore(tmp_path / "attempts.sqlite3").get("a" * 32)
        assert manifest["sources"]["engine"] == "unavailable"
        assert any(
            "Engine log recorder failed" in issue for issue in manifest["issues"]
        )


def test_running_attempts_do_not_consume_completed_retention(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3", max_attempts=2)
    completed = []
    for _ in range(3):
        attempt = store.create("host1")
        store.update(attempt, status="complete")
        completed.append(attempt)
    running = [store.create("host1") for _ in range(3)]
    history = store.history("host1")
    assert {a["attempt_id"] for a in history["attempts"]} == set(
        completed[-2:] + running
    )
    assert history["evicted_attempts"] == 1
    store.retention_days = -1
    assert {a["attempt_id"] for a in store.history("host1")["attempts"]} == set(running)


def test_history_survives_eviction_after_its_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    store.update(attempt, status="complete")
    other_writer = AttemptLogStore(store.path)
    other_writer.retention_days = -1
    original_db = store._db

    @contextmanager
    def evict_after_read() -> Iterator[Any]:
        read_snapshot = False

        def trace(statement: str) -> None:
            nonlocal read_snapshot
            read_snapshot |= "SELECT * FROM attempts WHERE hostname" in statement

        with original_db() as db:
            db.set_trace_callback(trace)
            yield db
        if read_snapshot:
            other_writer.history("host1")

    monkeypatch.setattr(store, "_db", evict_after_read)
    result = store.history("host1")
    assert result["total"] == 1
    assert result["attempts"][0]["attempt_id"] == attempt
    assert other_writer.history("host1")["total"] == 0


@pytest.mark.parametrize("missing", [1, 4])
@pytest.mark.asyncio
async def test_unavailable_suffix_marker_does_not_overstate_gap(
    tmp_path: Path, missing: int
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    store.append(attempt, "received", remote_seq=0)
    collector = RemoteLogCollector(
        MagicMock(), store, ProvisioningLogBuffer(store=store), ProvisioningSettings()
    )
    remote = dict(store.get(attempt), next_seq=missing + 1, dropped_records=missing)
    page = dict(records=[], attempt=remote, has_more=False)
    await collector._ingest(attempt, page)
    records = store.read(attempt)["records"]
    marker = records[-1]
    assert marker["remote_seq"] == missing
    assert f"1..{missing}" not in marker["msg"]
    assert "unavailable" in marker["msg"]
    assert store.get(attempt)["remote_cursor"] == missing + 1
    if missing > 1:
        assert any(
            f"1..{missing - 1}" in issue for issue in store.get(attempt)["issues"]
        )
    await collector._ingest(attempt, page)
    assert store.read(attempt)["records"] == records


def test_raw_retention_tolerates_concurrent_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: dict[str, Any]
) -> None:
    tail = tmp_path / ("a" * 32 + ".engine.log")
    tail.touch()
    original_stat = Path.stat

    def removed_before_stat(path: Path, **kwargs: Any) -> Any:
        if path == tail:
            tail.unlink(missing_ok=True)
        return original_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", removed_before_stat)
    recorder["prune_raw_logs"](
        dict(root=str(tmp_path), max_attempts=1, retention_days=7), MagicMock()
    )


def test_timeout_remains_124_when_process_group_already_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: dict[str, Any]
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    command = MagicMock(returncode=0, pid=12345)
    command.poll.return_value = None
    journal = MagicMock(returncode=0)
    journal.poll.return_value = 0
    selector = MagicMock()
    selector.select.return_value = []
    selector.get_map.return_value = {}
    worker_globals = recorder["worker"].__globals__
    monkeypatch.setitem(worker_globals, "open_store", lambda config: store)
    monkeypatch.setitem(
        worker_globals,
        "selectors",
        SimpleNamespace(DefaultSelector=lambda: selector, EVENT_READ=1),
    )
    monkeypatch.setitem(
        worker_globals,
        "subprocess",
        SimpleNamespace(
            Popen=MagicMock(side_effect=[command, journal]),
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    killpg = MagicMock(side_effect=ProcessLookupError)
    monkeypatch.setitem(
        worker_globals,
        "os",
        SimpleNamespace(environ={}, set_blocking=lambda *args: None, killpg=killpg),
    )
    recorder["worker"](
        dict(
            attempt_id=attempt,
            phase="setup",
            stage="setup",
            timeout=-1,
            health_timeout=0,
            inactivity_timeout=10,
            command="already exited",
        )
    )
    manifest = store.get(attempt)
    assert manifest["phases"]["setup"]["exit_status"] == 124, manifest["issues"]
    assert manifest["phases"]["setup"]["recording"] is False
    assert any("deadline" in issue for issue in manifest["issues"])
    assert not any("Node recorder failed" in issue for issue in manifest["issues"])
    sigs = [c.args[1] for c in killpg.call_args_list]
    assert signal.SIGTERM in sigs
    assert signal.SIGKILL in sigs  # deadline SIGTERM, then final group SIGKILL.


def test_phase_updates_and_appends_preserve_concurrent_metadata(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")

    def write_phase(index: int) -> None:
        store.update_phase(attempt, "setup", **{f"field_{index}": index})
        store.append(attempt, str(index))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write_phase, range(12)))
    assert store.get(attempt)["phases"]["setup"] == {f"field_{i}": i for i in range(12)}
    assert store.get(attempt)["next_seq"] == 12
    with pytest.raises(KeyError):
        store.update_phase("missing", "setup", status="running")


def test_issue_growth_and_journal_replay_are_bounded(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    for i in range(40):
        store.issue(attempt, f"failure-{i}")
    store.issue(attempt, "failure-39", source="journal")
    assert store.get(attempt)["issues"] == [f"failure-{i}" for i in range(8, 40)]
    assert store.get(attempt)["sources"]["journal"] == "unavailable"
    store.issue(attempt, "x" * 3000)
    assert len(store.get(attempt)["issues"][-1]) == 2048
    store.append(attempt, "journal message", source="journal", journal_cursor="cursor1")
    assert store.append(attempt, "duplicate", journal_cursor="cursor1") is None
    store.append(attempt, "new message", journal_cursor="cursor2")
    assert store.get(attempt)["next_seq"] == 2
    with pytest.raises(KeyError):
        store.issue("missing", "unavailable")


def test_byte_accounting_survives_rotation_eviction_and_upgrade(tmp_path: Path) -> None:
    store = AttemptLogStore(
        tmp_path / "logs.sqlite3",
        max_bytes=1800,
        attempt_max_bytes=1200,
        max_attempts=2,
    )
    for _ in range(3):
        attempt = store.create("host1")
        for seq in range(5):
            store.append(attempt, f"line-{seq}")
        store.update(attempt, status="complete")
    history = store.history("host1")
    expected = sum(a["retained_bytes"] for a in history["attempts"])
    assert 0 < expected <= store.max_bytes
    assert history["evicted_attempts"] == 1
    with store._db() as db:
        assert (
            db.execute(
                "SELECT value FROM statistics WHERE name='retained_bytes'"
            ).fetchone()[0]
            == expected
        )
        # Model an existing database from before the accounting migration.
        db.execute("DROP TRIGGER record_bytes_insert")
        db.execute("DROP TRIGGER record_bytes_delete")
        db.execute("DELETE FROM statistics WHERE name='retained_bytes'")
    upgraded = AttemptLogStore(store.path)
    with upgraded._db() as db:
        assert (
            db.execute(
                "SELECT value FROM statistics WHERE name='retained_bytes'"
            ).fetchone()[0]
            == expected
        )
    upgraded.retention_days = -1
    assert upgraded.history("host1")["total"] == 0
    with upgraded._db() as db:
        assert (
            db.execute(
                "SELECT value FROM statistics WHERE name='retained_bytes'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_stream_rejects_mismatched_resume_and_reports_eviction(
    tmp_path: Path,
) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    store.append(attempt, "evidence")
    provisioner = MagicMock(log_buffer=ProvisioningLogBuffer(store=store))
    with pytest.raises(HTTPException) as error:
        await stream_provisioning_logs("host1", attempt, 0, "different:0", provisioner)
    assert error.value.status_code == 400
    response = await stream_provisioning_logs("host1", attempt, 0, None, provisioner)
    stream = aiter(response.body_iterator)
    assert "evidence" in str(await anext(stream))
    store.update(attempt, status="complete")
    store.retention_days = -1
    store.history("host1")
    remaining = "".join([str(chunk) async for chunk in stream])
    assert "event: unavailable" in remaining


@pytest.mark.asyncio
async def test_bundle_reports_eviction_during_download(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "logs.sqlite3")
    attempt = store.create("host1")
    store.append(attempt, "evidence")
    store.update(attempt, status="complete")
    provisioner = MagicMock(log_buffer=ProvisioningLogBuffer(store=store))
    response = await download_attempt_logs("host1", attempt, provisioner)
    store.retention_days = -1
    store.history("host1")
    chunks = [chunk async for chunk in response.body_iterator]
    data = b"".join(c.encode() if isinstance(c, str) else bytes(c) for c in chunks)
    footer = json.loads(gzip.decompress(data).splitlines()[-1])["export"]
    assert footer["incomplete"] is True
    assert footer["expected_records"] == 1
    assert footer["exported_records"] == 0


@pytest.mark.asyncio
async def test_stream_in_memory_fallback_resumes_from_offset() -> None:
    buffer = ProvisioningLogBuffer()
    buffer.create("host1")
    buffer.append("host1", "info", "earlier")
    buffer.append("host1", "info", "resumed")
    buffer.mark_complete("host1")
    response = await stream_provisioning_logs(
        "host1", None, 1, None, MagicMock(log_buffer=buffer)
    )
    content = "".join([str(chunk) async for chunk in response.body_iterator])
    assert "resumed" in content
    assert "earlier" not in content
