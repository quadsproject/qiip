#!/usr/bin/env python3
"""Node recorder. Standard library only; launched with the uploaded log_store.py.

Workers have no SSH-owned pipes. A disconnect stops retrieval, never recording.
Commands are sent to the worker's stdin and are never written into the log DB.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from log_store import AttemptLogStore, timestamp


def open_store(config):
    return AttemptLogStore(
        Path(config["root"]) / "attempts.sqlite3",
        max_bytes=config["max_bytes"] // 2,
        attempt_max_bytes=config["attempt_max_bytes"],
        max_attempts=config["max_attempts"],
        retention_days=config["retention_days"],
        max_record_bytes=config["max_record_bytes"],
    )


def prune_raw_logs(config, store):
    root = Path(config["root"])
    paths = []
    for path in root.glob("*.engine.log"):
        if re.fullmatch(r"[a-f0-9]{32}\.engine\.log", path.name):
            with suppress(FileNotFoundError):
                paths.append((path.stat().st_mtime, path))
    paths.sort(key=lambda item: item[0], reverse=True)
    cutoff = time.time() - config["retention_days"] * 86400
    for index, (modified, path) in enumerate(paths):
        if index >= config["max_attempts"] - 1 or modified < cutoff:
            with suppress(KeyError):
                store.issue(
                    path.name.split(".")[0], "Raw engine tail evicted by node retention"
                )
            path.unlink(missing_ok=True)


def _acquire_host_lock(root: str | Path | None) -> tuple[int | None, bool]:
    """Take the host-scoped mutation lock.

    Returns ``(fd, True)`` on success, ``(None, True)`` when another worker
    holds the lock, and ``(None, False)`` when locking is unavailable (no
    root, read-only root, or recorder running without a writable filesystem).
    One lock path is shared by both engine setup paths: workers with a live
    mutating command must never run concurrently on one node. The kernel
    releases the lock when the holder exits, so a crashed worker cannot wedge
    the host forever; a surviving orphan is still caught by the phase probe
    and blocks a retry. Unavailable locking disables fencing but never fails
    the recorder, preserving behavior for configs without a remote log root.
    """
    try:
        if root is None:
            return None, False
        lock_path = Path(root) / "host.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except Exception:
        return None, False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return None, True
        print(
            f"provision-logs: host lock unavailable ({exc}); "
            "continuing without fencing",
            file=sys.stderr,
        )
        return None, False
    return fd, True


def _release_host_lock(fd: int | None) -> None:
    """Close the supervisor's lock copy without unlocking.

    The flock is per open-file-description: when the lock fd is inherited by
    the command process (pass_fds), closing the supervisor copy keeps the
    lock held until the command exits, so a crashed supervisor cannot free
    the fence while the mutation is still running. This file must never be
    deleted while a worker could hold it (see _acquire_host_lock).
    """
    if fd is None:
        return
    with suppress(OSError):
        os.close(fd)


def _group_dead(pgid: int) -> bool:
    """Return whether no process remains in the group *pgid*."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def worker(config):
    store = open_store(config)
    attempt, phase = config["attempt_id"], config["phase"]
    stage = config["stage"]
    current_stage = stage
    selector = selectors.DefaultSelector()
    processes = []
    engine_path = Path(config["engine_log"]) if config.get("engine_log") else None
    command_done = False
    cancelled = False
    timed_out = False
    last_command_output = time.monotonic()
    deadline = time.monotonic() + config["timeout"]
    collection_deadline = deadline + config["health_timeout"]
    records = []
    lock_fd = None
    launched = False

    def record(msg, source, level="info"):
        records.append(
            dict(
                msg=msg,
                source=source,
                stage=current_stage,
                level=level,
                stream=source.split(".")[-1],
            )
        )

    def emit(source, raw):
        nonlocal current_stage
        message = raw.decode("utf-8", errors="replace")
        if source == "journal":
            try:
                item = json.loads(message)
                records.append(
                    dict(
                        msg=message,
                        source="journal",
                        stage=stage,
                        stream="journal",
                        journal_cursor=item.get("__CURSOR"),
                    )
                )
            except (ValueError, AttributeError):
                record(message, "journal", "warning")
                store.issue(attempt, "Malformed or truncated journal record")
        else:
            marker = re.search(r"\[STEP:(\w+):(START|OK|WARN|FAIL)\]", message)
            if source == "setup.stdout" and marker:
                current_stage = marker.group(1)
            reject = re.search(r"\[REJECT:unsupported_hardware:(.*?)\]", message)
            if reject and source.endswith("stderr"):
                store.issue(attempt, f"unsupported_hardware: {reject.group(1)}")
            record(message, source, "warning" if source.endswith("stderr") else "info")
            if source == "journal.stderr":
                store.issue(
                    attempt,
                    "Service journal reported an error; collection may be incomplete",
                    source="journal",
                )

    try:
        if engine_path:
            engine_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_fd, lock_available = _acquire_host_lock(config.get("root"))
        if lock_available and lock_fd is None:
            holder = store.latest_running(attempt, exclude=attempt)
            message = "Host busy with another provisioning operation"
            if holder:
                message += f" (attempt {holder})"
            store.issue(attempt, message)
            store.update_phase(
                attempt, phase, status="complete", exit_status=126, recording=False
            )
            store.update(attempt, status="failed", failure_summary=message)
            return
        command = config.pop("command")
        environment = dict(os.environ, QIIP_LOG_CONFIG=json.dumps(config))
        if lock_fd is not None:
            os.set_inheritable(lock_fd, True)
            # The start scripts close this fd at the serving-process handoff so
            # the long-lived engine cannot hold the fence past the launch step.
            environment["QIIP_LOCK_FD"] = str(lock_fd)
        proc = subprocess.Popen(
            ["bash", "-c", command],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=[lock_fd] if lock_fd is not None else [],
        )
        launched = True
        processes.append(proc)
        for source, pipe in (
            (stage + ".stdout", proc.stdout),
            (stage + ".stderr", proc.stderr),
        ):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, [source, bytearray()])
        store.update_phase(attempt, phase, status="running", pid=proc.pid)
        # Cursor-based journal catch-up is done on-node before rejoining follow.
        # journalctl returns an error when an expired cursor cannot be sought.
        cursor = store.get(attempt).get("journal_cursor")
        journal_args = [
            "journalctl",
            "--no-pager",
            "--output=json",
            "--follow",
            "--unit=vllm.service",
            "--unit=llamacpp.service",
            "--unit=nvidia-fabricmanager.service",
        ]
        journal_args += (
            ["--after-cursor=" + cursor]
            if cursor
            else ["--since=" + store.get(attempt)["started_at"]]
        )
        try:
            journal = subprocess.Popen(
                journal_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            processes.append(journal)
            for source, pipe in (
                ("journal", journal.stdout),
                ("journal.stderr", journal.stderr),
            ):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, [source, bytearray()])
        except OSError as exc:
            store.issue(
                attempt, "Service journal unavailable: " + str(exc), source="journal"
            )

        while True:
            for key, _ in selector.select(0.1):
                source, pending = key.data
                chunk = os.read(key.fd, 8192)
                if chunk and source.startswith(stage + "."):
                    last_command_output = time.monotonic()
                pending.extend(chunk)
                while b"\n" in pending or len(pending) >= config["max_record_bytes"]:
                    newline = pending.find(b"\n")
                    length = min(
                        newline if newline >= 0 else len(pending),
                        config["max_record_bytes"],
                    )
                    emit(source, bytes(pending[:length]))
                    del pending[: length + (1 if newline == length else 0)]
                    if newline < 0 or newline > length:
                        store.issue(
                            attempt, "Long remote line split by record byte limit"
                        )
                if not chunk:
                    if pending:
                        emit(source, bytes(pending))
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            store.append_many(attempt, records)
            records.clear()
            now = time.monotonic()
            if store.get(attempt).get("cancel_command"):
                cancelled = True
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                _release_host_lock(lock_fd)
                if _group_dead(proc.pid):
                    store.update_phase(
                        attempt, phase, status="complete", exit_status=130
                    )
                    store.issue(attempt, "Remote command cancelled by gateway")
                else:
                    store.update_phase(
                        attempt,
                        phase,
                        status="survivor",
                        exit_status=130,
                        recording=False,
                    )
                    store.issue(
                        attempt,
                        f"Remote process group {proc.pid} survived cancellation; "
                        "conflicting retries blocked",
                    )
                break
            pipes_open = any(
                key.data[0].startswith(stage + ".")
                for key in selector.get_map().values()
            )
            if not command_done and proc.poll() is not None and not pipes_open:
                command_done = True
                _release_host_lock(lock_fd)
                collection_deadline = now + config["health_timeout"]
                store.update_phase(
                    attempt,
                    phase,
                    status="complete",
                    exit_status=proc.returncode,
                )
                if not engine_path:
                    break
            if not command_done and (
                now >= deadline
                or now - last_command_output >= config["inactivity_timeout"]
            ):
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                _release_host_lock(lock_fd)
                if _group_dead(proc.pid):
                    store.update_phase(
                        attempt, phase, status="complete", exit_status=124
                    )
                else:
                    store.update_phase(
                        attempt,
                        phase,
                        status="survivor",
                        exit_status=124,
                        recording=False,
                    )
                    store.issue(
                        attempt,
                        f"Remote process group {proc.pid} survived timeout; "
                        "conflicting retries blocked",
                    )
                store.issue(
                    attempt, "Remote command exceeded its total or inactivity deadline"
                )
                break
            if command_done and (
                store.get(attempt).get("stop_collection") or now >= collection_deadline
            ):
                if now >= collection_deadline:
                    store.issue(attempt, "Node collection deadline reached")
                break
    except Exception as exc:
        store.issue(attempt, "Node recorder failed: " + str(exc), source="recorder")
        if launched:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            if _group_dead(proc.pid):
                store.update_phase(attempt, phase, status="complete", exit_status=125)
            else:
                store.update_phase(
                    attempt,
                    phase,
                    status="survivor",
                    exit_status=125,
                    recording=False,
                )
                store.issue(
                    attempt,
                    f"Remote process group {proc.pid} survived recorder failure; "
                    "conflicting retries blocked",
                )
        else:
            store.update_phase(attempt, phase, status="complete", exit_status=125)
    finally:
        _release_host_lock(lock_fd)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if process is processes[0]:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
        if cancelled or timed_out:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        # Preserve bytes already written before stopping the journal/command.
        # Draining both channels here also covers a failure just before EOF.
        for key in list(selector.get_map().values()):
            source, pending = key.data
            while True:
                try:
                    chunk = os.read(key.fd, 8192)
                except BlockingIOError:
                    store.issue(
                        attempt,
                        "Source pipe still open at collection end",
                        source=source,
                    )
                    break
                if not chunk:
                    break
                pending.extend(chunk)
                while b"\n" in pending or len(pending) >= config["max_record_bytes"]:
                    newline = pending.find(b"\n")
                    length = min(
                        newline if newline >= 0 else len(pending),
                        config["max_record_bytes"],
                    )
                    emit(source, bytes(pending[:length]))
                    del pending[: length + (1 if newline == length else 0)]
                    if newline < 0 or newline > length:
                        store.issue(
                            attempt, "Long remote line split by record byte limit"
                        )
            if pending:
                emit(source, bytes(pending))
            key.fileobj.close()
        selector.close()
        store.append_many(attempt, records)
        if engine_path:
            store.update(attempt, stop_collection=True)
        if len(processes) > 1 and processes[1].returncode not in (0, -signal.SIGTERM):
            store.issue(
                attempt, "Service journal unavailable (nonzero exit)", source="journal"
            )
        store.update_phase(attempt, phase, recording=False)
        if launched:
            store.update(
                attempt,
                status="complete"
                if store.get(attempt).get("stop_collection")
                else "recorded",
            )


def engine_sink():
    """Keep draining the engine pipe even if recording or raw-tail storage fails."""
    store = None
    config = {}
    previous_handlers = {}

    def stop_recording(signum, frame):
        raise SystemExit(f"Recorder received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, stop_recording)
    try:
        config = json.loads(os.environ["QIIP_LOG_CONFIG"])
        store = open_store(config)
        capture_engine_output(config, store)
    except BaseException as exc:
        # Repeated catchable signals must not close the pipe during the drain.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if store is not None:
            with suppress(Exception):
                store.issue(
                    config["attempt_id"],
                    "Engine log recorder failed: " + str(exc),
                    source="engine",
                )
        # Even reporting the failure can fail (e.g. a full log filesystem).
        with suppress(Exception):
            print("Engine logging stopped: " + str(exc), file=sys.stderr)
    finally:
        if store is not None and config.get("phase"):
            with suppress(Exception):
                store.update_phase(
                    config["attempt_id"], config["phase"], engine_recording=False
                )
    # Never SIGPIPE an otherwise healthy engine because its evidence expired
    # or the recorder failed, including during initialization or file close.
    for sig in previous_handlers:
        signal.signal(sig, signal.SIG_IGN)
    try:
        while sys.stdin.buffer.read(8192):
            pass
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def capture_engine_output(config, store):
    """Drain up to 64 KiB / 100 ms of output per durable transaction."""
    path = Path(config["engine_log"])
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = bytearray()
    batch = []
    issues = set()
    buffered = 0
    recording = True
    flush_at = time.monotonic() + 0.1
    tail_limit = max(
        1,
        min(
            config["attempt_max_bytes"],
            config["max_bytes"] // (2 * config["max_attempts"]),
        ),
    )

    def line(raw):
        batch.append(
            dict(
                msg=raw.decode("utf-8", errors="replace"),
                source="engine",
                stage="start",
                stream=config["engine"],
            )
        )

    def flush(*, eof=False):
        nonlocal buffered, flush_at, recording
        if recording:
            try:
                stopping = bool(store.get(config["attempt_id"]).get("stop_collection"))
            except KeyError:
                recording = False
                stopping = True
            if recording:
                # A stop request closes durable capture after the buffered
                # batch, never before it. Include an unterminated final line.
                if stopping and pending:
                    line(bytes(pending))
                    pending.clear()
                if batch:
                    store.append_many(config["attempt_id"], batch)
                for issue in issues:
                    store.issue(config["attempt_id"], issue)
                if stopping or eof:
                    if config.get("phase"):
                        store.update_phase(
                            config["attempt_id"],
                            config["phase"],
                            engine_recording=False,
                        )
                    recording = False
        batch.clear()
        issues.clear()
        buffered = 0
        flush_at = time.monotonic() + 0.1

    with (
        path.open("wb", buffering=0) as output,
        selectors.DefaultSelector() as selector,
    ):
        selector.register(sys.stdin.buffer, selectors.EVENT_READ)
        size = 0
        while True:
            ready = selector.select(max(0, flush_at - time.monotonic()))
            if ready:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    if pending:
                        line(bytes(pending))
                        pending.clear()
                    flush(eof=True)
                    return
                if not path.exists():
                    return  # The caller continues draining an evicted raw tail.
                pending.extend(chunk)
                buffered += len(chunk)
                while b"\n" in pending or len(pending) >= config["max_record_bytes"]:
                    newline = pending.find(b"\n")
                    length = min(
                        newline if newline >= 0 else len(pending),
                        config["max_record_bytes"],
                    )
                    line(bytes(pending[:length]))
                    del pending[: length + (1 if newline == length else 0)]
                    if newline < 0 or newline > length:
                        issues.add("Long engine line split by record byte limit")
                tail = chunk[-tail_limit:]
                if size + len(tail) > tail_limit:
                    output.seek(0)
                    output.truncate()
                    size = 0
                    issues.add("Raw engine log rotated by node byte limit")
                output.write(tail)
                size += len(tail)
            if buffered >= 65536 or time.monotonic() >= flush_at:
                flush()


def main():
    action = sys.argv[1]
    if action == "engine":
        engine_sink()
        return
    config = json.loads(sys.stdin.read())
    if action == "worker":
        worker(config)
        return
    store = open_store(config)
    if action == "active":
        # Host-level authority: report whether any attempt on this host still
        # has a live phase. A stale row (recorder died, group gone) is
        # re-probed and closed so an orphan can never wedge the host forever.
        live: dict[str, object] = {}
        for attempt_id, metadata in store.attempts_metadata_by_host(config["hostname"]):
            for phase, info in (metadata.get("phases") or {}).items():
                if info.get("status") not in {"running", "launching", "survivor"}:
                    continue
                pid = info.get("pid")
                if pid and _group_dead(pid):
                    store.update_phase(
                        attempt_id,
                        phase,
                        status="complete",
                        exit_status=137,
                        recording=False,
                    )
                    continue
                if not pid:
                    launcher = info.get("launcher_pid")
                    if launcher and not _group_dead(launcher):
                        # Worker is still starting; keep the intent live.
                        live = {
                            "holder": attempt_id,
                            "phase": phase,
                            "status": info["status"],
                            "pid": None,
                        }
                        continue
                    # No command pid and no live launcher: an abandoned launch
                    # intent (worker died or the node rebooted before it could
                    # record the command pid).
                    store.update_phase(
                        attempt_id,
                        phase,
                        status="complete",
                        exit_status=137,
                        recording=False,
                    )
                    continue
                if not live:
                    live = {
                        "holder": attempt_id,
                        "phase": phase,
                        "status": info["status"],
                        "pid": pid,
                    }
        print(json.dumps({"active": bool(live), **live}))
        return
    attempt = config["attempt_id"]
    if action == "launch":
        try:
            store.get(attempt)
        except KeyError:
            prune_raw_logs(config, store)
            store.create(
                config["hostname"],
                engine=config["engine"],
                model=config["model"],
                bundle_version=config["bundle_version"],
                attempt_id=attempt,
            )
        phase = config["phase"]
        metadata = store.get(attempt)
        if phase not in metadata.get("phases", {}):
            store.update(
                attempt, status="running", stop_collection=False, cancel_command=False
            )
            store.update_phase(
                attempt,
                phase,
                status="launching",
                recording=True,
                **({"engine_recording": True} if config.get("engine_log") else {}),
            )
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # The launcher pid lets the active probe tell an in-flight start
            # from an intent abandoned before the command pid was recorded.
            store.update_phase(attempt, phase, launcher_pid=proc.pid)
            # communicate() waits for the worker; only send and close stdin.
            proc.stdin.write(json.dumps(config).encode())
            proc.stdin.close()
    elif action in {"finish", "cancel"}:
        store.update(
            attempt,
            stop_collection=True,
            cancel_command=action == "cancel",
            status="complete",
            finished_at=timestamp(),
        )
        deadline = time.monotonic() + 5
        while (
            any(
                p.get("recording") or p.get("engine_recording")
                for p in store.get(attempt).get("phases", {}).values()
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        metadata = store.get(attempt)
        if any(p.get("engine_recording") for p in metadata.get("phases", {}).values()):
            store.issue(
                attempt, "Engine log flush acknowledgement unavailable", source="engine"
            )
        elif (
            any("engine_recording" in p for p in metadata.get("phases", {}).values())
            and "engine" not in metadata["sources"]
        ):
            store.issue(attempt, "Engine startup log unavailable", source="engine")
    elif action != "read":
        raise ValueError("Unknown recorder action")
    try:
        page = store.read(attempt, after=config.get("after", 0), limit=128)
        print(json.dumps(page))
    except KeyError:
        print(json.dumps({"unavailable": "Remote attempt evicted or unavailable"}))


if __name__ == "__main__":
    os.umask(0o077)
    main()
