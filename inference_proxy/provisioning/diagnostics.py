"""Bounded failure snapshots, also uploaded to nodes (Python 3.9+, stdlib only).

Diagnostic payloads use the existing log store and its retention budget. Only
small source manifests live in attempt metadata. A retry keeps completed sources
and timestamps new snapshots honestly; historical journals keep the failure window.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

LOG_SOURCES = ("setup.stdout", "setup.stderr", "start.stdout", "start.stderr", "engine")
SYSTEM_SOURCES = (
    "nvidia_services",
    "kernel_gpu_oom",
    "os",
    "kernel",
    "gpu",
    "driver",
    "toolkit",
    "runtime",
    "ram",
    "disk",
    "mounts",
)
SOURCE_NAMES = (*LOG_SOURCES, *SYSTEM_SOURCES)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - node Python 3.9


def source_commands(config: dict[str, Any]) -> dict[str, list[str]]:
    failure = config["failure"]
    # Epoch timestamps avoid journalctl's locale-dependent date parsing.
    since = "@" + str(datetime.fromisoformat(failure["started_at"]).timestamp())
    until = "@" + str(datetime.fromisoformat(failure["failed_at"]).timestamp() + 1)
    journal = [
        "journalctl",
        "--no-pager",
        "--output=short-iso",
        "--lines=201",
        "--since=" + since,
        "--until=" + until,
    ]
    runtime = (
        ["/usr/local/bin/llama-server", "--version"]
        if config["engine"] == "llama_cpp"
        else [
            "/opt/vllm-venv/bin/python",
            "-c",
            "import importlib.metadata as m; import torch; "
            "print('vllm=' + m.version('vllm')); "
            "print('torch=' + torch.__version__); print('CUDA runtime=' + str(torch.version.cuda))",
        ]
    )
    return {
        "nvidia_services": [
            *journal,
            "--unit=nvidia-persistenced.service",
            "--unit=nvidia-fabricmanager.service",
            "--unit=nvidia.service",
        ],
        "kernel_gpu_oom": [
            *journal,
            "--kernel",
            "--case-sensitive=no",
            "--grep=NVRM|Xid|nvidia|GPU|out of memory|oom|killed process",
        ],
        "os": ["cat", "/etc/os-release"],
        "kernel": ["uname", "-a"],
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
            "--format=csv",
        ],
        "driver": ["cat", "/proc/driver/nvidia/version"],
        "toolkit": ["/usr/local/cuda/bin/nvcc", "--version"],
        "runtime": runtime,
        "ram": ["cat", "/proc/meminfo"],
        "disk": ["df", "-hT", "/", config["mount_point"]],
        "mounts": [
            "findmnt",
            "--target",
            config["mount_point"],
            "--output=SOURCE,TARGET,FSTYPE,OPTIONS",
        ],
    }


def capture(command: list[str], timeout: float, max_bytes: int) -> dict[str, Any]:
    """Drain both streams with constant memory and kill descendants on timeout."""
    started = time.monotonic()
    result: dict[str, Any] = {
        "collected_at": timestamp(),
        "exit_code": None,
        "signal": None,
        "truncated": False,
    }
    tail = bytearray()
    seen = 0
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                deadline = started + timeout
                try:
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        for key, _ in selector.select(min(remaining, 0.1)):
                            chunk = os.read(key.fd, 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            seen += len(chunk)
                            tail.extend(chunk)
                            del tail[:-max_bytes]
                    proc.wait(timeout=max(0.001, deadline - time.monotonic()))
                    result["status"] = (
                        "collected" if proc.returncode == 0 else "unavailable"
                    )
                except (TimeoutError, subprocess.TimeoutExpired):
                    result.update(
                        status="timed_out", reason="Source collection deadline exceeded"
                    )
                finally:
                    # A child may keep the pipe open after its parent exits.
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    # A storage probe stuck in kernel I/O must not hold up
                    # other sources; even cleanup has a bounded wait.
                    with suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=0.1)
                code = proc.returncode
                result["exit_code"] = code if code is not None and code >= 0 else None
                result["signal"] = -code if code is not None and code < 0 else None
        finally:
            proc.stdout.close()
    except OSError as exc:
        result.update(status="unavailable", reason=str(exc)[:512])
    decoded = tail.decode("utf-8", errors="replace").encode("utf-8")
    text = decoded[-max_bytes:].decode("utf-8", errors="ignore")
    lines = text.splitlines()
    result["truncated"] = (
        seen > max_bytes or len(decoded) > max_bytes or len(lines) > 200
    )
    result["output"] = "\n".join(lines[-200:])
    if result["status"] == "collected" and result["truncated"]:
        result["status"] = "truncated"
    if result["status"] == "unavailable" and "reason" not in result:
        result["reason"] = f"Collector exited with status {result['exit_code']}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def collect(config: dict[str, Any], store: Any) -> None:
    """Persist each source before moving on, preserving useful partial results."""
    attempt_id = config["attempt_id"]
    sources = store.get(attempt_id).get("diagnostics", {}).get("sources", {})
    deadline = time.monotonic() + config["diagnostics_timeout"]
    commands = source_commands(config)
    max_bytes = config["diagnostics_source_max_bytes"]
    for name in SOURCE_NAMES:
        if sources.get(name, {}).get("status") in {"collected", "truncated"}:
            continue
        remaining = deadline - time.monotonic()
        result: dict[str, Any]
        if remaining <= 0:
            result = dict(
                status="timed_out", reason="Overall collection deadline exceeded"
            )
        else:
            try:
                if name in LOG_SOURCES:
                    result = store.tail(attempt_id, name, max_bytes=max_bytes)
                else:
                    result = capture(
                        commands[name],
                        min(remaining, config["diagnostics_source_timeout"]),
                        max_bytes,
                    )
            except Exception as exc:
                result = dict(
                    status="unavailable",
                    reason=f"Collector failed: {type(exc).__name__}",
                )
        output = result.pop("output", "")
        result.setdefault("collected_at", timestamp())
        result["deferred"] = result["status"] in {"unavailable", "timed_out"}
        if output:
            # Split UTF-8 text without exceeding the store's individual record cap.
            chunk = max(1, store.max_record_bytes // 4)
            entries = store.append_many(
                attempt_id,
                [
                    dict(
                        msg=output[pos : pos + chunk],
                        source="diagnostics." + name,
                        stage="diagnostics",
                        ts=result["collected_at"],
                    )
                    for pos in range(0, len(output), chunk)
                ],
            )
            result["first_seq"] = entries[0]["seq"]
            result["last_seq"] = entries[-1]["seq"]
        sources[name] = result
        store.update(
            attempt_id, diagnostics=dict(sources=sources, updated_at=timestamp())
        )
