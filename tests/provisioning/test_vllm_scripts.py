"""Behavioral tests for verified vLLM process replacement."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(os.environ.get("AUTOVLLM_TEST_SCRIPT_ROOT", REPO_ROOT))
START_SCRIPT = SCRIPT_ROOT / "auto-vllm" / "start-vllm.sh"
STOP_SCRIPT = SCRIPT_ROOT / "auto-vllm" / "stop-vllm.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _wait_for_line(path: Path, expected: str, *, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and expected in path.read_text().splitlines():
            return
        time.sleep(0.01)
    pytest.fail(f"{expected!r} did not appear in {path}")


def _script_environment(
    tmp_path: Path,
    *,
    vllm_bin: Path,
    process_log: Path,
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    nvidia_smi = bin_dir / "nvidia-smi"
    _write_executable(
        nvidia_smi,
        """#!/bin/bash
case "$*" in
  *"--query-gpu=name"*) echo "NVIDIA A100" ;;
  *"--list-gpus"*) echo "GPU 0: NVIDIA A100" ;;
  *"--query-gpu=memory.total"*) echo "81920" ;;
esac
""",
    )
    flashinfer_python = tmp_path / "fake-python"
    _write_executable(
        flashinfer_python,
        """#!/bin/bash
[[ "${AUTOVLLM_FLASHINFER_AVAILABLE:-1}" == "1" ]]
""",
    )

    cache_dir = tmp_path / "nfs-cache"
    cache_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "AUTOVLLM_BIN": str(vllm_bin),
            "AUTOVLLM_SCRIPT_DIR": str(SCRIPT_ROOT / "auto-vllm"),
            "AUTOVLLM_COMMAND_PATTERN": f"{vllm_bin} serve",
            "AUTOVLLM_PID_FILE": str(tmp_path / "vllm.pid"),
            "AUTOVLLM_HF_CACHE_LINK": str(tmp_path / "cache" / "huggingface"),
            "AUTOVLLM_LOG_FILE": str(tmp_path / "vllm-serve.log"),
            "AUTOVLLM_PYTHON": str(flashinfer_python),
            "AUTOVLLM_STARTUP_GRACE_PERIOD": "0.05",
            "AUTOVLLM_STOP_TIMEOUT": "2",
            "AUTOVLLM_STOP_INTERVAL": "0.01",
            "AUTOVLLM_TEST_LOG": str(process_log),
            "AUTOVLLM_NFS_MOUNT_POINT": str(cache_dir),
        }
    )
    return env


def _configured_profile(
    *,
    gpu_model: str,
    gpu_count: int,
    gpu_vram_gb: int,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    env = os.environ.copy()
    for name in (
        "VLLM_MODEL",
        "VLLM_TENSOR_PARALLEL",
        "VLLM_GPU_MEM_UTIL",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_BATCHED_TOKENS",
        "VLLM_EXTRA_ARGS",
        "AUTOVLLM_MODEL",
        "AUTOVLLM_TENSOR_PARALLEL",
        "AUTOVLLM_GPU_MEM_UTIL",
        "AUTOVLLM_MAX_MODEL_LEN",
        "AUTOVLLM_MAX_BATCHED_TOKENS",
        "AUTOVLLM_EXTRA_ARGS",
    ):
        env.pop(name, None)
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    env.update(overrides or {})
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
GPU_MODEL={gpu_model!r}
GPU_COUNT={gpu_count}
GPU_VRAM_GB={gpu_vram_gb}
configure_vllm_params
printf '%s|%s|%s|%s|%s|%s\n' \
    "$MODEL" "$TENSOR_PARALLEL" "$GPU_MEM_UTIL" \
    "$MAX_MODEL_LEN" "$MAX_BATCHED_TOKENS" "$EXTRA_ARGS"
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()[-1].split("|")


def _start_fake_vllm(
    vllm_bin: Path,
    model: str,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(vllm_bin), "serve", model],
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_start_vllm_stops_existing(tmp_path: Path) -> None:
    """A stale PID cannot hide an orphan, and no unrelated PID is killed."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
model="$2"
echo "start:${model}" >> "$AUTOVLLM_TEST_LOG"
trap 'echo "stop:'"${model}"'" >> "$AUTOVLLM_TEST_LOG"; exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    old_vllm = _start_fake_vllm(vllm_bin, "old-model", env)
    unrelated = subprocess.Popen(["sleep", "30"])
    pid_file = Path(env["AUTOVLLM_PID_FILE"])

    try:
        _wait_for_line(process_log, "start:old-model")
        # Simulate PID reuse: the file names a live but unrelated process,
        # while the real old vLLM must be discovered from its command line.
        pid_file.write_text(str(unrelated.pid))
        env["AUTOVLLM_MODEL"] = "new-model"

        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        _wait_for_line(process_log, "start:new-model")
        old_vllm.wait(timeout=2)
        assert unrelated.poll() is None
        assert process_log.read_text().splitlines()[:3] == [
            "start:old-model",
            "stop:old-model",
            "start:new-model",
        ]
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if old_vllm.poll() is None:
            old_vllm.kill()
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_start_vllm_aborts_when_existing_process_cannot_stop(
    tmp_path: Path,
) -> None:
    """A process surviving SIGTERM prevents a replacement launch."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "stubborn-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
model="$2"
echo "start:${model}" >> "$AUTOVLLM_TEST_LOG"
trap '' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    old_vllm = _start_fake_vllm(vllm_bin, "old-model", env)
    pid_file = Path(env["AUTOVLLM_PID_FILE"])

    try:
        _wait_for_line(process_log, "start:old-model")
        pid_file.write_text(str(old_vllm.pid))
        env["AUTOVLLM_MODEL"] = "new-model"
        env["AUTOVLLM_STOP_TIMEOUT"] = "1"

        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode != 0
        assert "Timed out waiting for vLLM PID" in result.stderr
        assert process_log.read_text().splitlines() == ["start:old-model"]
        assert pid_file.read_text() == str(old_vllm.pid)
        assert old_vllm.poll() is None
    finally:
        old_vllm.send_signal(signal.SIGKILL)
        old_vllm.wait(timeout=2)
        pid_file.unlink(missing_ok=True)


def test_stop_vllm_stale_pid_does_not_kill_unrelated_process(
    tmp_path: Path,
) -> None:
    """PID existence alone is never treated as vLLM identity."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(vllm_bin, "#!/bin/bash\nexit 0\n")
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    unrelated = subprocess.Popen(["sleep", "30"])
    pid_file = Path(env["AUTOVLLM_PID_FILE"])
    pid_file.write_text(str(unrelated.pid))

    try:
        result = subprocess.run(
            ["bash", str(STOP_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0
        assert unrelated.poll() is None
        assert not pid_file.exists()
        assert "Ignoring stale vLLM PID file" in result.stderr
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_force_stop_kills_stubborn_process_and_removes_pidfile(
    tmp_path: Path,
) -> None:
    """Force mode verifies SIGKILL completion before deleting the PID file."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "stubborn-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo "start:$2" >> "$AUTOVLLM_TEST_LOG"
trap '' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    process = _start_fake_vllm(vllm_bin, "old-model", env)
    pid_file = Path(env["AUTOVLLM_PID_FILE"])

    try:
        _wait_for_line(process_log, "start:old-model")
        pid_file.write_text(str(process.pid))

        result = subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        process.wait(timeout=2)
        assert not pid_file.exists()
        assert "Stopped vLLM process" in result.stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_start_requires_flashinfer_aot_before_launch(tmp_path: Path) -> None:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo launched >> "$AUTOVLLM_TEST_LOG"
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    env["AUTOVLLM_FLASHINFER_AVAILABLE"] = "0"

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "matching FlashInfer AOT kernels are unavailable" in result.stderr
    assert not process_log.exists()
    assert not Path(env["AUTOVLLM_PID_FILE"]).exists()


def test_vllm_launch_disables_flashinfer_jit(tmp_path: Path) -> None:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo "jit:${FLASHINFER_DISABLE_JIT:-unset}" >> "$AUTOVLLM_TEST_LOG"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )

    try:
        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert process_log.read_text().splitlines() == ["jit:1"]
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )


def test_hf_cache_real_directory_aborts_before_launch(tmp_path: Path) -> None:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo launched >> "$AUTOVLLM_TEST_LOG"
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    cache_target = Path(env["AUTOVLLM_HF_CACHE_LINK"])
    cache_target.mkdir(parents=True)
    sentinel = cache_target / "locally-cached-model"
    sentinel.write_text("keep")

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "is a real directory" in result.stderr
    assert cache_target.is_dir()
    assert not cache_target.is_symlink()
    assert sentinel.read_text() == "keep"
    assert not (cache_target / Path(env["AUTOVLLM_NFS_MOUNT_POINT"]).name).exists()
    assert not process_log.exists()


def test_hf_cache_stale_symlink_is_replaced_exactly(tmp_path: Path) -> None:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    cache_target = Path(env["AUTOVLLM_HF_CACHE_LINK"])
    cache_target.parent.mkdir(parents=True)
    cache_target.symlink_to(tmp_path / "old-cache")

    try:
        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert cache_target.is_symlink()
        assert os.readlink(cache_target) == env["AUTOVLLM_NFS_MOUNT_POINT"]
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )


@pytest.mark.parametrize(
    ("gpu_model", "gpu_count", "gpu_vram_gb", "expected"),
    [
        (
            "NVIDIA A100",
            3,
            80,
            ["Qwen/Qwen2.5-72B-Instruct", "3", "0.90", "32768", "32768", ""],
        ),
        (
            "NVIDIA A100",
            2,
            80,
            ["Qwen/Qwen2.5-32B-Instruct", "2", "0.90", "32768", "32768", ""],
        ),
        (
            "NVIDIA A100",
            1,
            40,
            ["Qwen/Qwen2.5-14B-Instruct", "1", "0.90", "32768", "32768", ""],
        ),
        (
            "Tesla V100",
            2,
            32,
            [
                "Qwen/Qwen2.5-14B-Instruct",
                "2",
                "0.85",
                "8192",
                "32768",
                "--dtype float16",
            ],
        ),
        (
            "Tesla V100",
            3,
            32,
            [
                "Qwen/Qwen2.5-32B-Instruct",
                "3",
                "0.85",
                "8192",
                "32768",
                "--dtype float16",
            ],
        ),
        (
            "NVIDIA GeForce RTX 4090",
            1,
            24,
            [
                "Qwen/Qwen2.5-7B-Instruct",
                "1",
                "0.80",
                "4096",
                "32768",
                "--enforce-eager",
            ],
        ),
        (
            "NVIDIA RTX 6000 Ada",
            1,
            48,
            [
                "Qwen/Qwen2.5-14B-Instruct",
                "1",
                "0.80",
                "4096",
                "32768",
                "--enforce-eager",
            ],
        ),
    ],
)
def test_gpu_profile_matrix_selects_runnable_configuration(
    gpu_model: str,
    gpu_count: int,
    gpu_vram_gb: int,
    expected: list[str],
) -> None:
    assert (
        _configured_profile(
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            gpu_vram_gb=gpu_vram_gb,
        )
        == expected
    )


def test_explicit_vllm_overrides_still_win() -> None:
    assert _configured_profile(
        gpu_model="NVIDIA A100",
        gpu_count=2,
        gpu_vram_gb=80,
        overrides={
            "AUTOVLLM_MODEL": "example/custom-model",
            "AUTOVLLM_TENSOR_PARALLEL": "1",
            "AUTOVLLM_GPU_MEM_UTIL": "0.73",
            "AUTOVLLM_MAX_MODEL_LEN": "1234",
            "AUTOVLLM_MAX_BATCHED_TOKENS": "5678",
            "AUTOVLLM_EXTRA_ARGS": "--dtype float16",
        },
    ) == [
        "example/custom-model",
        "1",
        "0.73",
        "1234",
        "5678",
        "--dtype float16",
    ]


def test_retired_vllm_overrides_warn_and_are_ignored() -> None:
    env = os.environ.copy()
    retired = {
        "VLLM_TENSOR_PARALLEL": "1",
        "VLLM_GPU_MEM_UTIL": "0.11",
        "VLLM_MAX_MODEL_LEN": "123",
        "VLLM_MAX_BATCHED_TOKENS": "456",
        "VLLM_EXTRA_ARGS": "--dtype float16",
    }
    for name in (
        *retired,
        "AUTOVLLM_TENSOR_PARALLEL",
        "AUTOVLLM_GPU_MEM_UTIL",
        "AUTOVLLM_MAX_MODEL_LEN",
        "AUTOVLLM_MAX_BATCHED_TOKENS",
        "AUTOVLLM_EXTRA_ARGS",
    ):
        env.pop(name, None)
    env.update(retired)
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
GPU_MODEL='NVIDIA A100'
GPU_COUNT=2
GPU_VRAM_GB=80
configure_vllm_params
printf '%s|%s|%s|%s|%s\n' \
    "$TENSOR_PARALLEL" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" \
    "$MAX_BATCHED_TOKENS" "$EXTRA_ARGS"
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "2|0.90|32768|32768|"
    for legacy, replacement in zip(
        retired,
        (
            "AUTOVLLM_TENSOR_PARALLEL",
            "AUTOVLLM_GPU_MEM_UTIL",
            "AUTOVLLM_MAX_MODEL_LEN",
            "AUTOVLLM_MAX_BATCHED_TOKENS",
            "AUTOVLLM_EXTRA_ARGS",
        ),
        strict=True,
    ):
        assert f"{legacy} is ignored; use {replacement} instead" in result.stderr


def test_vllm_env_does_not_leak_script_params(tmp_path: Path) -> None:
    captured_env = tmp_path / "vllm.env"
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "environment-capturing-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
env | sort > "$AUTOVLLM_CAPTURE_ENV"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )
    env.update(
        {
            "AUTOVLLM_CAPTURE_ENV": str(captured_env),
            "AUTOVLLM_API_PORT": "8123",
            "AUTOVLLM_MODEL": "example/model",
            "AUTOVLLM_TENSOR_PARALLEL": "1",
            "AUTOVLLM_GPU_MEM_UTIL": "0.73",
            "AUTOVLLM_MAX_MODEL_LEN": "1234",
            "AUTOVLLM_MAX_BATCHED_TOKENS": "5678",
            "AUTOVLLM_EXTRA_ARGS": "--dtype float16",
            "VLLM_PORT": "8123",
            "VLLM_TENSOR_PARALLEL": "99",
            "VLLM_GPU_MEM_UTIL": "0.01",
            "VLLM_MAX_MODEL_LEN": "1",
            "VLLM_MAX_BATCHED_TOKENS": "1",
            "VLLM_EXTRA_ARGS": "--should-not-leak",
            "HF_TOKEN": "hf_secret",
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        captured = {
            line.partition("=")[0]: line.partition("=")[2]
            for line in captured_env.read_text().splitlines()
        }
        forbidden = {
            "AUTOVLLM_API_PORT",
            "AUTOVLLM_NFS_MOUNT_POINT",
            "AUTOVLLM_MODEL",
            "AUTOVLLM_TENSOR_PARALLEL",
            "AUTOVLLM_GPU_MEM_UTIL",
            "AUTOVLLM_MAX_MODEL_LEN",
            "AUTOVLLM_MAX_BATCHED_TOKENS",
            "AUTOVLLM_EXTRA_ARGS",
            "AUTOVLLM_SCRIPT_DIR",
            "AUTOVLLM_BIN",
            "AUTOVLLM_PID_FILE",
            "AUTOVLLM_HF_CACHE_LINK",
            "AUTOVLLM_LOG_FILE",
            "AUTOVLLM_PYTHON",
            "AUTOVLLM_PROC_ROOT",
            "AUTOVLLM_COMMAND_PATTERN",
            "AUTOVLLM_STARTUP_GRACE_PERIOD",
            "AUTOVLLM_STARTUP_LOG_LINES",
            "AUTOVLLM_STOP_TIMEOUT",
            "AUTOVLLM_STOP_INTERVAL",
            "VLLM_PORT",
            "VLLM_TENSOR_PARALLEL",
            "VLLM_GPU_MEM_UTIL",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_MAX_BATCHED_TOKENS",
            "VLLM_EXTRA_ARGS",
        }
        assert forbidden.isdisjoint(captured)
        assert captured["HF_TOKEN"] == "hf_secret"
        assert captured["FLASHINFER_DISABLE_JIT"] == "1"
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )


def test_start_failure_returns_log_tail_immediately(tmp_path: Path) -> None:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "failing-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo 'CUDA initialization failed' >&2
exit 7
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
    )

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "exited during startup" in result.stderr
    assert "CUDA initialization failed" in result.stderr
    assert "vLLM started" not in result.stdout
    assert not Path(env["AUTOVLLM_PID_FILE"]).exists()
