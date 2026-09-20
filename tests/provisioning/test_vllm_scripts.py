"""Behavioral tests for verified vLLM process replacement."""

from __future__ import annotations

import json
import os
import re
import shlex
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
    gpu_model: str = "NVIDIA A100",
    gpu_vram_mb: int = 81920,
    gpu_compute_cap: str = "8.0",
    gpu_count: int = 1,
    device_count: int | None = None,
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    nvidia_smi = bin_dir / "nvidia-smi"
    _write_executable(
        nvidia_smi,
        f"""#!/bin/bash
case "$*" in
  *"--query-gpu=name"*) echo "{gpu_model}" ;;
  *"--list-gpus"*)
    for i in $(seq "${{AUTOVLLM_TEST_GPU_COUNT:-1}}"); do
        echo "GPU $((i - 1)): {gpu_model}"
    done
    ;;
  *"--query-gpu=memory.total"*) echo "{gpu_vram_mb}" ;;
  *"--query-gpu=compute_cap"*) echo "{gpu_compute_cap}" ;;
  *"--query-gpu=driver_version"*) echo "580.126.09" ;;
  *"--query-gpu=persistence_mode"*) echo "Enabled" ;;
esac
""",
    )
    # verify_cuda_execution compiles and runs a tiny probe; fake nvcc emits a
    # runnable probe binary so the real proof path is exercised.
    _write_executable(
        bin_dir / "nvcc",
        """#!/bin/bash
out=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        out="$2"
        shift 2
    else
        shift
    fi
done
printf '#!/bin/bash\\nexit 0\\n' > "$out"
chmod +x "$out"
""",
    )
    _write_executable(
        bin_dir / "lspci",
        "#!/bin/bash\necho ''\n",
    )
    (tmp_path / "os-release").write_text("ID=rhel\nVERSION_ID=9.5\n")
    flashinfer_python = tmp_path / "fake-python"
    _write_executable(
        flashinfer_python,
        r"""#!/bin/bash
if [[ "$1" == "-" ]]; then
    # Topology inspection runs the real Python program piped to stdin.
    shift
    exec python3 - "$@"
fi
if [[ "$*" == *'torch.cuda.device_count()'* ]]; then
    echo "${AUTOVLLM_TEST_CUDA_COUNT:-1}"
    exit 0
fi
if [[ "$*" == *'import json'* ]]; then
    config_path=""
    for arg in "$@"; do
        case "$arg" in
            *.json) config_path="$arg" ;;
        esac
    done
    python3 -c 'import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("num_attention_heads", 0), cfg.get("num_key_value_heads", 0))' \
        "$config_path" 2>/dev/null || echo "0 0"
    exit 0
fi
if [[ "$*" == *'local_files_only'* ]]; then
    case "${AUTOVLLM_TEST_SNAPSHOT_STATE:-complete}" in
        complete) echo complete; exit 0 ;;
        incomplete) echo "incomplete" >&2; exit 2 ;;
        hung) echo "hung" >&2; sleep 30; exit 3 ;;
        download_fails) echo "missing" >&2; exit 3 ;;
        missing)
            if [[ -f "${AUTOVLLM_TEST_SNAPSHOT_FLAG:-/nonexistent}" ]]; then
                echo complete; exit 0
            fi
            echo "missing" >&2; exit 3
            ;;
    esac
fi
if [[ "$*" == *'snapshot_download'* ]]; then
    if [[ "${AUTOVLLM_TEST_SNAPSHOT_STATE:-complete}" != "download_fails" ]]; then
        touch "${AUTOVLLM_TEST_SNAPSHOT_FLAG:-/nonexistent}" 2>/dev/null || true
    fi
    if [[ -n "${AUTOVLLM_TEST_SNAPSHOT_CONFIG_HEADS:-}" ]]; then
        cache_root="${AUTOVLLM_NFS_MOUNT_POINT}/hub/models--org--model"
        mkdir -p "$cache_root/refs" "$cache_root/trees" \
            "$cache_root/snapshots/mainrev" "$cache_root/snapshots/oldrev"
        echo mainrev > "$cache_root/refs/main"
        : > "$cache_root/trees/mainrev.json"
        printf '{"num_attention_heads": %s, "num_key_value_heads": %s}' \
            "${AUTOVLLM_TEST_SNAPSHOT_CONFIG_HEADS}" \
            "${AUTOVLLM_TEST_SNAPSHOT_CONFIG_KV:-0}" \
            > "$cache_root/snapshots/mainrev/config.json"
        printf '{"num_attention_heads": 64, "num_key_value_heads": 8}' \
            > "$cache_root/snapshots/oldrev/config.json"
    fi
    exit 0
fi
[[ "${AUTOVLLM_FLASHINFER_AVAILABLE:-1}" == "1" ]]
""",
    )

    cache_dir = tmp_path / "nfs-cache"
    cache_dir.mkdir(exist_ok=True)
    flashinfer_dir = tmp_path / "flashinfer-cache"
    flashinfer_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.pop("INVOCATION_ID", None)
    env.pop("AUTOVLLM_DTYPE", None)
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
            "AUTOVLLM_FLASHINFER_CACHE_DIR": str(flashinfer_dir),
            "AUTOVLLM_MIN_FREE_GB": "1",
            "AUTOVLLM_STARTUP_GRACE_PERIOD": "0.05",
            "AUTOVLLM_STOP_TIMEOUT": "2",
            "AUTOVLLM_STOP_INTERVAL": "0.01",
            "AUTOVLLM_TEST_LOG": str(process_log),
            "AUTOVLLM_NFS_MOUNT_POINT": str(cache_dir),
            "AUTOVLLM_TEST_GPU_COUNT": str(gpu_count),
            "PROFILE_OS_RELEASE": str(tmp_path / "os-release"),
        }
    )
    if device_count is not None:
        env["AUTOVLLM_TEST_CUDA_COUNT"] = str(device_count)
    return env


def _configured_profile(
    *,
    profile_bucket: str,
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
        "AUTOVLLM_DTYPE",
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
PROFILE_BUCKET={profile_bucket!r}
GPU_COUNT={gpu_count}
GPU_VRAM_GB={gpu_vram_gb}
configure_vllm_params
printf '%s|%s|%s|%s|%s|%s|%s\\n' \
    "$MODEL" "$TENSOR_PARALLEL" "$GPU_MEM_UTIL" \
    "$MAX_MODEL_LEN" "$MAX_BATCHED_TOKENS" "$EXTRA_ARGS" "$EFFECTIVE_DTYPE"
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


def test_start_requires_flashinfer_jit_toolchain_when_aot_unavailable(
    tmp_path: Path,
) -> None:
    """When FlashInfer is selected (SM90+) and AOT kernels are missing,
    the script must verify the JIT toolchain and fail if it is absent."""
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
    env["AUTOVLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "FlashInfer" in result.stdout + result.stderr
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
    env["AUTOVLLM_ATTENTION_BACKEND"] = "FLASHINFER"

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


def _captured_vllm_argv(
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    gpu_model: str = "NVIDIA A100",
    gpu_compute_cap: str = "8.0",
) -> str:
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
printf '%s\\n' "$*" >> "$AUTOVLLM_TEST_LOG"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
        gpu_model=gpu_model,
        gpu_compute_cap=gpu_compute_cap,
    )
    env.update(extra_env or {})
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
        return process_log.read_text()
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )


def test_dtype_override_is_passed_to_vllm_serve(tmp_path: Path) -> None:
    argv = _captured_vllm_argv(tmp_path, {"AUTOVLLM_DTYPE": "bfloat16"})
    assert " --dtype bfloat16" in f" {argv}"
    assert argv.count("--dtype") == 1


def test_empty_dtype_is_omitted_from_vllm_serve(tmp_path: Path) -> None:
    argv = _captured_vllm_argv(tmp_path)
    assert "--dtype" not in argv


def test_t4_default_emits_single_float16_dtype(tmp_path: Path) -> None:
    # T4 forces float16 by default; it must reach vLLM as exactly one --dtype
    # argv value (it is no longer smuggled through EXTRA_ARGS).
    argv = _captured_vllm_argv(tmp_path, gpu_model="Tesla T4", gpu_compute_cap="7.5")
    assert " --dtype float16" in f" {argv}"
    assert argv.count("--dtype") == 1


def test_dtype_override_wins_over_t4_default_without_duplicate(tmp_path: Path) -> None:
    # The reviewer-raised conflict: T4 defaults to float16 via EXTRA_ARGS and a
    # dtype override would previously produce two --dtype flags. Now the
    # override wins and the default is dropped: exactly one --dtype emitted.
    argv = _captured_vllm_argv(
        tmp_path,
        {"AUTOVLLM_DTYPE": "bfloat16"},
        gpu_model="Tesla T4",
        gpu_compute_cap="7.5",
    )
    assert argv.count("--dtype") == 1
    assert " --dtype bfloat16" in f" {argv}"
    assert "--dtype float16" not in argv


def test_invalid_dtype_override_fails_before_launch(tmp_path: Path) -> None:
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
    env["AUTOVLLM_DTYPE"] = "float16 --seed 0"

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "float16 --seed 0" in result.stderr
    assert "unsupported vLLM dtype" in result.stderr
    assert not process_log.exists()


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
    ("profile_bucket", "gpu_count", "gpu_vram_gb", "expected"),
    [
        (
            "ampere-a100",
            3,
            80,
            ["Qwen/Qwen2.5-72B-Instruct", "3", "0.90", "32768", "32768", "", ""],
        ),
        (
            "ampere-a100",
            2,
            80,
            ["Qwen/Qwen2.5-32B-Instruct", "2", "0.90", "32768", "32768", "", ""],
        ),
        (
            "ampere-a100",
            1,
            40,
            ["Qwen/Qwen2.5-14B-Instruct", "1", "0.90", "32768", "32768", "", ""],
        ),
        (
            "volta",
            2,
            32,
            [
                "Qwen/Qwen2.5-14B-Instruct",
                "2",
                "0.85",
                "8192",
                "32768",
                "",
                "float16",
            ],
        ),
        (
            "volta",
            3,
            32,
            [
                "Qwen/Qwen2.5-32B-Instruct",
                "3",
                "0.85",
                "8192",
                "32768",
                "",
                "float16",
            ],
        ),
        (
            "consumer-ada",
            1,
            24,
            [
                "Qwen/Qwen2.5-7B-Instruct",
                "1",
                "0.80",
                "4096",
                "32768",
                "--enforce-eager",
                "",
            ],
        ),
        (
            "consumer-ada",
            1,
            48,
            [
                "Qwen/Qwen2.5-14B-Instruct",
                "1",
                "0.80",
                "4096",
                "32768",
                "--enforce-eager",
                "",
            ],
        ),
        (
            "ga102-dc",
            1,
            48,
            ["Qwen/Qwen2.5-14B-Instruct", "1", "0.90", "32768", "32768", "", ""],
        ),
        (
            "ampere-a30",
            1,
            24,
            ["Qwen/Qwen2.5-7B-Instruct", "1", "0.90", "32768", "32768", "", ""],
        ),
        (
            "consumer-ampere",
            1,
            24,
            [
                "Qwen/Qwen2.5-7B-Instruct",
                "1",
                "0.80",
                "4096",
                "32768",
                "--enforce-eager",
                "",
            ],
        ),
        (
            "turing",
            1,
            16,
            ["Qwen/Qwen3-14B-AWQ", "1", "0.90", "8192", "8192", "", "float16"],
        ),
    ],
)
def test_gpu_profile_matrix_selects_runnable_configuration(
    profile_bucket: str,
    gpu_count: int,
    gpu_vram_gb: int,
    expected: list[str],
) -> None:
    assert (
        _configured_profile(
            profile_bucket=profile_bucket,
            gpu_count=gpu_count,
            gpu_vram_gb=gpu_vram_gb,
        )
        == expected
    )


def test_explicit_vllm_overrides_still_win() -> None:
    assert _configured_profile(
        profile_bucket="ampere-a100",
        gpu_count=2,
        gpu_vram_gb=80,
        overrides={
            "AUTOVLLM_MODEL": "example/custom-model",
            "AUTOVLLM_TENSOR_PARALLEL": "1",
            "AUTOVLLM_GPU_MEM_UTIL": "0.73",
            "AUTOVLLM_MAX_MODEL_LEN": "1234",
            "AUTOVLLM_MAX_BATCHED_TOKENS": "5678",
            "AUTOVLLM_EXTRA_ARGS": "--enforce-eager",
            "AUTOVLLM_DTYPE": "bfloat16",
        },
    ) == [
        "example/custom-model",
        "1",
        "0.73",
        "1234",
        "5678",
        "--enforce-eager",
        "bfloat16",
    ]


def test_explicit_dtype_override_wins_over_v100_default() -> None:
    # V100 normally defaults to float16; an explicit override must replace it
    # rather than producing two conflicting --dtype arguments.
    assert _configured_profile(
        profile_bucket="volta",
        gpu_count=2,
        gpu_vram_gb=32,
        overrides={"AUTOVLLM_DTYPE": "bfloat16"},
    ) == [
        "Qwen/Qwen2.5-14B-Instruct",
        "2",
        "0.85",
        "8192",
        "32768",
        "",
        "bfloat16",
    ]


def test_invalid_dtype_override_is_rejected_by_allowlist() -> None:
    # "float16 --seed 0" contains whitespace and is not a supported dtype; it
    # must fail closed instead of becoming an additional vLLM flag.
    env = os.environ.copy()
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    env["AUTOVLLM_DTYPE"] = "float16 --seed 0"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
PROFILE_BUCKET='ampere-a100'
GPU_COUNT=1
GPU_VRAM_GB=80
configure_vllm_params
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "unsupported vLLM dtype 'float16 --seed 0'" in result.stderr


def test_extra_args_containing_dtype_is_rejected() -> None:
    env = os.environ.copy()
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    env["AUTOVLLM_EXTRA_ARGS"] = "--enforce-eager --dtype float16"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
PROFILE_BUCKET='ampere-a100'
GPU_COUNT=1
GPU_VRAM_GB=80
configure_vllm_params
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "--dtype inside AUTOVLLM_EXTRA_ARGS" in result.stderr


def test_shell_and_python_dtype_allowlists_are_identical() -> None:
    """The auto-vLLM start script and the API boundary must accept exactly the
    same --dtype values: the six that the pinned vLLM 0.26.0 accepts. The
    float8_* values are KV-cache dtype settings and must never drift into
    either allowlist."""
    from inference_proxy.models.node import SUPPORTED_VLLM_DTYPES

    match = re.search(
        r'^SUPPORTED_VLLM_DTYPES="([^"]*)"',
        START_SCRIPT.read_text(),
        re.MULTILINE,
    )
    assert match is not None
    assert set(match.group(1).split()) == SUPPORTED_VLLM_DTYPES
    assert {
        "auto",
        "half",
        "float16",
        "bfloat16",
        "float",
        "float32",
    } == SUPPORTED_VLLM_DTYPES


def test_reserved_vllm_names_are_ignored_without_compatibility_warnings() -> None:
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
PROFILE_BUCKET='ampere-a100'
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
    assert result.stderr == ""


def test_vllm_env_does_not_leak_script_params(tmp_path: Path) -> None:
    captured_env = tmp_path / "captured.env"
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
        gpu_count=4,
        device_count=2,
    )
    env.update(
        {
            "AUTOVLLM_CAPTURE_ENV": str(captured_env),
            "AUTOVLLM_ATTENTION_BACKEND": "FLASHINFER",
            "AUTOVLLM_API_PORT": "8123",
            "AUTOVLLM_MODEL": "example/model",
            "AUTOVLLM_TENSOR_PARALLEL": "1",
            "AUTOVLLM_GPU_MEM_UTIL": "0.73",
            "AUTOVLLM_MAX_MODEL_LEN": "1234",
            "AUTOVLLM_MAX_BATCHED_TOKENS": "5678",
            "AUTOVLLM_EXTRA_ARGS": "--enforce-eager",
            "AUTOVLLM_DTYPE": "bfloat16",
            "AUTOVLLM_GPU_DEVICES": "1,3",
            "AUTOVLLM_ENV_FILE": str(tmp_path / "vllm.env"),
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
            "AUTOVLLM_DTYPE",
            "AUTOVLLM_GPU_DEVICES",
            "AUTOVLLM_ENV_FILE",
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
            "AUTOVLLM_ATTENTION_BACKEND",
            "VLLM_PORT",
            "VLLM_TENSOR_PARALLEL",
            "VLLM_GPU_MEM_UTIL",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_MAX_BATCHED_TOKENS",
            "VLLM_EXTRA_ARGS",
        }
        assert forbidden.isdisjoint(captured)
        assert captured["CUDA_VISIBLE_DEVICES"] == "1,3"
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


def _run_preflight_command(
    tmp_path: Path,
    *,
    snapshot_state: str = "complete",
    model: str = "org/model",
    gpu_count: int = 1,
    device_count: int | None = None,
    gpu_devices: str | None = None,
    tensor_parallel: str | None = None,
    local_model_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
        gpu_count=gpu_count,
        device_count=device_count,
    )
    env["AUTOVLLM_MODEL"] = (
        str(local_model_dir) if local_model_dir is not None else model
    )
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = snapshot_state
    if gpu_devices is not None:
        env["AUTOVLLM_GPU_DEVICES"] = gpu_devices
    if tensor_parallel is not None:
        env["AUTOVLLM_TENSOR_PARALLEL"] = tensor_parallel
    model_path = str(local_model_dir) if local_model_dir is not None else model
    return subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH={shlex.quote(model_path)}
export MODEL_PATH
run_preflight
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def test_preflight_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    result = _run_preflight_command(tmp_path, snapshot_state="incomplete")

    assert result.returncode != 0
    assert "incomplete" in result.stdout + result.stderr


def test_preflight_accepts_verified_complete_snapshot(tmp_path: Path) -> None:
    result = _run_preflight_command(tmp_path, snapshot_state="complete")

    assert result.returncode == 0, result.stderr
    assert "verified complete" in result.stdout


def test_prestage_skips_download_when_snapshot_verified(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = "complete"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "verified complete" in result.stdout
    assert "Preparing model weights" not in result.stdout


def test_prestage_downloads_then_verifies_when_missing(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = "missing"
    env["AUTOVLLM_TEST_SNAPSHOT_FLAG"] = str(tmp_path / "downloaded.flag")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Preparing model weights" in result.stdout
    assert "staged and verified" in result.stdout


def test_preflight_warns_when_snapshot_probe_times_out(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_MODEL"] = "org/model"
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = "hung"
    env["AUTOVLLM_PROBE_TIMEOUT"] = "1"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH='org/model'
export MODEL_PATH
run_preflight
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "timed out" in result.stdout + result.stderr


def test_prestage_fails_when_verification_times_out(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = "hung"
    env["AUTOVLLM_PROBE_TIMEOUT"] = "1"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot verify cached model" in result.stderr


def test_preflight_snapshot_probe_failure_fails_closed(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_MODEL"] = "org/model"
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "fake-python",
        """#!/bin/bash
if [[ "$*" == *'torch.cuda.device_count()'* ]]; then
    echo 1
    exit 0
fi
if [[ "$*" == *'local_files_only'* ]]; then
    echo "import error" >&2
    exit 1
fi
exit 0
""",
    )
    env["AUTOVLLM_PYTHON"] = str(bin_dir / "fake-python")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH='org/model'
export MODEL_PATH
run_preflight
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "could not be verified" in result.stdout + result.stderr


def test_preflight_verifies_nfs_export_match_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    for expect_ok in (True, False):
        env = _script_environment(
            tmp_path,
            vllm_bin=tmp_path / "fake-vllm",
            process_log=tmp_path / "process.log",
        )
        env["AUTOVLLM_MODEL"] = "org/model"
        env["AUTOVLLM_NFS_EXPORT"] = "storage.example:/exports/huggingface"
        mounts = tmp_path / f"mounts-{expect_ok}"
        source = (
            "storage.example:/exports/huggingface"
            if expect_ok
            else "other.example:/exports/other"
        )
        mounts.write_text(
            f"{source} {env['AUTOVLLM_NFS_MOUNT_POINT']} nfs "
            "rw,vers=3,hard,proto=tcp,timeo=600,retrans=3,sec=sys 0 0\n"
        )
        env["AUTOVLLM_MOUNTS_FILE"] = str(mounts)
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH='org/model'
export MODEL_PATH
run_preflight
""",
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if expect_ok:
            assert result.returncode == 0, result.stderr
            assert "NFS mount verified" in result.stdout
        else:
            assert result.returncode != 0
            assert "verification failed" in result.stdout + result.stderr


def test_preflight_fails_on_proven_flashinfer_capacity_shortage(
    tmp_path: Path,
) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_MODEL"] = "org/model"
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "df",
        """#!/bin/bash
if [[ "$1" == "--output=target" ]]; then
    echo 'Mounted on'
    echo '/fixture'
    exit 0
fi
echo 'Avail'
echo '0'
exit 0
""",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH='org/model'
export MODEL_PATH
run_preflight
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "Insufficient free space" in result.stdout + result.stderr


def test_prestage_fails_when_download_is_not_verifiable(tmp_path: Path) -> None:
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_TEST_SNAPSHOT_STATE"] = "download_fails"
    env["AUTOVLLM_TEST_SNAPSHOT_FLAG"] = str(tmp_path / "downloaded.flag")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "did not produce a verifiable snapshot" in result.stderr


def test_prestage_verifies_against_hub_cache_dir(tmp_path: Path) -> None:
    """Verification must use the hub cache root (<mount>/hub), not the mount."""
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    cache = Path(env["AUTOVLLM_NFS_MOUNT_POINT"]) / "hub"
    repo = cache / "models--org--model"
    commit = "a" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "snapshots" / commit).mkdir(parents=True)
    (repo / "snapshots" / commit / "config.json").write_text("{}")
    (repo / "trees").mkdir()
    (repo / "trees" / f"{commit}.json").write_text("{}")
    observer = tmp_path / "probed-cache-dir"
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "fake-python",
        f"""#!/bin/bash
if [[ "$*" == *'torch.cuda.device_count()'* ]]; then
    echo 1
    exit 0
fi
if [[ "$*" == *'local_files_only'* ]]; then
    printf '%s\\n' "$4" >> '{observer}'
    if [[ -d "$4/models--org--model/snapshots" ]]; then
        echo complete
        exit 0
    fi
    echo missing >&2
    exit 3
fi
exit 0
""",
    )
    env["AUTOVLLM_PYTHON"] = str(bin_dir / "fake-python")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "verified complete" in result.stdout
    assert observer.read_text().splitlines() == [str(cache)]


def test_preflight_verifies_against_hub_cache_dir(tmp_path: Path) -> None:
    """Preflight resolves the repo under <hub>/models--<slug> too."""
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_MODEL"] = "org/model"
    cache = Path(env["AUTOVLLM_NFS_MOUNT_POINT"]) / "hub"
    repo = cache / "models--org--model"
    commit = "a" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "snapshots" / commit).mkdir(parents=True)
    (repo / "snapshots" / commit / "config.json").write_text("{}")
    (repo / "trees").mkdir()
    (repo / "trees" / f"{commit}.json").write_text("{}")
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "fake-python",
        """#!/bin/bash
if [[ "$*" == *'torch.cuda.device_count()'* ]]; then
    echo 1
    exit 0
fi
if [[ "$*" == *'local_files_only'* ]]; then
    if [[ -d "$4/models--org--model/snapshots" ]]; then
        echo complete
        exit 0
    fi
    echo missing >&2
    exit 3
fi
exit 0
""",
    )
    env["AUTOVLLM_PYTHON"] = str(bin_dir / "fake-python")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
MODEL_PATH='org/model'
export MODEL_PATH
run_preflight
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "verified complete" in result.stdout


def test_prestage_repairs_snapshot_without_tree_manifest(tmp_path: Path) -> None:
    """Without a cached tree manifest a snapshot is unverifiable: preparation
    must run instead of selecting the offline path. huggingface_hub 1.26
    returns an existing snapshot dir as-is when the manifest is absent."""
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    cache = Path(env["AUTOVLLM_NFS_MOUNT_POINT"]) / "hub"
    repo = cache / "models--org--model"
    commit = "a" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "snapshots" / commit).mkdir(parents=True)
    (repo / "snapshots" / commit / "config.json").write_text("{}")
    # No trees/<commit>.json on purpose.
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "fake-python",
        f"""#!/bin/bash
if [[ "$*" == *'local_files_only'* ]]; then
    echo complete
    exit 0
fi
if [[ "$*" == *'snapshot_download'* ]]; then
    mkdir -p '{repo}/trees'
    printf '{{}}\\n' > '{repo}/trees/{commit}.json'
    exit 0
fi
exit 0
""",
    )
    env["AUTOVLLM_PYTHON"] = str(bin_dir / "fake-python")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
prestage_model_weights 'org/model'
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Preparing model weights" in result.stdout
    assert "staged and verified" in result.stdout


def _write_model_config(model_dir: Path, heads: int, kv: int) -> Path:
    model_dir.mkdir(exist_ok=True)
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps({"num_attention_heads": heads, "num_key_value_heads": kv})
    )
    return config_path


def test_preflight_allows_tp1_on_multigpu_t4_default(tmp_path: Path) -> None:
    """The reported false rejection: 4 visible GPUs with default TP=1 passes."""
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=4,
        tensor_parallel="1",
    )

    assert result.returncode == 0, result.stderr
    assert "physical=4 (nvidia-smi), CUDA-visible=4, allocated=1" in result.stdout
    assert "using 1 of 4 CUDA-visible device(s)" in result.stdout


def test_preflight_accepts_explicit_subset(tmp_path: Path) -> None:
    """A single selected device on a 4-GPU host is a valid TP=1 launch."""
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=1,
        gpu_devices="1",
        tensor_parallel="1",
    )

    assert result.returncode == 0, result.stderr
    assert "CUDA-visible=1" in result.stdout


def test_preflight_exports_cuda_visible_devices(tmp_path: Path) -> None:
    """The selected subset is exported so torch and the engine agree."""
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
        gpu_count=4,
        device_count=1,
    )
    env["AUTOVLLM_GPU_DEVICES"] = "1"

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
echo "CVD=${{CUDA_VISIBLE_DEVICES:-unset}}"
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "CVD=1" in result.stdout


def test_preflight_rejects_insufficient_devices(tmp_path: Path) -> None:
    """TP larger than the CUDA-visible subset fails with an actionable error."""
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=2,
        gpu_devices="1,2",
        tensor_parallel="4",
    )

    assert result.returncode != 0
    assert "tensor-parallel-size 4 exceeds 2 CUDA-visible device(s) of 4 physical" in (
        result.stdout + result.stderr
    )


def test_preflight_rejects_device_index_out_of_range(tmp_path: Path) -> None:
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=1,
        gpu_devices="7",
        tensor_parallel="1",
    )

    assert result.returncode != 0
    assert "device 7 not present" in result.stdout + result.stderr


def test_preflight_rejects_malformed_device_list(tmp_path: Path) -> None:
    for malformed in ("abc", "1,,2", "1,", "-1"):
        result = _run_preflight_command(
            tmp_path,
            model="org/model",
            gpu_count=4,
            device_count=1,
            gpu_devices=malformed,
            tensor_parallel="1",
        )

        assert result.returncode != 0, malformed
        assert "not a comma-separated list of numeric indices" in (
            result.stdout + result.stderr
        )


def test_preflight_rejects_duplicate_devices(tmp_path: Path) -> None:
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=1,
        gpu_devices="1,1",
        tensor_parallel="1",
    )

    assert result.returncode != 0
    assert "listed more than once" in result.stdout + result.stderr


def test_preflight_rejects_incompatible_attention_heads(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model_config(model_dir, heads=28, kv=4)

    result = _run_preflight_command(
        tmp_path,
        gpu_count=8,
        device_count=8,
        tensor_parallel="8",
        local_model_dir=model_dir,
    )

    assert result.returncode != 0
    assert "28 attention head(s), not divisible by tensor-parallel-size 8" in (
        result.stdout + result.stderr
    )


def test_preflight_accepts_divisible_topology_and_kv_replication(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_model_config(model_dir, heads=28, kv=4)

    divisible = _run_preflight_command(
        tmp_path,
        gpu_count=4,
        device_count=4,
        tensor_parallel="4",
        local_model_dir=model_dir,
    )
    assert divisible.returncode == 0, divisible.stderr
    assert "Model topology verified for TP=4" in divisible.stdout

    replication_dir = tmp_path / "model-replicated"
    _write_model_config(replication_dir, heads=64, kv=8)
    replicated = _run_preflight_command(
        tmp_path,
        gpu_count=16,
        device_count=16,
        tensor_parallel="16",
        local_model_dir=replication_dir,
    )
    assert replicated.returncode == 0, replicated.stderr


def test_preflight_warns_when_topology_unverifiable(tmp_path: Path) -> None:
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=4,
        tensor_parallel="4",
    )

    assert result.returncode == 0, result.stderr
    assert "Model topology not verifiable pre-launch" in result.stdout


def test_preflight_does_not_eval_model_metadata(tmp_path: Path) -> None:
    """String-valued model metadata must not reach shell arithmetic: command
    substitution embedded in config.json stays inert. A nonnumeric heads
    value is unverifiable; a nonnumeric KV value falls back to heads like a
    missing one, neither evaluating the string."""
    marker = tmp_path / "pwned"
    heads_dir = tmp_path / "model-heads"
    heads_dir.mkdir()
    (heads_dir / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": f"1[$(touch {marker})]",
                "num_key_value_heads": 4,
            }
        )
    )
    heads_result = _run_preflight_command(
        tmp_path,
        gpu_count=2,
        device_count=2,
        tensor_parallel="2",
        local_model_dir=heads_dir,
    )

    assert heads_result.returncode == 0, heads_result.stderr
    assert not marker.exists()
    assert "Model topology not verifiable pre-launch" in heads_result.stdout

    kv_dir = tmp_path / "model-kv"
    kv_dir.mkdir()
    (kv_dir / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 28,
                "num_key_value_heads": f"1[$(touch {marker})]",
            }
        )
    )
    kv_result = _run_preflight_command(
        tmp_path,
        gpu_count=2,
        device_count=2,
        tensor_parallel="2",
        local_model_dir=kv_dir,
    )

    assert kv_result.returncode == 0, kv_result.stderr
    assert not marker.exists()
    assert "Model topology verified for TP=2" in kv_result.stdout


def test_preflight_ignores_unrelated_cached_revision(tmp_path: Path) -> None:
    """A cached snapshot for an unrelated revision must not judge the
    requested revision; a pending main revision reports unverifiable instead
    of being rejected by another branch's config."""
    cache_root = tmp_path / "nfs-cache" / "hub" / "models--org--model"
    (cache_root / "refs").mkdir(parents=True)
    (cache_root / "snapshots" / "oldrev").mkdir(parents=True)
    (cache_root / "refs" / "main").write_text("mainrev\n")
    (cache_root / "snapshots" / "oldrev" / "config.json").write_text(
        json.dumps({"num_attention_heads": 28, "num_key_value_heads": 8})
    )

    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=8,
        device_count=8,
        tensor_parallel="8",
    )

    assert result.returncode == 0, result.stderr
    assert "Model topology not verifiable pre-launch" in result.stdout


def test_preflight_defaults_tp_to_selected_subset(tmp_path: Path) -> None:
    """Standalone preflight must derive the launcher's default TP (the
    selected device count) instead of the physical inventory, so a valid
    subset passes the dry run."""
    result = _run_preflight_command(
        tmp_path,
        model="org/model",
        gpu_count=4,
        device_count=2,
        gpu_devices="0,2",
    )

    assert result.returncode == 0, result.stderr
    assert "physical=4 (nvidia-smi), CUDA-visible=2, allocated=2" in result.stdout


def test_failed_launch_preserves_previous_env_file(tmp_path: Path) -> None:
    """A replacement that exits during startup must not replace the saved
    working settings; a later restart keeps the working configuration."""
    env_file = tmp_path / "vllm.env"
    env_file.write_text("AUTOVLLM_MODEL=old/model\nAUTOVLLM_GPU_DEVICES=0,2\n")
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "failing-vllm"
    _write_executable(
        vllm_bin,
        "#!/bin/bash\necho 'CUDA initialization failed' >&2\nexit 7\n",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
        gpu_count=4,
        device_count=2,
    )
    env.update(
        {
            "AUTOVLLM_MODEL": "new/model",
            "AUTOVLLM_GPU_DEVICES": "1,3",
            "AUTOVLLM_ENV_FILE": str(env_file),
        }
    )

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "exited during startup" in result.stderr
    assert (
        env_file.read_text() == "AUTOVLLM_MODEL=old/model\nAUTOVLLM_GPU_DEVICES=0,2\n"
    )


def test_profile_sizes_model_by_effective_tensor_parallel() -> None:
    """TP=1 on a 4x H100 host must pick a single-card model, not the 72B
    profile that would OOM on the allocated device."""
    result = _configured_profile(
        profile_bucket="hopper",
        gpu_count=4,
        gpu_vram_gb=80,
        overrides={"AUTOVLLM_TENSOR_PARALLEL": "1"},
    )

    assert result[0] == "Qwen/Qwen2.5-32B-Instruct"
    assert result[1] == "1"


def test_start_vllm_persists_effective_launch_env(tmp_path: Path) -> None:
    env_file = tmp_path / "vllm.env"
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
        gpu_count=4,
        device_count=1,
    )
    env.update(
        {
            "AUTOVLLM_MODEL": "example/model",
            "AUTOVLLM_GPU_DEVICES": "1",
            "AUTOVLLM_ENV_FILE": str(env_file),
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
PROFILE_BUCKET='ampere-a100'
GPU_MODEL='NVIDIA A100'
GPU_COUNT=4
GPU_DEVICE_COUNT=1
GPU_VRAM_GB=80
configure_vllm_params
persist_vllm_env
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    content = env_file.read_text()
    assert "AUTOVLLM_TENSOR_PARALLEL=1" in content
    assert "AUTOVLLM_MODEL=example/model" in content
    assert "AUTOVLLM_GPU_DEVICES=1" in content


def test_persist_vllm_env_rewrites_stale_device_keys(tmp_path: Path) -> None:
    env_file = tmp_path / "vllm.env"
    env_file.write_text("AUTOVLLM_GPU_DEVICES=0,2\n")
    env = _script_environment(
        tmp_path,
        vllm_bin=tmp_path / "fake-vllm",
        process_log=tmp_path / "process.log",
    )
    env["AUTOVLLM_ENV_FILE"] = str(env_file)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source <(sed '/^main$/d' {START_SCRIPT!s})
PROFILE_BUCKET='ampere-a100'
GPU_MODEL='NVIDIA A100'
GPU_COUNT=2
GPU_DEVICE_COUNT=2
GPU_VRAM_GB=80
configure_vllm_params
persist_vllm_env
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "AUTOVLLM_GPU_DEVICES" not in env_file.read_text()


def test_main_rechecks_topology_after_download(tmp_path: Path) -> None:
    """The post-prestage hard check rejects an incompatible model even when
    preflight could only WARN before the weights were downloaded."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
echo launched >> "$AUTOVLLM_TEST_LOG"
exit 0
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
        gpu_count=8,
        device_count=8,
    )
    env.update(
        {
            "AUTOVLLM_MODEL": "org/model",
            "AUTOVLLM_TENSOR_PARALLEL": "8",
            "AUTOVLLM_TEST_SNAPSHOT_STATE": "missing",
            "AUTOVLLM_TEST_SNAPSHOT_FLAG": str(tmp_path / "downloaded.flag"),
            "AUTOVLLM_TEST_SNAPSHOT_CONFIG_HEADS": "28",
            "AUTOVLLM_TEST_SNAPSHOT_CONFIG_KV": "4",
            "AUTOVLLM_ENV_FILE": str(tmp_path / "vllm.env"),
        }
    )

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "28 attention head(s), not divisible by tensor-parallel-size 8" in (
        result.stdout + result.stderr
    )
    assert not process_log.exists()


def test_launcher_rejects_hidden_devices_with_ambient_cvd(tmp_path: Path) -> None:
    """Ambient CUDA_VISIBLE_DEVICES (no AUTOVLLM_GPU_DEVICES) hides devices
    from torch; the physical count must not silently override that."""
    process_log = tmp_path / "process.log"
    vllm_bin = tmp_path / "fake-vllm"
    _write_executable(vllm_bin, "#!/bin/bash\nexit 0\n")
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
        gpu_count=4,
        device_count=2,
    )
    env["CUDA_VISIBLE_DEVICES"] = "0,2"

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "tensor-parallel-size 4 exceeds 2 CUDA-visible device(s) of 4 physical" in (
        result.stdout + result.stderr
    )


def test_launcher_passes_valid_subset_through_full_main(tmp_path: Path) -> None:
    """A valid TP=2 subset on a 4-GPU host starts through the real main()."""
    process_log = tmp_path / "process.log"
    captured_env = tmp_path / "vllm.env"
    vllm_bin = tmp_path / "capturing-vllm"
    _write_executable(
        vllm_bin,
        """#!/bin/bash
env | sort > "$AUTOVLLM_CAPTURE_ENV"
echo "start:$2" >> "$AUTOVLLM_TEST_LOG"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    env = _script_environment(
        tmp_path,
        vllm_bin=vllm_bin,
        process_log=process_log,
        gpu_count=4,
        device_count=2,
    )
    env.update(
        {
            "AUTOVLLM_CAPTURE_ENV": str(captured_env),
            "AUTOVLLM_MODEL": "example/model",
            "AUTOVLLM_GPU_DEVICES": "0,2",
            "AUTOVLLM_TENSOR_PARALLEL": "2",
            "AUTOVLLM_ENV_FILE": str(tmp_path / "persisted.env"),
        }
    )

    try:
        result = subprocess.run(
            ["bash", str(START_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        _wait_for_line(process_log, "start:example/model")
        captured = {
            line.partition("=")[0]: line.partition("=")[2]
            for line in captured_env.read_text().splitlines()
        }
        assert captured["CUDA_VISIBLE_DEVICES"] == "0,2"
        assert "AUTOVLLM_GPU_DEVICES" not in captured
        assert "physical=4 (nvidia-smi), CUDA-visible=2, allocated=2" in (result.stdout)
    finally:
        subprocess.run(
            ["bash", str(STOP_SCRIPT), "--force"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
