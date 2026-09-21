"""Behavioral tests for measured runtime profile selection and validation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(os.environ.get("AUTOVLLM_TEST_SCRIPT_ROOT", REPO_ROOT))
PROFILES = SCRIPT_ROOT / "common" / "profiles.sh"
SETUP_BASE = SCRIPT_ROOT / "common" / "setup-base.sh"
VLLM_SETUP = SCRIPT_ROOT / "auto-vllm" / "setup.sh"
VLLM_START = SCRIPT_ROOT / "auto-vllm" / "start-vllm.sh"
LLAMACPP_SETUP = SCRIPT_ROOT / "auto-llamacpp" / "setup.sh"
LLAMACPP_START = SCRIPT_ROOT / "auto-llamacpp" / "start-llamacpp.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _hardware_env(
    tmp_path: Path,
    *,
    cap: str = "8.0",
    vram_mb: int = 40960,
    count: int = 1,
    nvswitch: bool = False,
    os_id: str = "rhel",
    os_ver: str = "9.5",
    glibc: str = "2.34",
    cap_alt: str = "9.0",
    vram_mb_alt: int = 81920,
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gpus = "".join(f"GPU {i}: stub\n" for i in range(count))
    if count > 1:
        # Emulate per-card answers so -i <idx> selects the alt card (used to
        # prove subset selection sizes against the first SELECTED device).
        nvidia_body = f"""#!/bin/bash
index="0"
prev=""
for arg in "$@"; do
    if [ "$prev" = "-i" ]; then index="$arg"; fi
    prev="$arg"
done
if [ "$index" = "1" ]; then
    cap="{cap_alt}"; vram="{vram_mb_alt}"
else
    cap="{cap}"; vram="{vram_mb}"
fi
case "$*" in
  *"--list-gpus"*) printf '{gpus}' ;;
  *"--query-gpu=name"*) echo "NVIDIA stub" ;;
  *"--query-gpu=memory.total"*) echo "$vram" ;;
  *"--query-gpu=compute_cap"*) echo "$cap" ;;
  *"--query-gpu=driver_version"*) echo "580.126.09" ;;
  *"-q"*) echo "Fabric" ; echo "    State: Completed" ;;
esac
"""
    else:
        nvidia_body = f"""#!/bin/bash
case "$*" in
  *"--list-gpus"*) printf '{gpus}' ;;
  *"--query-gpu=name"*) echo "NVIDIA stub" ;;
  *"--query-gpu=memory.total"*) echo "{vram_mb}" ;;
  *"--query-gpu=compute_cap"*) echo "{cap}" ;;
  *"--query-gpu=driver_version"*) echo "580.126.09" ;;
  *"-q"*) echo "Fabric" ; echo "    State: Completed" ;;
esac
"""
    _write_executable(bin_dir / "nvidia-smi", nvidia_body)
    lspci_out = "NVIDIA Corporation NVSwitch\n" if nvswitch else ""
    _write_executable(bin_dir / "lspci", f"#!/bin/bash\necho '{lspci_out}'\n")
    _write_executable(bin_dir / "ldd", f"#!/bin/bash\necho 'ldd (GNU libc) {glibc}'\n")
    (tmp_path / "os-release").write_text(f"ID={os_id}\nVERSION_ID={os_ver}\n")
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PROFILE_OS_RELEASE"] = str(tmp_path / "os-release")
    return env


def _source_and_call(
    script: Path,
    command: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {script!s}\n{command}",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


@pytest.mark.parametrize(
    ("cap", "vram_mb", "expected_bucket"),
    [
        ("9.0", 81920, "hopper"),
        ("9.0", 98304, "hopper"),
        ("8.0", 40960, "ampere-a100"),
        ("8.0", 81920, "ampere-a100"),
        ("8.0", 24576, "ampere-a30"),
        ("8.6", 49152, "ga102-dc"),
        ("8.6", 24576, "consumer-ampere"),
        ("8.9", 12288, "consumer-ada"),
        ("7.5", 16384, "turing"),
        ("7.5", 15079, "turing"),
        ("7.5", 12288, "consumer-turing"),
    ],
)
def test_selection_uses_measured_capability(
    tmp_path: Path, cap: str, vram_mb: int, expected_bucket: str
) -> None:
    env = _hardware_env(tmp_path, cap=cap, vram_mb=vram_mb)
    result = _source_and_call(
        PROFILES,
        'select_runtime_profile vllm\necho "$PROFILE_BUCKET|$PROFILE_REASON"',
        env,
    )
    assert result.returncode == 0, result.stderr
    bucket, reason = result.stdout.splitlines()[-1].split("|", 1)
    assert bucket == expected_bucket
    assert f"sm={cap}" in reason
    assert f"vram={(vram_mb + 512) // 1024}GB" in reason
    assert "engine=vllm" in reason
    assert "cpu-only" not in reason


def test_selection_uses_first_selected_device_subset(tmp_path: Path) -> None:
    # Device 0 is ampere (8.0/40GB), device 1 is hopper (9.0/80GB). Pinning
    # AUTOVLLM_GPU_DEVICES=1 must size against device 1, not device 0.
    env = _hardware_env(
        tmp_path,
        count=2,
        cap="8.0",
        vram_mb=40960,
        cap_alt="9.0",
        vram_mb_alt=81920,
    )
    env["GPU_DEVICES_OVERRIDE"] = "1"
    result = _source_and_call(
        PROFILES,
        'select_runtime_profile vllm\necho "$PROFILE_BUCKET"',
        env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "hopper"


def test_volta_llamacpp_pins_cuda12_toolkit(tmp_path: Path) -> None:
    # CUDA 13.0 dropped Volta (SM70); llama.cpp is source-built so it pins the
    # last 12.x toolkit instead of the 13.0 default.
    env = _hardware_env(tmp_path, cap="7.0", vram_mb=32768)
    result = _source_and_call(
        PROFILES,
        "select_runtime_profile llamacpp\n"
        'echo "$PROFILE_BUCKET|$PROFILE_CUDA_TOOLKIT_VERSION"',
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "PROFILE:select:llamacpp-volta" in result.stdout
    assert result.stdout.splitlines()[-1] == "volta|12.9"


def test_volta_vllm_rejected_for_cuda13_stack(tmp_path: Path) -> None:
    # The frozen vLLM stack is torch 2.11.0 CUDA-13.0, which cannot target
    # Volta; reject up front instead of failing the CUDA probe later.
    env = _hardware_env(tmp_path, cap="7.0", vram_mb=32768)
    result = _source_and_call(PROFILES, "select_runtime_profile vllm", env)
    assert result.returncode == 3
    assert "[REJECT:unsupported_hardware:" in result.stderr
    assert "sm=7.0" in result.stderr
    assert "CUDA 13.0" in result.stderr


def test_unknown_architecture_rejected_with_actionable_message(
    tmp_path: Path,
) -> None:
    env = _hardware_env(tmp_path, cap="6.1", vram_mb=8192)
    result = _source_and_call(PROFILES, "select_runtime_profile vllm", env)
    assert result.returncode == 3
    assert "[REJECT:unsupported_hardware:" in result.stderr
    assert "sm=6.1" in result.stderr
    assert "supported:" in result.stderr


def test_insufficient_vram_for_hopper_is_rejected(tmp_path: Path) -> None:
    env = _hardware_env(tmp_path, cap="9.0", vram_mb=40960)
    result = _source_and_call(PROFILES, "select_runtime_profile vllm", env)
    assert result.returncode == 3
    assert "no tested profile" in result.stderr


@pytest.mark.parametrize(
    ("os_id", "os_ver", "architecture", "glibc", "engine", "expected_rc"),
    [
        ("rhel", "9.5", "x86_64", "2.34", "vllm", 0),
        ("rhel", "8.10", "x86_64", "2.34", "vllm", 3),
        ("centos", "9.0", "x86_64", "2.34", "vllm", 3),
        ("rhel", "9.5", "aarch64", "2.34", "vllm", 3),
        ("rhel", "9.5", "x86_64", "2.28", "vllm", 3),
        ("rhel", "9.5", "x86_64", "2.28", "llamacpp", 0),
    ],
)
def test_os_abi_gate(
    tmp_path: Path,
    os_id: str,
    os_ver: str,
    architecture: str,
    glibc: str,
    engine: str,
    expected_rc: int,
) -> None:
    env = _hardware_env(tmp_path, os_id=os_id, os_ver=os_ver, glibc=glibc)
    fake_uname = Path(env["PATH"].split(":")[0]) / "uname"
    _write_executable(fake_uname, f"#!/bin/bash\necho '{architecture}'\n")
    result = _source_and_call(PROFILES, f"select_runtime_profile {engine}", env)
    assert result.returncode == expected_rc, result.stderr
    if expected_rc == 3:
        assert "[REJECT:unsupported_hardware:" in result.stderr


def test_match_requires_os_abi_and_engine_specific_glibc(tmp_path: Path) -> None:
    # llama.cpp is source-built: its profile tolerates older glibc, vLLM does
    # not (manylinux_2_34 wheel ABI).
    env = _hardware_env(tmp_path, glibc="2.28")
    result = _source_and_call(PROFILES, "select_runtime_profile llamacpp", env)
    assert result.returncode == 0, result.stderr


def _toolkit_env(tmp_path: Path, *, nvcc_version: str | None) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "ops.log"
    _write_executable(
        bin_dir / "sudo",
        f"""#!/bin/bash
echo "$*" >> "{log}"
exit 0
""",
    )
    if nvcc_version is not None:
        _write_executable(
            bin_dir / "nvcc",
            f"""#!/bin/bash
if [[ "$1" == "--version" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "V{nvcc_version}.123"
fi
""",
        )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CUDA_NVCC"] = str(bin_dir / "nvcc") if nvcc_version else ""
    env["AUTOVLLM_TEST_LOG"] = str(log)
    return env


def test_toolkit_accepts_exact_profile_version(tmp_path: Path) -> None:
    env = _toolkit_env(tmp_path, nvcc_version="13.0")
    result = _source_and_call(
        SETUP_BASE,
        "install_cuda_toolkit\necho DONE",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "already installed" in result.stdout
    assert "dnf" not in result.stdout


def test_toolkit_installs_exact_when_nvcc_mismatches(tmp_path: Path) -> None:
    env = _toolkit_env(tmp_path, nvcc_version="12.4")
    result = _source_and_call(
        SETUP_BASE,
        "install_cuda_toolkit\necho DONE",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "installing exact 13.0" in result.stdout
    assert (
        "dnf -y install cuda-toolkit-13-0" in Path(env["AUTOVLLM_TEST_LOG"]).read_text()
    )


def test_toolkit_installs_exact_when_absent(tmp_path: Path) -> None:
    env = _toolkit_env(tmp_path, nvcc_version=None)
    result = _source_and_call(
        SETUP_BASE,
        "install_cuda_toolkit\necho DONE",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "cuda-toolkit-13-0" in Path(env["AUTOVLLM_TEST_LOG"]).read_text()


def _proof_env(
    tmp_path: Path, *, probe_ok: bool | None = None, compile_ok: bool = True
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    probe_body = (
        "printf '#!/bin/bash\\nexit 3\\n'"
        if probe_ok is False
        else "printf '#!/bin/bash\\nexit 0\\n'"
    )
    compile_line = (
        'echo "nvcc fatal: unsupported host compiler" >&2\nexit 1'
        if not compile_ok
        else f'{probe_body} > "$out"\nchmod +x "$out"'
    )
    _write_executable(
        bin_dir / "nvcc",
        f"""#!/bin/bash
if [[ "$1" == "--version" ]]; then
    echo "V13.0.123"
    exit 0
fi
out=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then out="$2"; shift 2; else shift; fi
done
{compile_line}
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CUDA_NVCC"] = str(bin_dir / "nvcc")
    return env


def test_verify_cuda_execution_compiles_and_runs_probe(tmp_path: Path) -> None:
    env = _proof_env(tmp_path)
    result = _source_and_call(
        SETUP_BASE,
        "verify_cuda_execution\necho DONE",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA execution verified" in result.stdout


def test_cuda_probe_compiles_with_real_nvcc(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler unavailable")
    env = _proof_env(tmp_path)
    # Compile the actual generated source without needing a GPU to execute it.
    _write_executable(
        Path(env["CUDA_NVCC"]),
        f'''#!/bin/bash
set -e
"{nvcc}" -c "$3" -o "$2.o"
printf '#!/bin/bash\\nexit 0\\n' > "$2"
chmod +x "$2"
''',
    )
    env["INSTALL_TMP_DIR"] = str(tmp_path)
    result = _source_and_call(SETUP_BASE, "verify_cuda_execution", env)
    assert result.returncode == 0, result.stderr
    assert "CUDA execution verified" in result.stdout
    assert not list(tmp_path.glob("cuda-probe.*"))


def test_verify_cuda_execution_fails_closed_when_probe_fails(
    tmp_path: Path,
) -> None:
    env = _proof_env(tmp_path, probe_ok=False)
    result = _source_and_call(
        SETUP_BASE,
        'verify_cuda_execution\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "FATAL: CUDA execution probe failed" in result.stderr


def test_verify_cuda_execution_requires_nvcc(tmp_path: Path) -> None:
    env = _hardware_env(tmp_path)
    bin_dir = env["PATH"].split(":")[0]
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"  # no nvcc anywhere
    env["CUDA_NVCC"] = ""
    result = _source_and_call(
        SETUP_BASE,
        'verify_cuda_execution\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "nvcc not found" in result.stderr


def test_fabric_helpers_noop_without_nvswitch(tmp_path: Path) -> None:
    env = _hardware_env(tmp_path, nvswitch=False)
    result = _source_and_call(
        SETUP_BASE,
        "wait_nvswitch_fabric && fabric_ready && echo READY",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "READY" in result.stdout


def test_verify_cuda_execution_fails_on_compile_error(tmp_path: Path) -> None:
    env = _proof_env(tmp_path, compile_ok=False)
    env["INSTALL_TMP_DIR"] = str(tmp_path)
    result = _source_and_call(
        SETUP_BASE,
        'verify_cuda_execution\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "nvcc fatal: unsupported host compiler" in result.stderr
    assert "FATAL: nvcc failed to compile" in result.stderr
    assert not list(tmp_path.glob("cuda-probe.*"))


def test_toolkit_install_failure_is_transient(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "ops.log"
    _write_executable(
        bin_dir / "sudo",
        f"""#!/bin/bash
echo "$*" >> "{log}"
exit 1
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CUDA_NVCC"] = ""
    result = _source_and_call(
        SETUP_BASE,
        'install_cuda_toolkit\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "FATAL: could not install cuda-toolkit-13-0" in result.stderr


def _fabric_env(
    tmp_path: Path,
    *,
    fm_version: str = "580.126.09",
    fm_installed: bool = True,
    service_active: bool = True,
    service_type: str = "simple",
    fabric_state: str = "Completed",
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_executable(
        bin_dir / "nvidia-smi",
        f"""#!/bin/bash
case "$*" in
  *"--query-gpu=name"*) echo "NVIDIA stub" ;;
  *"--list-gpus"*) echo "GPU 0: stub" ;;
  *"--query-gpu=memory.total"*) echo "81920" ;;
  *"--query-gpu=compute_cap"*) echo "9.0" ;;
  *"--query-gpu=driver_version"*) echo "580.126.09" ;;
  *"-q"*) echo "Fabric" ; echo "    State: {fabric_state}" ;;
esac
""",
    )
    _write_executable(
        bin_dir / "lspci", "#!/bin/bash\necho 'NVIDIA Corporation NVSwitch'\n"
    )
    rpm_body = (
        f'echo "nvidia-fabricmanager-{fm_version}-1"\nexit 0'
        if fm_installed
        else "exit 1"
    )
    _write_executable(
        bin_dir / "rpm",
        f"""#!/bin/bash
if [[ "$*" == *"--qf"* ]]; then
    {f'echo "{fm_version}"' if fm_installed else "exit 1"}
    exit 0
fi
if [[ "$*" == *"-q"* ]]; then
    {rpm_body}
fi
exit 1
""",
    )
    _write_executable(
        bin_dir / "systemctl",
        f"""#!/bin/bash
if [[ "$*" == *"is-active"* ]]; then
    exit {0 if service_active else 1}
fi
if [[ "$*" == *"-p Type"* ]]; then
    echo "{service_type}"
elif [[ "$*" == *"-p ActiveState"* ]]; then
    echo "active"
fi
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env


def test_fabric_ready_rejects_version_mismatch(tmp_path: Path) -> None:
    env = _fabric_env(tmp_path, fm_version="579.0.0")
    result = _source_and_call(
        SETUP_BASE,
        'fabric_ready\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "version mismatch" in result.stderr


def test_fabric_ready_reports_missing_manager(tmp_path: Path) -> None:
    env = _fabric_env(tmp_path, fm_installed=False)
    result = _source_and_call(
        SETUP_BASE,
        'fabric_ready\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "nvidia-fabricmanager not installed" in result.stderr


def test_fabric_ready_accepts_oneshot_exit(tmp_path: Path) -> None:
    env = _fabric_env(tmp_path, service_type="oneshot", fabric_state="N/A")
    result = _source_and_call(
        SETUP_BASE,
        'fabric_ready\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=0", result.stderr


def test_wait_fabric_fails_fast_on_version_mismatch(tmp_path: Path) -> None:
    env = _fabric_env(tmp_path, fm_version="579.0.0")
    result = _source_and_call(
        SETUP_BASE,
        'wait_nvswitch_fabric\necho "rc=$?"',
        env,
    )
    assert result.stdout.splitlines()[-1] == "rc=1"
    assert "version mismatch" in result.stderr
    assert "did not complete within" not in result.stderr


def test_wait_fabric_returns_immediately_when_training_complete(
    tmp_path: Path,
) -> None:
    env = _fabric_env(tmp_path)
    result = _source_and_call(
        SETUP_BASE,
        "wait_nvswitch_fabric && echo READY",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "READY" in result.stdout


def _setup_main_order(script: Path, engine: str, env: dict[str, str]) -> list[str]:
    """Run the setup main with every real install step stubbed so the wiring
    order (profile select before toolkit/proof/fabric) is observable."""
    return subprocess.run(
        [
            "bash",
            "-c",
            f"""
source {script!s}
NFS_EXPORT=storage.example.com:/exports/hf
require_sha256() {{ :; }}
reject_retired_flashinfer_index() {{ :; }}
step() {{ echo "STEP:$1"; }}
soft_step() {{ echo "STEP:$1"; }}
select_runtime_profile() {{ echo "PROFILE:$1"; return 0; }}
install_nvidia_driver() {{ :; }}
install_cuda_toolkit() {{ :; }}
verify_cuda_execution() {{ :; }}
ensure_fabric_manager() {{ :; }}
install_vllm() {{ :; }}
install_vllm_unit() {{ :; }}
install_llamacpp() {{ :; }}
mount_nfs_cache() {{ :; }}
configure_firewall() {{ :; }}
install_llmfit() {{ :; }}
main
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    ).stdout.splitlines()


def test_vllm_setup_selects_profile_before_toolkit(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    lines = _setup_main_order(VLLM_SETUP, "vllm", env)
    profile_at = lines.index("PROFILE:vllm")
    toolkit_at = lines.index("STEP:cuda_toolkit")
    fabric_at = lines.index("STEP:fabric_manager")
    proof_at = lines.index("STEP:cuda_proof")
    assert profile_at < toolkit_at < fabric_at < proof_at


def test_llamacpp_setup_selects_profile_and_prepares_fabric(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    lines = _setup_main_order(LLAMACPP_SETUP, "llamacpp", env)
    profile_at = lines.index("PROFILE:llamacpp")
    toolkit_at = lines.index("STEP:cuda_toolkit")
    fabric_at = lines.index("STEP:fabric_manager")
    proof_at = lines.index("STEP:cuda_proof")
    assert profile_at < toolkit_at < fabric_at < proof_at


def test_llamacpp_cpu_only_main_skips_profile_selection(tmp_path: Path) -> None:
    # CPU-only standalone start (REQUIRE_CUDA=0) must not attempt profile
    # selection: an unavailable GPU means a probe failure, not a rejection.
    # Mask real workstation GPUs as well as supporting CPU-only CI runners.
    _write_executable(tmp_path / "nvidia-smi", "#!/bin/bash\nexit 1\n")
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:/usr/bin:/bin"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source {LLAMACPP_START!s}
run_llamacpp() {{ echo RAN; }}
REQUIRE_CUDA=0 main
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RAN" in result.stdout
    assert "[PROFILE:" not in result.stdout
    assert "[REJECT:" not in result.stderr


def test_llamacpp_param_selection_by_bucket(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source {LLAMACPP_START!s}
PROFILE_BUCKET='hopper'
GPU_COUNT=2
GPU_VRAM_GB=80
configure_llamacpp_params
printf '%s|%s|%s\\n' "$N_GPU_LAYERS" "$CTX_SIZE" "$PARALLEL"
""",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "99|32768|8"


def test_profiles_marker_from_start_script_chain(tmp_path: Path) -> None:
    # The full sourcing chain (start-vllm.sh -> preflight -> setup-base ->
    # profiles) exposes select_runtime_profile with the measured signature
    # and prints the record marker used by attempt logs.
    env = _hardware_env(tmp_path, cap="8.0", vram_mb=81920)
    env["AUTOVLLM_SCRIPT_DIR"] = str(SCRIPT_ROOT / "auto-vllm")
    env.pop("PROFILE_OS_RELEASE", None)
    env["PROFILE_OS_RELEASE"] = str(tmp_path / "os-release")
    result = _source_and_call(
        VLLM_START,
        'select_runtime_profile vllm\necho "$PROFILE_NAME"',
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "[PROFILE:select:vllm-ampere-a100" in result.stdout
    assert result.stdout.splitlines()[-1] == "vllm-ampere-a100"
