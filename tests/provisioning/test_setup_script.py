"""Behavioral tests for the node bootstrap setup script."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(os.environ.get("AUTOVLLM_TEST_SCRIPT_ROOT", REPO_ROOT))
SETUP_SCRIPT = SCRIPT_ROOT / "auto-vllm" / "setup.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_shell(
    command: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )


def _source_and(
    command: str,
    *,
    replacements: dict[str, str] | None = None,
) -> str:
    script = shlex.quote(str(SETUP_SCRIPT))
    # Keep the function-level tests executable against the pre-fix script,
    # whose setup sequence ran unconditionally when sourced.
    sed_expressions = ["/^# --- Main ---/,$d"]
    for old, new in (replacements or {}).items():
        escaped_old = old.replace("\\", "\\\\").replace("|", "\\|")
        escaped_new = new.replace("\\", "\\\\").replace("&", "\\&").replace("|", "\\|")
        sed_expressions.append(f"s|{escaped_old}|{escaped_new}|g")
    sed_args = " ".join(f"-e {shlex.quote(item)}" for item in sed_expressions)
    return f"source <(sed {sed_args} {script})\n{command}"


def test_setup_requires_nfs_export_before_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AUTOVLLM_NFS_EXPORT", raising=False)
    attempted_remote_work = tmp_path / "attempted-remote-work"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        f"#!/bin/bash\ntouch {shlex.quote(str(attempted_remote_work))}\nexit 99\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "AUTOVLLM_NFS_EXPORT is required" in result.stderr
    assert "[STEP:" not in result.stdout
    assert not attempted_remote_work.exists()


def test_hard_step_stops_at_first_failed_command(tmp_path: Path) -> None:
    reached_later_command = tmp_path / "reached-later-command"

    result = _run_shell(
        _source_and(
            f"""
failing_step() {{
    false
    touch {shlex.quote(str(reached_later_command))}
}}
step example failing_step
"""
        )
    )

    assert result.returncode != 0
    assert result.stdout.splitlines() == [
        "[STEP:example:START]",
        "[STEP:example:FAIL]",
    ]
    assert not reached_later_command.exists()


def test_soft_step_reports_early_failure_as_warning(tmp_path: Path) -> None:
    reached_later_command = tmp_path / "reached-later-command"

    result = _run_shell(
        _source_and(
            f"""
failing_step() {{
    false
    touch {shlex.quote(str(reached_later_command))}
}}
soft_step optional failing_step
echo continued
"""
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "[STEP:optional:START]",
        "[STEP:optional:WARN] (non-fatal, continuing)",
        "continued",
    ]
    assert not reached_later_command.exists()


def test_flashinfer_aot_package_matches_runtime_version(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    operation_log = tmp_path / "operations.log"
    installed_marker = tmp_path / "cubin-installed"
    _write_executable(bin_dir / "vllm", "#!/bin/bash\nexit 0\n")

    _write_executable(
        bin_dir / "python",
        """#!/bin/bash
code="$2"
if [[ "$code" == *'print(Version(version("flashinfer-python")).public)'* ]]; then
    echo '0.6.15.post1'
elif [[ "$code" == *'Version(version("flashinfer-cubin")).public'* ]]; then
    if test -f "$AUTOVLLM_CUBIN_MARKER"; then
        echo verify-final >> "$AUTOVLLM_TEST_LOG"
    fi
    test -f "$AUTOVLLM_CUBIN_MARKER"
else
    echo "unexpected python invocation: $*" >&2
    exit 2
fi
""",
    )
    _write_executable(
        bin_dir / "pip",
        """#!/bin/bash
echo "$*" >> "$AUTOVLLM_TEST_LOG"
touch "$AUTOVLLM_CUBIN_MARKER"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "AUTOVLLM_VENV": str(venv),
            "AUTOVLLM_TEST_LOG": str(operation_log),
            "AUTOVLLM_CUBIN_MARKER": str(installed_marker),
        }
    )

    result = _run_shell(
        _source_and(
            "install_vllm",
            replacements={"/opt/vllm-venv": str(venv)},
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert operation_log.exists()
    assert operation_log.read_text().splitlines() == [
        "install --no-deps --index-url https://flashinfer.ai/whl "
        "flashinfer-cubin==0.6.15.post1",
        "verify-final",
    ]


@pytest.mark.parametrize(
    ("installed_version", "expected_returncode", "expected_marker"),
    [
        ("580.126.09", 0, "[STEP:nvidia_driver:OK]"),
        ("570.172.08", 1, "[STEP:nvidia_driver:FAIL]"),
    ],
)
def test_existing_nvidia_driver_version_matrix(
    tmp_path: Path,
    installed_version: str,
    expected_returncode: int,
    expected_marker: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/bin/bash
if [[ "$*" == *'--query-gpu=driver_version'* ]]; then
    echo "$AUTOVLLM_INSTALLED_DRIVER"
fi
exit 0
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_INSTALLED_DRIVER": installed_version,
            "AUTOVLLM_NVIDIA_DRIVER_VERSION": "580.126.09",
        }
    )

    result = _run_shell(
        _source_and("step nvidia_driver install_nvidia_driver"), env=env
    )

    assert result.returncode == expected_returncode
    assert expected_marker in result.stdout
    if installed_version != "580.126.09":
        assert "installed NVIDIA driver 570.172.08" in result.stderr
        assert "requested 580.126.09" in result.stderr
        assert "NVIDIA-driver.run" not in result.stdout + result.stderr


@pytest.mark.parametrize("installed_version", ["1.1.6", "1.0.0"])
def test_existing_llmfit_version_matrix(
    tmp_path: Path,
    installed_version: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    llmfit_bin = tmp_path / "installed" / "llmfit"
    llmfit_bin.parent.mkdir()
    operation_log = tmp_path / "operations.log"
    extract_dir = tmp_path / "llmfit-new"
    _write_executable(
        llmfit_bin,
        f"#!/bin/bash\necho 'llmfit {installed_version}'\n",
    )
    _write_executable(
        bin_dir / "wget",
        """#!/bin/bash
echo "wget:$*" >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-O' ]]; then
        touch "$2"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        bin_dir / "tar",
        f"""#!/bin/bash
echo "tar:$*" >> "$AUTOVLLM_TEST_LOG"
mkdir -p {shlex.quote(str(extract_dir))}
printf '#!/bin/bash\necho "llmfit 1.1.6"\n' > {shlex.quote(str(extract_dir / "llmfit"))}
chmod +x {shlex.quote(str(extract_dir / "llmfit"))}
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "sudo:$*" >> "$AUTOVLLM_TEST_LOG"
if [[ "$1" == 'install' ]]; then
    exec /usr/bin/install "${@:2}"
fi
exit 2
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_LLMFIT_BIN": str(llmfit_bin),
            "AUTOVLLM_TMP_DIR": str(tmp_path),
            "AUTOVLLM_TEST_LOG": str(operation_log),
            "AUTOVLLM_LLMFIT_VERSION": "1.1.6",
        }
    )

    result = _run_shell(
        _source_and(
            "install_llmfit",
            replacements={
                "/tmp/": f"{tmp_path}/",
                "/usr/local/bin/llmfit": str(llmfit_bin),
            },
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    if installed_version == "1.1.6":
        assert "already installed, skipping" in result.stdout
        assert not operation_log.exists()
    else:
        assert "Replacing llmfit 1.0.0 with requested 1.1.6" in result.stdout
        assert operation_log.read_text().splitlines()[0].startswith("wget:")
        version_result = subprocess.run(
            [str(llmfit_bin), "--version"],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        assert version_result.stdout.strip() == "llmfit 1.1.6"


def test_system_update_pins_running_kernel(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    operation_log = tmp_path / "operations.log"
    _write_executable(
        bin_dir / "uname",
        """#!/bin/bash
[[ "$1" == '-r' ]] && echo '5.14.0-test'
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "$*" >> "$AUTOVLLM_TEST_LOG"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_TEST_LOG": str(operation_log),
        }
    )

    result = _run_shell(_source_and("run_system_update"), env=env)

    assert result.returncode == 0, result.stderr
    operations = operation_log.read_text().splitlines()
    assert operations[0].startswith(
        "dnf -y install kernel-devel-5.14.0-test kernel-headers-5.14.0-test"
    )
    assert operations[1] == "dnf -y update --exclude=kernel*"


@pytest.mark.parametrize("backend", ["firewalld", "iptables", "none"])
def test_firewall_backend_matrix(tmp_path: Path, backend: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    operation_log = tmp_path / "operations.log"
    if backend == "firewalld":
        _write_executable(bin_dir / "firewall-cmd", "#!/bin/bash\nexit 0\n")
    elif backend == "iptables":
        _write_executable(bin_dir / "iptables", "#!/bin/bash\nexit 0\n")

    _write_executable(
        bin_dir / "systemctl",
        """#!/bin/bash
if [[ "$1" == 'is-active' ]]; then
    [[ "$AUTOVLLM_FIREWALL_BACKEND" == 'firewalld' ]]
elif [[ "$1" == 'list-unit-files' && "$AUTOVLLM_FIREWALL_BACKEND" == 'iptables' ]]; then
    echo 'iptables.service disabled'
else
    exit 1
fi
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "$*" >> "$AUTOVLLM_TEST_LOG"
if [[ "$1 $2" == 'firewall-cmd --query-port=8000/tcp' || "$1 $2" == 'iptables -C' ]]; then
    exit 1
fi
exit 0
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_FIREWALL_BACKEND": backend,
            "AUTOVLLM_TEST_LOG": str(operation_log),
        }
    )

    result = _run_shell(_source_and("configure_firewall"), env=env)

    assert result.returncode == 0, result.stderr
    operations = (
        operation_log.read_text().splitlines() if operation_log.exists() else []
    )
    if backend == "firewalld":
        assert "firewall-cmd --add-port=8000/tcp --permanent" in operations
        assert "firewall-cmd --reload" in operations
    elif backend == "iptables":
        assert any(line.startswith("iptables -I INPUT") for line in operations)
        assert "systemctl restart iptables" in operations
    else:
        assert operations == []
        assert "no firewall rule required" in result.stdout
