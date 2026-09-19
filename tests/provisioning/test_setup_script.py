"""Behavioral tests for the node bootstrap setup script."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from inference_proxy.config.settings import (
    DEFAULT_LLMFIT_SHA256,
    DEFAULT_LLMFIT_VERSION,
    DEFAULT_NVIDIA_DRIVER_SHA256,
    DEFAULT_NVIDIA_DRIVER_VERSION,
)

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
    # Source only the function definitions; running the main setup sequence
    # would attempt to modify the test host.
    sed_expressions = ["/^# --- Main ---/,$d"]
    for old, new in (replacements or {}).items():
        escaped_old = old.replace("\\", "\\\\").replace("|", "\\|")
        escaped_new = new.replace("\\", "\\\\").replace("&", "\\&").replace("|", "\\|")
        sed_expressions.append(f"s|{escaped_old}|{escaped_new}|g")
    sed_args = " ".join(f"-e {shlex.quote(item)}" for item in sed_expressions)
    script_dir = shlex.quote(str(SCRIPT_ROOT / "auto-vllm"))
    return (
        f"AUTOVLLM_SCRIPT_DIR={script_dir}\n"
        f"source <(sed {sed_args} {script})\n{command}"
    )


def _uv_bootstrap_fixture(
    tmp_path: Path,
    *,
    valid_checksum: bool,
) -> tuple[dict[str, str], Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    archive_content = "verified uv fixture"
    digest = hashlib.sha256(archive_content.encode()).hexdigest()
    if not valid_checksum:
        digest = "0" * 64
    (bundle / ".uv-version").write_text("0.12.17\n")
    (bundle / "uv-x86_64-unknown-linux-gnu.tar.gz.sha256").write_text(
        f"{digest}  uv-x86_64-unknown-linux-gnu.tar.gz\n"
    )
    (bundle / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")

    operation_log = tmp_path / "operations.log"
    uv_bin = tmp_path / "installed" / "uv"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "wget",
        """#!/bin/bash
echo wget >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-O' ]]; then
        printf '%s' "$AUTOVLLM_TEST_ARCHIVE_CONTENT" > "$2"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        fake_bin / "sha256sum",
        """#!/bin/bash
echo sha256sum >> "$AUTOVLLM_TEST_LOG"
exec /usr/bin/sha256sum "$@"
""",
    )
    _write_executable(
        fake_bin / "tar",
        """#!/bin/bash
echo tar >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-C' ]]; then
        mkdir -p "$2/uv-x86_64-unknown-linux-gnu"
        cat > "$2/uv-x86_64-unknown-linux-gnu/uv" <<'EOF'
#!/bin/bash
echo 'uv 0.12.17 (fixture)'
EOF
        chmod +x "$2/uv-x86_64-unknown-linux-gnu/uv"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        fake_bin / "sudo",
        """#!/bin/bash
echo "sudo:$*" >> "$AUTOVLLM_TEST_LOG"
if [[ "$1" == 'install' ]]; then
    mkdir -p "$(dirname "${@: -1}")"
    exec /usr/bin/install "${@:2}"
fi
exit 99
""",
    )
    common_dir = bundle.parent / "common"
    common_dir.mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "common" / "setup-base.sh", common_dir / "setup-base.sh")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_SCRIPT_DIR": str(bundle),
        "AUTOVLLM_UV_BIN": str(uv_bin),
        "AUTOVLLM_TMP_DIR": str(tmp_path),
        "AUTOVLLM_TEST_LOG": str(operation_log),
        "AUTOVLLM_TEST_ARCHIVE_CONTENT": archive_content,
    }
    return env, operation_log, uv_bin


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


def test_script_defaults_match_gateway_artifact_pairs() -> None:
    result = _run_shell(
        _source_and(
            'printf \'%s\\n\' "$DRIVER_VERSION" "$DRIVER_SHA256" '
            '"$LLMFIT_RELEASE" "$LLMFIT_SHA256"'
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        DEFAULT_NVIDIA_DRIVER_VERSION,
        DEFAULT_NVIDIA_DRIVER_SHA256,
        DEFAULT_LLMFIT_VERSION,
        DEFAULT_LLMFIT_SHA256,
    ]


def test_uv_bootstrap_verifies_digest_before_install(tmp_path: Path) -> None:
    env, operation_log, uv_bin = _uv_bootstrap_fixture(tmp_path, valid_checksum=True)

    result = _run_shell(
        f"source {shlex.quote(str(SETUP_SCRIPT))}\ninstall_uv",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    operations = operation_log.read_text().splitlines()
    assert operations[:3] == ["wget", "sha256sum", "tar"]
    assert operations[3].startswith("sudo:install -m 755 ")
    assert operations[3].endswith(f" {uv_bin}")
    installed = subprocess.run(
        [str(uv_bin), "--version"],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert installed.stdout.startswith("uv 0.12.17")


def test_uv_bootstrap_checksum_mismatch_never_installs(tmp_path: Path) -> None:
    env, operation_log, uv_bin = _uv_bootstrap_fixture(tmp_path, valid_checksum=False)

    result = _run_shell(
        f"source {shlex.quote(str(SETUP_SCRIPT))}\ninstall_uv",
        env=env,
    )

    assert result.returncode != 0
    assert "SHA-256 verification failed" in result.stderr
    assert operation_log.read_text().splitlines() == ["wget", "sha256sum"]
    assert not uv_bin.exists()


@pytest.mark.parametrize(
    ("installed_version", "expected_bootstrap"),
    [("0.12.17", False), ("0.11.0", True)],
)
def test_uv_bootstrap_version_matrix(
    tmp_path: Path,
    installed_version: str,
    expected_bootstrap: bool,
) -> None:
    env, operation_log, uv_bin = _uv_bootstrap_fixture(tmp_path, valid_checksum=True)
    uv_bin.parent.mkdir()
    _write_executable(
        uv_bin,
        f"#!/bin/bash\necho 'uv {installed_version} (existing)'\n",
    )

    result = _run_shell(
        f"source {shlex.quote(str(SETUP_SCRIPT))}\ninstall_uv",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert operation_log.exists() is expected_bootstrap
    installed = subprocess.run(
        [str(uv_bin), "--version"],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert installed.stdout.startswith("uv 0.12.17")


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


def test_vllm_environment_syncs_with_frozen_uv_lock(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    operation_log = tmp_path / "operations.log"
    uv_bin = tmp_path / "uv"

    _write_executable(
        bin_dir / "python",
        """#!/bin/bash
[[ "$*" == *'Version(version("flashinfer-cubin")).public'* ]]
""",
    )
    _write_executable(
        uv_bin,
        """#!/bin/bash
if [[ "$1" == '--version' ]]; then
    echo 'uv 0.12.17 (test)'
    exit 0
fi
echo "env:$UV_PROJECT_ENVIRONMENT" >> "$AUTOVLLM_TEST_LOG"
echo "$*" >> "$AUTOVLLM_TEST_LOG"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "AUTOVLLM_VENV": str(venv),
            "AUTOVLLM_UV_BIN": str(uv_bin),
            "AUTOVLLM_TEST_LOG": str(operation_log),
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
    assert operation_log.read_text().splitlines() == [
        f"env:{venv}",
        f"sync --project {SCRIPT_ROOT / 'auto-vllm'} --frozen --no-dev "
        "--no-install-project --no-build --python /usr/bin/python3.12 "
        "--python-platform x86_64-manylinux_2_34",
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
    operation_log = tmp_path / "operations.log"
    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/bin/bash
if [[ "$*" == *'--query-gpu=driver_version'* ]]; then
    echo "$AUTOVLLM_INSTALLED_DRIVER"
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "$*" >> "$AUTOVLLM_TEST_LOG"
""",
    )
    _write_executable(bin_dir / "modinfo", "#!/bin/bash\nexit 1\n")
    _write_executable(bin_dir / "ls", "#!/bin/bash\nexit 1\n")
    _write_executable(bin_dir / "wget", "#!/bin/bash\nexit 1\n")
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_INSTALLED_DRIVER": installed_version,
            "AUTOVLLM_NVIDIA_DRIVER_VERSION": "580.126.09",
            "AUTOVLLM_TEST_LOG": str(operation_log),
        }
    )

    result = _run_shell(
        _source_and("step nvidia_driver install_nvidia_driver"), env=env
    )

    assert result.returncode == expected_returncode
    assert expected_marker in result.stdout
    if installed_version != "580.126.09":
        assert "does not match requested" in result.stdout
        operations = operation_log.read_text().splitlines()
        assert any("remove" in op and "nvidia" in op for op in operations)


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_nvidia_driver_verification_precedes_privileged_changes(
    tmp_path: Path,
    valid_checksum: bool,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    operation_log = tmp_path / "operations.log"
    archive_content = "verified nvidia fixture"
    digest = hashlib.sha256(archive_content.encode()).hexdigest()
    if not valid_checksum:
        digest = "0" * 64

    for command in ("nvidia-smi", "modinfo", "ls"):
        _write_executable(fake_bin / command, "#!/bin/bash\nexit 1\n")
    _write_executable(
        fake_bin / "wget",
        """#!/bin/bash
echo wget >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-O' ]]; then
        printf '%s' "$AUTOVLLM_TEST_ARCHIVE_CONTENT" > "$2"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        fake_bin / "sha256sum",
        """#!/bin/bash
echo sha256sum >> "$AUTOVLLM_TEST_LOG"
exec /usr/bin/sha256sum "$@"
""",
    )
    _write_executable(
        fake_bin / "sudo",
        """#!/bin/bash
echo "sudo:$*" >> "$AUTOVLLM_TEST_LOG"
exit 0
""",
    )
    env = {
        **{k: v for k, v in os.environ.items() if k != "BASH_ENV"},
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_TMP_DIR": str(tmp_path),
        "AUTOVLLM_NVIDIA_DRIVER_SHA256": digest,
        "AUTOVLLM_TEST_LOG": str(operation_log),
        "AUTOVLLM_TEST_ARCHIVE_CONTENT": archive_content,
    }

    result = _run_shell(_source_and("install_nvidia_driver"), env=env)
    operations = operation_log.read_text().splitlines()

    if valid_checksum:
        assert result.returncode == 0, result.stderr
        assert operations[:2] == ["wget", "sha256sum"]
        assert operations[-1].startswith("sudo:sh ")
    else:
        assert result.returncode != 0
        assert "SHA-256 verification failed" in result.stderr
        assert operations == ["wget", "sha256sum"]


@pytest.mark.parametrize("digest", [None, ""])
def test_custom_nvidia_driver_without_digest_fails_before_probe(
    tmp_path: Path,
    digest: str | None,
) -> None:
    attempted_probe = tmp_path / "attempted-probe"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "nvidia-smi",
        f"#!/bin/bash\ntouch {shlex.quote(str(attempted_probe))}\nexit 1\n",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_NVIDIA_DRIVER_VERSION": "999.1",
    }
    if digest is not None:
        env["AUTOVLLM_NVIDIA_DRIVER_SHA256"] = digest

    result = _run_shell(_source_and("install_nvidia_driver"), env=env)

    assert result.returncode == 2
    assert "AUTOVLLM_NVIDIA_DRIVER_SHA256" in result.stderr
    assert not attempted_probe.exists()


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
        printf '%s' "$AUTOVLLM_TEST_ARCHIVE_CONTENT" > "$2"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        bin_dir / "tar",
        """#!/bin/bash
echo "tar:$*" >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-C' ]]; then
        mkdir -p "$2/llmfit-new"
        printf '#!/bin/bash\necho "llmfit 1.1.6"\n' > "$2/llmfit-new/llmfit"
        chmod +x "$2/llmfit-new/llmfit"
        exit 0
    fi
    shift
done
exit 2
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
    archive_bytes = b"verified llmfit fixture"
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_LLMFIT_BIN": str(llmfit_bin),
            "AUTOVLLM_TMP_DIR": str(tmp_path),
            "AUTOVLLM_TEST_LOG": str(operation_log),
            "AUTOVLLM_LLMFIT_VERSION": "1.1.6",
            "AUTOVLLM_LLMFIT_SHA256": archive_sha256,
            "AUTOVLLM_TEST_ARCHIVE_CONTENT": archive_bytes.decode(),
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


def test_llmfit_checksum_mismatch_never_extracts_or_installs(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    operation_log = tmp_path / "operations.log"
    _write_executable(
        fake_bin / "wget",
        """#!/bin/bash
echo wget >> "$AUTOVLLM_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-O' ]]; then printf bad > "$2"; exit 0; fi
    shift
done
exit 2
""",
    )
    for command in ("tar", "sudo"):
        _write_executable(
            fake_bin / command,
            f'#!/bin/bash\necho {command} >> "$AUTOVLLM_TEST_LOG"\nexit 99\n',
        )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_TMP_DIR": str(tmp_path),
        "AUTOVLLM_LLMFIT_BIN": str(tmp_path / "llmfit"),
        "AUTOVLLM_LLMFIT_SHA256": "a" * 64,
        "AUTOVLLM_TEST_LOG": str(operation_log),
    }

    result = _run_shell(_source_and("install_llmfit"), env=env)

    assert result.returncode != 0
    assert "SHA-256 verification failed" in result.stderr
    assert operation_log.read_text().splitlines() == ["wget"]


@pytest.mark.parametrize("digest", [None, ""])
def test_custom_llmfit_without_digest_fails_before_binary_probe(
    tmp_path: Path,
    digest: str | None,
) -> None:
    attempted_probe = tmp_path / "attempted-probe"
    llmfit = tmp_path / "llmfit"
    _write_executable(
        llmfit,
        f"#!/bin/bash\ntouch {shlex.quote(str(attempted_probe))}\nexit 0\n",
    )
    env = {
        **os.environ,
        "AUTOVLLM_LLMFIT_BIN": str(llmfit),
        "AUTOVLLM_LLMFIT_VERSION": "9.9.9",
    }
    if digest is not None:
        env["AUTOVLLM_LLMFIT_SHA256"] = digest

    result = _run_shell(_source_and("install_llmfit"), env=env)

    assert result.returncode == 2
    assert "AUTOVLLM_LLMFIT_SHA256" in result.stderr
    assert not attempted_probe.exists()


def test_runtime_flashinfer_index_override_fails_before_setup_work(
    tmp_path: Path,
) -> None:
    attempted_work = tmp_path / "attempted-work"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        f"#!/bin/bash\ntouch {shlex.quote(str(attempted_work))}\nexit 99\n",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_NFS_EXPORT": "storage.example:/exports/hf",
        "FLASHINFER_INDEX_URL": "https://mirror.example/flashinfer",
    }

    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "FLASHINFER_INDEX_URL is retired" in result.stderr
    assert "regenerated auto-vllm bundle" in result.stderr
    assert not attempted_work.exists()


def test_system_update_pins_running_kernel_and_build_packages(tmp_path: Path) -> None:
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
    env.pop("BASH_ENV", None)
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_TEST_LOG": str(operation_log),
        }
    )

    result = _run_shell(_source_and("run_system_update\ninstall_cuda_toolkit"), env=env)

    assert result.returncode == 0, result.stderr
    operations = operation_log.read_text().splitlines()
    assert operations[0] == (
        "dnf -y install kernel-devel-5.14.0-test kernel-headers-5.14.0-test"
    )
    assert operations[1] == (
        "dnf -y install cmake gcc gcc-c++ make wget nfs-utils elfutils-libelf-devel "
        "python3.12 python3.12-devel"
    )
    assert operations[2] == "dnf -y update --exclude=kernel*"
    assert operations[3] == "dnf -y install dnf-plugins-core"
    assert "ninja" not in "\n".join(operations)


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
    env.pop("BASH_ENV", None)
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


NFS_OPTS = "rw,vers=3,hard,proto=tcp,timeo=600,retrans=3,sec=sys"


def _nfs_env(
    tmp_path: Path,
    *,
    export: str = "storage.example:/exports/huggingface",
    mounts_line: str | None = None,
) -> tuple[dict[str, str], Path]:
    """Env for storage-logic tests: permissive sudo + injectable fixtures.

    fstab and autofs fixtures live under tmp_path so no test can touch the
    host's /etc/fstab or /etc/auto.* files.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    operation_log = tmp_path / "operations.log"
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "sudo:$*" >> "$AUTOVLLM_TEST_LOG"
case "$1" in
    umount|mount|systemctl) exit 0 ;;
esac
exec "$@"
""",
    )
    _write_executable(
        bin_dir / "mountpoint",
        """#!/bin/bash
if grep -qF "$2 " "$AUTOVLLM_MOUNTS_FILE"; then
    exit 0
fi
exit 1
""",
    )
    _write_executable(bin_dir / "fuser", "#!/bin/bash\nexit 1\n")
    mounts_file = tmp_path / "mounts"
    mount_point = str(tmp_path / "srv" / "hf-cache")
    if mounts_line is None:
        mounts_line = f"{export} {mount_point} nfs {NFS_OPTS} 0 0"
    mounts_file.write_text(mounts_line + "\n")
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "AUTOVLLM_TEST_LOG": str(operation_log),
            "AUTOVLLM_NFS_EXPORT": export,
            "AUTOVLLM_NFS_MOUNT_POINT": mount_point,
            "AUTOVLLM_MOUNTS_FILE": str(mounts_file),
            "AUTOVLLM_FSTAB_FILE": str(tmp_path / "fstab"),
            "AUTOVLLM_AUTO_MAP_DIR": str(tmp_path / "auto"),
            "AUTOVLLM_MIN_FREE_GB": "1",
        }
    )
    (tmp_path / "auto").mkdir(exist_ok=True)
    (tmp_path / "fstab").touch()
    (tmp_path / "srv" / "hf-cache").mkdir(parents=True, exist_ok=True)
    return env, operation_log


def _storage_output(result: subprocess.CompletedProcess[str]) -> tuple[str, int]:
    return result.stdout + result.stderr, result.returncode


def test_verify_nfs_storage_accepts_exact_export_and_options(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 0, result.stderr
    assert "verified" in result.stdout


def test_verify_nfs_storage_rejects_wrong_export_without_touch(
    tmp_path: Path,
) -> None:
    env, log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"other.example:/exports/other {mp} nfs rw,vers=3,hard,proto=tcp,timeo=600,retrans=3 0 0\n"
    )
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 2
    assert "is not the expected export" in result.stderr


@pytest.mark.parametrize(
    ("fstype", "opts", "needle"),
    [
        ("nfs4", NFS_OPTS, "unsupported"),
        ("nfs", "rw,vers=3,hard,proto=tcp,timeo=600,sec=sys", "retrans=3"),
        ("nfs", "rw,hard,proto=tcp,timeo=600,retrans=3,sec=sys", "vers=3"),
        ("nfs", "rw,vers=3,soft,proto=tcp,timeo=600,retrans=3,sec=sys", "soft"),
        ("nfs", "rw,vers=3,hard,proto=tcp,timeo=300,retrans=3,sec=sys", "timeo=300"),
    ],
)
def test_verify_nfs_storage_rejects_wrong_fstype_or_options(
    tmp_path: Path,
    fstype: str,
    opts: str,
    needle: str,
) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    mounts_line = f"storage.example:/exports/huggingface {mp} {fstype} {opts} 0 0"
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(mounts_line + "\n")
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 3
    assert needle in result.stderr


def test_verify_nfs_storage_decodes_octal_escapes(tmp_path: Path) -> None:
    env, _log = _nfs_env(
        tmp_path,
        export="storage.example:/exports/with space",
        mounts_line=(
            r"storage.example:/exports/with\040space /srv/hf\040cache nfs "
            + NFS_OPTS
            + " 0 0"
        ),
    )
    env["AUTOVLLM_NFS_EXPORT"] = "storage.example:/exports/with space"
    env["AUTOVLLM_NFS_MOUNT_POINT"] = "/srv/hf cache"
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 0, result.stderr


def test_check_storage_capacity_fails_on_proven_shortage(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "df",
        """#!/bin/bash
if [[ "$1" == "--output=target" ]]; then
    echo 'Mounted on'
    echo '/fixture'
    exit 0
fi
if [[ "$1" == "--output=avail" ]]; then
    echo 'Avail'
    echo '1073741824'
    exit 0
fi
exit 1
""",
    )
    env["AUTOVLLM_MIN_FREE_GB"] = "20"
    result = _run_shell(
        _source_and('check_storage_capacity "$NFS_MOUNT_POINT"'), env=env
    )
    assert result.returncode == 1
    assert "1GB free but 20GB is required" in result.stderr


def test_check_storage_capacity_warns_on_probe_failure(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(bin_dir / "df", "#!/bin/bash\nexit 1\n")
    result = _run_shell(
        _source_and('check_storage_capacity "$NFS_MOUNT_POINT"'), env=env
    )
    assert result.returncode == 2
    assert "cannot determine free space" in result.stderr


def test_mount_nfs_cache_accepts_matching_mount_without_churn(
    tmp_path: Path,
) -> None:
    env, log = _nfs_env(tmp_path)
    result = _run_shell(_source_and("mount_nfs_cache"), env=env)
    assert result.returncode == 0, result.stderr
    operations = log.read_text().splitlines()
    assert not [op for op in operations if "umount" in op or " mount " in op]


def test_mount_nfs_cache_refuses_wrong_export_without_umount(
    tmp_path: Path,
) -> None:
    env, log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"other.example:/exports/other {mp} nfs rw,vers=3,hard,proto=tcp,timeo=600,retrans=3 0 0\n"
    )
    result = _run_shell(_source_and("mount_nfs_cache"), env=env)
    assert result.returncode != 0
    assert "refusing to remount" in result.stderr
    operations = log.read_text().splitlines() if log.exists() else []
    assert "umount" not in operations


def test_mount_nfs_cache_busy_mount_fails_before_umount(tmp_path: Path) -> None:
    env, log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"storage.example:/exports/huggingface {mp} nfs rw,vers=3,hard,proto=tcp,timeo=300,retrans=3 0 0\n"
    )
    bin_dir = Path(env["PATH"].split(":")[0])
    probe = tmp_path / "probe-bash"
    _write_executable(probe, "#!/bin/bash\nexit 0\n")
    env["AUTOVLLM_NFS_PROBE_BASH"] = str(probe)
    _write_executable(
        bin_dir / "fuser",
        """#!/bin/bash
if [[ "$1" == "-vm" ]]; then
    echo '1234 holder'
    exit 1
fi
exit 0
""",
    )
    result = _run_shell(_source_and("mount_nfs_cache"), env=env)
    assert result.returncode != 0
    assert "is busy" in result.stderr
    operations = log.read_text().splitlines() if log.exists() else []
    assert "umount" not in operations


def test_ensure_persistence_appends_marked_fstab_entry(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    fstab = Path(env["AUTOVLLM_FSTAB_FILE"])
    fstab.write_text("# existing comment\n")
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    lines = fstab.read_text().splitlines()
    assert any("# qiip-managed" in line and mp in line for line in lines)


def test_ensure_persistence_is_idempotent(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    fstab = Path(env["AUTOVLLM_FSTAB_FILE"])
    fstab.write_text("# existing comment\n")
    cmd = 'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    assert sum(1 for line in fstab.read_text().splitlines() if mp in line) == 1


def test_ensure_persistence_adopts_matching_unmarked_entry(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    fstab = Path(env["AUTOVLLM_FSTAB_FILE"])
    fstab.write_text(
        f"storage.example:/exports/huggingface {mp} nfs "
        "vers=3,hard,proto=tcp,timeo=600,retrans=3,_netdev,nofail 0 0\n"
    )
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "# qiip-managed" in fstab.read_text()


def test_ensure_persistence_refuses_conflicting_unmarked_entry(
    tmp_path: Path,
) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    fstab = Path(env["AUTOVLLM_FSTAB_FILE"])
    fstab.write_text(
        f"other.example:/exports/other {mp} nfs "
        "vers=3,hard,proto=tcp,timeo=600,retrans=3 0 0\n"
    )
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode != 0
    assert "will not overwrite" in result.stderr
    assert "other.example:/exports/other" in fstab.read_text()


def test_ensure_persistence_updates_autofs_map_with_marker(tmp_path: Path) -> None:
    env, log = _nfs_env(tmp_path)
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    (auto_dir / "auto.srv").write_text(
        "# managed by ops\n"
        f"{mp} -fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "storage.example:/exports/huggingface\n"
    )
    env["AUTOVLLM_AUTO_MAP_DIR"] = str(auto_dir)
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    text = (auto_dir / "auto.srv").read_text()
    assert f"# qiip-managed {mp}" in text
    assert "systemctl reload autofs" in log.read_text()


def test_ensure_persistence_leaves_foreign_autofs_map_untouched(
    tmp_path: Path,
) -> None:
    """QIIP manages only map entries for its own export; foreign maps stay put."""
    env, _log = _nfs_env(tmp_path)
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    foreign = auto_dir / "auto.srv"
    original = (
        "/srv/hf-cache -fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "other.example:/exports/other\n"
    )
    foreign.write_text(original)
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert foreign.read_text() == original


def test_verify_nfs_storage_absent_mount_reports_not_found(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text("proc /proc proc rw 0 0\n")
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_verify_nfs_storage_accepts_ipv6_bracketed_export(tmp_path: Path) -> None:
    env, _log = _nfs_env(
        tmp_path,
        export="[2001:db8::1]:/exports/huggingface",
        mounts_line="[2001:db8::1]:/exports/huggingface /srv/hf-cache nfs "
        + NFS_OPTS
        + " 0 0",
    )
    env["AUTOVLLM_NFS_MOUNT_POINT"] = "/srv/hf-cache"
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"[2001:db8::1]:/exports/huggingface /srv/hf-cache nfs {NFS_OPTS} 0 0\n"
    )
    result = _run_shell(_source_and("verify_nfs_storage"), env=env)
    assert result.returncode == 0, result.stderr


def test_ensure_persistence_write_failure_is_fatal(tmp_path: Path) -> None:
    env, log = _nfs_env(tmp_path)
    bin_dir = Path(env["PATH"].split(":")[0])
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
echo "sudo:$*" >> "$AUTOVLLM_TEST_LOG"
if [[ "$1" == "tee" && "${AUTOVLLM_FAIL_TEE:-0}" == "1" ]]; then
    exit 1
fi
case "$1" in
    umount|mount|systemctl) exit 0 ;;
esac
exec "$@"
""",
    )
    env["AUTOVLLM_FAIL_TEE"] = "1"
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode != 0
    assert "write failed" in result.stderr
    assert "Updated QIIP" not in result.stdout


def test_mount_nfs_cache_remounts_stale_options_when_server_up(
    tmp_path: Path,
) -> None:
    env, log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"storage.example:/exports/huggingface {mp} nfs rw,vers=3,hard,proto=tcp,timeo=300,retrans=3 0 0\n"
    )
    bin_dir = Path(env["PATH"].split(":")[0])
    probe = tmp_path / "probe-bash"
    _write_executable(probe, "#!/bin/bash\nexit 0\n")
    env["AUTOVLLM_NFS_PROBE_BASH"] = str(probe)
    _write_executable(
        bin_dir / "mount",
        f"""#!/bin/bash
echo "mount:$*" >> "$AUTOVLLM_TEST_LOG"
cat > "$AUTOVLLM_MOUNTS_FILE" <<EOF
storage.example:/exports/huggingface {mp} nfs rw,vers=3,hard,proto=tcp,timeo=600,retrans=3 0 0
EOF
exit 0
""",
    )
    result = _run_shell(_source_and("mount_nfs_cache"), env=env)
    assert result.returncode == 0, result.stderr
    assert "umount" in log.read_text()
    assert "capacity" in result.stdout.lower()


def test_mount_nfs_cache_fails_fast_when_server_unreachable(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    Path(env["AUTOVLLM_MOUNTS_FILE"]).write_text(
        f"storage.example:/exports/huggingface {mp} nfs rw,vers=3,hard,proto=tcp,timeo=300,retrans=3 0 0\n"
    )
    probe = tmp_path / "probe-bash"
    _write_executable(probe, "#!/bin/bash\nexit 1\n")
    env["AUTOVLLM_NFS_PROBE_BASH"] = str(probe)
    result = _run_shell(_source_and("mount_nfs_cache"), env=env)
    assert result.returncode != 0
    assert "not reachable" in result.stderr


def test_ensure_autofs_update_is_idempotent(tmp_path: Path) -> None:
    env, _log = _nfs_env(tmp_path)
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    (auto_dir / "auto.srv").write_text(
        f"{mp} -fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "storage.example:/exports/huggingface\n"
    )
    cmd = 'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    text = (auto_dir / "auto.srv").read_text()
    assert text.count(f"# qiip-managed {mp}") == 1
    assert text.count("storage.example:/exports/huggingface") == 1


def test_ensure_persistence_updates_only_matching_autofs_key(
    tmp_path: Path,
) -> None:
    """A second entry mounting the same export under another key stays intact."""
    env, _log = _nfs_env(tmp_path)
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    other = str(tmp_path / "mnt" / "other-cache")
    entry = (
        "-fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "storage.example:/exports/huggingface"
    )
    (auto_dir / "auto.srv").write_text(f"{mp} {entry}\n{other} {entry}\n")
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    text = (auto_dir / "auto.srv").read_text()
    assert f"# qiip-managed {mp}" in text
    assert f"{other} {entry}" in text
    assert text.count("storage.example:/exports/huggingface") == 2


def test_ensure_persistence_selects_autofs_map_by_key_not_export(
    tmp_path: Path,
) -> None:
    """An unrelated map sharing the export is never the one QIIP edits."""
    env, _log = _nfs_env(tmp_path)
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    mp = env["AUTOVLLM_NFS_MOUNT_POINT"]
    other = str(tmp_path / "mnt" / "other-cache")
    entry = (
        "-fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "storage.example:/exports/huggingface"
    )
    (auto_dir / "auto.other").write_text(f"{other} {entry}\n")
    (auto_dir / "auto.srv").write_text(f"{mp} {entry}\n")
    result = _run_shell(
        _source_and(
            'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
        ),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert f"# qiip-managed {mp}" in (auto_dir / "auto.srv").read_text()
    assert (auto_dir / "auto.other").read_text() == f"{other} {entry}\n"


def test_ensure_persistence_preserves_escaped_fields_on_reupdate(
    tmp_path: Path,
) -> None:
    """Repeated updates keep \\040/\\011 escapes instead of literal whitespace."""
    env, _log = _nfs_env(
        tmp_path,
        export="storage.example:/exports/with space",
    )
    env["AUTOVLLM_NFS_MOUNT_POINT"] = "/srv/hf cache"
    cmd = 'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    text = Path(env["AUTOVLLM_FSTAB_FILE"]).read_text()
    assert r"storage.example:/exports/with\040space /srv/hf\040cache nfs" in text
    assert text.count(r"storage.example:/exports/with\040space") == 1


def test_ensure_persistence_preserves_escaped_autofs_fields_on_reupdate(
    tmp_path: Path,
) -> None:
    """Repeated autofs updates keep \\040 escapes in key and export."""
    env, _log = _nfs_env(tmp_path, export="storage.example:/exports/with space")
    env["AUTOVLLM_NFS_MOUNT_POINT"] = "/srv/hf cache"
    auto_dir = Path(env["AUTOVLLM_AUTO_MAP_DIR"])
    (auto_dir / "auto.srv").write_text(
        "/srv/hf\\040cache -fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        "storage.example:/exports/with\\040space\n"
    )
    cmd = 'ensure_nfs_persistence "vers=3,hard,proto=tcp,timeo=600,retrans=3"'
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    assert _run_shell(_source_and(cmd), env=env).returncode == 0
    text = (auto_dir / "auto.srv").read_text()
    assert (
        r"-fstype=nfs,vers=3,hard,proto=tcp,timeo=600,retrans=3 "
        r"storage.example:/exports/with\040space"
    ) in text
    assert text.count(r"storage.example:/exports/with\040space") == 1


def test_nfs_server_reachable_probes_bare_ipv6_address(tmp_path: Path) -> None:
    """The TCP probe gets the unbracketed address; display keeps the brackets."""
    env, _log = _nfs_env(
        tmp_path,
        export="[2001:db8::1]:/exports/huggingface",
    )
    probe = tmp_path / "probe-bash"
    _write_executable(
        probe,
        """#!/bin/bash
if [[ "$2" == *"/dev/tcp/2001:db8::1/2049"* ]] \\
    && [[ "$2" != *"[2001:db8::1]"* ]]; then
    exit 0
fi
exit 1
""",
    )
    env["AUTOVLLM_NFS_PROBE_BASH"] = str(probe)
    result = _run_shell(_source_and("nfs_server_reachable"), env=env)
    assert result.returncode == 0, result.stderr
    assert "reachable at [2001:db8::1]:2049" in result.stdout
