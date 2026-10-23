"""Fault injection across the shipped setup/publication/launch boundary."""

from __future__ import annotations

import errno
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from common import generations
from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning.bundles import bundle_files, write_bundle
from inference_proxy.provisioning.diagnostics import source_commands
from inference_proxy.provisioning.ssh_client import SSHConnectionError
from tests.provisioning.test_attempt_logs import LocalNodeSSH
from tests.provisioning.test_provisioner import _make_provisioner
from tests.provisioning.test_vllm_scripts import _script_environment, _write_executable

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "common/generations.py"


def _bundle(tmp_path: Path, label: str) -> Path:
    bundle = tmp_path / label
    bundle.mkdir()
    files = {
        "auto-vllm/start-vllm.sh": (
            f'#!/bin/bash\necho script:{label}\nexec "$AUTOVLLM_BIN"\n'
        ).encode(),
        "common/generations.py": TOOL.read_bytes(),
    }
    write_bundle(bundle, files)
    return bundle


def _runtime(tmp_path: Path, label: str) -> Path:
    runtime = tmp_path / label
    (runtime / "bin").mkdir(parents=True)
    _write_executable(runtime / "bin/vllm", f"#!/bin/bash\necho runtime:{label}\n")
    generations.seal_runtime(runtime, label, ["vllm"])
    return runtime


def _activate(root: Path, bundle: Path, runtime: Path) -> dict[str, Any]:
    return generations.activate(
        root, bundle, "auto-vllm", runtime, {"QIIP_ENGINE": "vllm"}
    )


def _launch(root: Path, bundle: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "exec",
            str(root),
            str(bundle / "auto-vllm"),
            "start-vllm.sh",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("boundary", ["generation", "previous", "current", "committed"])
def test_interrupted_activation_selects_a_whole_generation_and_retry_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    root = tmp_path / "engine"
    first_bundle = _bundle(tmp_path, "old-bundle")
    first_runtime = _runtime(tmp_path, "old-runtime")
    prior = _activate(root, first_bundle, first_runtime)
    second_bundle = _bundle(tmp_path, "new-bundle")
    second_runtime = _runtime(tmp_path, "new-runtime")
    replace = os.replace
    rename = os.rename
    sync = generations.sync_directory
    committed = False

    def interrupt_replace(source: Any, destination: Any) -> None:
        nonlocal committed
        destination = Path(destination)
        if destination.name == boundary:
            raise OSError(errno.ENOSPC, "controlled full activation filesystem")
        replace(source, destination)
        committed = destination == root / "current"

    def interrupt_rename(source: Any, destination: Any) -> None:
        if (
            boundary == "generation"
            and Path(destination).parent == root / "generations"
        ):
            raise OSError(errno.ENOSPC, "controlled interrupted generation publication")
        rename(source, destination)

    def interrupt_sync(path: Path) -> None:
        if boundary == "committed" and committed and path == root:
            raise OSError(errno.EIO, "controlled lost activation acknowledgement")
        sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_replace)
        patch.setattr(os, "rename", interrupt_rename)
        patch.setattr(generations, "sync_directory", interrupt_sync)
        with pytest.raises(OSError):
            _activate(root, second_bundle, second_runtime)

    output = _launch(root, second_bundle).stdout
    if boundary == "committed":
        assert "script:new-bundle\nruntime:new-runtime" in output
    else:
        assert "script:old-bundle\nruntime:old-runtime" in output
    selected = _activate(root, second_bundle, second_runtime)
    assert generations.evidence((root / "current").resolve()) == selected
    assert (root / "previous").resolve().name == prior["generation_id"]
    assert (
        "script:new-bundle\nruntime:new-runtime" in _launch(root, first_bundle).stdout
    )
    subprocess.run(
        [sys.executable, str(TOOL), "rollback", str(root)],
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert (
        "script:old-bundle\nruntime:old-runtime" in _launch(root, second_bundle).stdout
    )


def test_partial_upload_is_inactive_and_retry_publishes_verified_snapshot(
    tmp_path: Path,
) -> None:
    files = bundle_files(ROOT / "auto-vllm", ROOT / "common")
    staged = tmp_path / ".upload"
    staged.mkdir()
    identity = write_bundle(staged, files)
    removed = staged / "common/profiles.sh"
    removed.unlink()
    final = tmp_path / identity
    with pytest.raises(ValueError, match="Incomplete"):
        generations.publish_bundle(staged, final, identity)
    assert not final.exists()
    removed.write_bytes(files["common/profiles.sh"])
    generations.publish_bundle(staged, final, identity)
    assert generations.digest(generations.verify_bundle(final)) == identity


def test_bundle_snapshot_includes_sources_reached_through_file_symlinks(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "auto-vllm"
    common = tmp_path / "common"
    engine.mkdir()
    common.mkdir()
    target = tmp_path / "script-source"
    target.write_text("#!/bin/bash\nexit 0\n")
    (engine / "setup.sh").symlink_to(target)
    (common / "generations.py").symlink_to(TOOL)
    staged = tmp_path / "staged"
    staged.mkdir()
    identity = write_bundle(staged, bundle_files(engine, common))
    assert (staged / "auto-vllm/setup.sh").read_bytes() == target.read_bytes()
    assert not (staged / "auto-vllm/setup.sh").is_symlink()
    generations.verify_bundle(staged, identity)
    standalone = generations.setup_bundle(tmp_path / "standalone", engine)
    generations.verify_bundle(standalone)


@pytest.mark.parametrize("boundary", ["previous", "current"])
def test_interrupted_rollback_preserves_target_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    root = tmp_path / "engine"
    first_bundle = _bundle(tmp_path, "old-bundle")
    prior = _activate(root, first_bundle, _runtime(tmp_path, "old-runtime"))
    new = _activate(
        root, _bundle(tmp_path, "new-bundle"), _runtime(tmp_path, "new-runtime")
    )
    replace = os.replace

    def interrupt(source: Any, destination: Any) -> None:
        if Path(destination) == root / boundary:
            raise OSError(errno.EIO, "controlled interrupted rollback")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(OSError):
            generations.rollback(root)
    assert (root / "current").resolve().name == new["generation_id"]
    assert (root / "ROLLBACK.json").is_file()
    assert generations.rollback(root) == prior
    assert not (root / "ROLLBACK.json").exists()
    assert (root / "previous").resolve().name == new["generation_id"]
    assert (
        "script:old-bundle\nruntime:old-runtime" in _launch(root, first_bundle).stdout
    )


def test_disk_full_completion_marker_does_not_seal_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    _write_executable(runtime / "bin/vllm", "#!/bin/bash\nexit 0\n")
    replace = os.replace

    def no_space(source: Any, destination: Any) -> None:
        raise OSError(errno.ENOSPC, "controlled full disk")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", no_space)
        with pytest.raises(OSError):
            generations.seal_runtime(runtime, "locked-profile", ["vllm"])
    assert not (runtime / "RUNTIME.json").exists()
    assert os.replace is replace
    generations.seal_runtime(runtime, "locked-profile", ["vllm"])
    assert generations.verify_runtime(runtime)["identity"] == "locked-profile"


@pytest.mark.asyncio
async def test_provisioner_upload_failure_preserves_pinned_bundle_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = tmp_path / "node"
    node.mkdir()
    ssh = LocalNodeSSH(node, dict(os.environ))
    provisioner = _make_provisioner(ssh_client=ssh)
    upload = ssh.upload

    async def partial_upload(host: str, source: Path, destination: str) -> None:
        target = node / destination / source.name / "auto-vllm"
        target.mkdir(parents=True)
        shutil.copyfile(source / "auto-vllm/setup.sh", target / "setup.sh")
        raise SSHConnectionError(host, "controlled interrupted upload")

    with monkeypatch.context() as patch:
        patch.setattr(ssh, "upload", partial_upload)
        with pytest.raises(SSHConnectionError):
            await provisioner._upload_scripts("host1")
    assert not provisioner._remote_bundles
    assert all(
        path.name.startswith(".upload-") for path in (node / ".qiip/bundles").iterdir()
    )
    assert ssh.upload == upload
    await provisioner._upload_scripts("host1")
    published = node / provisioner._remote_bundles["host1", InferenceEngine.VLLM]
    generations.verify_bundle(published, published.name)


def _vllm_setup_environment(tmp_path: Path) -> dict[str, str]:
    original_engine = tmp_path / "source-vllm"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    options = "--host --port --tensor-parallel-size --gpu-memory-utilization --max-model-len --max-num-batched-tokens --enable-auto-tool-choice --tool-call-parser --reasoning-parser --dtype --enforce-eager"
    _write_executable(
        original_engine,
        f"""#!/bin/bash
if [ "$*" = "serve --help=all" ]; then
    [ "${{TEST_CLI_FAIL:-0}}" = 0 ] || exit 43
    echo {shlex.quote(options)}
    exit 0
fi
ln -sfn /proc/$$ {shlex.quote(str(proc_root))}/$$
echo fixture-engine-start
sleep 2
""",
    )
    env = _script_environment(
        tmp_path, vllm_bin=original_engine, process_log=tmp_path / "engine-events"
    )
    _write_executable(
        Path(env["PATH"].split(":")[0]) / "sudo", '#!/bin/bash\nexec "$@"\n'
    )
    env.pop("AUTOVLLM_COMMAND_PATTERN", None)
    mounts = tmp_path / "mounts"
    mounts.write_text(
        f"fixture:/cache {env['AUTOVLLM_NFS_MOUNT_POINT']} nfs rw,vers=3,hard,proto=tcp,timeo=600,retrans=3,sec=sys 0 0\n"
    )
    env.update(
        {
            "AUTOVLLM_MOUNTS_FILE": str(mounts),
            "AUTOVLLM_PROC_ROOT": str(proc_root),
            "AUTOVLLM_NFS_EXPORT": "fixture:/cache",
            "AUTOVLLM_UV_BIN": str(tmp_path / "uv"),
            "AUTOVLLM_BOOTSTRAP_PYTHON": sys.executable,
            "AUTOVLLM_RUNTIME_ROOT": str(tmp_path / "runtimes"),
            "QIIP_GENERATION_ROOT": str(tmp_path / "generations"),
            "AUTOVLLM_TMP_DIR": str(tmp_path),
            "TEST_SOURCE_PYTHON": env["AUTOVLLM_PYTHON"],
            "TEST_SOURCE_VLLM": str(original_engine),
            "TEST_PROFILE": "fixture-old",
            "TEST_SYNC_LOG": str(tmp_path / "sync.log"),
        }
    )
    _write_executable(
        tmp_path / "uv",
        """#!/bin/bash
set -e
if [ "$1" = --version ]; then echo 'uv 0.12.17 (fixture)'; exit 0; fi
echo "$UV_PROJECT_ENVIRONMENT" >> "$TEST_SYNC_LOG"
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
cp "$TEST_SOURCE_PYTHON" "$UV_PROJECT_ENVIRONMENT/bin/python"
if [ "${TEST_SYNC_FAIL:-0}" != 0 ]; then exit "$TEST_SYNC_FAIL"; fi
cp "$TEST_SOURCE_VLLM" "$UV_PROJECT_ENVIRONMENT/bin/vllm"
printf '#!/bin/bash\necho 1.13\n' > "$UV_PROJECT_ENVIRONMENT/bin/ninja"
chmod +x "$UV_PROJECT_ENVIRONMENT/bin/"*
""",
    )
    return env


def _setup(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command = f"""
source {shlex.quote(str(ROOT / "auto-vllm/setup.sh"))}
prepare_runtime() {{ PROFILE_NAME="$TEST_PROFILE"; PROFILE_CUDA_TOOLKIT_VERSION=13.0; }}
mount_nfs_cache() {{ :; }}
configure_firewall() {{ :; }}
install_llmfit() {{ :; }}
install_vllm_unit() {{ :; }}
main
"""
    return subprocess.run(
        ["bash", "-c", command], env=env, text=True, capture_output=True, timeout=15
    )


@pytest.mark.parametrize("fault", ["install", "cli", "import", "disk"])
def test_real_vllm_setup_never_syncs_active_runtime_and_launches_selected_generation(
    tmp_path: Path, fault: str
) -> None:
    env = _vllm_setup_environment(tmp_path)
    result = _setup(env)
    assert result.returncode == 0, result.stderr
    root = Path(env["QIIP_GENERATION_ROOT"])
    original = (root / "current").resolve()
    original_runtime = Path(generations.evidence(original)["runtime_path"])
    env["TEST_PROFILE"] = "fixture-new"
    if fault == "install":
        env["TEST_SYNC_FAIL"] = "42"
    elif fault == "cli":
        env["TEST_CLI_FAIL"] = "1"
    elif fault == "import":
        env["AUTOVLLM_FLASHINFER_AVAILABLE"] = "0"
    else:
        # ENOSPC during package materialisation at the real uv sync boundary.
        env["TEST_SYNC_FAIL"] = "28"
    failed = _setup(env)
    assert failed.returncode != 0
    assert (root / "current").resolve() == original
    assert "[STEP:vllm_install:FAIL]" in failed.stdout
    assert len(list(Path(env["AUTOVLLM_RUNTIME_ROOT"]).glob("*/RUNTIME.json"))) == 1
    env.pop("TEST_SYNC_FAIL", None)
    env.pop("TEST_CLI_FAIL", None)
    env.pop("AUTOVLLM_FLASHINFER_AVAILABLE", None)
    retried = _setup(env)
    assert retried.returncode == 0, retried.stderr
    selected = generations.evidence((root / "current").resolve())
    assert Path(selected["runtime_path"]) != original_runtime
    assert (root / "previous").resolve() == original
    paths = Path(env["TEST_SYNC_LOG"]).read_text().splitlines()
    assert paths.count(str(original_runtime)) == 1
    before_retry = len(paths)
    assert _setup(env).returncode == 0
    assert len(Path(env["TEST_SYNC_LOG"]).read_text().splitlines()) == before_retry

    launched = subprocess.run(
        ["bash", str(ROOT / "auto-vllm/start-vllm.sh")],
        env={**env, "AUTOVLLM_MODEL": "org/model"},
        text=True,
        capture_output=True,
        timeout=15,
    )
    try:
        assert launched.returncode == 0, launched.stderr
        assert selected["generation_id"] in launched.stdout
        assert "# Model:" in launched.stdout
        assert (root / "current/vllm.env").is_file()
    finally:
        stopped = subprocess.run(
            ["bash", str(ROOT / "auto-vllm/stop-vllm.sh"), "--force"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert stopped.returncode == 0, stopped.stderr


def test_generation_refuses_corrupt_runtime_without_repairing_active_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "engine"
    bundle = _bundle(tmp_path, "bundle")
    runtime = _runtime(tmp_path, "runtime")
    _activate(root, bundle, runtime)
    (runtime / "bin/vllm").write_text("corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        _activate(root, bundle, runtime)
    assert (runtime / "bin/vllm").read_text() == "corrupted"


def test_vllm_missing_completion_manifest_cannot_trigger_sync_into_published_runtime(
    tmp_path: Path,
) -> None:
    env = _vllm_setup_environment(tmp_path)
    result = _setup(env)
    assert result.returncode == 0, result.stderr
    root = Path(env["QIIP_GENERATION_ROOT"])
    selected = generations.evidence((root / "current").resolve())
    runtime = Path(selected["runtime_path"])
    original_binary = (runtime / "bin/vllm").read_bytes()
    (runtime / "RUNTIME.json").unlink()
    failed = _setup(env)
    assert failed.returncode != 0
    assert "refusing to modify a published runtime" in failed.stderr
    assert (runtime / "bin/vllm").read_bytes() == original_binary
    assert Path(env["TEST_SYNC_LOG"]).read_text().splitlines() == [str(runtime)]


@pytest.mark.parametrize("value", ["relative", "/", "/opt/../qiip", "/opt/qiip\n"])
def test_generation_root_rejects_non_dedicated_or_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        ProvisioningSettings(generation_root=Path(value))


@pytest.mark.parametrize(
    "engine,binary", [("vllm", "python"), ("llama_cpp", "llama-server")]
)
def test_diagnostics_use_the_attempts_selected_runtime(
    engine: str, binary: str
) -> None:
    commands = source_commands(
        {
            "engine": engine,
            "mount_point": "/srv/hf-cache",
            "failure": {
                "started_at": "2026-10-07T10:00:00+00:00",
                "failed_at": "2026-10-07T10:00:01+00:00",
            },
            "selected_generation": {"runtime_path": "/opt/retained-runtime"},
        }
    )
    assert commands["runtime"][0] == f"/opt/retained-runtime/bin/{binary}"


@pytest.mark.parametrize(
    "generation,subcommand,expected",
    [
        ("previous", "serve", 0),
        ("previous", "serve-extra", 1),
        ("unrelated", "serve", 1),
    ],
)
def test_vllm_stop_identifies_prior_generations_by_exact_argv(
    tmp_path: Path, generation: str, subcommand: str, expected: int
) -> None:
    proc = tmp_path / "proc/1234"
    proc.mkdir(parents=True)
    runtime_root = tmp_path / "runtimes"
    executable = (
        runtime_root if generation == "previous" else tmp_path / "outside"
    ) / "old/bin/vllm"
    (proc / "cmdline").write_bytes(
        f"python\0{executable}\0{subcommand}\0org/model\0".encode()
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source {shlex.quote(str(ROOT / "auto-vllm/vllm-process.sh"))}
PROC_ROOT={shlex.quote(str(tmp_path / "proc"))}
AUTOVLLM_RUNTIME_ROOT={shlex.quote(str(runtime_root))}
COMMAND_PATTERN='/new/bin/vllm serve'
is_vllm_pid 1234
""",
        ],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == expected


def test_external_interpreter_update_does_not_corrupt_sealed_runtime(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, "runtime")
    interpreter = tmp_path / "system-python"
    _write_executable(interpreter, "#!/bin/bash\necho python-before-update\n")
    (runtime / "bin/python").symlink_to(interpreter)
    generations.seal_runtime(runtime, "same-version", ["python", "vllm"])
    manifest = generations.verify_runtime(runtime)
    assert "bin/python" not in manifest["files"]
    assert manifest["external_links"] == {"bin/python": str(interpreter)}
    _write_executable(interpreter, "#!/bin/bash\necho python-after-update\n")
    assert generations.verify_runtime(runtime, "same-version") == manifest
    root = tmp_path / "engine"
    bundle = _bundle(tmp_path, "bundle")
    _activate(root, bundle, runtime)
    assert "runtime:runtime" in _launch(root, bundle).stdout
    interpreter.unlink()
    with pytest.raises(ValueError, match="external executable"):
        generations.verify_runtime(runtime)


@pytest.mark.parametrize(
    "engine,script", [("vllm", "stop-vllm.sh"), ("llama_cpp", "stop-llamacpp.sh")]
)
@pytest.mark.parametrize("damage", ["binary", "manifest", "runtime"])
def test_stop_dispatch_remains_available_when_runtime_is_damaged(
    tmp_path: Path, engine: str, script: str, damage: str
) -> None:
    directory = "auto-vllm" if engine == "vllm" else "auto-llamacpp"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_bundle(
        bundle,
        {
            f"{directory}/{script}": b"#!/bin/bash\necho stopped\n",
            "common/generations.py": TOOL.read_bytes(),
        },
    )
    runtime = _runtime(tmp_path, "runtime")
    root = tmp_path / "engine"
    generations.activate(root, bundle, directory, runtime, {"QIIP_ENGINE": engine})
    if damage == "binary":
        (runtime / "bin/vllm").write_text("corrupt")
    elif damage == "manifest":
        (runtime / "RUNTIME.json").unlink()
    else:
        shutil.rmtree(runtime)
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "exec",
            str(root),
            str(bundle / directory),
            script,
            "--force",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "stopped" in result.stdout


@pytest.mark.parametrize(
    "action,script",
    [
        ("exec", "start-vllm.sh"),
        ("exec-service", "start-vllm.sh"),
        ("exec-service", "preflight.sh"),
        ("exec-service", "wait-fabric.sh"),
    ],
)
def test_saved_launch_settings_apply_only_to_every_service_exec(
    tmp_path: Path, action: str, script: str
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_bundle(
        bundle,
        {
            "common/generations.py": TOOL.read_bytes(),
            **{
                f"auto-vllm/{name}": b"#!/bin/bash\nenv\n"
                for name in ("start-vllm.sh", "preflight.sh", "wait-fabric.sh")
            },
        },
    )
    root = tmp_path / "engine"
    selected = _activate(root, bundle, _runtime(tmp_path, "runtime"))
    saved = {
        "AUTOVLLM_MODEL": "old/model",
        "AUTOVLLM_GPU_DEVICES": "1",
        "AUTOVLLM_TENSOR_PARALLEL": "1",
        "AUTOVLLM_MAX_MODEL_LEN": "16384",
        "AUTOVLLM_TOOL_CALL_PARSER": "old-parser",
        "AUTOVLLM_REASONING_PARSER": "old-reasoning",
        "AUTOVLLM_EXTRA_ARGS": "--old-arg",
        "AUTOVLLM_MAX_BATCHED_TOKENS": "1000",
        "AUTOVLLM_GPU_MEM_UTIL": "0.8",
        "AUTOVLLM_DTYPE": "float16",
    }
    (root / "current/vllm.env").write_text(
        "\n".join(f"{key}={value}" for key, value in saved.items())
    )
    env = {"PATH": os.environ["PATH"]}
    if action == "exec":
        env["AUTOVLLM_MODEL"] = "new/model"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            action,
            str(root),
            str(bundle / "auto-vllm"),
            script,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if action == "exec":
        assert values["AUTOVLLM_MODEL"] == "new/model"
        assert not (saved.keys() - {"AUTOVLLM_MODEL"}) & values.keys()
    else:
        assert {key: values[key] for key in saved} == saved
    unit = (
        root / "generations" / selected["generation_id"] / "vllm.service"
    ).read_text()
    assert unit.count(" exec-service ") == 3


def test_setup_reactivation_clears_saved_settings_and_preserves_previous(
    tmp_path: Path,
) -> None:
    root = tmp_path / "engine"
    first_bundle = _bundle(tmp_path, "old-bundle")
    old = _activate(root, first_bundle, _runtime(tmp_path, "old-runtime"))
    old_env = root / "current/vllm.env"
    old_env.write_text("AUTOVLLM_MODEL=old/model\n")
    second_bundle = _bundle(tmp_path, "new-bundle")
    runtime = _runtime(tmp_path, "new-runtime")
    new = _activate(root, second_bundle, runtime)
    assert (root / "previous/vllm.env").read_text() == "AUTOVLLM_MODEL=old/model\n"
    (root / "current/vllm.env").write_text("AUTOVLLM_MODEL=new/model\n")
    assert _activate(root, second_bundle, runtime) == new
    assert not (root / "current/vllm.env").exists()
    assert (root / "previous").resolve().name == old["generation_id"]
    assert generations.rollback(root) == old
    assert (root / "current/vllm.env").read_text() == "AUTOVLLM_MODEL=old/model\n"


@pytest.mark.parametrize("recover", ["rollback", "activate"])
def test_rollback_journal_can_abandon_corrupt_current_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recover: str
) -> None:
    root = tmp_path / "engine"
    prior = _activate(
        root, _bundle(tmp_path, "old-bundle"), _runtime(tmp_path, "old-runtime")
    )
    runtime = _runtime(tmp_path, "broken-runtime")
    broken = _activate(root, _bundle(tmp_path, "broken-bundle"), runtime)
    (runtime / "RUNTIME.json").unlink()
    replace = os.replace

    def interrupt(source: Any, destination: Any) -> None:
        if Path(destination) == root / "current":
            raise OSError(errno.EIO, "controlled interrupted rollback")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(OSError):
            generations.rollback(root)
    assert json.loads((root / "ROLLBACK.json").read_text())["previous"].endswith(
        broken["generation_id"]
    )
    if recover == "rollback":
        assert generations.rollback(root) == prior
    else:
        replacement = _activate(
            root,
            _bundle(tmp_path, "replacement"),
            _runtime(tmp_path, "replacement-runtime"),
        )
        assert (root / "current").resolve().name == replacement["generation_id"]
        assert (root / "previous").resolve().name == prior["generation_id"]
    assert not (root / "ROLLBACK.json").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_publish_recovers_damaged_completed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    files = {"common/tool.py": b"print('verified')\n"}
    staged = tmp_path / ".upload-1"
    staged.mkdir()
    identity = write_bundle(staged, files)
    final = tmp_path / identity
    generations.publish_bundle(staged, final, identity)
    (final / "common/tool.py").unlink()
    staged.mkdir()
    write_bundle(staged, files)
    rename = os.rename

    def fail_publication(source: Any, destination: Any) -> None:
        if Path(destination) == final:
            raise OSError(errno.EIO, "controlled interrupted repair")
        rename(source, destination)

    if interrupt:
        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", fail_publication)
            with pytest.raises(OSError):
                generations.publish_bundle(staged, final, identity)
    generations.publish_bundle(staged, final, identity)
    generations.verify_bundle(final, identity)
    assert not staged.exists()
    assert len(list(tmp_path.glob(".corrupt-*"))) == 1


def test_reactivation_through_symlink_preserves_rollback_target(tmp_path: Path) -> None:
    target = tmp_path / "actual-root"
    target.mkdir()
    root = tmp_path / "symlink-root"
    root.symlink_to(target)
    prior = _activate(
        root, _bundle(tmp_path, "old-bundle"), _runtime(tmp_path, "old-runtime")
    )
    bundle = _bundle(tmp_path, "new-bundle")
    runtime = _runtime(tmp_path, "new-runtime")
    selected = _activate(root, bundle, runtime)
    assert _activate(root, bundle, runtime) == selected
    assert (root / "previous").resolve().name == prior["generation_id"]
    assert generations.rollback(root) == prior


def test_standalone_llamacpp_recovers_dangling_first_install_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "install-root"
    runtime = _runtime(root, "installed")
    links = tmp_path / "links"
    links.mkdir()
    for binary in ("llama-server", "llama-fit-params", "llama-quantize"):
        (links / binary).symlink_to(root / "current/bin" / binary)
        _write_executable(runtime / "bin" / binary, "#!/bin/bash\nexit 0\n")
    env = _script_environment(
        tmp_path, vllm_bin=runtime / "bin/vllm", process_log=tmp_path / "events"
    )
    _write_executable(
        Path(env["PATH"].split(":")[0]) / "sudo", '#!/bin/bash\nexec "$@"\n'
    )
    command = f"""
source {shlex.quote(str(ROOT / "auto-llamacpp/setup.sh"))}
LLAMACPP_INSTALL_ROOT={shlex.quote(str(root))}
LLAMACPP_LINK_DIR={shlex.quote(str(links))}
select_llamacpp_runtime {shlex.quote(str(runtime))}
"""
    result = subprocess.run(
        ["bash", "-c", command], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert (root / "current").resolve() == runtime
    assert all(
        (links / name).is_file()
        for name in ("llama-server", "llama-fit-params", "llama-quantize")
    )


def test_setup_generation_mutations_use_sudo(tmp_path: Path) -> None:
    env = _vllm_setup_environment(tmp_path)
    sudo = Path(env["PATH"].split(":")[0]) / "sudo"
    operations = tmp_path / "sudo.log"
    _write_executable(
        sudo,
        f'#!/bin/bash\nprintf \'%s\\n\' "$*" >> {shlex.quote(str(operations))}\nexec "$@"\n',
    )
    result = _setup(env)
    assert result.returncode == 0, result.stderr
    commands = operations.read_text().splitlines()
    for action in ("seal-runtime", "setup-bundle", "activate"):
        assert any(f"generations.py {action} " in command for command in commands)
