"""Artifact hits, compatibility failures and resource-aware source recovery."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from common import generations
from common import llamacpp_artifacts as artifacts
from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.models.node import InferenceEngine
from tests.provisioning.test_llamacpp_scripts import (
    SETUP_SCRIPT,
    _build_fixture,
    _run_shell,
    _source_setup,
    _source_start,
    _write_executable,
)
from tests.provisioning.test_provisioner import _make_provisioner


def _hardware(env: dict[str, str], root: Path) -> None:
    binaries = Path(env["PATH"].split(":")[0])
    library = root / "libcudart.so.13"
    library.write_bytes(b"CUDA library fixture")
    _write_executable(
        binaries / "ldd",
        f"#!/bin/bash\nif [ \"$*\" = --version ]; then echo 'ldd (GNU libc) 2.34'; else echo 'libcudart.so.13 => {library} (0x1234)'; fi\n",
    )
    _write_executable(binaries / "lspci", "#!/bin/bash\nexit 0\n")
    _write_executable(
        binaries / "nvidia-smi",
        """#!/bin/bash
case "$*" in
  *'--list-gpus'*) printf 'GPU 0: fixture\\nGPU 1: fixture\\n' ;;
  *'--query-gpu=name'*) echo 'NVIDIA fixture' ;;
  *'--query-gpu=memory.total'*) echo 81920 ;;
  *'--query-gpu=compute_cap'*)
    if [[ "$*" == *' -i '* ]]; then echo 8.0; else printf '8.0\\n9.0\\n'; fi ;;
  *'--query-gpu=driver_version'*) echo 580.126.09 ;;
esac
""",
    )
    (root / "os-release").write_text('ID=rhel\nVERSION_ID="9.5"\n')
    env["PROFILE_OS_RELEASE"] = str(root / "os-release")
    env["AUTOVLLM_DRIVER_STATE_DIR"] = str(root / "driver-state")


def _published(tmp_path: Path) -> tuple[dict[str, str], Path, dict[str, Any]]:
    producer = tmp_path / "producer"
    producer.mkdir()
    env, _log, links = _build_fixture(producer)
    _hardware(env, producer)
    result = _run_shell(
        _source_setup("select_runtime_profile llamacpp\ninstall_llamacpp"), env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    runtime = (links / "llama-server").resolve().parents[1]
    published = tmp_path / "published"
    result = _run_shell(
        _source_setup(
            f"llamacpp_artifact_tool publish {shlex.quote(str(runtime))} "
            f"{shlex.quote(str(published))} https://artifacts.example/llamacpp"
        ),
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    publication = json.loads(result.stdout.splitlines()[-1])
    # The node downloads through the real setup boundary, with a controlled
    # mirror instead of network/GPU/package mutation.
    binaries = Path(env["PATH"].split(":")[0])
    _write_executable(
        binaries / "wget",
        """#!/bin/bash
echo "wget $*" >> "$AUTOLLAMACPP_TEST_LOG"
url=''; output=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        -O) output="$2"; shift 2 ;;
        https://*) url="$1"; shift ;;
        *) shift ;;
    esac
done
case "$url" in
    */catalog.json) cp "$ARTIFACT_FIXTURE/catalog.json" "$output" ;;
    https://artifacts.example/*) cp "$ARTIFACT_FIXTURE/${url##*/}" "$output" ;;
    *) printf '%s' "$AUTOLLAMACPP_TEST_ARCHIVE" > "$output" ;;
esac
""",
    )
    target = tmp_path / "target"
    target.mkdir()
    env.update(
        AUTOVLLM_TMP_DIR=str(target),
        AUTOVLLM_MIN_FREE_GB="1",
        AUTOVLLM_NFS_EXPORT="storage:/cache",
        AUTOVLLM_NFS_MOUNT_POINT=str(target / "cache"),
        AUTOLLAMACPP_INSTALL_ROOT=str(target / "runtime"),
        AUTOLLAMACPP_LINK_DIR=str(target / "bin"),
        AUTOLLAMACPP_ARTIFACT_CATALOG_URL="https://artifacts.example/llamacpp/catalog.json",
        AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256=publication["catalog_sha256"],
        AUTOLLAMACPP_ALLOW_SOURCE_BUILD="0",
        QIIP_GENERATION_ROOT=str(target / "generations"),
        AUTOLLAMACPP_TEST_LOG=str(target / "operations.log"),
        ARTIFACT_FIXTURE=str(published),
    )
    return env, published, publication["artifact"]


def _install(env: dict[str, str]) -> Any:
    return _run_shell(
        _source_setup("select_runtime_profile llamacpp\ninstall_llamacpp"), env=env
    )


def test_documented_producer_command_uses_the_selected_script_bundle(
    tmp_path: Path,
) -> None:
    env, _published_dir, _entry = _published(tmp_path)
    result = _run_shell(
        f"source {shlex.quote(str(SETUP_SCRIPT))}\n"
        + """
mount_nfs_cache() { :; }
configure_firewall() { :; }
install_llmfit() { :; }
main
""",
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    readme = (SETUP_SCRIPT.parent / "README.md").read_text()
    command = readme.split("```bash\n", 1)[1].split("```", 1)[0]
    command = command.replace(
        "/opt/qiip/llama_cpp", env["QIIP_GENERATION_ROOT"]
    ).replace("/srv/llamacpp-artifacts", str(tmp_path / "republished"))
    result = _run_shell(command, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    catalog = tmp_path / "republished/catalog.json"
    assert json.loads(result.stdout.splitlines()[-1])[
        "catalog_sha256"
    ] == generations.file_digest(catalog)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"identity": 7},
        {"identity": "publication_schema=2"},
        {"sha256": "bad"},
        {"url": "file:///artifact"},
        {"url": None},
        {"archive_bytes": 0},
        {"unpacked_bytes": True},
    ],
)
def test_catalog_skips_bad_entries_before_a_valid_match(
    tmp_path: Path, bad: Any
) -> None:
    _env, published, entry = _published(tmp_path)
    malformed = {**entry, **bad} if isinstance(bad, dict) and bad else bad
    catalog = published / "catalog.json"
    catalog.write_text(json.dumps({"schema": 1, "artifacts": [malformed, entry]}))
    host = artifacts.metadata(entry["identity"])
    host.update(cuda_toolkit="13.0", cmake_cuda_architectures="native")
    assert artifacts.select(catalog, host) == entry
    catalog.write_text(json.dumps({"schema": 1, "artifacts": [malformed]}))
    assert artifacts.select(catalog, host) is None


@pytest.mark.parametrize("failure", ["library", "manifest", "incomplete", "inventory"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_damaged_install_is_quarantined_and_publication_retry_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, interrupted: bool
) -> None:
    env, published, entry = _published(tmp_path)
    staged = tmp_path / "staged"
    artifacts.unpack(published / Path(entry["url"]).name, staged, entry)
    final = Path(env["AUTOLLAMACPP_INSTALL_ROOT"]) / "runtime"
    artifacts.install(staged, final)
    if failure == "library":
        (final / "lib/libcudart.so.13").write_bytes(b"damaged library")
    elif failure == "manifest":
        (final / "RUNTIME.json").write_text("not json")
    elif failure == "inventory":
        damaged = json.loads((final / "RUNTIME.json").read_text())
        damaged["files"] = list(damaged["files"])
        (final / "RUNTIME.json").write_text(json.dumps(damaged))
    else:
        (final / "RUNTIME.json").unlink()
    rename = os.rename

    def interrupted_rename(source: Any, destination: Any) -> None:
        if Path(destination) == final:
            raise OSError("controlled interrupted publication")
        rename(source, destination)

    if interrupted:
        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", interrupted_rename)
            with pytest.raises(OSError, match="interrupted publication"):
                artifacts.install(staged, final)
        assert not final.exists()
        assert not list(final.parent.glob(".install-*"))
    artifacts.install(staged, final)
    artifacts.verify_package(final, entry["identity"])
    artifacts.install(staged, final)  # An intact installed copy is idempotent.
    assert len(list(final.parent.glob(".corrupt-*"))) == 1
    assert not list(final.parent.glob(".install-*"))
    assert not list(final.parent.glob("*.incomplete-*"))


@pytest.mark.parametrize("catalog", [False, True])
def test_setup_recovers_a_corrupt_inactive_runtime(
    tmp_path: Path, catalog: bool
) -> None:
    env, published, entry = _published(tmp_path)
    identity = hashlib.sha256(entry["identity"].encode()).hexdigest()[:16]
    final = Path(env["AUTOLLAMACPP_INSTALL_ROOT"]) / f"v0.4.1-{identity}"
    final.parent.mkdir()
    artifacts.unpack(published / Path(entry["url"]).name, final, entry)
    (final / "lib/libcudart.so.13").write_bytes(b"damaged")
    if not catalog:
        env.update(
            AUTOLLAMACPP_ARTIFACT_CATALOG_URL="",
            AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256="",
            AUTOLLAMACPP_ALLOW_SOURCE_BUILD="1",
        )
    result = _install(env)
    assert result.returncode == 0, result.stdout + result.stderr
    artifacts.verify_package(final, entry["identity"])
    assert (Path(env["AUTOLLAMACPP_LINK_DIR"]) / "llama-server").resolve().parents[
        1
    ] == final
    assert len(list(final.parent.glob(".corrupt-*"))) == 1


@pytest.mark.parametrize("stack_ok", [False, True])
def test_downloaded_cuda_failure_uses_fresh_proof_before_source_build(
    tmp_path: Path, stack_ok: bool
) -> None:
    env, published, entry = _published(tmp_path)
    archive = published / Path(entry["url"]).name
    extracted = tmp_path / "modified"
    artifacts.unpack(archive, extracted, entry)
    (extracted / "bin/cuda-probe").write_text("#!/bin/bash\nexit 10\n")
    artifacts.seal(extracted, entry["identity"])
    with tarfile.open(archive, "w:gz") as stream:
        for path in extracted.rglob("*"):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(extracted))
    entry.update(
        sha256=generations.file_digest(archive),
        archive_bytes=archive.stat().st_size,
        unpacked_bytes=sum(
            p.stat().st_size for p in extracted.rglob("*") if p.is_file()
        ),
    )
    (published / "catalog.json").write_text(
        json.dumps({"schema": 1, "artifacts": [entry]})
    )
    env.update(
        AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256=generations.file_digest(
            published / "catalog.json"
        ),
        AUTOLLAMACPP_ALLOW_SOURCE_BUILD="1",
    )
    if not stack_ok:
        nvcc = Path(env["AUTOLLAMACPP_NVCC"])
        nvcc.write_text(nvcc.read_text().replace("echo CUDA_EXECUTED", "exit 10"))
    result = _install(env)
    assert "[BUILD:fallback:artifact_cuda_proof_failed]" in result.stdout
    log = Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text()
    if stack_ok:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[BUILD:compile:OK:" in result.stdout
    else:
        assert result.returncode == 10, result.stdout + result.stderr
        assert "CUDA execution probe failed" in result.stderr
        assert "cmake" not in log
        assert "https://mirror.example/" not in log
        assert not list(Path(env["AUTOLLAMACPP_INSTALL_ROOT"]).glob("*/RUNTIME.json"))


@pytest.mark.parametrize("selection", ["native", "80-real;90-virtual"])
def test_engine_and_probe_share_explicit_targets_on_a_mixed_producer(
    tmp_path: Path, selection: str
) -> None:
    env, log, links = _build_fixture(tmp_path)
    env["AUTOLLAMACPP_CUDA_ARCHITECTURES"] = selection
    nvcc = Path(env["AUTOLLAMACPP_NVCC"])
    nvcc.write_text(
        nvcc.read_text().replace(
            "else\n",
            'else\n    printf \'probe <%s>\\n\' "$@" >> "$AUTOLLAMACPP_TEST_LOG"\n',
            1,
        )
    )
    result = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = "80;90" if selection == "native" else selection
    operations = log.read_text()
    assert f"<-DCMAKE_CUDA_ARCHITECTURES={expected}>" in operations
    for sm in ("80", "90"):
        code = (
            f"[sm_{sm},compute_{sm}]"
            if selection == "native"
            else (f"sm_{sm}" if sm == "80" else f"compute_{sm}")
        )
        assert f"<arch=compute_{sm},code={code}>" in operations
    runtime = (links / "llama-server").resolve().parents[1]
    info = artifacts.metadata(artifacts.verify_package(runtime)["identity"])
    assert info["compiled_cuda_architectures"] == "80,90"
    assert info["cmake_cuda_architectures"] == expected


@pytest.mark.parametrize("catalog", [True, False])
def test_download_failure_is_reported_without_claiming_an_incompatible_artifact(
    tmp_path: Path, catalog: bool
) -> None:
    env, _published_dir, _entry = _published(tmp_path)
    wget = Path(env["PATH"].split(":")[0]) / "wget"
    wget.write_text(
        wget.read_text().replace(
            'case "$url" in',
            f'if [[ "$url" {"==" if catalog else "!="} */catalog.json ]]; then exit 8; fi\ncase "$url" in',
        )
    )
    result = _install(env)
    assert result.returncode != 0
    assert "[ARTIFACT:miss:artifact_download_failed]" in result.stdout
    assert "artifact_download_failed" in result.stderr
    assert "source fallback is disabled" in result.stderr
    assert "cmake" not in Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text()


@pytest.mark.parametrize("cap, expected", [(None, 2), ("0", 2), ("64", 2), ("1", 1)])
def test_automatic_job_cap_still_obeys_cpu_and_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str | None, expected: int
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {10 * 1024**2} kB\n")
    monkeypatch.setenv("AUTOLLAMACPP_MEMINFO", str(meminfo))
    monkeypatch.delenv("AUTOLLAMACPP_BUILD_JOBS", raising=False)
    if cap is not None:
        monkeypatch.setenv("AUTOLLAMACPP_BUILD_JOBS", cap)
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "64")
    monkeypatch.setattr(artifacts, "capacity", lambda _requirements: None)
    assert artifacts.resources(tmp_path, tmp_path / "install")["jobs"] == expected


def test_compatibility_uses_compile_targets_instead_of_producer_gpu_inventory(
    tmp_path: Path,
) -> None:
    _env, _published_dir, entry = _published(tmp_path)
    info = artifacts.metadata(entry["identity"])
    info.update(
        compute_capabilities="8.0,9.0",
        cmake_cuda_architectures="80-real",
        compiled_cuda_architectures="80",
    )
    identity = "\n".join(f"{key}={value}" for key, value in info.items())
    host = {**info, "cuda_toolkit": "13.0", "cmake_cuda_architectures": "native"}
    assert not artifacts.compatible(identity, host)
    host["compute_capabilities"] = "8.0"
    assert artifacts.compatible(identity, host)
    info["compiled_cuda_architectures"] = "80,90"
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        artifacts.metadata("\n".join(f"{key}={value}" for key, value in info.items()))


def test_published_artifact_activates_without_a_compiler_and_launch_finds_libraries(
    tmp_path: Path,
) -> None:
    env, _published_dir, entry = _published(tmp_path)
    Path(env["AUTOLLAMACPP_NVCC"]).unlink()
    # Any attempt to install a compiler/toolkit is a test failure.
    _write_executable(Path(env["PATH"].split(":")[0]) / "dnf", "#!/bin/bash\nexit 99\n")
    result = _run_shell(
        f"source {shlex.quote(str(SETUP_SCRIPT))}\n"
        "mount_nfs_cache() { :; }\nconfigure_firewall() { :; }\n"
        "install_llmfit() { :; }\nmain",
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ARTIFACT:hit:catalog]" in result.stdout
    assert "[STEP:cuda_proof:OK]" in result.stdout
    assert "[TIMING:llamacpp:artifact_seconds=" in result.stdout
    selected = generations.evidence(
        (Path(env["QIIP_GENERATION_ROOT"]) / "current").resolve()
    )
    runtime = Path(selected["runtime_path"])
    assert artifacts.verify_package(runtime)["identity"] == entry["identity"]
    operations = Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text()
    assert "cmake" not in operations and "dnf" not in operations
    launch = _run_shell(
        _source_start('"$LLAMACPP_BIN" --version\nprintf "%s" "$LD_LIBRARY_PATH"'),
        env={
            **env,
            "QIIP_GENERATION_PINNED": "1",
            "AUTOLLAMACPP_BIN": str(runtime / "bin/llama-server"),
        },
    )
    assert launch.returncode == 0, launch.stderr
    assert str(runtime / "lib") in launch.stdout
    retry = _install(env)
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert "[ARTIFACT:hit:local]" in retry.stdout
    assert Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text().count("wget") == 2


@pytest.mark.parametrize(
    "mismatch",
    [
        "source_sha256",
        "fit_cli_patch_sha256",
        "profile",
        "cuda_toolkit",
        "compute_capabilities",
        "os_major",
        "glibc",
        "arch",
    ],
)
def test_catalog_compatibility_requires_source_transform_hardware_and_abi(
    tmp_path: Path, mismatch: str
) -> None:
    identity = "\n".join(
        f"{key}={value}"
        for key, value in {
            "publication_schema": "3",
            "version": "v0.4.1",
            "source_sha256": "a" * 64,
            "fit_cli_patch_sha256": "b" * 64,
            "build_profile": "portable",
            "compute_capabilities": "8.0",
            "compiled_cuda_architectures": "80",
            "cmake_cuda_architectures": "80",
            "cuda_toolkit": "13.0.88",
            "profile": "llamacpp-ampere-a100",
            "compiler": "GNU-11.5.0",
            "os_id": "rhel",
            "os_major": "9",
            "arch": "x86_64",
            "glibc": "2.34",
        }.items()
    )
    host = artifacts.metadata(identity)
    host["cuda_toolkit"] = "13.0"
    host["cmake_cuda_architectures"] = "native"
    assert artifacts.compatible(identity, host)
    host[mismatch] = {
        "source_sha256": "c" * 64,
        "fit_cli_patch_sha256": "c" * 64,
        "profile": "llamacpp-hopper",
        "cuda_toolkit": "12.9",
        "compute_capabilities": "9.0",
        "os_major": "8",
        "glibc": "2.28",
        "arch": "aarch64",
    }[mismatch]
    assert not artifacts.compatible(identity, host)


@pytest.mark.parametrize("failure", ["digest", "catalog", "version", "cli", "cuda"])
def test_invalid_artifact_never_activates(tmp_path: Path, failure: str) -> None:
    env, published, entry = _published(tmp_path)
    archive = published / Path(entry["url"]).name
    if failure == "digest":
        archive.write_bytes(b"damaged")
    elif failure == "catalog":
        (published / "catalog.json").write_text("{}")
    else:
        extracted = tmp_path / "modified"
        artifacts.unpack(archive, extracted, entry)
        if failure == "version":
            binary = extracted / "bin/llama-server"
            binary.write_text(
                binary.read_text().replace("version: 0.4.1", "version: 0.4.2")
            )
        elif failure == "cli":
            binary = extracted / "bin/llama-fit-params"
            binary.write_text(
                binary.read_text().replace('"$metadata"|"$estimate"', '"$metadata"')
            )
        else:
            (extracted / "bin/cuda-probe").write_text("#!/bin/bash\nexit 10\n")
        artifacts.seal(extracted, entry["identity"])
        with tarfile.open(archive, "w:gz") as stream:
            for path in extracted.rglob("*"):
                if path.is_file():
                    stream.add(path, arcname=path.relative_to(extracted))
        entry.update(
            sha256=generations.file_digest(archive),
            archive_bytes=archive.stat().st_size,
            unpacked_bytes=sum(
                p.stat().st_size for p in extracted.rglob("*") if p.is_file()
            ),
        )
        (published / "catalog.json").write_text(
            json.dumps({"schema": 1, "artifacts": [entry]})
        )
        env["AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256"] = generations.file_digest(
            published / "catalog.json"
        )
    result = _install(env)
    assert result.returncode != 0
    assert not (Path(env["AUTOLLAMACPP_LINK_DIR"]) / "llama-server").exists()
    assert not list(Path(env["AUTOLLAMACPP_INSTALL_ROOT"]).glob("*/RUNTIME.json"))
    assert "cmake" not in Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text()
    assert list(Path(env["AUTOVLLM_TMP_DIR"]).glob("llamacpp-artifact.*")) == []


def test_artifact_mismatch_uses_pinned_source_only_when_policy_allows(
    tmp_path: Path,
) -> None:
    env, _published_dir, _entry = _published(tmp_path)
    env["AUTOLLAMACPP_TEST_ARCHIVE"] += " different pin"
    env["AUTOLLAMACPP_SHA256"] = hashlib.sha256(
        env["AUTOLLAMACPP_TEST_ARCHIVE"].encode()
    ).hexdigest()
    denied = _install(env)
    assert denied.returncode != 0
    assert "source fallback is disabled" in denied.stderr
    assert "cmake" not in Path(env["AUTOLLAMACPP_TEST_LOG"]).read_text()
    env["AUTOLLAMACPP_ALLOW_SOURCE_BUILD"] = "1"
    allowed = _install(env)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "[BUILD:fallback:no_compatible_artifact]" in allowed.stdout
    runtime = (Path(env["AUTOLLAMACPP_LINK_DIR"]) / "llama-server").resolve().parents[1]
    assert env["AUTOLLAMACPP_SHA256"] in artifacts.verify_package(runtime)["identity"]


def test_low_memory_limits_actual_cmake_jobs_and_refuses_less_than_one_job(
    tmp_path: Path,
) -> None:
    env, log, _links = _build_fixture(tmp_path)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {10 * 1024**2} kB\n")
    env.update(AUTOLLAMACPP_MEMINFO=str(meminfo), AUTOLLAMACPP_BUILD_JOBS="64")
    _write_executable(
        Path(env["PATH"].split(":")[0]) / "nproc", "#!/bin/bash\necho 64\n"
    )
    built = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert built.returncode == 0, built.stdout + built.stderr
    assert "<--parallel> <2>" in log.read_text()
    env["AUTOLLAMACPP_SHA256"] = "f" * 64  # Force a miss on the local seal.
    meminfo.write_text(f"MemAvailable: {5 * 1024**2} kB\n")
    log.write_text("")
    refused = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert refused.returncode != 0
    assert "Insufficient available build memory" in refused.stderr
    assert "wget" not in log.read_text() and "cmake" not in log.read_text()


def test_source_build_and_package_target_only_selected_gpus(tmp_path: Path) -> None:
    env, log, links = _build_fixture(tmp_path)
    env["AUTOVLLM_GPU_DEVICES"] = "0"
    cmake = Path(env["PATH"].split(":")[0]) / "cmake"
    cmake.write_text(
        cmake.read_text().replace(
            "printf 'cmake'",
            'echo "CUDA_MASK=$CUDA_VISIBLE_DEVICES" >> "$AUTOLLAMACPP_TEST_LOG"\nprintf \'cmake\'',
        )
    )
    result = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    runtime = (links / "llama-server").resolve().parents[1]
    info = artifacts.metadata(artifacts.verify_package(runtime)["identity"])
    assert info["compute_capabilities"] == "8.0"
    assert info["compiled_cuda_architectures"] == "80"
    assert info["cmake_cuda_architectures"] == "80"
    assert "CUDA_MASK=0" in log.read_text()


def test_source_capacity_is_rechecked_after_toolchain_installation(
    tmp_path: Path,
) -> None:
    env, log, _links = _build_fixture(tmp_path)
    result = _run_shell(
        _source_setup("""
llamacpp_artifact_tool() {
    if [ "$1" = resources ]; then
        if [ -e "$INSTALL_TMP_DIR/resource-checked" ]; then
            echo 'FATAL: toolkit consumed required build/install space' >&2
            return 1
        fi
        touch "$INSTALL_TMP_DIR/resource-checked"
    fi
    python3 "${SCRIPT_DIR}/../common/llamacpp_artifacts.py" "$@"
}
install_llamacpp
"""),
        env=env,
    )
    assert result.returncode != 0
    assert "[STEP:cuda_toolkit:OK]" in result.stdout
    assert "toolkit consumed required build/install space" in result.stderr
    assert not log.exists() or (
        "wget" not in log.read_text() and "cmake" not in log.read_text()
    )


def test_capacity_accounts_for_build_and_install_on_the_same_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100, 30, 70)
    )
    with pytest.raises(ValueError, match="80 bytes required"):
        artifacts.capacity([(tmp_path, 50), (tmp_path / "missing/install", 30)])


def test_archive_links_and_traversal_are_rejected_before_extraction(
    tmp_path: Path,
) -> None:
    for index, filename in enumerate(
        ("../escape", "/escape", "bin/../../escape", "bin/tool")
    ):
        archive = tmp_path / f"bad-{index}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo(filename)
            if filename == "bin/tool":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            else:
                member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError, match="Unsafe artifact"):
            artifacts.unpack(
                archive,
                tmp_path / f"out-{index}",
                {
                    "sha256": generations.file_digest(archive),
                    "archive_bytes": archive.stat().st_size,
                    "unpacked_bytes": 1,
                },
            )
        assert not (tmp_path / f"out-{index}").exists()


def test_interrupted_package_copy_is_inactive_and_retry_can_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, published, entry = _published(tmp_path)
    staged = tmp_path / "staged"
    artifacts.unpack(published / Path(entry["url"]).name, staged, entry)
    final = Path(env["AUTOLLAMACPP_INSTALL_ROOT"]) / "new-runtime"
    original = shutil.copytree

    def interrupted_copy(source: Path, destination: Path, **kwargs: Any) -> None:
        (destination / "bin").mkdir()
        shutil.copy2(source / "bin/llama-server", destination / "bin/llama-server")
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", interrupted_copy)
    with pytest.raises(OSError, match="No space left"):
        artifacts.install(staged, final)
    assert not final.exists()
    assert not list(final.parent.glob(".install-*"))
    monkeypatch.setattr(shutil, "copytree", original)
    artifacts.install(staged, final)
    artifacts.verify_package(final)
    library = final / "lib/libcudart.so.13"
    library.write_bytes(b"corrupted library")
    with pytest.raises(ValueError, match="runtime data is corrupt"):
        generations.verify_runtime(final)


def test_local_lookup_never_selects_an_interrupted_publication(tmp_path: Path) -> None:
    env, published, entry = _published(tmp_path)
    root = Path(env["AUTOLLAMACPP_INSTALL_ROOT"])
    root.mkdir()
    staged = root / ".install-interrupted"
    artifacts.unpack(published / Path(entry["url"]).name, staged, entry)
    host = artifacts.metadata(entry["identity"])
    host["cuda_toolkit"] = "13.0"
    result = _run_shell(
        _source_setup(
            f"llamacpp_artifact_tool local {shlex.quote(str(root))} {shlex.quote(json.dumps(host))}"
        ),
        env=env,
    )
    assert result.returncode == 4, result.stderr
    assert result.stdout == ""


def test_interrupted_source_retry_preserves_prior_selection(tmp_path: Path) -> None:
    env, _log, links = _build_fixture(tmp_path)
    first = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    previous = (links / "llama-server").resolve().parents[1]
    env["AUTOLLAMACPP_TEST_ARCHIVE"] += " next source"
    env["AUTOLLAMACPP_SHA256"] = hashlib.sha256(
        env["AUTOLLAMACPP_TEST_ARCHIVE"].encode()
    ).hexdigest()
    sudo = Path(env["PATH"].split(":")[0]) / "sudo"
    original_sudo = sudo.read_text()
    sudo.write_text(
        original_sudo.replace(
            'exec "$@"',
            'if [[ "$*" == *"llamacpp_artifacts.py install"* ]]; then exit 143; fi\nexec "$@"',
        )
    )
    interrupted = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert interrupted.returncode == 143
    assert (links / "llama-server").resolve().parents[1] == previous
    artifacts.verify_package(previous)
    assert list(tmp_path.glob("auto-llamacpp.*")) == []
    sudo.write_text(original_sudo)
    retry = _run_shell(_source_setup("install_llamacpp"), env=env)
    assert retry.returncode == 0, retry.stdout + retry.stderr
    selected = (links / "llama-server").resolve().parents[1]
    assert selected != previous
    assert (previous.parent / "previous").resolve() == previous
    artifacts.verify_package(selected)


@pytest.mark.parametrize(
    "settings",
    [
        {"llamacpp_artifact_catalog_url": "https://mirror/catalog.json"},
        {"llamacpp_artifact_catalog_sha256": "a" * 64},
        {
            "llamacpp_artifact_catalog_url": "file:///catalog.json",
            "llamacpp_artifact_catalog_sha256": "a" * 64,
        },
        {
            "llamacpp_artifact_catalog_url": "https://user:secret@mirror/catalog.json",
            "llamacpp_artifact_catalog_sha256": "a" * 64,
        },
        {
            "llamacpp_artifact_catalog_url": "https://mirror/catalog.json",
            "llamacpp_artifact_catalog_sha256": "broken",
        },
        {"llamacpp_build_jobs": -1},
    ],
)
def test_catalog_configuration_must_pin_trust_and_bound_jobs(
    settings: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ProvisioningSettings(**settings)


def test_gateway_passes_artifact_policy_to_only_llamacpp_setup() -> None:
    settings = ProvisioningSettings(
        llamacpp_artifact_catalog_url="https://mirror/catalog.json",
        llamacpp_artifact_catalog_sha256="a" * 64,
        llamacpp_allow_source_build=False,
        llamacpp_build_jobs=3,
    )
    provisioner = _make_provisioner(settings=settings)
    env = provisioner._setup_script_env(InferenceEngine.LLAMA_CPP)
    assert (
        env["AUTOLLAMACPP_ARTIFACT_CATALOG_URL"]
        == settings.llamacpp_artifact_catalog_url
    )
    assert env["AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256"] == "a" * 64
    assert env["AUTOLLAMACPP_ALLOW_SOURCE_BUILD"] == "0"
    assert env["AUTOLLAMACPP_BUILD_JOBS"] == "3"
    assert "AUTOLLAMACPP_BUILD_JOBS" not in provisioner._setup_script_env()
