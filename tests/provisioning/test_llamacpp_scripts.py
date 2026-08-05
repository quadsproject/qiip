"""Behavioral tests for the managed llama.cpp CUDA bundle."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from inference_proxy.config.settings import (
    DEFAULT_LLAMACPP_SHA256,
    DEFAULT_LLAMACPP_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(os.environ.get("AUTOVLLM_TEST_SCRIPT_ROOT", REPO_ROOT))
SETUP_SCRIPT = SCRIPT_ROOT / "auto-llamacpp" / "setup.sh"
START_SCRIPT = SCRIPT_ROOT / "auto-llamacpp" / "start-llamacpp.sh"
PROCESS_SCRIPT = SCRIPT_ROOT / "auto-llamacpp" / "llamacpp-process.sh"


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
        timeout=10,
        check=False,
    )


def _source_setup(command: str) -> str:
    setup_script = shlex.quote(str(SETUP_SCRIPT))
    script_dir = shlex.quote(str(SCRIPT_ROOT / "auto-llamacpp"))
    return (
        f"AUTOLLAMACPP_SCRIPT_DIR={script_dir}\n"
        f"source <(sed '/^# --- Main ---/,$d' {setup_script})\n"
        f"{command}"
    )


def _source_start(command: str) -> str:
    start_script = shlex.quote(str(START_SCRIPT))
    return f"source {start_script}\n{command}"


def _write_fake_fit_planner(path: Path, *, train_context: int = 128000) -> None:
    _write_executable(
        path,
        f"""#!/bin/bash
if [[ " $* " != *' --kv-unified '* ]] || [[ " $* " != *' --gpu-layers all '* ]]; then
    exit 44
fi
context={train_context}
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ctx-size)
            context="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
echo 'llama_model_loader: n_ctx_train = {train_context}' >&2
context_mib=$((context * 1500 / {train_context}))
printf 'CUDA0 8500 %s 200\n' "$context_mib"
printf 'Host 0 0 100\n'
""",
    )


def test_start_resolves_exact_artifact_and_preserves_alias(tmp_path: Path) -> None:
    model = tmp_path / "gguf" / "artifact -- id" / "files" / "model---Q4.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    alias = "org/model--with---separators"
    env = {
        **os.environ,
        "AUTOLLAMACPP_NFS_MOUNT_POINT": str(tmp_path),
        "AUTOLLAMACPP_GGUF_PATH": str(model.relative_to(tmp_path)),
        "AUTOLLAMACPP_MODEL_ALIAS": alias,
    }

    result = _run_shell(
        _source_start(
            'resolve_gguf_artifact; printf "%s\\n%s\\n" "$GGUF_PATH" "$MODEL_ALIAS"'
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(model.resolve()), alias]


def test_start_preserves_split_entrypoint_filename_and_siblings(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "hub" / "models--org--split-model" / "snapshots" / ("a" * 40)
    blobs = tmp_path / "hub" / "models--org--split-model" / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)

    shards: list[Path] = []
    for index in (1, 2):
        blob = blobs / hashlib.sha256(str(index).encode()).hexdigest()
        blob.write_bytes(f"shard {index}".encode())
        shard = snapshot / f"model-{index:05d}-of-00002.gguf"
        shard.symlink_to(Path(os.path.relpath(blob, snapshot)))
        shards.append(shard)

    env = {
        **os.environ,
        "AUTOLLAMACPP_NFS_MOUNT_POINT": str(tmp_path),
        "AUTOLLAMACPP_GGUF_PATH": str(shards[0].relative_to(tmp_path)),
        "AUTOLLAMACPP_MODEL_ALIAS": "org/split-model",
    }
    result = _run_shell(
        _source_start(
            "\n".join(
                (
                    "resolve_gguf_artifact",
                    'sibling="$(dirname -- "$GGUF_PATH")/model-00002-of-00002.gguf"',
                    'test -f "$sibling"',
                    'printf "%s\\n%s\\n" "$GGUF_PATH" "$sibling"',
                )
            )
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(shards[0]), str(shards[1])]
    assert shards[0].resolve() != shards[0]


def test_managed_launch_uses_exact_artifact_and_explicit_alias(tmp_path: Path) -> None:
    """Exercise the public launch path without relying on the new helper name."""
    exact = tmp_path / "gguf" / "artifact -- id" / "files" / "model---q4_k_m.gguf"
    exact.parent.mkdir(parents=True)
    blob = tmp_path / "blobs" / hashlib.sha256(b"selected").hexdigest()
    blob.parent.mkdir()
    blob.write_bytes(b"selected")
    exact.symlink_to(Path(os.path.relpath(blob, exact.parent)))
    decoy = tmp_path / "gguf" / "unrelated" / "wrong-q4_k_m.gguf"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"wrong")
    alias = "org/model--with---separators"

    fake_server = tmp_path / "fake llama-server"
    fake_fit = tmp_path / "fake llama-fit-params"
    fake_tools = tmp_path / "fake tools"
    fake_tools.mkdir()
    args_file = tmp_path / "llama args"
    _write_executable(
        fake_server,
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$AUTOLLAMACPP_TEST_ARGS"\n',
    )
    _write_fake_fit_planner(fake_fit)
    _write_executable(
        fake_tools / "nvidia-smi",
        "#!/bin/bash\nprintf '22367\\n'\n",
    )
    script_bundle = tmp_path / "script bundle"
    script_bundle.mkdir()
    _write_executable(script_bundle / "stop-llamacpp.sh", "#!/bin/bash\nexit 0\n")
    env = {
        **os.environ,
        "PATH": f"{fake_tools}:/usr/bin:/bin",
        "AUTOLLAMACPP_NFS_MOUNT_POINT": str(tmp_path),
        "AUTOLLAMACPP_GGUF_PATH": str(exact.relative_to(tmp_path)),
        "AUTOLLAMACPP_MODEL_ALIAS": alias,
        "AUTOLLAMACPP_MANAGED": "1",
        "AUTOLLAMACPP_FIT_TARGET_MIB": "1536",
        "AUTOLLAMACPP_BIN": str(fake_server),
        "AUTOLLAMACPP_FIT_BIN": str(fake_fit),
        "AUTOLLAMACPP_PID_FILE": str(tmp_path / "llama.pid"),
        "AUTOLLAMACPP_LOG_FILE": str(tmp_path / "llama.log"),
        "AUTOLLAMACPP_TEST_ARGS": str(args_file),
    }
    command = "\n".join(
        (
            f"SCRIPT_DIR={shlex.quote(str(script_bundle))}",
            'GPU_COUNT=1; GPU_MODEL="fixture"; GPU_VRAM_GB=80',
            "configure_llamacpp_params",
            'verify_llamacpp_started() { wait "$1"; }',
            "run_llamacpp",
        )
    )

    result = _run_shell(_source_start(command), env=env)

    assert result.returncode == 0, result.stderr
    args = args_file.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--model") + 1] == str(exact)
    assert args[args.index("--alias") + 1] == alias
    assert args[args.index("--fit") + 1] == "off"
    assert "--fit-target" not in args
    assert args[args.index("--ctx-size") + 1] == "1024000"
    assert args[args.index("--parallel") + 1] == "8"
    assert args[args.index("--gpu-layers") + 1] == "all"
    assert "--kv-unified" in args
    assert args[args.index("--verbosity") + 1] == "4"
    assert "--verbose" not in args
    assert not ({"-b", "--batch-size"} & set(args))
    assert "--flash-attn" not in args
    assert "--cont-batching" not in args
    assert (tmp_path / "llama.log").read_text(encoding="utf-8") == (
        "qiip_fit_plan: context_per_slot=128000 slots=8 "
        "aggregate_context=1024000 fit_target_mib=1536\n"
    )


@pytest.mark.parametrize(
    ("train_context", "free_mib", "expected"),
    [
        (8192, 10500, "4096 1 4096"),
        (128000, 1000000, "128000 256 32768000"),
    ],
)
def test_managed_planner_reduces_context_only_when_needed_and_uses_library_limit(
    tmp_path: Path,
    train_context: int,
    free_mib: int,
    expected: str,
) -> None:
    fake_fit = tmp_path / "llama-fit-params"
    fake_tools = tmp_path / "bin"
    fake_tools.mkdir()
    _write_fake_fit_planner(fake_fit, train_context=train_context)
    _write_executable(
        fake_tools / "nvidia-smi",
        f"#!/bin/bash\nprintf '{free_mib}\\n'\n",
    )
    result = _run_shell(
        _source_start(
            "\n".join(
                (
                    "GPU_COUNT=1",
                    "GGUF_PATH=/cache/model.gguf",
                    "plan_managed_configuration",
                    'printf "%s %s %s\\n" "$MANAGED_CONTEXT_PER_SLOT" '
                    '"$MANAGED_PARALLEL" "$MANAGED_AGGREGATE_CONTEXT"',
                )
            )
        ),
        env={
            **os.environ,
            "PATH": f"{fake_tools}:/usr/bin:/bin",
            "AUTOLLAMACPP_FIT_BIN": str(fake_fit),
            "AUTOLLAMACPP_FIT_TARGET_MIB": "1024",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == expected


def test_managed_planner_names_missing_cuda_estimates(tmp_path: Path) -> None:
    fake_fit = tmp_path / "llama-fit-params"
    _write_executable(fake_fit, "#!/bin/bash\nprintf 'Vulkan0 10 20 30\\n'\n")

    result = _run_shell(
        _source_start(
            "\n".join(
                (
                    "GGUF_PATH=/cache/model.gguf",
                    "MANAGED_GPU_FREE_MIB=(4096)",
                    "managed_candidate_fits 4096 1",
                )
            )
        ),
        env={**os.environ, "AUTOLLAMACPP_FIT_BIN": str(fake_fit)},
    )

    assert result.returncode != 0
    assert "returned no CUDA device memory estimates" in result.stderr


@pytest.mark.parametrize(
    "override",
    [
        "AUTOLLAMACPP_GPU_LAYERS",
        "AUTOLLAMACPP_CTX_SIZE",
        "AUTOLLAMACPP_PARALLEL",
        "AUTOLLAMACPP_BATCH_SIZE",
        "AUTOLLAMACPP_EXTRA_ARGS",
    ],
)
def test_managed_start_rejects_manual_sizing_overrides(override: str) -> None:
    env = {
        **os.environ,
        "AUTOLLAMACPP_MANAGED": "1",
        override: "1",
    }

    result = _run_shell(
        _source_start("configure_llamacpp_params"),
        env=env,
    )

    assert result.returncode != 0
    assert f"{override} is not supported for managed llama.cpp" in result.stderr


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_managed_start_requires_positive_fit_target(value: str) -> None:
    result = _run_shell(
        _source_start("configure_llamacpp_params"),
        env={
            **os.environ,
            "AUTOLLAMACPP_MANAGED": "1",
            "AUTOLLAMACPP_FIT_TARGET_MIB": value,
        },
    )

    assert result.returncode != 0
    assert "must be a positive integer MiB value" in result.stderr


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("../outside.gguf", "canonical relative POSIX path"),
        ("/absolute.gguf", "relative to the NFS mount"),
        ("not-a-model.bin", "must end in .gguf"),
    ],
)
def test_start_rejects_unsafe_artifact_paths(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    env = {
        **os.environ,
        "AUTOLLAMACPP_NFS_MOUNT_POINT": str(tmp_path),
        "AUTOLLAMACPP_GGUF_PATH": relative_path,
        "AUTOLLAMACPP_MODEL_ALIAS": "org/model",
    }

    result = _run_shell(_source_start("resolve_gguf_artifact"), env=env)

    assert result.returncode != 0
    assert message in result.stderr


def test_start_rejects_artifact_symlink_escaping_mount(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.gguf"
    outside.write_bytes(b"outside")
    link = tmp_path / "gguf" / "artifact" / "files" / "model.gguf"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    env = {
        **os.environ,
        "AUTOLLAMACPP_NFS_MOUNT_POINT": str(tmp_path),
        "AUTOLLAMACPP_GGUF_PATH": str(link.relative_to(tmp_path)),
        "AUTOLLAMACPP_MODEL_ALIAS": "org/model",
    }

    result = _run_shell(_source_start("resolve_gguf_artifact"), env=env)

    assert result.returncode != 0
    assert "escapes the NFS mount" in result.stderr


def test_start_has_no_global_search_alias_derivation_or_quantization_selector() -> None:
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert "find_gguf_model" not in source
    assert "derive_model_alias" not in source
    assert "AUTOLLAMACPP_QUANTIZATION" not in source
    assert 'find "$gguf_dir"' not in source


def _build_fixture(
    tmp_path: Path, *, valid_checksum: bool = True
) -> tuple[dict[str, str], Path, Path]:
    operation_log = tmp_path / "operations.log"
    fake_bin = tmp_path / "fake bin -- tools"
    fake_bin.mkdir()
    archive_content = "verified llama.cpp source fixture"
    digest = hashlib.sha256(archive_content.encode()).hexdigest()
    if not valid_checksum:
        digest = "0" * 64

    _write_executable(
        fake_bin / "wget",
        """#!/bin/bash
printf 'wget' >> "$AUTOLLAMACPP_TEST_LOG"
printf ' <%s>' "$@" >> "$AUTOLLAMACPP_TEST_LOG"
printf '\n' >> "$AUTOLLAMACPP_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-O' ]]; then
        printf '%s' "$AUTOLLAMACPP_TEST_ARCHIVE" > "$2"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        fake_bin / "tar",
        """#!/bin/bash
echo tar >> "$AUTOLLAMACPP_TEST_LOG"
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == '-C' ]]; then
        mkdir -p "$2/common"
        printf 'fixture' > "$2/CMakeLists.txt"
        printf '%s\n' ').set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));' > "$2/common/arg.cpp"
        exit 0
    fi
    shift
done
exit 2
""",
    )
    _write_executable(
        fake_bin / "cmake",
        """#!/bin/bash
printf 'cmake' >> "$AUTOLLAMACPP_TEST_LOG"
printf ' <%s>' "$@" >> "$AUTOLLAMACPP_TEST_LOG"
printf '\n' >> "$AUTOLLAMACPP_TEST_LOG"
if [[ "${AUTOLLAMACPP_TEST_CMAKE_FAIL:-}" == 'configure' && "${1:-}" != '--build' ]]; then
    exit 42
fi
if [[ "${1:-}" != '--build' ]]; then
    while [[ "$#" -gt 0 ]]; do
        if [[ "$1" == '-S' ]]; then
            grep -Fq 'LLAMA_EXAMPLE_PARALLEL, LLAMA_EXAMPLE_FIT_PARAMS' "$2/common/arg.cpp" || exit 43
            break
        fi
        shift
    done
fi
if [[ "${1:-}" == '--build' ]]; then
    build_dir="$2"
    mkdir -p "$build_dir/bin"
    cat > "$build_dir/bin/llama-server" <<'EOF'
#!/bin/bash
echo 'version: 10242 (b10242)' >&2
EOF
    cat > "$build_dir/bin/llama-fit-params" <<'EOF'
#!/bin/bash
printf 'fit' >> "$AUTOLLAMACPP_TEST_LOG"
printf ' <%s>' "$@" >> "$AUTOLLAMACPP_TEST_LOG"
printf '\n' >> "$AUTOLLAMACPP_TEST_LOG"
expected='--ctx-size 4096 --parallel 2 --kv-unified --gpu-layers all --fit-print on --verbosity 4 --version'
if [ "$*" != "$expected" ]; then
    exit 45
fi
echo 'version: 10242 (b10242)' >&2
EOF
    cat > "$build_dir/bin/llama-quantize" <<'EOF'
#!/bin/bash
exit 0
EOF
    chmod +x "$build_dir/bin/llama-server" "$build_dir/bin/llama-fit-params" "$build_dir/bin/llama-quantize"
fi
""",
    )
    _write_executable(fake_bin / "ninja", "#!/bin/bash\nexit 0\n")
    _write_executable(fake_bin / "nvcc", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/bin/bash
if [[ "$*" == *'--query-gpu=compute_cap'* ]]; then
    printf '8.0\n9.0\n'
    exit 0
fi
exit 1
""",
    )
    _write_executable(
        fake_bin / "sudo",
        """#!/bin/bash
printf 'sudo' >> "$AUTOLLAMACPP_TEST_LOG"
printf ' <%s>' "$@" >> "$AUTOLLAMACPP_TEST_LOG"
printf '\n' >> "$AUTOLLAMACPP_TEST_LOG"
exec "$@"
""",
    )

    install_root = tmp_path / "install -- root"
    link_dir = tmp_path / "links -- bin"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOVLLM_TMP_DIR": str(tmp_path),
        "AUTOLLAMACPP_VERSION": "b10242",
        "AUTOLLAMACPP_SHA256": digest,
        "AUTOLLAMACPP_SOURCE_URL": (
            "https://mirror.example/llama.cpp/b10242/source.tar.gz"
        ),
        "AUTOLLAMACPP_INSTALL_ROOT": str(install_root),
        "AUTOLLAMACPP_LINK_DIR": str(link_dir),
        "AUTOLLAMACPP_NVCC": str(fake_bin / "nvcc"),
        "AUTOLLAMACPP_TEST_LOG": str(operation_log),
        "AUTOLLAMACPP_TEST_ARCHIVE": archive_content,
    }
    return env, operation_log, link_dir


def test_setup_defaults_match_gateway_source_pair() -> None:
    result = _run_shell(
        _source_setup(
            'printf \'%s\\n\' "$LLAMACPP_VERSION" "$LLAMACPP_SHA256" '
            '"$LLAMACPP_SOURCE_URL"'
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        DEFAULT_LLAMACPP_VERSION,
        DEFAULT_LLAMACPP_SHA256,
        ("https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10242.tar.gz"),
    ]


def test_install_builds_verified_cuda_source_with_minimal_targets(
    tmp_path: Path,
) -> None:
    env, operation_log, link_dir = _build_fixture(tmp_path)

    result = _run_shell(_source_setup("install_llamacpp"), env=env)

    assert result.returncode == 0, result.stderr
    operations = operation_log.read_text().splitlines()
    assert operations[0].startswith("wget <-q> <https://mirror.example/")
    assert operations[1] == "tar"
    configure = next(line for line in operations if line.startswith("cmake <-S>"))
    assert "<-DGGML_CUDA=ON>" in configure
    assert "<-DGGML_NATIVE=OFF>" in configure
    assert "<-DCMAKE_CUDA_ARCHITECTURES=native>" in configure
    assert "<-DBUILD_SHARED_LIBS=OFF>" in configure
    assert "<-DLLAMA_BUILD_TESTS=OFF>" in configure
    assert "<-DLLAMA_BUILD_EXAMPLES=OFF>" in configure
    assert "<-DLLAMA_BUILD_SERVER=ON>" in configure
    assert "<-DLLAMA_BUILD_UI=OFF>" in configure
    assert "<-DLLAMA_BUILD_NUMBER=10242>" in configure
    assert "<-DLLAMA_BUILD_COMMIT=b10242>" in configure
    build = next(line for line in operations if line.startswith("cmake <--build>"))
    assert "<llama-server> <llama-fit-params> <llama-quantize>" in build
    assert (
        "fit <--ctx-size> <4096> <--parallel> <2> <--kv-unified> "
        "<--gpu-layers> <all> <--fit-print> <on> <--verbosity> <4> <--version>"
    ) in operations
    assert (link_dir / "llama-server").resolve().is_file()
    assert (link_dir / "llama-fit-params").resolve().is_file()
    assert (link_dir / "llama-quantize").resolve().is_file()
    marker = (link_dir / "llama-server").resolve().parents[1] / "BUILD-INFO"
    marker_text = marker.read_text()
    assert "build_profile=cuda-portable-cpu-v2-fit-concurrency" in marker_text
    assert (
        "fit_cli_patch_sha256="
        "58917efc78ca760a2a1dd162d84e6cf1930c5b62a8dd9710bb4579ca4f2d69dc"
    ) in marker_text
    assert "compute_capabilities=8.0,9.0" in marker_text


def test_fit_cli_patch_body_is_digest_verified() -> None:
    result = _run_shell(
        _source_setup(
            'LLAMACPP_FIT_PATCH_TO="${LLAMACPP_FIT_PATCH_TO} changed"\n'
            "verify_fit_params_patch_identity"
        )
    )

    assert result.returncode != 0
    assert "source transformation digest mismatch" in result.stderr


def test_install_is_idempotent_for_source_and_gpu_identity(tmp_path: Path) -> None:
    env, operation_log, _link_dir = _build_fixture(tmp_path)

    first = _run_shell(_source_setup("install_llamacpp"), env=env)
    second = _run_shell(_source_setup("install_llamacpp"), env=env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "already installed for CUDA capabilities 8.0,9.0" in second.stdout
    assert (
        sum(line.startswith("wget") for line in operation_log.read_text().splitlines())
        == 1
    )


def test_fit_concurrency_profile_invalidates_v1_build(tmp_path: Path) -> None:
    env, operation_log, link_dir = _build_fixture(tmp_path)
    old_marker = "\n".join(
        (
            "version=b10242",
            f"source_sha256={env['AUTOLLAMACPP_SHA256']}",
            "build_profile=cuda-portable-cpu-v1",
            "compute_capabilities=8.0,9.0",
            "cmake_cuda_architectures=native",
        )
    )
    old_identity = hashlib.sha256(old_marker.encode()).hexdigest()[:16]
    old_install = Path(env["AUTOLLAMACPP_INSTALL_ROOT"]) / f"b10242-{old_identity}"
    old_bin = old_install / "bin"
    old_bin.mkdir(parents=True)
    _write_executable(
        old_bin / "llama-server",
        "#!/bin/bash\necho 'version: 10242 (b10242)' >&2\n",
    )
    _write_executable(old_bin / "llama-quantize", "#!/bin/bash\nexit 0\n")
    (old_install / "BUILD-INFO").write_text(f"{old_marker}\n", encoding="utf-8")

    result = _run_shell(_source_setup("install_llamacpp"), env=env)

    assert result.returncode == 0, result.stderr
    assert (link_dir / "llama-server").resolve().parent != old_bin
    assert (
        sum(line.startswith("wget") for line in operation_log.read_text().splitlines())
        == 1
    )


def test_checksum_mismatch_never_extracts_builds_or_installs(tmp_path: Path) -> None:
    env, operation_log, link_dir = _build_fixture(tmp_path, valid_checksum=False)

    result = _run_shell(_source_setup("install_llamacpp"), env=env)

    assert result.returncode != 0
    assert "SHA-256 verification failed" in result.stderr
    operations = operation_log.read_text().splitlines()
    assert len([line for line in operations if line.startswith("wget")]) == 1
    assert "tar" not in operations
    assert not any(line.startswith("cmake") for line in operations)
    assert not any(line.startswith("sudo") for line in operations)
    assert not link_dir.exists()


def test_failed_build_removes_temporary_work_directory(tmp_path: Path) -> None:
    env, _operation_log, link_dir = _build_fixture(tmp_path)
    env["AUTOLLAMACPP_TEST_CMAKE_FAIL"] = "configure"

    result = _run_shell(
        _source_setup('run_with_errexit install_llamacpp\nexit "$STEP_STATUS"'),
        env=env,
    )

    assert result.returncode == 42
    assert list(tmp_path.glob("auto-llamacpp.*")) == []
    assert not link_dir.exists()


def test_empty_digest_fails_before_download(tmp_path: Path) -> None:
    env, operation_log, _link_dir = _build_fixture(tmp_path)
    env["AUTOLLAMACPP_SHA256"] = ""

    result = _run_shell(_source_setup("install_llamacpp"), env=env)

    assert result.returncode == 2
    assert "requires a 64-character SHA-256" in result.stderr
    assert not operation_log.exists()


def test_version_parser_reads_real_stderr_shape(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    _write_executable(
        binary,
        "#!/bin/bash\necho 'version: 10242 (fixture)' >&2\n",
    )

    result = _run_shell(
        _source_setup(f"installed_llamacpp_version {shlex.quote(str(binary))}")
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "b10242\n"


def test_managed_start_refuses_missing_nvidia_driver(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "nvidia-smi", "#!/bin/bash\nexit 1\n")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOLLAMACPP_SCRIPT_DIR": str(SCRIPT_ROOT / "auto-llamacpp"),
    }

    result = subprocess.run(
        ["bash", str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert "managed llama.cpp requires a working NVIDIA GPU" in result.stderr


def test_standalone_start_can_explicitly_allow_cpu_only(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "nvidia-smi", "#!/bin/bash\nexit 1\n")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "AUTOLLAMACPP_REQUIRE_CUDA": "0",
        "AUTOLLAMACPP_SCRIPT_DIR": str(SCRIPT_ROOT / "auto-llamacpp"),
    }

    result = _run_shell(
        f"source {shlex.quote(str(START_SCRIPT))}\n"
        "detect_gpu_info\n"
        'printf "%s\\n" "$GPU_COUNT"',
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\n"


def test_start_clears_llama_argument_namespace_including_context_size() -> None:
    env = {
        **os.environ,
        "AUTOLLAMACPP_SCRIPT_DIR": str(SCRIPT_ROOT / "auto-llamacpp"),
        "LLAMA_ARG_MODEL": "attacker/model.gguf",
        "LLAMA_ARG_PORT": "9999",
        "LLAMA_ARG_CTX_SIZE": "64",
    }
    result = _run_shell(
        f"source {shlex.quote(str(START_SCRIPT))}\n"
        "clear_script_environment\n"
        "if compgen -A variable LLAMA_ARG_ >/dev/null; then exit 9; fi",
        env=env,
    )

    assert result.returncode == 0, result.stderr


def _write_process_record(
    proc_root: Path,
    pid: int,
    *,
    executable: Path,
    argv: list[str],
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "exe").symlink_to(executable)
    (process_dir / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in argv) + b"\0"
    )


@pytest.mark.parametrize(
    ("same_executable", "argv", "expected"),
    [
        (True, ["llama-server", "--model", "/cache/model.gguf"], True),
        (True, ["llama-server", "--help"], False),
        (False, ["logger", "--file", "llama-server-output.log"], False),
    ],
)
def test_process_identity_uses_executable_and_exact_model_token(
    tmp_path: Path,
    *,
    same_executable: bool,
    argv: list[str],
    expected: bool,
) -> None:
    bin_dir = tmp_path / "bin -- path with spaces"
    bin_dir.mkdir()
    llama_binary = bin_dir / "llama-server"
    other_binary = bin_dir / "not-llama-server"
    _write_executable(llama_binary, "#!/bin/bash\nexit 0\n")
    _write_executable(other_binary, "#!/bin/bash\nexit 0\n")
    proc_root = tmp_path / "proc"
    _write_process_record(
        proc_root,
        123,
        executable=llama_binary if same_executable else other_binary,
        argv=argv,
    )

    result = _run_shell(
        f"PROC_ROOT={shlex.quote(str(proc_root))}\n"
        f"LLAMACPP_BIN={shlex.quote(str(llama_binary))}\n"
        f"source {shlex.quote(str(PROCESS_SCRIPT))}\n"
        "is_llamacpp_pid 123",
    )

    assert (result.returncode == 0) is expected


def test_process_identity_survives_managed_binary_relink(tmp_path: Path) -> None:
    install_root = tmp_path / "install -- root"
    old_binary = install_root / "b10242-old" / "bin" / "llama-server"
    new_binary = install_root / "b10243-new" / "bin" / "llama-server"
    old_binary.parent.mkdir(parents=True)
    new_binary.parent.mkdir(parents=True)
    _write_executable(old_binary, "#!/bin/bash\nexit 0\n")
    _write_executable(new_binary, "#!/bin/bash\nexit 0\n")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    current_binary = link_dir / "llama-server"
    current_binary.symlink_to(old_binary)
    proc_root = tmp_path / "proc"
    _write_process_record(
        proc_root,
        123,
        executable=old_binary,
        argv=["llama-server", "--model", "/cache/model.gguf"],
    )
    command = (
        f"PROC_ROOT={shlex.quote(str(proc_root))}\n"
        f"LLAMACPP_BIN={shlex.quote(str(current_binary))}\n"
        f"LLAMACPP_INSTALL_ROOT={shlex.quote(str(install_root))}\n"
        f"source {shlex.quote(str(PROCESS_SCRIPT))}\n"
        "is_llamacpp_pid 123"
    )

    before_relink = _run_shell(command)
    current_binary.unlink()
    current_binary.symlink_to(new_binary)
    after_relink = _run_shell(command)

    assert before_relink.returncode == 0
    assert after_relink.returncode == 0
