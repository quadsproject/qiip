"""start-llamacpp.sh in catalog-profile mode, run as real bash.

The arguments the script renders are pinned to the commands that were
measured, token for token, so a catalog edit cannot silently drift away from
what ``GPU-MODEL-SELECTION.md`` validated.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from inference_proxy.placement.catalog import BUILTIN_PROFILES, ModelProfile
from inference_proxy.provisioning.provisioner import NodeProvisioner

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = SCRIPT_ROOT / "auto-llamacpp" / "start-llamacpp.sh"
GPU_UUID = "GPU-1b2808fa-8417-2e5f-4664-7b30c5e1f2d8"

# The measured commands, verbatim from GPU-MODEL-SELECTION.md sections 3.6, 4.3
# and 5.3 (24 GB class).
MEASURED_COMMANDS = {
    "qwen3.8-27b-24g": (
        "llama-server -m Qwen3.8-27B-UD-Q4_K_S.gguf "
        "-c 262144 -np 1 -kvu -ngl 999 --fit off "
        "-fa on -ctk q4_0 -ctv q4_0 -ub 256 "
        "--spec-type draft-mtp --spec-draft-n-max 2 -ctkd f16 -ctvd f16 "
        "--no-mmproj --jinja "
        "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0"
    ),
    "qwen3.6-35b-a3b-24g": (
        "llama-server -m Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf "
        "-c 262144 -np 1 -kvu -ngl 999 --fit off "
        "-fa on -ctk q8_0 -ctv q8_0 -ub 256 "
        "--spec-type draft-mtp --spec-draft-n-max 2 -ctkd f16 -ctvd f16 "
        "--no-mmproj --jinja "
        "--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0"
    ),
    "muse-glimmer-30b-24g": (
        "llama-server -m Muse-Glimmer-30B-UD-Q5_K_M.gguf "
        "-md Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash "
        "--spec-draft-n-max 7 "
        "-c 131072 -np 1 -kvu -ngl 999 --fit off -fa on -ub 256 "
        "--no-mmproj --jinja "
        "--temp 1.0 --top-p 0.95 --top-k 64"
    ),
    # Measured on a real L4 on 2026-09-20 (gpu-model-research/validation).
    # Gemma's assistant is an MTP head in its own file: no draft cache flags.
    "gemma-4-31b-24g": (
        "llama-server -m gemma-4-31B-it-IQ4_XS.gguf "
        "-c 131072 -np 1 -kvu -ngl 999 --fit off "
        "-fa on -ctk q4_0 -ctv q4_0 -ub 256 "
        "--spec-type draft-mtp -md mtp-gemma-4-31B-it.gguf --spec-draft-n-max 4 "
        "--no-mmproj --jinja "
        "--temp 1.0 --top-p 0.95 --top-k 64"
    ),
}
_LONG = {
    "-m": "--model",
    "-md": "--model-draft",
    "-c": "--ctx-size",
    "-np": "--parallel",
    "-kvu": "--kv-unified",
    "-ngl": "--gpu-layers",
    "-fa": "--flash-attn",
    "-ctk": "--cache-type-k",
    "-ctv": "--cache-type-v",
    "-ctkd": "--cache-type-k-draft",
    "-ctvd": "--cache-type-v-draft",
    "-ub": "--ubatch-size",
}
_FLAGS = {"--kv-unified", "--no-mmproj", "--jinja", "--metrics"}
# What the managed launch adds around the measured command. The measured Muse
# command left the KV cache at llama.cpp's default, f16; the script states it.
_MANAGED_ONLY = {"--host", "--port", "--alias", "--verbosity", "--metrics"}


def _options(tokens: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        name = _LONG.get(tokens[index], tokens[index])
        assert name.startswith("--"), tokens[index]
        assert name not in options, f"{name} given twice"
        if name in _FLAGS:
            options[name] = ""
            index += 1
        else:
            options[name] = tokens[index + 1]
            index += 2
    for key in ("--model", "--model-draft"):
        if key in options:
            options[key] = Path(options[key]).name
    # "-ngl 999" and "--gpu-layers all" both mean every layer on the GPU.
    if options.get("--gpu-layers") == "999":
        options["--gpu-layers"] = "all"
    return options


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _profile(profile_id: str) -> ModelProfile:
    return next(item for item in BUILTIN_PROFILES if item.profile_id == profile_id)


class _Rig:
    def __init__(self, tmp_path: Path, *, gpus: int = 1, free_mib: int = 22564) -> None:
        self.root = tmp_path
        self.args_file = tmp_path / "server args"
        self.env_file = tmp_path / "server env"
        self.log_file = tmp_path / "llama.log"
        tools = tmp_path / "tools"
        tools.mkdir()
        rows = "\\n".join([str(free_mib)] * gpus)
        uuids = "\\n".join([GPU_UUID] * gpus)
        _write_executable(
            tools / "nvidia-smi",
            f"""#!/bin/bash
case "$*" in
    "") exit 0 ;;
    *--list-gpus*) for _ in $(seq {gpus}); do echo "GPU 0: NVIDIA L4"; done ;;
    *--query-gpu=name*) echo "NVIDIA L4" ;;
    *--query-gpu=memory.total*) echo 23034 ;;
    *--query-gpu=memory.free*) printf '{rows}\\n' ;;
    *--query-gpu=uuid*) printf '{uuids}\\n' ;;
    *) exit 9 ;;
esac
""",
        )
        self.fit = tmp_path / "llama-fit-params"
        _write_executable(
            self.fit,
            "#!/bin/bash\necho 'print_info: n_ctx_train = 262144' >&2\n",
        )
        self.server = tmp_path / "llama-server"
        _write_executable(
            self.server,
            "#!/bin/bash\n"
            'printf \'%s\\n\' "$@" > "$AUTOLLAMACPP_TEST_ARGS"\n'
            "printf '%s\\n' \"${GGML_CUDA_DISABLE_GRAPHS:-unset}\" "
            '> "$AUTOLLAMACPP_TEST_ENV"\n',
        )
        self.bundle = tmp_path / "bundle"
        self.bundle.mkdir()
        _write_executable(self.bundle / "stop-llamacpp.sh", "#!/bin/bash\nexit 0\n")
        self.tools = tools

    def env(self, profile: ModelProfile, **overrides: str) -> dict[str, str]:
        target = self.root / "hub" / profile.target.filename
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"weights")
        draft_artifact = None
        if profile.draft is not None:
            (self.root / "hub" / profile.draft.filename).write_bytes(b"draft")
        request = profile.runtime_request(
            reserve_mib=256,
            draft_artifact_id="d" * 64 if profile.draft is not None else None,
            gpu_class="l4",
        )
        env = {
            **{
                k: v for k, v in os.environ.items() if not k.startswith("AUTOLLAMACPP_")
            },
            "PATH": f"{self.tools}:/usr/bin:/bin",
            "AUTOLLAMACPP_NFS_MOUNT_POINT": str(self.root),
            "AUTOLLAMACPP_GGUF_PATH": f"hub/{profile.target.filename}",
            "AUTOLLAMACPP_MODEL_ALIAS": profile.target.repo_id,
            "AUTOLLAMACPP_MANAGED": "1",
            "AUTOLLAMACPP_FIT_TARGET_MIB": str(request.fit_target_mib),
            "AUTOLLAMACPP_MANAGED_SIZING": "profile",
            "AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT": str(request.context_per_slot),
            "AUTOLLAMACPP_MANAGED_PARALLEL": "1",
            "AUTOLLAMACPP_MANAGED_CACHE_TYPE": profile.cache_type.value,
            "AUTOLLAMACPP_BIN": str(self.server),
            "AUTOLLAMACPP_FIT_BIN": str(self.fit),
            "AUTOLLAMACPP_PID_FILE": str(self.root / "llama.pid"),
            "AUTOLLAMACPP_LOG_FILE": str(self.log_file),
            "AUTOLLAMACPP_TEST_ARGS": str(self.args_file),
            "AUTOLLAMACPP_TEST_ENV": str(self.env_file),
        }
        # The production mapping, not a copy of it.
        draft_artifact = None
        if profile.draft is not None:
            from inference_proxy.huggingface.artifacts import (
                GGUFArtifact,
                ResolvedGGUFArtifact,
            )

            draft_artifact = ResolvedGGUFArtifact(
                artifact=GGUFArtifact(
                    artifact_id="d" * 64,
                    repo_id=profile.draft.repo_id,
                    resolved_revision=profile.draft.revision,
                    files=(profile.draft.filename,),
                    entrypoint=profile.draft.filename,
                    model_alias=profile.draft.repo_id,
                    file_sizes={profile.draft.filename: profile.draft.size_bytes},
                ),
                node_relative_entrypoint=f"hub/{profile.draft.filename}",
            )
        env.update(NodeProvisioner._profile_script_env(request, draft_artifact))
        env.update(overrides)
        return env

    def run(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        command = "\n".join(
            (
                f"source {shlex.quote(str(START_SCRIPT))}",
                f"SCRIPT_DIR={shlex.quote(str(self.bundle))}",
                "detect_gpu_info",
                "configure_llamacpp_params",
                'verify_llamacpp_started() { wait "$1"; }',
                "run_llamacpp",
            )
        )
        return subprocess.run(
            ["bash", "-c", command],
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )


@pytest.mark.parametrize("profile_id", sorted(MEASURED_COMMANDS))
def test_profile_launch_matches_the_measured_command(
    tmp_path: Path, profile_id: str
) -> None:
    rig = _Rig(tmp_path)
    profile = _profile(profile_id)

    result = rig.run(rig.env(profile))

    assert result.returncode == 0, result.stderr
    rendered = _options(rig.args_file.read_text(encoding="utf-8").splitlines())
    measured = _options(shlex.split(MEASURED_COMMANDS[profile_id])[1:])
    if profile_id == "muse-glimmer-30b-24g":
        measured.update({"--cache-type-k": "f16", "--cache-type-v": "f16"})
    assert {k: v for k, v in rendered.items() if k not in _MANAGED_ONLY} == measured
    assert set(rendered) - set(measured) == _MANAGED_ONLY
    assert rig.env_file.read_text(encoding="utf-8").strip() == "unset"


def test_profile_plan_lines_record_the_launch_gate(tmp_path: Path) -> None:
    rig = _Rig(tmp_path, free_mib=22564)
    profile = _profile("qwen3.8-27b-24g")

    result = rig.run(rig.env(profile))

    assert result.returncode == 0, result.stderr
    lines = rig.log_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "qiip_fit_plan: sizing=profile train_context=262144 "
        "context_per_slot=262144 slots=1 aggregate_context=262144 "
        "fit_target_mib=256 cache_type_k=q4_0 cache_type_v=q4_0 flash_attn=on "
        "estimator_overrun_used=false"
    )
    assert lines[1] == (
        "qiip_profile_plan: profile_id=qwen3.8-27b-24g profile_version=1 "
        "ubatch=256 spec_type=draft-mtp spec_draft_n_max=2 draft_cache_type=f16 "
        f"draft_gguf=none required_free_mib={profile.required_free_mib} "
        f"gpu_free_mib=22564 gpu_uuid={GPU_UUID} cuda_graphs=on"
    )


def test_profile_refuses_a_gpu_without_the_measured_headroom(tmp_path: Path) -> None:
    profile = _profile("qwen3.8-27b-24g")
    rig = _Rig(tmp_path, free_mib=profile.required_free_mib + 255)

    result = rig.run(rig.env(profile))

    assert result.returncode != 0
    assert "plus a 256 MiB reserve" in result.stderr
    assert not rig.args_file.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"AUTOLLAMACPP_PROFILE_GPU_NAME": "NVIDIA GeForce RTX 3090"},
            "planned for 'NVIDIA GeForce RTX 3090', but this host has 'NVIDIA L4'",
        ),
        ({"AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB": "24000"}, "at least 24000 MiB"),
        ({"AUTOLLAMACPP_PROFILE_GPU_NAME": "NVIDIA L4; id"}, "nvidia-smi product name"),
        ({"AUTOLLAMACPP_PROFILE_GPU_NAME": ""}, "nvidia-smi product name"),
    ],
)
def test_profile_refuses_a_gpu_it_was_not_planned_for(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    rig = _Rig(tmp_path)

    result = rig.run(rig.env(_profile("qwen3.8-27b-24g"), **overrides))

    assert result.returncode != 0
    assert message in result.stderr
    assert not rig.args_file.exists()


def test_profile_refuses_a_multi_gpu_host(tmp_path: Path) -> None:
    rig = _Rig(tmp_path, gpus=2)

    result = rig.run(rig.env(_profile("qwen3.8-27b-24g")))

    assert result.returncode != 0
    assert "exactly one GPU per host; found 2" in result.stderr
    assert not rig.args_file.exists()


def test_profile_context_cannot_exceed_the_training_context(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    env = rig.env(
        _profile("qwen3.8-27b-24g"), AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT="524288"
    )

    result = rig.run(env)

    assert result.returncode != 0
    assert "exceeds model training context 262144" in result.stderr


def test_profile_can_disable_cuda_graphs(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    env = rig.env(
        _profile("qwen3.8-27b-24g"), AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS="1"
    )

    result = rig.run(env)

    assert result.returncode == 0, result.stderr
    assert rig.env_file.read_text(encoding="utf-8").strip() == "1"
    assert "cuda_graphs=off" in rig.log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"AUTOLLAMACPP_PROFILE_SPEC_TYPE": "draft-mtp --port 1"}, "must be draft-mtp"),
        ({"AUTOLLAMACPP_PROFILE_SPEC_TYPE": "ngram-mod"}, "must be draft-mtp"),
        ({"AUTOLLAMACPP_PROFILE_TEMPERATURE": "1.0 --api-key x"}, "plain decimals"),
        ({"AUTOLLAMACPP_PROFILE_TOP_K": "20;id"}, "top_k must be an integer"),
        ({"AUTOLLAMACPP_PROFILE_UBATCH": "8"}, "between 32 and 4096"),
        ({"AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX": "17"}, "between 1 and 16"),
        ({"AUTOLLAMACPP_MANAGED_PARALLEL": "2"}, "exactly one slot"),
        ({"AUTOLLAMACPP_MANAGED_CACHE_TYPE": "q5_1"}, "f16, q8_0 or q4_0"),
        ({"AUTOLLAMACPP_PROFILE_ID": "Bad Id"}, "catalog profile id"),
        ({"AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH": "hub/x.gguf"}, "takes no draft"),
        ({"AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE": ""}, "must state"),
        ({"AUTOLLAMACPP_EXTRA_ARGS": "--api-key x"}, "not supported for managed"),
        ({"AUTOLLAMACPP_MANAGED_ALLOW_ESTIMATOR_OVERRUN": "1"}, "estimator overrun"),
    ],
)
def test_profile_inputs_cannot_carry_extra_arguments(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    rig = _Rig(tmp_path)

    result = rig.run(rig.env(_profile("qwen3.8-27b-24g"), **overrides))

    assert result.returncode != 0
    assert message in result.stderr
    assert not rig.args_file.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH": ""},
            "MTP assistant profiles require",
        ),
        (
            {"AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE": "f16"},
            "shares the target KV cache",
        ),
        (
            {"AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH": "../outside.gguf"},
            "canonical relative POSIX path",
        ),
    ],
)
def test_mtp_assistant_inputs_are_checked(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    rig = _Rig(tmp_path)

    result = rig.run(rig.env(_profile("gemma-4-31b-24g"), **overrides))

    assert result.returncode != 0
    assert message in result.stderr
    assert not rig.args_file.exists()


def test_the_plan_line_names_the_assistant_type_not_the_server_flag(
    tmp_path: Path,
) -> None:
    """The plan keeps qiip's type, so the verifier knows a shared cache is due."""
    rig = _Rig(tmp_path)

    result = rig.run(rig.env(_profile("gemma-4-31b-24g")))

    assert result.returncode == 0, result.stderr
    log = rig.log_file.read_text(encoding="utf-8")
    assert "spec_type=draft-mtp-assistant " in log
    assert "draft_cache_type=default " in log


def test_dflash_draft_must_stay_inside_the_shared_cache(tmp_path: Path) -> None:
    rig = _Rig(tmp_path)
    env = rig.env(
        _profile("muse-glimmer-30b-24g"),
        AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH="../outside.gguf",
    )

    result = rig.run(env)

    assert result.returncode != 0
    assert "canonical relative POSIX path" in result.stderr
    assert not rig.args_file.exists()


@pytest.mark.parametrize("sizing", ["auto", "custom"])
def test_planner_sizing_rejects_profile_inputs(tmp_path: Path, sizing: str) -> None:
    rig = _Rig(tmp_path)
    env = rig.env(_profile("qwen3.8-27b-24g"), AUTOLLAMACPP_MANAGED_SIZING=sizing)
    if sizing == "auto":
        for name in ("CONTEXT_PER_SLOT", "PARALLEL", "CACHE_TYPE"):
            env[f"AUTOLLAMACPP_MANAGED_{name}"] = ""

    result = rig.run(env)

    assert result.returncode != 0
    assert "only valid with profile sizing" in result.stderr
    assert not rig.args_file.exists()


@pytest.mark.parametrize(
    "script", ["auto-llamacpp/start-llamacpp.sh", "auto-vllm/start-vllm.sh"]
)
def test_engine_log_sink_does_not_keep_a_wrapper_shell(script: str) -> None:
    """bash 5.1 keeps a wrapper shell for `>(cmd)` unless cmd is exec'ed.

    That wrapper holds the start script's stdout, so the gateway's command
    worker never saw the start command end (observed on a RHEL 9 A30 host:
    healthy server, provisioning stuck in starting_llamacpp). Newer bash execs
    by itself, so only the source can be asserted everywhere.
    """
    source = (SCRIPT_ROOT / script).read_text(encoding="utf-8")

    assert '>(exec python3 "${SCRIPT_DIR}/../common/provision-logs.py" engine' in source
    assert '>(python3 "${SCRIPT_DIR}/../common/provision-logs.py" engine' not in source


def test_start_command_output_ends_while_the_server_keeps_running(
    tmp_path: Path,
) -> None:
    """With remote logging on, whoever reads the script's stdout must get EOF."""
    rig = _Rig(tmp_path)
    _write_executable(
        rig.server, "#!/bin/bash\nexec sleep 20\n"
    )  # a server that stays up
    common = tmp_path / "common"
    common.mkdir()
    # Stand-in for the node-side sink: drains its input for as long as it is open.
    (common / "provision-logs.py").write_text(
        "import sys\nfor _ in sys.stdin:\n    pass\n", encoding="utf-8"
    )
    env = rig.env(_profile("qwen3.8-27b-24g"), QIIP_LOG_CONFIG="{}")
    command = "\n".join(
        (
            f"source {shlex.quote(str(START_SCRIPT))}",
            f"SCRIPT_DIR={shlex.quote(str(rig.bundle))}",
            "detect_gpu_info",
            "configure_llamacpp_params",
            "verify_llamacpp_started() { :; }",
            "run_llamacpp",
        )
    )
    process = subprocess.Popen(
        ["bash", "-c", command],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # communicate() returns only at EOF, i.e. when nothing holds the pipe.
        output, _ = process.communicate(timeout=8)
    finally:
        pid_file = tmp_path / "llama.pid"
        if pid_file.exists():
            subprocess.run(["kill", pid_file.read_text().strip()], check=False)
        process.kill()

    assert process.returncode == 0, output
    assert "llama-server started" in output


def test_profile_draft_preserves_split_artifact_storage_checks(tmp_path: Path) -> None:
    """The shared resolver must validate draft shards as well as target shards."""
    rig = _Rig(tmp_path)
    profile = _profile("muse-glimmer-30b-24g")
    env = rig.env(profile)
    first = tmp_path / "hub" / "draft-00001-of-00002.gguf"
    second = tmp_path / "hub" / "draft-00002-of-00002.gguf"
    first.write_bytes(b"first shard")
    env["AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH"] = str(first.relative_to(tmp_path))

    result = rig.run(env)
    assert result.returncode != 0
    assert "split GGUF shard missing" in result.stderr
    assert not rig.args_file.exists()

    second.write_bytes(b"second shard")
    result = rig.run(env)
    assert result.returncode == 0, result.stderr
    args = _options(rig.args_file.read_text().splitlines())
    assert args["--model-draft"] == first.name
