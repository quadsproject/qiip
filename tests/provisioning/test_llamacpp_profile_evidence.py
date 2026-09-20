"""Profile-launch verification against real llama.cpp v0.4.1 startup logs.

The fixtures are genuine engine logs (see their README). Each rejection test
changes exactly one fact in a real log, so it proves the verifier notices that
fact; none of them invents a log shape.

These tests qualify the parser against the pinned llama.cpp revision. They say
nothing about whether a profile fits an L4 or an A30.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.huggingface.artifacts import GGUFArtifact, ResolvedGGUFArtifact
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import (
    LlamaCppCacheType,
    LlamaCppRuntimeRequest,
    LlamaCppSizingMode,
    LlamaCppSpeculativeType,
    NodeGPU,
)
from inference_proxy.placement.catalog import BUILTIN_PROFILES, ModelProfile
from inference_proxy.provisioning.llamacpp_profile import (
    ProfileEvidenceError,
    parse_profile_evidence,
)
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
    _parse_llamacpp_runtime_fit,
)

FIXTURES = Path(__file__).parent / "fixtures" / "llamacpp-v0.4.1"
GPU_UUID = "GPU-17c2f9f6-51c5-83cb-7ad6-71d9ec81806e"
# The Gemma log was captured on the workstation's other GPU.
GEMMA_GPU_UUID = "GPU-df7b0ed0-b4dc-5aad-acfe-659e2877b6f3"
GEMMA = "gemma-4-31b-24g"
DRAFT_ID = "d" * 64
DRAFT_PATH = (
    "hub/models--z-lab--Muse-Glimmer-30B-DFlash2-GGUF/snapshots/"
    "880882627431093d99d3b2368efb4a6fcf12d4cb/Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf"
)
GEMMA_DRAFT_PATH = (
    "hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/"
    "c1ac76e99d5513b141e8adde7288b85c3f9c32ec/mtp-gemma-4-31B-it.gguf"
)
CASES = {
    "qwen3.8-27b-24g": ("qwen3.8-27b-mtp.engine.log", 22000),
    "qwen3.6-35b-a3b-24g": ("qwen3.6-35b-a3b-mtp.engine.log", 20800),
    "muse-glimmer-30b-24g": ("muse-glimmer-30b-dflash.engine.log", 21600),
    GEMMA: ("gemma-4-31b-assistant.engine.log", 21897),
}


def _gpu_uuid(profile_id: str) -> str:
    return GEMMA_GPU_UUID if profile_id == GEMMA else GPU_UUID


def _draft_path(profile_id: str) -> str:
    return GEMMA_DRAFT_PATH if profile_id == GEMMA else DRAFT_PATH


def _log(profile_id: str) -> str:
    return (FIXTURES / CASES[profile_id][0]).read_text(encoding="utf-8")


def _profile(profile_id: str) -> ModelProfile:
    return next(item for item in BUILTIN_PROFILES if item.profile_id == profile_id)


def _request(profile_id: str) -> LlamaCppRuntimeRequest:
    """The catalog request, with the launch gate the capture run used."""
    profile = _profile(profile_id)
    request = profile.runtime_request(
        reserve_mib=512,
        draft_artifact_id=DRAFT_ID if profile.draft is not None else None,
        gpu_class="l4",
    )
    assert request.profile is not None
    return request.model_copy(
        update={
            "profile": request.profile.model_copy(
                update={"required_free_mib": CASES[profile_id][1]}
            )
        }
    )


def _draft_artifact(profile_id: str) -> ResolvedGGUFArtifact | None:
    draft = _profile(profile_id).draft
    if draft is None:
        return None
    return ResolvedGGUFArtifact(
        artifact=GGUFArtifact(
            artifact_id=DRAFT_ID,
            repo_id=draft.repo_id,
            resolved_revision=draft.revision,
            files=(draft.filename,),
            entrypoint=draft.filename,
            model_alias=draft.repo_id,
            file_sizes={draft.filename: draft.size_bytes},
        ),
        node_relative_entrypoint=_draft_path(profile_id),
    )


def _provisioner(
    log_text: str, *, memory: str = "0, 97887, 30000, 67887"
) -> NodeProvisioner:
    provisioner = NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=MagicMock(),
        settings=ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=EndpointPolicy.from_values(
            allowed_hosts=["host1"], allowed_networks=[], allowed_ports=[8000]
        ),
    )
    provisioner._ssh_run_command = AsyncMock(  # type: ignore[method-assign]
        side_effect=[log_text, memory, ""]
    )
    return provisioner


GPUS = (NodeGPU(index=0, uuid=GPU_UUID, name="NVIDIA L4", total_mib=97887),)


def _gpus(profile_id: str) -> tuple[NodeGPU, ...]:
    return (GPUS[0].model_copy(update={"uuid": _gpu_uuid(profile_id)}),)


@pytest.mark.parametrize("profile_id", sorted(CASES))
def test_real_logs_yield_separate_target_and_draft_evidence(profile_id: str) -> None:
    profile = _profile(profile_id)

    fit = _parse_llamacpp_runtime_fit(_log(profile_id))

    assert fit.sizing == "profile"
    assert fit.context_per_slot == fit.aggregate_context == profile.context
    assert fit.slots == 1 and fit.kv_unified
    assert fit.cache_type_k == fit.cache_type_v == profile.cache_type.value
    assert fit.flash_attn == "on"
    assert fit.gpu_layers == fit.total_layers
    evidence = fit.profile
    assert evidence is not None
    assert evidence.ubatch == 256
    assert evidence.spec_type == profile.speculative_type.value
    assert evidence.spec_draft_n_max == profile.speculative_draft_n_max
    assert evidence.gpu_uuid == _gpu_uuid(profile_id)
    if profile.draft is None:
        # MTP reuses the offloaded target: no second model is loaded.
        assert evidence.draft_gpu_layers is None
        assert evidence.draft_gguf is None
    elif profile_id == GEMMA:
        assert evidence.draft_gpu_layers == evidence.draft_total_layers == 5
        assert evidence.draft_gguf == GEMMA_DRAFT_PATH
        # The target's layer count, not the assistant's 5/5 that follows it.
        assert (fit.gpu_layers, fit.total_layers) == (61, 61)
    else:
        assert evidence.draft_gpu_layers == evidence.draft_total_layers == 6
        assert evidence.draft_gguf == DRAFT_PATH
        # The target's layer count, not the draft's 6/6 that follows it.
        assert (fit.gpu_layers, fit.total_layers) == (53, 53)
    if profile_id == GEMMA:
        # The assistant reads the target's cache: no draft cache type exists.
        assert evidence.draft_shares_target_cache
        assert evidence.draft_cache_type_k is evidence.draft_cache_type_v is None
    else:
        assert not evidence.draft_shares_target_cache
        assert evidence.draft_cache_type_k == evidence.draft_cache_type_v == "f16"


def test_gemma_has_two_target_caches_and_both_are_checked() -> None:
    """Gemma's global and sliding-window caches are separate records."""
    fit = _parse_llamacpp_runtime_fit(_log(GEMMA))

    assert fit.profile is not None
    assert fit.profile.target_cache_types == (("q4_0", "q4_0"), ("q4_0", "q4_0"))
    assert fit.cache_type_k == fit.cache_type_v == "q4_0"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            # The label llama.cpp prints for the assistant is f16 while the
            # target runs q4_0: only real sharing makes that acceptable.
            "layer   3: sharing with layer 59.",
            "layer   3: not shared.",
            "does not share every KV layer",
        ),
        (
            "offloaded 5/5 layers to GPU",
            "offloaded 4/5 layers to GPU",
            "did not fully offload the draft model",
        ),
        (
            "mtp-gemma-4-31B-it.gguf'",
            "some-other-assistant.gguf'",
            "draft model other than the planned artifact",
        ),
        (
            "- n_max=4, n_min=0",
            "- n_max=2, n_min=0",
            "speculative draft length differs",
        ),
        (
            # The sliding-window cache alone at another precision.
            "50 layers,  1/1 seqs), K (q4_0):",
            "50 layers,  1/1 seqs), K (q8_0):",
            "KV cache",
        ),
    ],
)
def test_one_changed_fact_in_the_real_gemma_log_is_rejected(
    old: str, new: str, message: str
) -> None:
    text = _log(GEMMA)
    assert old in text, "fixture no longer contains the fact under test"

    with pytest.raises(ProvisioningError, match=message):
        _parse_llamacpp_runtime_fit(text.replace(old, new))


def test_an_assistant_that_allocates_its_own_cache_is_rejected() -> None:
    text = _log(GEMMA)
    marker = "creating non-SWA KV cache, size = 131072 cells"
    assert text.count(marker) == 2
    head, _, tail = text.rpartition(marker)
    injected = (
        marker
        + "\n0.00.000.000 I llama_kv_cache:      CUDA0 KV buffer size =   288.00 MiB"
    )

    with pytest.raises(ProvisioningError, match="allocated a KV cache"):
        _parse_llamacpp_runtime_fit(head + injected + tail)


def test_the_target_cache_type_is_not_read_from_the_draft() -> None:
    """The draft's f16 record comes last; the target really runs q4_0."""
    fit = _parse_llamacpp_runtime_fit(_log("qwen3.8-27b-24g"))

    assert fit.cache_type_k == "q4_0"
    assert fit.profile is not None
    assert fit.profile.draft_cache_type_k == "f16"


@pytest.mark.parametrize(
    ("profile_id", "old", "new", "message"),
    [
        (
            "qwen3.8-27b-24g",
            "offloaded 66/66 layers to GPU",
            "offloaded 60/66 layers to GPU",
            "did not fully offload the model",
        ),
        (
            "muse-glimmer-30b-24g",
            "offloaded 6/6 layers to GPU",
            "offloaded 5/6 layers to GPU",
            "did not fully offload the draft model",
        ),
        (
            "qwen3.8-27b-24g",
            "K (q4_0): 2304.00 MiB, V (q4_0): 2304.00 MiB",
            "K (q8_0): 2304.00 MiB, V (q8_0): 2304.00 MiB",
            "KV cache types differ from its VRAM plan",
        ),
        (
            "qwen3.8-27b-24g",
            "K (f16):  512.00 MiB, V (f16):  512.00 MiB",
            "K (q8_0):  512.00 MiB, V (q8_0):  512.00 MiB",
            "draft KV cache type differs",
        ),
        (
            "qwen3.8-27b-24g",
            "adding speculative implementation 'draft-mtp'",
            "adding speculative implementation 'ngram-mod'",
            "speculative implementation differs",
        ),
        (
            "qwen3.8-27b-24g",
            "- n_max=2, n_min=0",
            "- n_max=5, n_min=0",
            "speculative draft length differs",
        ),
        (
            "qwen3.8-27b-24g",
            "load_model: speculative decoding context initialized",
            "load_model: speculative decoding context failed",
            "initialized speculative decoding context",
        ),
        (
            "muse-glimmer-30b-24g",
            "Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf'",
            "Some-Other-Draft.gguf'",
            "draft model other than the planned artifact",
        ),
        (
            "qwen3.8-27b-24g",
            "n_ubatch              = 256",
            "n_ubatch              = 512",
            "micro-batch",
        ),
        (
            "qwen3.8-27b-24g",
            "kv_unified = 'true'",
            "kv_unified = 'false'",
            "requires unified KV cache",
        ),
    ],
)
def test_one_changed_fact_in_a_real_log_is_rejected(
    profile_id: str, old: str, new: str, message: str
) -> None:
    text = _log(profile_id)
    assert old in text, "fixture no longer contains the fact under test"
    # Change the first occurrence only for target facts, every one otherwise.
    mutated = text.replace(old, new, 1) if "66/66" in old else text.replace(old, new)

    with pytest.raises(ProvisioningError, match=message):
        _parse_llamacpp_runtime_fit(mutated)


def test_a_log_without_a_draft_is_not_a_profile_launch() -> None:
    text = _log("qwen3.8-27b-24g")
    cut = text[: text.index("common_speculative_init_result")]

    with pytest.raises(ProfileEvidenceError, match="exactly one speculative draft"):
        parse_profile_evidence(cut)


def test_only_the_latest_launch_counts() -> None:
    """An earlier, complete launch cannot vouch for a later, broken one."""
    good = _log("qwen3.8-27b-24g")
    broken = good[: good.index("common_speculative_init_result")]

    with pytest.raises(ProvisioningError, match="exactly one speculative draft"):
        _parse_llamacpp_runtime_fit(good + "\n" + broken)
    assert _parse_llamacpp_runtime_fit(broken + "\n" + good).profile is not None


def test_planner_sizing_still_rejects_a_q4_0_cache() -> None:
    text = _log("qwen3.8-27b-24g").replace("sizing=profile", "sizing=custom", 1)

    with pytest.raises(ProvisioningError, match="does not support a q4_0"):
        _parse_llamacpp_runtime_fit(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_id", sorted(CASES))
async def test_verified_runtime_records_the_speculative_side(profile_id: str) -> None:
    provisioner = _provisioner(_log(profile_id))
    profile = _profile(profile_id)

    runtime = await provisioner._verify_llamacpp_runtime(
        "host1",
        expected_request=_request(profile_id),
        draft_artifact=_draft_artifact(profile_id),
        gpus=_gpus(profile_id),
    )

    assert runtime.requested.sizing is LlamaCppSizingMode.PROFILE
    assert runtime.effective.cache_type_k is profile.cache_type
    assert runtime.effective.ubatch == 256
    speculative = runtime.effective.speculative
    assert speculative is not None
    assert speculative.type is profile.speculative_type
    assert speculative.draft_n_max == profile.speculative_draft_n_max
    if profile_id == GEMMA:
        assert speculative.draft_shares_target_cache
        assert speculative.draft_cache_type_k is None
    else:
        assert not speculative.draft_shares_target_cache
        assert speculative.draft_cache_type_k is LlamaCppCacheType.F16
    assert (speculative.draft_gpu_layers is None) == (
        profile.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP
    )


def _changed(name: str, changes: dict[str, object]) -> LlamaCppRuntimeRequest:
    request = _request(name)
    assert request.profile is not None
    return request.model_copy(
        update={"profile": request.profile.model_copy(update=changes)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": "another-profile"},
        {"profile_version": 2},
        {"ubatch": 512},
        {"speculative_draft_n_max": 3},
        {"draft_cache_type": LlamaCppCacheType.Q8_0},
        {"required_free_mib": 21000},
        {"disable_cuda_graphs": True},
    ],
)
async def test_a_profile_label_alone_proves_nothing(changes: dict[str, object]) -> None:
    """The log says "qwen3.8-27b-24g", but the request asked for something else."""
    provisioner = _provisioner(_log("qwen3.8-27b-24g"))

    with pytest.raises(ProvisioningError, match="requested catalog profile"):
        await provisioner._verify_llamacpp_runtime(
            "host1",
            expected_request=_changed("qwen3.8-27b-24g", changes),
            gpus=GPUS,
        )


@pytest.mark.asyncio
async def test_a_different_context_is_rejected() -> None:
    provisioner = _provisioner(_log("qwen3.8-27b-24g"))
    request = _request("qwen3.8-27b-24g").model_copy(
        update={"context_per_slot": 131072}
    )

    with pytest.raises(ProvisioningError, match="requested profile sizing"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=request, gpus=GPUS
        )


@pytest.mark.asyncio
async def test_a_launch_on_another_gpu_is_rejected() -> None:
    provisioner = _provisioner(_log("qwen3.8-27b-24g"))
    other = (GPUS[0].model_copy(update={"uuid": "GPU-00000000-0000"}),)

    with pytest.raises(ProvisioningError, match="other than the inventoried one"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=_request("qwen3.8-27b-24g"), gpus=other
        )


@pytest.mark.asyncio
async def test_a_launch_on_another_gpu_product_is_rejected() -> None:
    provisioner = _provisioner(_log("qwen3.8-27b-24g"))
    other = (GPUS[0].model_copy(update={"name": "NVIDIA GeForce RTX 3090"}),)

    with pytest.raises(ProvisioningError, match="other than the profile's product"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=_request("qwen3.8-27b-24g"), gpus=other
        )


@pytest.mark.asyncio
async def test_a_second_gpu_is_rejected() -> None:
    provisioner = _provisioner(
        _log("qwen3.8-27b-24g"),
        memory="0, 97887, 30000, 67887\n1, 97887, 0, 97887",
    )

    with pytest.raises(ProvisioningError, match="exactly one GPU per host"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=_request("qwen3.8-27b-24g"), gpus=GPUS
        )


@pytest.mark.asyncio
async def test_the_post_load_reserve_still_applies() -> None:
    provisioner = _provisioner(_log("qwen3.8-27b-24g"), memory="0, 23034, 22600, 434")

    with pytest.raises(ProvisioningError, match="below the requested fit target"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=_request("qwen3.8-27b-24g"), gpus=GPUS
        )


@pytest.mark.asyncio
async def test_a_draft_that_was_not_requested_is_rejected() -> None:
    provisioner = _provisioner(_log("muse-glimmer-30b-24g"))

    with pytest.raises(ProvisioningError, match="requested catalog profile"):
        await provisioner._verify_llamacpp_runtime(
            "host1",
            expected_request=_request("muse-glimmer-30b-24g"),
            draft_artifact=None,
            gpus=GPUS,
        )


@pytest.mark.asyncio
async def test_a_planner_request_cannot_be_satisfied_by_a_profile_launch() -> None:
    provisioner = _provisioner(_log("qwen3.8-27b-24g"))
    request = LlamaCppRuntimeRequest(sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512)

    with pytest.raises(ProvisioningError, match="sizing mode differs"):
        await provisioner._verify_llamacpp_runtime(
            "host1", expected_request=request, gpus=GPUS
        )
