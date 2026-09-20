"""The built-in profile catalog and its resolution against the GGUF cache."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_proxy.huggingface.artifacts import GGUFArtifact
from inference_proxy.models.node import (
    LlamaCppCacheType,
    LlamaCppSizingMode,
    LlamaCppSpeculativeType,
)
from inference_proxy.placement.catalog import (
    BUILTIN_PROFILES,
    ArtifactRole,
    GGUFRef,
    ModelProfile,
    resolve_catalog,
    validate_catalog,
)


def _artifact(
    ref: GGUFRef, *, size: int | None = None, marker: str = "a"
) -> GGUFArtifact:
    return GGUFArtifact(
        artifact_id=(marker * 64)[:64],
        repo_id=ref.repo_id,
        resolved_revision=ref.revision,
        files=(ref.filename,),
        entrypoint=ref.filename,
        model_alias=ref.repo_id,
        file_sizes={ref.filename: ref.size_bytes if size is None else size},
    )


def _all_artifacts() -> list[GGUFArtifact]:
    artifacts = []
    for index, profile in enumerate(BUILTIN_PROFILES):
        artifacts.append(_artifact(profile.target, marker="abce"[index]))
        if profile.draft is not None:
            artifacts.append(_artifact(profile.draft, marker="00df"[index]))
    return artifacts


def test_catalog_order_and_scope() -> None:
    assert [item.profile_id for item in BUILTIN_PROFILES] == [
        "qwen3.8-27b-24g",
        "qwen3.6-35b-a3b-24g",
        "muse-glimmer-30b-24g",
        "gemma-4-31b-24g",
    ]
    for profile in BUILTIN_PROFILES:
        # 24 GB only: no profile may be eligible for a 16 GB card.
        assert {gpu.key for gpu in profile.gpu_classes} == {"l4", "a30"}
        assert all(gpu.min_total_mib > 20_000 for gpu in profile.gpu_classes)
    assert BUILTIN_PROFILES[0].preferred_gpu_class == "a30"
    # Both products were validated on real cards on 2026-09-19. Gemma has only
    # been run on an L4, so automatic placement keeps it off the A30.
    assert all(item.qualified_gpus == ("l4", "a30") for item in BUILTIN_PROFILES[:3])
    assert BUILTIN_PROFILES[3].qualified_gpus == ("l4",)


def test_every_profile_fits_the_l4_budget_it_was_selected_for() -> None:
    # 22,400 MiB is what an ECC-on L4 leaves llama.cpp at load.
    assert all(item.measured_need_mib < 22_400 for item in BUILTIN_PROFILES)


def test_runtime_request_is_the_profile_and_nothing_else() -> None:
    profile = BUILTIN_PROFILES[0]

    request = profile.runtime_request(
        reserve_mib=256, draft_artifact_id=None, gpu_class="l4"
    )

    assert request.sizing is LlamaCppSizingMode.PROFILE
    assert (request.context_per_slot, request.slots) == (262_144, 1)
    assert request.cache_type is LlamaCppCacheType.Q4_0
    assert request.fit_target_mib == 256
    assert request.profile is not None
    assert request.profile.required_free_mib == (
        profile.measured_need_mib + profile.cuda_context_allowance_mib
    )
    assert request.profile.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP
    # The request survives the etcd round trip it makes on the node record.
    assert type(request).model_validate(request.model_dump(mode="json")) == request


def test_all_files_present_resolves_every_profile() -> None:
    resolution = resolve_catalog(BUILTIN_PROFILES, tuple(_all_artifacts()))

    assert [item.profile.profile_id for item in resolution.resolved] == [
        item.profile_id for item in BUILTIN_PROFILES
    ]
    assert resolution.missing == ()
    muse = resolution.get("muse-glimmer-30b-24g")
    assert muse is not None and muse.draft_artifact_id == "d" * 64


def test_a_missing_draft_blocks_only_its_own_profile() -> None:
    artifacts = [item for item in _all_artifacts() if "DFlash2" not in item.repo_id]

    resolution = resolve_catalog(BUILTIN_PROFILES, tuple(artifacts))

    assert resolution.get("muse-glimmer-30b-24g") is None
    assert resolution.get("qwen3.8-27b-24g") is not None
    assert [(item.profile_id, item.role) for item in resolution.missing] == [
        ("muse-glimmer-30b-24g", ArtifactRole.DRAFT)
    ]


def test_a_file_of_the_wrong_size_or_revision_does_not_count() -> None:
    target = BUILTIN_PROFILES[0].target
    truncated = _artifact(target, size=target.size_bytes - 1)
    other_revision = _artifact(target).model_copy(
        update={"resolved_revision": "0" * 40}
    )

    resolution = resolve_catalog(BUILTIN_PROFILES[:1], (truncated, other_revision))

    assert resolution.resolved == ()
    assert resolution.missing[0].role is ArtifactRole.TARGET


def test_an_empty_cache_is_reported_not_raised() -> None:
    resolution = resolve_catalog(BUILTIN_PROFILES, ())

    assert resolution.resolved == ()
    assert len(resolution.missing) == 6


def test_profile_consistency_rules() -> None:
    base = BUILTIN_PROFILES[0]
    with pytest.raises(ValidationError, match="MTP profiles take no draft"):
        ModelProfile.model_validate(
            {**base.model_dump(), "draft": BUILTIN_PROFILES[2].draft}
        )
    with pytest.raises(ValidationError, match="must name one of"):
        ModelProfile.model_validate({**base.model_dump(), "qualified_gpus": ("t4",)})
    with pytest.raises(ValueError, match="unique"):
        validate_catalog((base, base))
