"""Versioned catalog of measured single-GPU llama.cpp profiles.

Each profile is one configuration that was measured as a whole: weights,
context, KV-cache precision, micro-batch and speculative decoding together.
A profile is typed data, never a command string: the start script renders the
``llama-server`` arguments from these fields, and a test pins that rendering
to the measured commands token for token.

Measurements and their method are in ``GPU-MODEL-SELECTION.md``. They were
taken on a 96 GB workstation GPU. ``qualified_gpus`` names the cards on which
a profile has since been run for real; placement uses only those.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_proxy.huggingface.artifacts import GGUFArtifact
from inference_proxy.models.node import (
    LlamaCppCacheType,
    LlamaCppProfileRuntime,
    LlamaCppRuntimeRequest,
    LlamaCppSampling,
    LlamaCppSizingMode,
    LlamaCppSpeculativeType,
)

CATALOG_VERSION = 1


class GGUFRef(BaseModel):
    """One exact file of one immutable Hugging Face revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str = Field(pattern=r"^[\w.-]+\.gguf$")
    size_bytes: int = Field(ge=1)
    # Provenance, not a runtime check: resolution matches repository, revision,
    # filename and byte size. The Hub cache stores each LFS file under its
    # SHA-256, and the qualification harness records that link for each run.
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def matches(self, artifact: GGUFArtifact) -> bool:
        """Whether *artifact* is this exact file, including its byte size."""
        return (
            artifact.repo_id == self.repo_id
            and artifact.resolved_revision == self.revision
            and artifact.files == (self.filename,)
            and artifact.file_sizes.get(self.filename) == self.size_bytes
        )


class GPUClass(BaseModel):
    """One GPU product a profile may run on.

    ``quads_model_token`` is matched against the QUADS inventory string before
    a host is touched. ``nvidia_smi_name`` and ``min_total_mib`` are checked on
    the node itself, because the inventory string is not evidence of what a
    booted host exposes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=r"^[a-z0-9]+$")
    quads_model_token: str = Field(min_length=2, max_length=64)
    nvidia_smi_name: str = Field(min_length=2, max_length=128)
    min_total_mib: int = Field(ge=1)


class ArtifactRole(StrEnum):
    """Which of a profile's files an artifact is."""

    TARGET = "target"
    DRAFT = "draft"


class ModelProfile(BaseModel):
    """One measured configuration of one model on one GPU."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    version: int = Field(ge=1)
    display_name: str
    target: GGUFRef
    draft: GGUFRef | None = None
    context: int = Field(ge=256)
    cache_type: LlamaCppCacheType
    ubatch: int = Field(ge=32, le=4096)
    speculative_type: LlamaCppSpeculativeType
    speculative_draft_n_max: int = Field(ge=1, le=16)
    draft_cache_type: LlamaCppCacheType | None = None
    sampling: LlamaCppSampling
    # Peak VRAM with the context 95% full, excluding the process's CUDA
    # context. nvidia-smi free memory before a launch does include it, so the
    # launch gate adds ``cuda_context_allowance_mib`` and the reserve.
    measured_need_mib: int = Field(ge=1)
    cuda_context_allowance_mib: int = Field(ge=0)
    gpu_classes: tuple[GPUClass, ...] = Field(min_length=1)
    preferred_gpu_class: str | None = None
    qualified_gpus: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_profile(self) -> ModelProfile:
        keys = [gpu.key for gpu in self.gpu_classes]
        if len(set(keys)) != len(keys):
            raise ValueError("a profile lists each GPU class once")
        for name, values in (
            ("preferred_gpu_class", (self.preferred_gpu_class,)),
            ("qualified_gpus", self.qualified_gpus),
        ):
            if any(value is not None and value not in keys for value in values):
                raise ValueError(f"{name} must name one of the profile's GPU classes")
        if (self.draft is None) != (
            self.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP
        ):
            raise ValueError(
                "in-file MTP profiles take no draft file; "
                "DFlash and MTP assistant profiles need one"
            )
        return self

    @property
    def required_free_mib(self) -> int:
        """Free VRAM, as ``nvidia-smi`` reports it, the launch itself needs."""
        return self.measured_need_mib + self.cuda_context_allowance_mib

    def gpu_class(self, key: str) -> GPUClass | None:
        return next((gpu for gpu in self.gpu_classes if gpu.key == key), None)

    def runtime_request(
        self, *, reserve_mib: int, draft_artifact_id: str | None, gpu_class: str
    ) -> LlamaCppRuntimeRequest:
        """Return the exact managed launch request of this profile on one GPU class."""
        gpu = self.gpu_class(gpu_class)
        if gpu is None:
            raise ValueError(
                f"profile {self.profile_id} does not run on GPU class {gpu_class!r}"
            )
        return LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.PROFILE,
            fit_target_mib=reserve_mib,
            context_per_slot=self.context,
            slots=1,
            cache_type=self.cache_type,
            profile=LlamaCppProfileRuntime(
                profile_id=self.profile_id,
                profile_version=self.version,
                ubatch=self.ubatch,
                required_free_mib=self.required_free_mib,
                gpu_name=gpu.nvidia_smi_name,
                gpu_min_total_mib=gpu.min_total_mib,
                speculative_type=self.speculative_type,
                speculative_draft_n_max=self.speculative_draft_n_max,
                draft_cache_type=self.draft_cache_type,
                draft_artifact_id=draft_artifact_id,
                sampling=self.sampling,
            ),
        )


_L4 = GPUClass(
    key="l4",
    quads_model_token="AD104GL",
    nvidia_smi_name="NVIDIA L4",
    min_total_mib=22_900,
)
_A30 = GPUClass(
    key="a30",
    quads_model_token="GA100GL",
    nvidia_smi_name="NVIDIA A30",
    min_total_mib=24_000,
)

# ``qualified_gpus`` below comes from real runs on 2026-09-19 (one L4, one A30,
# llama.cpp v0.4.1, context filled to 95%, receipts kept with the research
# notes). Peak device memory including the CUDA context, L4 / A30:
#   qwen3.8-27b-24g      21,818 / 21,865 MiB
#   qwen3.6-35b-a3b-24g  20,422 / 20,469 MiB
#   muse-glimmer-30b-24g 21,288 / 21,315 MiB
# That is at most 187 MiB above ``measured_need_mib``, so the 300 MiB
# CUDA-context allowance is conservative on both cards.
# Order is significant: it breaks ratio ties and decides which profile a fleet
# too small for one GPU per profile gets first.
BUILTIN_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        profile_id="qwen3.8-27b-24g",
        version=1,
        display_name="Qwen3.8-27B UD-Q4_K_S, 262K, q4_0 KV, MTP",
        target=GGUFRef(
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            revision="4ca720788d1e01f1bff70c033e0d0028fd02e502",
            filename="Qwen3.8-27B-UD-Q4_K_S.gguf",
            size_bytes=15_358_213_024,
            sha256="75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017",
        ),
        context=262_144,
        cache_type=LlamaCppCacheType.Q4_0,
        ubatch=256,
        speculative_type=LlamaCppSpeculativeType.DRAFT_MTP,
        speculative_draft_n_max=2,
        draft_cache_type=LlamaCppCacheType.F16,
        sampling=LlamaCppSampling(
            temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
        ),
        measured_need_mib=21_678,
        cuda_context_allowance_mib=300,
        gpu_classes=(_L4, _A30),
        qualified_gpus=("l4", "a30"),
        preferred_gpu_class="a30",
    ),
    ModelProfile(
        profile_id="qwen3.6-35b-a3b-24g",
        version=1,
        display_name="Qwen3.6-35B-A3B UD-Q3_K_XL, 262K, q8_0 KV, MTP",
        target=GGUFRef(
            repo_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            revision="5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            filename="Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
            size_bytes=17_227_569_440,
            sha256="3fba9ab57290b34726837521335bf9268397174f3dd662880e2d0c5264d17b81",
        ),
        context=262_144,
        cache_type=LlamaCppCacheType.Q8_0,
        ubatch=256,
        speculative_type=LlamaCppSpeculativeType.DRAFT_MTP,
        speculative_draft_n_max=2,
        draft_cache_type=LlamaCppCacheType.F16,
        sampling=LlamaCppSampling(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
        ),
        measured_need_mib=20_456,
        cuda_context_allowance_mib=300,
        gpu_classes=(_L4, _A30),
        qualified_gpus=("l4", "a30"),
    ),
    ModelProfile(
        profile_id="muse-glimmer-30b-24g",
        version=1,
        display_name="Muse Glimmer 30B UD-Q5_K_M, 131K, f16 KV, DFlash2",
        target=GGUFRef(
            repo_id="unsloth/Muse-Glimmer-30B-GGUF",
            revision="faa5b025c584459c13febfa5c59883516710ae39",
            filename="Muse-Glimmer-30B-UD-Q5_K_M.gguf",
            size_bytes=19_194_274_848,
            sha256="27c27bc0cc2591344a9ef977d57aa79a9d36ddebee59660bd2abbf738f940f5b",
        ),
        draft=GGUFRef(
            repo_id="z-lab/Muse-Glimmer-30B-DFlash2-GGUF",
            revision="880882627431093d99d3b2368efb4a6fcf12d4cb",
            filename="Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf",
            size_bytes=1_645_657_280,
            sha256="93dbfb6f88e4645dec1347cf93f9d6fc80b90d413038722385b2a8e53565c949",
        ),
        context=131_072,
        cache_type=LlamaCppCacheType.F16,
        ubatch=256,
        speculative_type=LlamaCppSpeculativeType.DRAFT_DFLASH,
        speculative_draft_n_max=7,
        sampling=LlamaCppSampling(temperature=1.0, top_p=0.95, top_k=64),
        measured_need_mib=21_218,
        cuda_context_allowance_mib=300,
        gpu_classes=(_L4, _A30),
        qualified_gpus=("l4", "a30"),
    ),
    # Gemma 4 31B. Its 10 global-attention layers cost 22.5 KiB of q4_0 KV per
    # token, and with a quantized cache llama.cpp v0.4.1 reserves an f16 copy
    # of one layer's K and V in the target's compute buffer and again in the
    # assistant's (about 1 GiB each at 131K, 2 GiB each at 262K). Measured
    # consequence on 24 GB: the native 262K context only loads with 3-bit
    # weights and no drafter (KLD 0.10, about 6 tok/s at depth on an L4), so
    # this profile stops at 131K with 4-bit weights and the assistant.
    # The drafter is Gemma's "assistant": an MTP head in its own GGUF that
    # reads the target's KV cache and allocates none.
    # Real L4 run, 2026-09-20: peak 21,750 MiB including the CUDA context,
    # 814 MiB free, 34 to 41 tok/s short prompts, 19.5 tok/s at 121K depth
    # (14.0 and 6.0 with speculation off). Not yet run on an A30.
    ModelProfile(
        profile_id="gemma-4-31b-24g",
        version=1,
        display_name="Gemma 4 31B IQ4_XS, 131K, q4_0 KV, MTP assistant",
        target=GGUFRef(
            repo_id="unsloth/gemma-4-31B-it-GGUF",
            revision="c1ac76e99d5513b141e8adde7288b85c3f9c32ec",
            filename="gemma-4-31B-it-IQ4_XS.gguf",
            size_bytes=16_372_460_480,
            sha256="e3d40b6e363954c993a6e28b160d881dfe5a243b62746bd4092b393d4f1d54e5",
        ),
        draft=GGUFRef(
            repo_id="unsloth/gemma-4-31B-it-GGUF",
            revision="c1ac76e99d5513b141e8adde7288b85c3f9c32ec",
            filename="mtp-gemma-4-31B-it.gguf",
            size_bytes=514_687_104,
            sha256="5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de",
        ),
        context=131_072,
        cache_type=LlamaCppCacheType.Q4_0,
        ubatch=256,
        speculative_type=LlamaCppSpeculativeType.DRAFT_MTP_ASSISTANT,
        speculative_draft_n_max=4,
        sampling=LlamaCppSampling(temperature=1.0, top_p=0.95, top_k=64),
        measured_need_mib=21_597,
        cuda_context_allowance_mib=300,
        gpu_classes=(_L4, _A30),
        qualified_gpus=("l4",),
    ),
)


@dataclass(frozen=True)
class MissingArtifact:
    """A catalog file that is not in the gateway's shared GGUF cache."""

    profile_id: str
    role: ArtifactRole
    ref: GGUFRef


@dataclass(frozen=True)
class ResolvedProfile:
    """A profile whose every file resolved to a content-addressed artifact."""

    profile: ModelProfile
    artifact_id: str
    draft_artifact_id: str | None


@dataclass(frozen=True)
class CatalogResolution:
    """Which profiles can be placed now, and which files the others lack."""

    resolved: tuple[ResolvedProfile, ...]
    missing: tuple[MissingArtifact, ...]

    def get(self, profile_id: str) -> ResolvedProfile | None:
        return next(
            (item for item in self.resolved if item.profile.profile_id == profile_id),
            None,
        )


def validate_catalog(profiles: tuple[ModelProfile, ...]) -> None:
    """Reject a catalog with duplicate or ambiguous identities."""
    ids = [profile.profile_id for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("catalog profile ids must be unique")


def resolve_catalog(
    profiles: tuple[ModelProfile, ...], artifacts: tuple[GGUFArtifact, ...]
) -> CatalogResolution:
    """Match every catalog file against the artifacts the gateway discovered.

    A missing file is a reported condition, never an error: the gateway keeps
    serving, and the affected profile is simply not placed.
    """
    resolved: list[ResolvedProfile] = []
    missing: list[MissingArtifact] = []

    def find(ref: GGUFRef) -> str | None:
        return next((item.artifact_id for item in artifacts if ref.matches(item)), None)

    for profile in profiles:
        target_id = find(profile.target)
        draft_id = find(profile.draft) if profile.draft is not None else None
        if target_id is None:
            missing.append(
                MissingArtifact(profile.profile_id, ArtifactRole.TARGET, profile.target)
            )
        if profile.draft is not None and draft_id is None:
            missing.append(
                MissingArtifact(profile.profile_id, ArtifactRole.DRAFT, profile.draft)
            )
        if target_id is not None and (profile.draft is None or draft_id is not None):
            resolved.append(ResolvedProfile(profile, target_id, draft_id))
    return CatalogResolution(tuple(resolved), tuple(missing))


validate_catalog(BUILTIN_PROFILES)
