"""Startup-log evidence for catalog-profile llama.cpp launches.

A profile launch loads two things: the target model and a speculative draft.
llama-server writes the same record kinds for both (``offloaded N/M layers``,
``llama_context: n_ctx``, ``llama_kv_cache: size``), so reading "the last
record" would describe the draft and hide the target. This module splits the
log at llama.cpp's own draft-initialization marker and checks each side
against the plan the start script recorded.

The record shapes were taken from real llama.cpp v0.4.1 logs, stored under
``tests/provisioning/fixtures/llamacpp-v0.4.1``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROFILE_PLAN_PATTERN = re.compile(
    r"^qiip_profile_plan: profile_id=(?P<profile_id>[a-z0-9][a-z0-9._-]{0,63}) "
    r"profile_version=(?P<profile_version>[1-9]\d{0,8}) "
    r"ubatch=(?P<ubatch>\d+) "
    r"spec_type=(?P<spec_type>draft-mtp-assistant|draft-mtp|draft-dflash) "
    r"spec_draft_n_max=(?P<spec_draft_n_max>\d+) "
    r"draft_cache_type=(?P<draft_cache_type>default|f16|q8_0|q4_0) "
    r"draft_gguf=(?P<draft_gguf>\S+) "
    r"required_free_mib=(?P<required_free_mib>\d+) "
    r"gpu_free_mib=(?P<gpu_free_mib>\d+) "
    r"gpu_uuid=(?P<gpu_uuid>GPU-[0-9a-fA-F-]{8,64}) "
    r"cuda_graphs=(?P<cuda_graphs>on|off)$",
    re.MULTILINE,
)
_FIT_PLAN_LINE = re.compile(r"^qiip_fit_plan: ", re.MULTILINE)
_DRAFT_MARKER = re.compile(
    r"common_speculative_init_result: (?:"
    r"creating MTP draft context against the target model '(?P<mtp>[^']+)'"
    r"|loading draft model '(?P<draft>[^']+)'"
    r")"
)
_DRAFT_READY = re.compile(r"load_model: speculative decoding context initialized")
_SPEC_IMPLEMENTATION = re.compile(
    r"adding speculative implementation '(?P<type>[^']+)'"
)
_SPEC_N_MAX = re.compile(r": - n_max=(?P<n_max>\d+), n_min=")
_OFFLOAD = re.compile(r"offloaded (?P<loaded>\d+)/(?P<total>\d+) layers to GPU")
_CONTEXT = re.compile(r"llama_context:\s+n_ctx\s*=\s*(?P<context>\d+)")
_UBATCH = re.compile(r"llama_context:\s+n_ubatch\s*=\s*(?P<ubatch>\d+)")
_KV_CACHE = re.compile(
    r"llama_kv_cache[^:]*:\s+size\s*=.*?"
    r"K \((?P<cache_type_k>[a-z0-9_]+)\):.*?"
    r"V \((?P<cache_type_v>[a-z0-9_]+)\):"
)
# An MTP assistant (Gemma 4) allocates no cache: llama.cpp reports, per draft
# layer, which target layer's cache it reads, and prints no draft buffer.
_KV_SHARED_LAYER = re.compile(r"llama_kv_cache: layer\s+\d+: sharing with layer\s+\d+")
_KV_LAYER_COUNT = re.compile(
    r"llama_kv_cache[^:]*:\s+size\s*=.*?,\s*(?P<layers>\d+) layers,"
)
_KV_BUFFER = re.compile(r"llama_kv_cache[^:]*:\s+\S+ KV buffer size\s*=")
_DEFAULT_DRAFT_CACHE_TYPE = "f16"
_ASSISTANT_SPEC_TYPE = "draft-mtp-assistant"
# llama-server has one MTP implementation for both the in-file head and the
# assistant file; the plan line is where qiip tells the two apart.
_SERVER_SPEC_TYPE = {_ASSISTANT_SPEC_TYPE: "draft-mtp"}


class ProfileEvidenceError(Exception):
    """The startup log does not prove the requested profile launch."""


@dataclass(frozen=True)
class ProfileSections:
    """The slice of one launch's log that describes each loaded model."""

    target: str
    draft: str
    draft_model_path: str
    draft_is_target_file: bool


@dataclass(frozen=True)
class ProfileEvidence:
    """What the start script planned and what llama-server then reported."""

    profile_id: str
    profile_version: int
    ubatch: int
    spec_type: str
    spec_draft_n_max: int
    draft_cache_type: str | None
    draft_gguf: str | None
    required_free_mib: int
    gpu_free_mib: int
    gpu_uuid: str
    cuda_graphs_disabled: bool
    target_gpu_layers: int
    target_total_layers: int
    target_context: int
    target_cache_types: tuple[tuple[str, str], ...]
    draft_gpu_layers: int | None
    draft_total_layers: int | None
    # None when the draft shares the target's cache (MTP assistant).
    draft_cache_type_k: str | None
    draft_cache_type_v: str | None
    draft_shares_target_cache: bool


def latest_launch(log_text: str) -> str:
    """Return the log from the last recorded plan onward.

    A log may hold more than one launch. Evidence is only ever read from the
    most recent one, so a record from an earlier launch can never stand in for
    one the current launch failed to write.
    """
    plans = list(_FIT_PLAN_LINE.finditer(log_text))
    if not plans:
        raise ProfileEvidenceError("llama.cpp startup log has no QIIP VRAM plan")
    return log_text[plans[-1].start() :]


def split_profile_sections(launch_text: str) -> ProfileSections:
    """Split one launch into its target and draft model sections."""
    markers = list(_DRAFT_MARKER.finditer(launch_text))
    if len(markers) != 1:
        raise ProfileEvidenceError(
            "llama.cpp startup log must show exactly one speculative draft "
            f"initialization; found {len(markers)}"
        )
    marker = markers[0]
    ready = _DRAFT_READY.search(launch_text, marker.end())
    if ready is None:
        raise ProfileEvidenceError(
            "llama.cpp did not report an initialized speculative decoding context"
        )
    mtp_path = marker.group("mtp")
    return ProfileSections(
        target=launch_text[: marker.start()],
        draft=launch_text[marker.end() : ready.start()],
        draft_model_path=mtp_path or marker.group("draft"),
        draft_is_target_file=mtp_path is not None,
    )


def _single(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ProfileEvidenceError(
            f"llama.cpp startup log must contain exactly one {what}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _cache_types(section: str, what: str) -> tuple[tuple[str, str], ...]:
    records = tuple(
        (match.group("cache_type_k"), match.group("cache_type_v"))
        for match in _KV_CACHE.finditer(section)
    )
    if not records:
        raise ProfileEvidenceError(
            f"llama.cpp startup log has no {what} KV cache type record"
        )
    return records


def _require_shared_draft_cache(draft_section: str, planned_cache_type: str) -> None:
    """Prove an MTP assistant allocated no cache and reads the target's.

    llama.cpp still prints a ``size = ... K (f16)`` record for the assistant.
    That type is a label on views of the target's buffers, so it is not
    evidence. The evidence is: every layer the record counts is reported as
    shared, and no draft KV buffer was allocated.
    """
    if planned_cache_type != "default":
        raise ProfileEvidenceError(
            "an MTP assistant profile plan must not set a draft cache type"
        )
    layers = sum(
        int(match.group("layers")) for match in _KV_LAYER_COUNT.finditer(draft_section)
    )
    shared = len(_KV_SHARED_LAYER.findall(draft_section))
    if layers < 1 or shared != layers:
        raise ProfileEvidenceError(
            "llama.cpp MTP assistant does not share every KV layer with the target "
            f"({shared}/{layers} layers shared)"
        )
    if _KV_BUFFER.search(draft_section):
        raise ProfileEvidenceError(
            "llama.cpp allocated a KV cache for an MTP assistant"
        )


def parse_profile_evidence(log_text: str) -> ProfileEvidence:
    """Return the verified-by-structure evidence of the latest profile launch.

    Structural checks live here: one plan, one draft, a fully offloaded target,
    a fully offloaded draft, uniform cache types per side. Comparing the values
    with the *requested* profile is the caller's job.
    """
    launch = latest_launch(log_text)
    plan = _single(PROFILE_PLAN_PATTERN, launch, "QIIP profile plan").groupdict()
    sections = split_profile_sections(launch)

    target_offload = _single(_OFFLOAD, sections.target, "target GPU offload record")
    target_context = _single(_CONTEXT, sections.target, "target context record")
    target_ubatch = _single(_UBATCH, sections.target, "target micro-batch record")
    target_cache = _cache_types(sections.target, "target")
    if int(target_ubatch.group("ubatch")) != int(plan["ubatch"]):
        raise ProfileEvidenceError(
            "llama.cpp target micro-batch differs from the profile plan"
        )

    draft_context = _single(_CONTEXT, sections.draft, "draft context record")
    if draft_context.group("context") != target_context.group("context"):
        raise ProfileEvidenceError(
            "llama.cpp draft context differs from the target context"
        )
    shares_target_cache = plan["spec_type"] == _ASSISTANT_SPEC_TYPE
    draft_cache_type: str | None
    if shares_target_cache:
        _require_shared_draft_cache(sections.draft, plan["draft_cache_type"])
        draft_cache_type = None
    else:
        draft_cache = _cache_types(sections.draft, "draft")
        if len(set(draft_cache)) != 1 or draft_cache[0][0] != draft_cache[0][1]:
            raise ProfileEvidenceError("llama.cpp draft KV cache types are not uniform")
        expected_draft_cache = (
            _DEFAULT_DRAFT_CACHE_TYPE
            if plan["draft_cache_type"] == "default"
            else plan["draft_cache_type"]
        )
        if draft_cache[0][0] != expected_draft_cache:
            raise ProfileEvidenceError(
                "llama.cpp draft KV cache type differs from the profile plan"
            )
        draft_cache_type = draft_cache[0][0]

    draft_offloads = list(_OFFLOAD.finditer(sections.draft))
    draft_gpu_layers: int | None = None
    draft_total_layers: int | None = None
    if plan["spec_type"] == "draft-mtp":
        # The MTP head lives in the target file: llama.cpp opens a second
        # context on the already-offloaded model and loads no second model.
        if not sections.draft_is_target_file or draft_offloads:
            raise ProfileEvidenceError(
                "llama.cpp MTP draft did not reuse the target model"
            )
        if plan["draft_gguf"] != "none":
            raise ProfileEvidenceError("an MTP profile plan must not name a draft")
    else:
        if sections.draft_is_target_file or len(draft_offloads) != 1:
            raise ProfileEvidenceError(
                "llama.cpp did not load exactly one separate draft model"
            )
        if plan["draft_gguf"] == "none" or not sections.draft_model_path.endswith(
            "/" + plan["draft_gguf"]
        ):
            raise ProfileEvidenceError(
                "llama.cpp loaded a draft model other than the planned artifact"
            )
        draft_gpu_layers = int(draft_offloads[0].group("loaded"))
        draft_total_layers = int(draft_offloads[0].group("total"))
        if draft_total_layers < 1 or draft_gpu_layers != draft_total_layers:
            raise ProfileEvidenceError(
                "llama.cpp did not fully offload the draft model to GPU "
                f"({draft_gpu_layers}/{draft_total_layers} layers)"
            )

    implementation = _single(
        _SPEC_IMPLEMENTATION, launch, "speculative implementation record"
    )
    if implementation.group("type") != _SERVER_SPEC_TYPE.get(
        plan["spec_type"], plan["spec_type"]
    ):
        raise ProfileEvidenceError(
            "llama.cpp speculative implementation differs from the profile plan"
        )
    n_max = _SPEC_N_MAX.search(launch, implementation.end())
    if n_max is None or int(n_max.group("n_max")) != int(plan["spec_draft_n_max"]):
        raise ProfileEvidenceError(
            "llama.cpp speculative draft length differs from the profile plan"
        )

    return ProfileEvidence(
        profile_id=plan["profile_id"],
        profile_version=int(plan["profile_version"]),
        ubatch=int(plan["ubatch"]),
        spec_type=plan["spec_type"],
        spec_draft_n_max=int(plan["spec_draft_n_max"]),
        draft_cache_type=(
            None if plan["draft_cache_type"] == "default" else plan["draft_cache_type"]
        ),
        draft_gguf=None if plan["draft_gguf"] == "none" else plan["draft_gguf"],
        required_free_mib=int(plan["required_free_mib"]),
        gpu_free_mib=int(plan["gpu_free_mib"]),
        gpu_uuid=plan["gpu_uuid"],
        cuda_graphs_disabled=plan["cuda_graphs"] == "off",
        target_gpu_layers=int(target_offload.group("loaded")),
        target_total_layers=int(target_offload.group("total")),
        target_context=int(target_context.group("context")),
        target_cache_types=target_cache,
        draft_gpu_layers=draft_gpu_layers,
        draft_total_layers=draft_total_layers,
        draft_cache_type_k=draft_cache_type,
        draft_cache_type_v=draft_cache_type,
        draft_shares_target_cache=shares_target_cache,
    )
