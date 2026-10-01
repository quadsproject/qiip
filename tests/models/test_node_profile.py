"""Profile sizing, placement provenance and GPU inventory on the node model."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from inference_proxy.models.node import (
    LlamaCppCacheType,
    LlamaCppProfileRuntime,
    LlamaCppRuntimeEffective,
    LlamaCppRuntimeRequest,
    LlamaCppSampling,
    LlamaCppSizingMode,
    LlamaCppSpeculativeType,
    Node,
    NodeGPU,
    NodePlacement,
)

SAMPLING = LlamaCppSampling(temperature=1.0, top_p=0.95, top_k=20)


def _profile(**changes: object) -> LlamaCppProfileRuntime:
    values: dict[str, object] = {
        "profile_id": "qwen3.8-27b-24g",
        "profile_version": 1,
        "ubatch": 256,
        "required_free_mib": 21978,
        "gpu_name": "NVIDIA L4",
        "gpu_min_total_mib": 22900,
        "speculative_type": LlamaCppSpeculativeType.DRAFT_MTP,
        "speculative_draft_n_max": 2,
        "draft_cache_type": LlamaCppCacheType.F16,
        "sampling": SAMPLING,
    }
    return LlamaCppProfileRuntime.model_validate({**values, **changes})


def _request(**changes: object) -> LlamaCppRuntimeRequest:
    values: dict[str, object] = {
        "sizing": LlamaCppSizingMode.PROFILE,
        "fit_target_mib": 256,
        "context_per_slot": 262144,
        "slots": 1,
        "cache_type": LlamaCppCacheType.Q4_0,
        "profile": _profile(),
    }
    return LlamaCppRuntimeRequest.model_validate({**values, **changes})


def test_profile_request_round_trips_through_json() -> None:
    request = _request()

    wire = request.model_dump(mode="json")

    assert wire["sizing"] == "profile" and wire["cache_type"] == "q4_0"
    assert "draft_artifact_id" not in wire["profile"]
    assert LlamaCppRuntimeRequest.model_validate(wire) == request


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"profile": None}, "requires a profile"),
        ({"slots": 2}, "exactly one slot"),
        ({"allow_estimator_overrun": True}, "no estimator"),
        ({"context_per_slot": None}, "requires context_per_slot"),
        ({"context_per_slot": 1000}, "256-token increments"),
    ],
)
def test_profile_sizing_rules(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(**changes)


def test_planner_sizing_cannot_reach_profile_only_values() -> None:
    with pytest.raises(ValidationError, match="supports f16 or q8_0"):
        _request(sizing=LlamaCppSizingMode.CUSTOM, profile=None)
    with pytest.raises(ValidationError, match="does not accept a profile"):
        _request(sizing=LlamaCppSizingMode.CUSTOM, cache_type=LlamaCppCacheType.Q8_0)
    with pytest.raises(ValidationError, match="does not accept custom values"):
        LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512, profile=_profile()
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"draft_artifact_id": "d" * 64}, "not an artifact"),
        ({"draft_cache_type": None}, "must state the draft cache type"),
        ({"speculative_type": "draft-dflash"}, "require a draft artifact"),
        (
            {"speculative_type": "draft-dflash", "draft_artifact_id": "d" * 64},
            "default draft cache",
        ),
        (
            {"speculative_type": "draft-mtp-assistant", "draft_cache_type": None},
            "assistant profiles require a draft artifact",
        ),
        (
            {"speculative_type": "draft-mtp-assistant", "draft_artifact_id": "d" * 64},
            "shares the target's KV cache",
        ),
        ({"profile_id": "Has Spaces --flag"}, "pattern"),
        ({"speculative_type": "ngram-mod"}, "Input should be"),
        ({"ubatch": 8}, "greater than or equal to 32"),
        ({"extra_args": "--api-key x"}, "Extra inputs are not permitted"),
        ({"gpu_name": "NVIDIA L4; rm -rf /"}, "pattern"),
    ],
)
def test_profile_runtime_is_closed_and_typed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _profile(**changes)


def test_mtp_assistant_takes_a_draft_file_and_no_draft_cache() -> None:
    profile = _profile(
        speculative_type="draft-mtp-assistant",
        draft_artifact_id="d" * 64,
        draft_cache_type=None,
    )
    assert profile.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP_ASSISTANT
    assert profile.draft_cache_type is None


def test_sampling_is_bounded() -> None:
    with pytest.raises(ValidationError):
        LlamaCppSampling(temperature=9.0, top_p=0.95, top_k=20)
    with pytest.raises(ValidationError):
        LlamaCppSampling.model_validate(
            {"temperature": "1.0 --port 1", "top_p": 0.95, "top_k": 20}
        )


def test_planner_nodes_keep_their_effective_wire_shape() -> None:
    effective = LlamaCppRuntimeEffective(
        train_context=4096,
        context_per_slot=4096,
        slot_context_limit=4096,
        slots=1,
        aggregate_context=4096,
        cache_type_k=LlamaCppCacheType.F16,
        cache_type_v=LlamaCppCacheType.F16,
        flash_attn="auto",
        kv_unified=True,
        gpu_layers=10,
        total_layers=10,
    )

    assert not {"ubatch", "speculative"} & set(effective.model_dump(mode="json"))


def test_placement_marks_only_managed_unowned_nodes() -> None:
    placement = NodePlacement(
        profile_id="qwen3.8-27b-24g", profile_version=1, claim_id="c" * 32
    )
    gpu = NodeGPU(index=0, uuid="GPU-33916087-1a73", name="NVIDIA L4", total_mib=23034)

    node = Node(
        node_id="h",
        endpoint="http://h:8000",
        managed=True,
        placement=placement,
        gpus=(gpu,),
        last_heartbeat=datetime.now(UTC),
    )

    assert Node.model_validate(node.model_dump(mode="json")) == node
    with pytest.raises(ValidationError, match="managed and unowned"):
        Node(node_id="h", endpoint="http://h:8000", placement=placement)
    with pytest.raises(ValidationError, match="managed and unowned"):
        Node(
            node_id="h",
            endpoint="http://h:8000",
            managed=True,
            owner="a@b.c",
            placement=placement,
        )


def test_records_written_before_this_change_still_load() -> None:
    node = Node.model_validate({"node_id": "h", "endpoint": "http://h:8000"})

    assert node.gpus == () and node.placement is None
