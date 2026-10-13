"""Tests for the memory-aware vLLM launch planner (RFE #130)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_proxy.provisioning import vllm_fit
from inference_proxy.provisioning.vllm_fit import (
    CONTEXT_LADDER,
    MIN_CONTEXT,
    VllmDeviceMemory,
    VllmRequested,
    estimate_model_profile,
    is_memory_failure,
    load_model_profile,
    model_profile_for,
    plan_vllm_launch,
    select_vllm_model,
    vllm_profile_bucket,
)


def _device(
    total_mib: int = 81_920, free_mib: int | None = None, index: int = 0
) -> VllmDeviceMemory:
    return VllmDeviceMemory(
        index=index,
        total_mib=total_mib,
        free_mib=total_mib if free_mib is None else free_mib,
    )


def test_select_vllm_model_matches_profile_and_vram_thresholds() -> None:
    assert select_vllm_model("hopper", 240) == "Qwen/Qwen2.5-72B-Instruct"
    assert select_vllm_model("hopper", 80) == "Qwen/Qwen2.5-32B-Instruct"
    assert select_vllm_model("ampere-a100", 40) == "Qwen/Qwen2.5-14B-Instruct"
    assert select_vllm_model("ampere-a30", 40) == "Qwen/Qwen2.5-14B-Instruct"
    assert select_vllm_model("turing", 16) == "Qwen/Qwen3-14B-AWQ"
    assert select_vllm_model("consumer-ada", 48) == "Qwen/Qwen2.5-14B-Instruct"
    assert select_vllm_model("consumer-ada", 24) == "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(ValueError):
        select_vllm_model("mystery", 80)


def test_select_vllm_model_covers_every_profile_branch() -> None:
    # ampere-a30 / ga102-dc
    assert select_vllm_model("ampere-a30", 80) == "Qwen/Qwen2.5-32B-Instruct"
    assert select_vllm_model("ampere-a30", 24) == "Qwen/Qwen2.5-7B-Instruct"
    assert select_vllm_model("ga102-dc", 24) == "Qwen/Qwen2.5-7B-Instruct"
    # turing above and below the datacenter floor
    assert select_vllm_model("turing", 24) == "Qwen/Qwen2.5-7B-Instruct"
    # volta at both thresholds
    assert select_vllm_model("volta", 96) == "Qwen/Qwen2.5-32B-Instruct"
    assert select_vllm_model("volta", 48) == "Qwen/Qwen2.5-14B-Instruct"


def test_tensor_parallel_and_backend_follow_profile_and_capability() -> None:
    two_a100 = (
        _device(total_mib=81_920, index=0),
        _device(total_mib=81_920, index=1),
    )
    datacenter = plan_vllm_launch(
        profile_bucket="ampere-a100", devices=two_a100, requested=VllmRequested()
    )
    assert datacenter.effective.tensor_parallel_size == 2
    assert datacenter.effective.attention_backend == "FLASH_ATTN"

    two_ada = (
        _device(total_mib=49_152, index=0),
        _device(total_mib=49_152, index=1),
    )
    consumer = plan_vllm_launch(
        profile_bucket="consumer-ada", devices=two_ada, requested=VllmRequested()
    )
    assert consumer.effective.tensor_parallel_size == 1
    assert consumer.effective.attention_backend == "FLASH_ATTN"

    older = plan_vllm_launch(
        profile_bucket="turing",
        devices=[_device(total_mib=16_384)],
        requested=VllmRequested(),
    )
    assert older.effective.attention_backend is None


def test_dtype_bytes_falls_back_for_an_unknown_dtype() -> None:
    assert vllm_fit._dtype_bytes("weird-dtype") == 2
    assert vllm_fit._dtype_bytes(None) == 2


def test_fitting_plan_reduces_context_to_real_free_memory() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=[_device()], requested=VllmRequested()
    )

    assert plan.fitting is True
    assert plan.model_selection == "automatic"
    assert plan.effective.model == "Qwen/Qwen2.5-32B-Instruct"
    # 64 GiB of weights plus overhead leaves room for 8K, not the historic 32K.
    assert plan.effective.max_model_len == 8192
    assert plan.effective.max_num_batched_tokens == 8192
    assert plan.effective.tensor_parallel_size == 1
    assert plan.effective.dtype == "bfloat16"
    assert plan.effective.attention_backend == "FLASHINFER"
    assert plan.estimate.weights_per_gpu_mib == 64_000
    assert plan.estimate.usable_budget_mib == 73_728
    assert plan.estimate.free_per_gpu_mib == 81_920
    assert plan.estimate.runtime_overhead_mib == 6144
    assert plan.estimate.total_mib <= plan.estimate.usable_budget_mib
    assert "reduced" in plan.reason


def test_multi_gpu_plan_shards_weights_and_kv() -> None:
    devices = [_device(index=index) for index in range(3)]
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested()
    )

    assert plan.fitting is True
    assert plan.model_selection == "automatic"
    assert plan.effective.model == "Qwen/Qwen2.5-72B-Instruct"
    assert plan.effective.tensor_parallel_size == 3
    # 145 GiB of weights split three ways leaves ample room for 32K.
    assert plan.effective.max_model_len == 32768
    assert plan.estimate.weights_per_gpu_mib == 48_334


def test_non_fitting_plan_reports_weights_over_budget() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=VllmRequested(model="Qwen/Qwen2.5-72B-Instruct"),
    )

    assert plan.fitting is False
    assert plan.model_selection == "explicit"
    assert "exceed the usable device budget" in plan.reason
    assert plan.estimate.weights_per_gpu_mib == 145_000
    assert plan.estimate.usable_budget_mib == 73_728


def test_busy_gpu_free_memory_caps_the_budget() -> None:
    # Another tenant already holds most of the device: only 5 GiB is free.
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device(total_mib=81_920, free_mib=5120)],
        requested=VllmRequested(),
    )

    assert plan.fitting is False
    assert plan.estimate.free_per_gpu_mib == 5120
    # The free cap, not 90% of total, decides the budget.
    assert plan.estimate.usable_budget_mib == 5120


def test_busy_gpu_that_still_fits_shrinks_context() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device(total_mib=81_920, free_mib=73_500)],
        requested=VllmRequested(),
    )

    assert plan.fitting is True
    assert plan.estimate.free_per_gpu_mib == 73_500
    assert plan.estimate.usable_budget_mib == 73_500
    assert plan.effective.max_model_len == 4096
    assert plan.effective.max_model_len < 8192


def test_explicit_model_and_settings_are_never_changed() -> None:
    request = VllmRequested(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        dtype="float16",
    )
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=[_device()], requested=request
    )

    assert plan.model_selection == "explicit"
    assert plan.effective.model == "Qwen/Qwen2.5-7B-Instruct"
    assert plan.effective.max_model_len == 32768
    assert plan.effective.gpu_memory_utilization == 0.85
    assert plan.effective.dtype == "float16"
    assert plan.fitting is True


def test_explicit_context_that_does_not_fit_fails_instead_of_shrinking() -> None:
    request = VllmRequested(max_model_len=65536)
    plan = plan_vllm_launch(
        profile_bucket="hopper", devices=[_device()], requested=request
    )

    assert plan.fitting is False
    assert plan.effective.max_model_len == 65536
    assert "explicit max_model_len 65536" in plan.reason


def test_exhausted_retry_policy_is_bounded_and_ordered() -> None:
    devices = [_device()]
    first = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested()
    )
    assert first.retry_available is True
    assert first.adjustments == ()

    second = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested(), attempt=1
    )
    assert [item.field for item in second.adjustments] == ["max_model_len"]
    assert second.effective.max_model_len == 4096
    assert second.retry_available is True

    third = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested(), attempt=2
    )
    assert [item.field for item in third.adjustments] == [
        "max_model_len",
        "max_num_batched_tokens",
    ]
    assert third.retry_available is True

    fourth = plan_vllm_launch(
        profile_bucket="hopper", devices=devices, requested=VllmRequested(), attempt=3
    )
    assert fourth.effective.gpu_memory_utilization == 0.85
    assert fourth.retry_available is False
    assert len(fourth.adjustments) == 3


def test_retry_skips_explicit_fields() -> None:
    request = VllmRequested(max_model_len=8192)
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=request,
        attempt=1,
    )

    assert [item.field for item in plan.adjustments] == ["max_num_batched_tokens"]
    assert plan.effective.max_model_len == 8192


def test_fully_explicit_configuration_has_no_retry() -> None:
    request = VllmRequested(
        max_model_len=8192,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
    )
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=request,
        attempt=1,
    )

    assert plan.adjustments == ()
    assert plan.retry_available is False


def test_plan_line_records_estimate_and_adjustment() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=VllmRequested(),
        attempt=1,
    )

    line = plan.plan_line()
    assert line.startswith("qiip_vllm_plan: ")
    assert "mode=automatic" in line
    assert "adjustment=max_model_len=4096:oom_retry" in line
    assert "budget_mib=73728" in line
    assert "free_mib=81920" in line
    assert "fitting=true" in line


def test_unsupported_profile_raises() -> None:
    with pytest.raises(ValueError):
        plan_vllm_launch(
            profile_bucket="mystery", devices=[_device()], requested=VllmRequested()
        )


def test_empty_device_list_raises() -> None:
    with pytest.raises(ValueError):
        plan_vllm_launch(profile_bucket="hopper", devices=[], requested=VllmRequested())


def test_model_profile_for_reads_a_cached_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "models--acme--fits"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16",
            }
        )
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"0" * 4096)

    profile = model_profile_for("acme/fits", cache_dir=tmp_path)
    assert profile.name == "acme/fits"
    assert profile.num_hidden_layers == 32
    assert profile.num_key_value_heads == 8
    assert profile.weights_mib == 1
    assert profile.approximate is False


def test_load_model_profile_rejects_missing_config(tmp_path: Path) -> None:
    assert load_model_profile(tmp_path) is None


def test_load_model_profile_rejects_geometry_without_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "num_attention_heads": 2,
            }
        )
    )
    assert load_model_profile(tmp_path) is None


def test_estimate_model_profile_infers_quantization() -> None:
    profile = estimate_model_profile("acme/Secret-13B-AWQ")

    assert profile.approximate is True
    assert profile.quantized is True
    assert profile.kv_dtype == "float16"
    assert profile.weights_mib < 13_000
    assert profile.max_position_embeddings == 8192


def test_estimate_model_profile_requires_a_parameter_count() -> None:
    with pytest.raises(ValueError):
        estimate_model_profile("acme/mystery-checkpoint")


def test_unknown_explicit_model_falls_back_to_a_conservative_estimate() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=VllmRequested(model="acme/Secret-13B-AWQ"),
    )

    assert plan.approximate is True
    assert plan.fitting is True
    assert plan.effective.model == "acme/Secret-13B-AWQ"


def test_automatic_context_can_satisfy_the_desired_context() -> None:
    # A small explicit model leaves enough KV budget to keep the desired
    # context instead of shrinking it.
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=VllmRequested(model="Qwen/Qwen2.5-7B-Instruct"),
    )

    assert plan.fitting is True
    assert plan.effective.max_model_len == 32_768
    assert "fits the KV budget" in plan.reason


def test_plan_rejects_a_negative_attempt() -> None:
    with pytest.raises(ValueError, match="attempt must not be negative"):
        plan_vllm_launch(
            profile_bucket="hopper",
            devices=[_device()],
            requested=VllmRequested(),
            attempt=-1,
        )


def test_model_profile_for_falls_back_when_the_cache_is_unusable(
    tmp_path: Path,
) -> None:
    # No cached repo at all: reading refs raises OSError.
    missing = model_profile_for("acme/Model-7B", cache_dir=tmp_path)
    assert missing.approximate is True

    # A refs/main file with no revision falls back too.
    repo = tmp_path / "models--acme--Model-7B"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("")
    assert model_profile_for("acme/Model-7B", cache_dir=tmp_path).approximate is True

    # A cached snapshot with a non-dict config is rejected, then estimated.
    (repo / "refs" / "main").write_text("rev1")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("[]")
    assert model_profile_for("acme/Model-7B", cache_dir=tmp_path).approximate is True


def test_load_model_profile_marks_quantized_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "quantization_config": {"quant_method": "awq"},
            }
        )
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"0" * 4096)

    profile = load_model_profile(tmp_path, name="acme/quant")
    assert profile is not None
    assert profile.quantized is True
    assert profile.kv_dtype == "float16"


def test_load_model_profile_rejects_config_without_geometry(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4096}))
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"0" * 4096)

    assert load_model_profile(tmp_path) is None


def test_a_budget_without_room_for_the_minimum_context_is_reported() -> None:
    plan = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device(free_mib=71_650)],
        requested=VllmRequested(),
    )

    assert plan.fitting is False
    assert plan.effective.max_model_len == MIN_CONTEXT
    assert "minimum tested context" in plan.reason


def test_adjusted_reflects_whether_any_retry_changed_a_field() -> None:
    initial = plan_vllm_launch(
        profile_bucket="hopper", devices=[_device()], requested=VllmRequested()
    )
    retried = plan_vllm_launch(
        profile_bucket="hopper",
        devices=[_device()],
        requested=VllmRequested(),
        attempt=1,
    )

    assert initial.adjusted is False
    assert retried.adjusted is True


def test_context_ladder_is_ascending_and_starts_at_minimum() -> None:
    assert tuple(sorted(CONTEXT_LADDER)) == CONTEXT_LADDER
    assert CONTEXT_LADDER[0] == MIN_CONTEXT
    assert vllm_fit.largest_ladder_value(5000, CONTEXT_LADDER) == 4096
    assert vllm_fit.largest_ladder_value(100, CONTEXT_LADDER) == 0


def test_vllm_profile_bucket_mirrors_the_node_profiles() -> None:
    assert vllm_profile_bucket("9.0", 81_920) == "hopper"
    assert vllm_profile_bucket("8.0", 40_960) == "ampere-a100"
    assert vllm_profile_bucket("8.0", 24_576) == "ampere-a30"
    assert vllm_profile_bucket("8.6", 49_152) == "ga102-dc"
    assert vllm_profile_bucket("8.6", 24_576) == "consumer-ampere"
    assert vllm_profile_bucket("8.9", 24_576) == "consumer-ada"
    assert vllm_profile_bucket("7.5", 16_384) == "turing"
    assert vllm_profile_bucket("7.5", 8_192) == "consumer-turing"
    assert vllm_profile_bucket("7.0", 16_384) == "volta"


def test_vllm_profile_bucket_rejects_unparseable_or_unsupported_hardware() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        vllm_profile_bucket("unknown", 81_920)
    with pytest.raises(ValueError, match="no tested runtime profile"):
        # An sm90 part below the Hopper memory floor has no tested bucket.
        vllm_profile_bucket("9.0", 4_096)


def test_is_memory_failure_recognizes_only_cuda_oom_signatures() -> None:
    assert is_memory_failure("torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert is_memory_failure("CUDA error: out of memory")
    assert is_memory_failure(
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    )
    assert is_memory_failure("RuntimeError: out of memory on device 0")
    assert not is_memory_failure("Connection refused")
    assert not is_memory_failure("")
    assert not is_memory_failure("the host is out of memory for disk")
