"""Unit tests for llmfit Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inference_proxy.models.llmfit import LLMFitResult, SystemInfo

FIXTURE_JSON = json.dumps(
    {
        "system": {
            "total_ram_gb": 64.0,
            "available_ram_gb": 58.24,
            "cpu_cores": 16,
            "cpu_name": "AMD EPYC 7742",
            "has_gpu": True,
            "gpu_vram_gb": 80.0,
            "unified_memory": False,
            "backend": "CUDA",
        },
        "models": [
            {
                "name": "llama-3.3-70b",
                "provider": "Meta",
                "parameter_count": "70B",
                "params_b": 70.0,
                "context_length": 131072,
                "use_case": "general",
                "category": "General",
                "release_date": "2024-12-06",
                "fit_level": "perfect",
                "run_mode": "gpu",
                "score": 95.2,
                "estimated_tps": 42.5,
                "runtime": "vLLM",
                "best_quant": "4bit",
                "memory_required_gb": 43.68,
                "utilization_pct": 68.2,
            },
            {
                "name": "qwen-2.5-72b-instruct",
                "provider": "Alibaba",
                "parameter_count": "72B",
                "params_b": 72.0,
                "context_length": 131072,
                "use_case": "general",
                "category": "General",
                "release_date": "2025-01-15",
                "fit_level": "good",
                "run_mode": "gpu",
                "score": 88.7,
                "estimated_tps": 38.1,
                "runtime": "vLLM",
                "best_quant": "4bit",
                "memory_required_gb": 45.2,
                "utilization_pct": 72.5,
            },
        ],
    }
)


class TestLLMFitResult:
    def test_parses_fixture(self) -> None:
        result = LLMFitResult.model_validate(json.loads(FIXTURE_JSON))
        assert result.system.has_gpu is True
        assert result.system.gpu_vram_gb == 80.0
        assert result.system.backend == "CUDA"
        assert len(result.models) == 2
        assert result.models[0].name == "llama-3.3-70b"
        assert result.models[0].score == 95.2
        assert result.models[1].fit_level == "good"


class TestSystemInfoDefaults:
    def test_minimal_construction(self) -> None:
        info = SystemInfo(has_gpu=False)
        assert info.gpu_vram_gb == 0.0
        assert info.gpu_name == ""
        assert info.cpu_name == ""


class TestFrozenModels:
    def test_assignment_raises(self) -> None:
        info = SystemInfo(has_gpu=True)
        with pytest.raises(ValidationError):
            info.has_gpu = False  # type: ignore[misc]


class TestExtraFieldsIgnored:
    def test_unknown_key_dropped(self) -> None:
        data = json.loads(FIXTURE_JSON)
        data["system"]["new_field"] = "surprise"
        result = LLMFitResult.model_validate(data)
        assert not hasattr(result.system, "new_field")
