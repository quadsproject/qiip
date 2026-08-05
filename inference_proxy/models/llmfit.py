"""Pydantic models for llmfit ``recommend --json`` output.

Parses system hardware info and ranked model recommendations into
typed, immutable objects.  ``extra="ignore"`` ensures forward
compatibility when llmfit adds new fields (T-25-01).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecommendationRuntime(StrEnum):
    """Canonical runtime vocabulary reported by llmfit recommendations.

    This is deliberately distinct from ``InferenceEngine``: recommendation
    output can name runtimes that QIIP does not provision.
    """

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    MLX = "mlx"
    UNKNOWN = "unknown"


_RUNTIME_ALIASES = {
    "vllm": RecommendationRuntime.VLLM,
    "llama.cpp": RecommendationRuntime.LLAMA_CPP,
    "llamacpp": RecommendationRuntime.LLAMA_CPP,
    "llama_cpp": RecommendationRuntime.LLAMA_CPP,
    "mlx": RecommendationRuntime.MLX,
}


class GGUFSource(BaseModel):
    """One GGUF repository candidate reported by llmfit."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    repo: str = Field(min_length=1)
    provider: str


class SystemInfo(BaseModel):
    """Hardware profile detected by llmfit."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    has_gpu: bool
    gpu_vram_gb: float = 0.0
    gpu_name: str = ""
    cpu_name: str = ""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    cpu_cores: int = 0
    unified_memory: bool = False
    backend: str = ""


class ModelRecommendation(BaseModel):
    """A single model recommendation from llmfit."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    score: float = 0.0
    fit_level: str = ""
    estimated_tps: float = 0.0
    memory_required_gb: float = 0.0
    provider: str = ""
    best_quant: str = ""
    run_mode: str = ""
    params_b: float = 0.0
    context_length: int = 0
    utilization_pct: float = 0.0
    category: str = ""
    runtime: RecommendationRuntime = RecommendationRuntime.UNKNOWN
    gguf_sources: tuple[GGUFSource, ...] = ()

    @field_validator("runtime", mode="before")
    @classmethod
    def normalize_runtime(cls, value: object) -> RecommendationRuntime:
        """Normalize llmfit spellings without making unknown runtimes fatal."""
        if isinstance(value, RecommendationRuntime):
            return value
        if not isinstance(value, str):
            return RecommendationRuntime.UNKNOWN
        return _RUNTIME_ALIASES.get(
            value.strip().lower(),
            RecommendationRuntime.UNKNOWN,
        )


class LLMFitResult(BaseModel):
    """Top-level result from ``llmfit recommend --json``."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    system: SystemInfo
    models: list[ModelRecommendation]
