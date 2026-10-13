"""Memory-aware automatic planning for vLLM launches (RFE #130).

The historic ``start-vllm.sh`` heuristics branch on the GPU marketing name and
total VRAM.  They pick a 32K context for almost every data-center bucket and
never ask whether the weights plus the KV cache actually fit the memory the
runtime can use, nor whether another tenant is already holding VRAM.

This module replaces that guesswork with a deterministic plan derived from the
selected devices' free memory, the model's real weight bytes and attention
geometry, a per-profile runtime-overhead estimate, and a safety reserve.  It
records every input it used so the gateway can persist the estimate and show it
in the dashboard.

The planner is pure: it never touches the network or a node.  Model geometry
comes from a small tested catalog for the models the automatic policy selects,
falls back to the on-disk Hugging Face snapshot when one is available, and only
then uses a conservative name-derived estimate for an unknown explicit model.

For a recognized out-of-memory failure in automatic mode the gateway re-plans
with the next bounded adjustment (see :data:`ADJUSTMENT_POLICY`).  A field the
operator set explicitly is never adjusted; if every field is explicit the plan
reports no applicable adjustment and the caller stops retrying.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MIB = 1024 * 1024
ModelSelection = Literal["explicit", "automatic"]
AdjustmentReason = Literal["oom_retry"]

# Tested context lengths. The plan never invents an untested value.
CONTEXT_LADDER: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
MIN_CONTEXT = CONTEXT_LADDER[0]
BATCH_LADDER: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)
MIN_GPU_MEMORY_UTILIZATION = 0.50
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

# Runtime overhead that is not weights or KV cache: CUDA context, activation
# peaks, CUDA graphs and framework allocator behavior. These are conservative
# per-profile estimates, recorded in every plan so drift is visible.
PROFILE_RUNTIME_OVERHEAD_MIB: dict[str, int] = {
    "hopper": 6144,
    "ampere-a100": 5120,
    "ampere-a30": 4096,
    "ga102-dc": 4096,
    "volta": 4096,
    "turing": 2048,
    "consumer-ada": 2048,
    "consumer-ampere": 2048,
    "consumer-turing": 2048,
}
DEFAULT_RUNTIME_OVERHEAD_MIB = 4096
# Safety headroom kept beyond the overhead estimate; at least this much and 2%
# of the usable budget, whichever is larger.
MIN_RESERVE_MIB = 256
RESERVE_FRACTION = 0.02

# Preferred context per measured profile before memory pressure shrinks it.
PROFILE_DESIRED_CONTEXT: dict[str, int] = {
    "hopper": 32768,
    "ampere-a100": 32768,
    "ampere-a30": 32768,
    "ga102-dc": 32768,
    "volta": 8192,
    "turing": 8192,
    "consumer-ada": 4096,
    "consumer-ampere": 4096,
    "consumer-turing": 4096,
}
DEFAULT_DESIRED_CONTEXT = 4096

# Profile bucket -> (SM major, SM minor). Mirrors common/profiles.sh so the
# attention-backend and dtype choices match the runtime profile selection.
PROFILE_COMPUTE_CAPABILITY: dict[str, tuple[int, int]] = {
    "hopper": (9, 0),
    "ampere-a100": (8, 0),
    "ampere-a30": (8, 0),
    "ga102-dc": (8, 6),
    "volta": (7, 0),
    "turing": (7, 5),
    "consumer-ada": (8, 9),
    "consumer-ampere": (8, 6),
    "consumer-turing": (7, 5),
}

# Buckets that spread a model across every selected device by default.
DATACENTER_BUCKETS = frozenset(
    {"hopper", "ampere-a100", "ampere-a30", "ga102-dc", "volta"}
)

_DTYPE_BYTES: dict[str, int] = {
    "auto": 2,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "float": 4,
    "float32": 4,
    "float8": 1,
    "int8": 1,
}


@dataclass(frozen=True)
class VllmModelProfile:
    """Weight footprint and attention geometry used to size a launch."""

    name: str
    weights_mib: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    kv_dtype: str
    quantized: bool = False
    approximate: bool = False

    @property
    def kv_bytes_per_token(self) -> int:
        """Full-precision KV bytes for one token, before tensor-parallel split."""
        return (
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * _dtype_bytes(self.kv_dtype)
        )


@dataclass(frozen=True)
class VllmDeviceMemory:
    """One selected device's memory at plan time."""

    index: int
    total_mib: int
    free_mib: int
    compute_capability: str | None = None


@dataclass(frozen=True)
class VllmRequested:
    """Explicit operator choices. ``None`` means "plan it automatically"."""

    model: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    gpu_devices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class VllmEffective:
    """The launch settings the plan authorizes."""

    model: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    dtype: str | None
    attention_backend: str | None


@dataclass(frozen=True)
class VllmEstimate:
    """Recorded memory arithmetic, all in MiB per allocated device."""

    weights_mib: int
    weights_per_gpu_mib: int
    kv_cache_mib: int
    runtime_overhead_mib: int
    reserve_mib: int
    usable_budget_mib: int
    free_per_gpu_mib: int

    @property
    def total_mib(self) -> int:
        return (
            self.weights_per_gpu_mib
            + self.kv_cache_mib
            + self.runtime_overhead_mib
            + self.reserve_mib
        )


@dataclass(frozen=True)
class VllmAdjustment:
    """One bounded change applied to an automatic field after an OOM."""

    field: str
    previous: str
    value: str
    reason: AdjustmentReason


@dataclass(frozen=True)
class VllmFitPlan:
    """The complete recorded outcome of one planning attempt."""

    model: str
    model_selection: ModelSelection
    model_reason: str
    attempt: int
    fitting: bool
    reason: str
    requested: VllmRequested
    effective: VllmEffective
    estimate: VllmEstimate
    adjustments: tuple[VllmAdjustment, ...] = ()
    retry_available: bool = False
    approximate: bool = False

    @property
    def adjusted(self) -> bool:
        return bool(self.adjustments)

    def plan_line(self) -> str:
        """Return the single-line card the node and its logs can retain."""
        adjustment = (
            ",".join(
                f"{item.field}={item.value}:{item.reason}" for item in self.adjustments
            )
            or "none"
        )
        effective = self.effective
        estimate = self.estimate
        return (
            "qiip_vllm_plan: "
            f"mode={self.model_selection} attempt={self.attempt} "
            f"model={effective.model} tp={effective.tensor_parallel_size} "
            f"max_model_len={effective.max_model_len} "
            f"max_num_batched_tokens={effective.max_num_batched_tokens} "
            f"gpu_memory_utilization={effective.gpu_memory_utilization:g} "
            f"dtype={effective.dtype or 'auto'} "
            f"attention_backend={effective.attention_backend or 'auto'} "
            f"weights_mib={estimate.weights_mib} "
            f"kv_mib={estimate.kv_cache_mib} "
            f"overhead_mib={estimate.runtime_overhead_mib} "
            f"reserve_mib={estimate.reserve_mib} "
            f"budget_mib={estimate.usable_budget_mib} "
            f"free_mib={estimate.free_per_gpu_mib} "
            f"fitting={'true' if self.fitting else 'false'} "
            f"adjustment={adjustment}"
        )


# ── Model catalog ───────────────────────────────────────────────────────

# Geometry for the models the automatic policy selects. Weights are the real
# on-disk checkpoint size in MiB (rounded up); changing a checkpoint size here
# is an intentional planning change.
_MODEL_CATALOG: dict[str, VllmModelProfile] = {
    "Qwen/Qwen2.5-72B-Instruct": VllmModelProfile(
        name="Qwen/Qwen2.5-72B-Instruct",
        weights_mib=145_000,
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        kv_dtype="bfloat16",
    ),
    "Qwen/Qwen2.5-32B-Instruct": VllmModelProfile(
        name="Qwen/Qwen2.5-32B-Instruct",
        weights_mib=64_000,
        num_hidden_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        kv_dtype="bfloat16",
    ),
    "Qwen/Qwen2.5-14B-Instruct": VllmModelProfile(
        name="Qwen/Qwen2.5-14B-Instruct",
        weights_mib=28_500,
        num_hidden_layers=48,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        kv_dtype="bfloat16",
    ),
    "Qwen/Qwen2.5-7B-Instruct": VllmModelProfile(
        name="Qwen/Qwen2.5-7B-Instruct",
        weights_mib=15_200,
        num_hidden_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
        kv_dtype="bfloat16",
    ),
    "Qwen/Qwen3-14B-AWQ": VllmModelProfile(
        name="Qwen/Qwen3-14B-AWQ",
        weights_mib=9_500,
        num_hidden_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        kv_dtype="float16",
        quantized=True,
    ),
}

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")
_QUANTIZED_MARKERS = ("awq", "gptq", "4bit", "4-bit", "int4", "fp8", "w4a16")
_PARAMETER_PATTERN = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*[bB](?![a-zA-Z])")
# Kept deliberately aligned with the retention signature in
# provisioning/reliability.py so the same evidence classifies identically.
_OOM_PATTERN = re.compile(
    r"cuda out of memory|cuda error: out of memory|"
    r"torch\.cuda\.OutOfMemoryError|out of memory on device",
    re.IGNORECASE,
)


def is_memory_failure(log_text: str) -> bool:
    """Return whether an engine log shows a recognized CUDA OOM failure."""
    return _OOM_PATTERN.search(log_text) is not None


def vllm_profile_bucket(compute_capability: str, total_mib: int) -> str:
    """Map measured hardware to a tested profile bucket.

    This mirrors ``common/profiles.sh::select_runtime_profile`` exactly so the
    gateway plans for the same bucket the node would select on its own. The
    shell stays authoritative for OS/ABI and driver gating, which setup.sh has
    already enforced by the time the gateway plans.
    """
    major_text, _, minor_text = compute_capability.strip().partition(".")
    if not major_text.isdigit() or not minor_text.isdigit():
        raise ValueError(f"cannot parse GPU compute capability {compute_capability!r}")
    sm = int(major_text) * 10 + int(minor_text)
    vram_gb = (total_mib + 512) // 1024
    bucket: str | None = None
    if sm == 90 and vram_gb >= 80:
        bucket = "hopper"
    elif sm == 80 and vram_gb >= 40:
        bucket = "ampere-a100"
    elif sm == 80 and vram_gb >= 24:
        bucket = "ampere-a30"
    elif sm == 86 and total_mib >= 45_056:
        bucket = "ga102-dc"
    elif sm == 86:
        bucket = "consumer-ampere"
    elif sm == 89:
        bucket = "consumer-ada"
    elif sm == 75 and total_mib >= 14_336:
        bucket = "turing"
    elif sm == 75:
        bucket = "consumer-turing"
    elif sm == 70 and vram_gb >= 16:
        bucket = "volta"
    if bucket is None:
        raise ValueError(f"no tested runtime profile for sm{sm} with {total_mib} MiB")
    return bucket


def _dtype_bytes(dtype: str | None) -> int:
    if dtype is None:
        return 2
    return _DTYPE_BYTES.get(dtype.strip().lower(), 2)


def select_vllm_model(profile_bucket: str, total_vram_gb: int) -> str:
    """Return the automatic model selection for a profile and VRAM.

    This is the public model-selection policy.  It mirrors the historical
    ``start-vllm.sh`` thresholds but lives here so the gateway can display and
    persist the choice.  An explicit model always bypasses this function.
    """
    if profile_bucket in {"hopper", "ampere-a100"}:
        if total_vram_gb >= 240:
            return "Qwen/Qwen2.5-72B-Instruct"
        if total_vram_gb >= 80:
            return "Qwen/Qwen2.5-32B-Instruct"
        return "Qwen/Qwen2.5-14B-Instruct"
    if profile_bucket in {"ampere-a30", "ga102-dc"}:
        if total_vram_gb >= 80:
            return "Qwen/Qwen2.5-32B-Instruct"
        if total_vram_gb >= 40:
            return "Qwen/Qwen2.5-14B-Instruct"
        return "Qwen/Qwen2.5-7B-Instruct"
    if profile_bucket in {"turing", "consumer-turing"}:
        if profile_bucket == "turing" and total_vram_gb <= 16:
            return "Qwen/Qwen3-14B-AWQ"
        return "Qwen/Qwen2.5-7B-Instruct"
    if profile_bucket == "volta":
        if total_vram_gb >= 96:
            return "Qwen/Qwen2.5-32B-Instruct"
        return "Qwen/Qwen2.5-14B-Instruct"
    if profile_bucket in {"consumer-ada", "consumer-ampere"}:
        if total_vram_gb >= 48:
            return "Qwen/Qwen2.5-14B-Instruct"
        return "Qwen/Qwen2.5-7B-Instruct"
    raise ValueError(f"no automatic model policy for profile {profile_bucket!r}")


def model_profile_for(
    model: str, *, cache_dir: str | Path | None = None
) -> VllmModelProfile:
    """Return a model profile from the catalog, cache, or a name estimate."""
    known = _MODEL_CATALOG.get(model)
    if known is not None:
        return known
    if cache_dir is not None:
        snapshot = _cached_snapshot(Path(cache_dir), model)
        if snapshot is not None:
            loaded = load_model_profile(snapshot, name=model)
            if loaded is not None:
                return loaded
    return estimate_model_profile(model)


def load_model_profile(
    snapshot_dir: Path, *, name: str | None = None
) -> VllmModelProfile | None:
    """Read a checkpoint's ``config.json`` and weight sizes, if present."""
    try:
        config = json.loads((snapshot_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None

    layers = _positive_int(config.get("num_hidden_layers"))
    hidden = _positive_int(config.get("hidden_size"))
    heads = _positive_int(config.get("num_attention_heads"))
    if layers is None or hidden is None or heads is None:
        return None
    kv_heads = _positive_int(config.get("num_key_value_heads")) or heads
    head_dim = _positive_int(config.get("head_dim")) or max(1, hidden // heads)
    max_positions = (
        _positive_int(config.get("max_position_embeddings"))
        or _positive_int(config.get("model_max_length"))
        or 8192
    )

    weights_mib = 0
    for path in sorted(snapshot_dir.rglob("*")):
        if path.is_file() and path.name.endswith(_WEIGHT_SUFFIXES):
            weights_mib += math.ceil(path.stat().st_size / MIB)
    if weights_mib == 0:
        return None

    raw_dtype = config.get("torch_dtype") or config.get("dtype")
    dtype = str(raw_dtype) if raw_dtype is not None else "bfloat16"
    quant_config = config.get("quantization_config")
    quantized = isinstance(quant_config, dict) or dtype.lower() in {"int8", "float8"}
    if quantized:
        dtype = "float16"
    return VllmModelProfile(
        name=name or snapshot_dir.name,
        weights_mib=weights_mib,
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_positions,
        kv_dtype=dtype,
        quantized=quantized,
    )


def estimate_model_profile(model: str) -> VllmModelProfile:
    """Conservative geometry for an explicit model with no cached checkpoint.

    Only the parameter count and quantization are inferred from the name, so
    the result is marked approximate.  The planner still subtracts the real
    free VRAM and a generous reserve, which is what keeps an approximate plan
    from over-committing memory.
    """
    match = _PARAMETER_PATTERN.search(model)
    if match is None:
        raise ValueError(f"cannot estimate the size of model {model!r}")
    params = float(match.group(1)) * 1e9
    lowered = model.lower()
    quantized = any(marker in lowered for marker in _QUANTIZED_MARKERS)
    bytes_per_param = 0.5 if quantized else 2.0
    weights_mib = math.ceil(params * bytes_per_param * 1.1 / MIB)
    # A real transformer keeps layer count proportional to depth; clamp the
    # guess to the range open models actually use instead of pretending to know
    # the exact architecture.
    layers = max(24, min(80, round(params / 1e9 + 12)))
    hidden = 4096
    heads = 32
    kv_heads = 8
    return VllmModelProfile(
        name=model,
        weights_mib=weights_mib,
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hidden // heads,
        max_position_embeddings=8192,
        kv_dtype="float16" if quantized else "bfloat16",
        quantized=quantized,
        approximate=True,
    )


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _cached_snapshot(cache_dir: Path, model: str) -> Path | None:
    repo = cache_dir / f"models--{model.replace('/', '--')}"
    refs = repo / "refs" / "main"
    try:
        revision = refs.read_text().strip()
    except OSError:
        return None
    if not revision:
        return None
    snapshot = repo / "snapshots" / revision
    return snapshot if (snapshot / "config.json").is_file() else None


# ── Planning ────────────────────────────────────────────────────────────


@dataclass
class _MutableEffective:
    tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    dtype: str | None
    attention_backend: str | None


@dataclass(frozen=True)
class _AdjustmentStep:
    field_name: str
    reason: AdjustmentReason
    apply: Callable[[_MutableEffective], None]

    def is_explicit(self, requested: VllmRequested) -> bool:
        return getattr(requested, self.field_name) is not None


def _context_halved(effective: _MutableEffective) -> None:
    effective.max_model_len = max(MIN_CONTEXT, effective.max_model_len // 2)
    effective.max_num_batched_tokens = min(
        effective.max_num_batched_tokens, effective.max_model_len
    )


def _batch_reduced(effective: _MutableEffective) -> None:
    effective.max_num_batched_tokens = min(
        effective.max_num_batched_tokens, MIN_CONTEXT
    )


def _utilization_reduced(effective: _MutableEffective) -> None:
    effective.gpu_memory_utilization = max(
        MIN_GPU_MEMORY_UTILIZATION,
        round(effective.gpu_memory_utilization - 0.05, 2),
    )


# Ordered bounded policy. Each successful OOM retry applies the next step that
# touches a field the operator left automatic.
ADJUSTMENT_POLICY: tuple[_AdjustmentStep, ...] = (
    _AdjustmentStep("max_model_len", "oom_retry", _context_halved),
    _AdjustmentStep("max_num_batched_tokens", "oom_retry", _batch_reduced),
    _AdjustmentStep("gpu_memory_utilization", "oom_retry", _utilization_reduced),
)


def largest_ladder_value(limit: int, ladder: Sequence[int]) -> int:
    """Return the largest tested value not greater than *limit*, or zero."""
    best = 0
    for value in ladder:
        if value <= limit:
            best = value
        else:
            break
    return best


def _default_tensor_parallel(profile_bucket: str, device_count: int) -> int:
    if profile_bucket in DATACENTER_BUCKETS:
        return max(1, device_count)
    return 1


def _choose_dtype(profile_bucket: str, model: VllmModelProfile) -> str:
    capability = PROFILE_COMPUTE_CAPABILITY.get(profile_bucket, (7, 5))
    if model.quantized or capability < (8, 0):
        return "float16"
    return model.kv_dtype if model.kv_dtype in {"bfloat16", "float16"} else "bfloat16"


def _choose_attention_backend(profile_bucket: str) -> str | None:
    capability = PROFILE_COMPUTE_CAPABILITY.get(profile_bucket, (7, 5))
    if capability >= (9, 0):
        return "FLASHINFER"
    if capability >= (8, 0):
        return "FLASH_ATTN"
    return None


def _format_setting(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def plan_vllm_launch(
    *,
    profile_bucket: str,
    devices: Sequence[VllmDeviceMemory],
    requested: VllmRequested | None = None,
    cache_dir: str | Path | None = None,
    attempt: int = 0,
    runtime_overhead_mib: int | None = None,
) -> VllmFitPlan:
    """Plan one vLLM launch attempt from real device memory.

    *attempt* is the zero-based OOM retry counter. Attempt zero uses the
    initial automatic sizing; each later attempt applies the next bounded
    adjustment that touches a field the operator left automatic.
    """
    if not devices:
        raise ValueError("a vLLM plan needs at least one selected device")
    if attempt < 0:
        raise ValueError("attempt must not be negative")
    request = requested or VllmRequested()

    model_name, selection, model_reason = _resolve_model(
        profile_bucket, devices, request
    )
    model = model_profile_for(model_name, cache_dir=cache_dir)

    device_count = len(devices)
    tensor_parallel = request.tensor_parallel_size or _default_tensor_parallel(
        profile_bucket, device_count
    )
    tensor_parallel = max(1, min(tensor_parallel, device_count))
    allocated = list(devices[:tensor_parallel])

    utilization = (
        request.gpu_memory_utilization
        if request.gpu_memory_utilization is not None
        else DEFAULT_GPU_MEMORY_UTILIZATION
    )
    dtype = request.dtype or _choose_dtype(profile_bucket, model)
    backend = _choose_attention_backend(profile_bucket)

    desired_context = min(
        PROFILE_DESIRED_CONTEXT.get(profile_bucket, DEFAULT_DESIRED_CONTEXT),
        model.max_position_embeddings,
    )
    context = (
        request.max_model_len
        if request.max_model_len is not None
        else max(MIN_CONTEXT, largest_ladder_value(desired_context, CONTEXT_LADDER))
    )
    batched = (
        request.max_num_batched_tokens
        if request.max_num_batched_tokens is not None
        else max(MIN_CONTEXT, largest_ladder_value(context, BATCH_LADDER))
    )

    effective = _MutableEffective(
        tensor_parallel_size=tensor_parallel,
        max_model_len=context,
        max_num_batched_tokens=batched,
        gpu_memory_utilization=utilization,
        dtype=dtype,
        attention_backend=backend,
    )
    applicable = [step for step in ADJUSTMENT_POLICY if not step.is_explicit(request)]
    steps_to_apply = applicable[:attempt]
    retry_available = len(applicable) > attempt

    overhead = (
        runtime_overhead_mib
        if runtime_overhead_mib is not None
        else PROFILE_RUNTIME_OVERHEAD_MIB.get(
            profile_bucket, DEFAULT_RUNTIME_OVERHEAD_MIB
        )
    )
    # A reduced utilization shrinks the usable budget, so apply that class of
    # adjustment before sizing. Context and batch steps lower the fitted result
    # afterwards so a retry is never re-expanded back to the fit maximum.
    utilization_before = effective.gpu_memory_utilization
    for step in steps_to_apply:
        if step.field_name == "gpu_memory_utilization":
            step.apply(effective)

    remaining, budget, free, weights_per_gpu, kv_per_token = _memory_budget(
        model,
        allocated,
        effective.tensor_parallel_size,
        effective.gpu_memory_utilization,
    )
    reserve = max(MIN_RESERVE_MIB, round(budget * RESERVE_FRACTION))
    kv_budget = remaining - overhead - reserve

    fitting, context, reason = _fit_context(
        model=model,
        requested=request,
        desired_context=desired_context,
        current_context=effective.max_model_len,
        kv_budget_mib=kv_budget,
        kv_per_token_mib=kv_per_token,
        weights_per_gpu_mib=weights_per_gpu,
    )
    effective.max_model_len = context
    effective.max_num_batched_tokens = min(effective.max_num_batched_tokens, context)

    adjustments: list[VllmAdjustment] = []
    for step in steps_to_apply:
        if step.field_name == "gpu_memory_utilization":
            if effective.gpu_memory_utilization != utilization_before:
                adjustments.append(
                    VllmAdjustment(
                        field=step.field_name,
                        previous=_format_setting(utilization_before),
                        value=_format_setting(effective.gpu_memory_utilization),
                        reason=step.reason,
                    )
                )
            continue
        previous = getattr(effective, step.field_name)
        step.apply(effective)
        value = getattr(effective, step.field_name)
        if value != previous:
            adjustments.append(
                VllmAdjustment(
                    field=step.field_name,
                    previous=_format_setting(previous),
                    value=_format_setting(value),
                    reason=step.reason,
                )
            )
    effective.max_num_batched_tokens = min(
        effective.max_num_batched_tokens, effective.max_model_len
    )
    if not fitting:
        retry_available = False

    kv_cache_mib = math.ceil(kv_per_token * effective.max_model_len)
    estimate = VllmEstimate(
        weights_mib=model.weights_mib,
        weights_per_gpu_mib=weights_per_gpu,
        kv_cache_mib=kv_cache_mib,
        runtime_overhead_mib=overhead,
        reserve_mib=reserve,
        usable_budget_mib=budget,
        free_per_gpu_mib=free,
    )
    return VllmFitPlan(
        model=model_name,
        model_selection=selection,
        model_reason=model_reason,
        attempt=attempt,
        fitting=fitting,
        reason=reason,
        requested=request,
        effective=VllmEffective(
            model=model_name,
            tensor_parallel_size=effective.tensor_parallel_size,
            max_model_len=effective.max_model_len,
            max_num_batched_tokens=effective.max_num_batched_tokens,
            gpu_memory_utilization=effective.gpu_memory_utilization,
            dtype=effective.dtype,
            attention_backend=effective.attention_backend,
        ),
        estimate=estimate,
        adjustments=tuple(adjustments),
        retry_available=retry_available,
        approximate=model.approximate,
    )


def _resolve_model(
    profile_bucket: str,
    devices: Sequence[VllmDeviceMemory],
    requested: VllmRequested,
) -> tuple[str, ModelSelection, str]:
    if requested.model:
        return (
            requested.model,
            "explicit",
            f"explicit model {requested.model}",
        )
    total_vram_gb = sum(device.total_mib for device in devices) // 1024
    model = select_vllm_model(profile_bucket, total_vram_gb)
    return (
        model,
        "automatic",
        f"automatic policy: profile {profile_bucket} with {total_vram_gb} GB",
    )


def _memory_budget(
    model: VllmModelProfile,
    devices: Sequence[VllmDeviceMemory],
    tensor_parallel: int,
    utilization: float,
) -> tuple[int, int, int, int, float]:
    """Return budget left after weights, usable budget, free floor, weights/GPU, KV/token."""
    usable = [
        min(device.free_mib, math.floor(device.total_mib * utilization))
        for device in devices
    ]
    budget = min(usable)
    free = min(device.free_mib for device in devices)
    weights_per_gpu = math.ceil(model.weights_mib / tensor_parallel)
    kv_shards = max(1, min(tensor_parallel, model.num_key_value_heads))
    kv_per_token_mib = model.kv_bytes_per_token / kv_shards / MIB
    remaining = budget - weights_per_gpu
    return remaining, budget, free, weights_per_gpu, kv_per_token_mib


def _fit_context(
    *,
    model: VllmModelProfile,
    requested: VllmRequested,
    desired_context: int,
    current_context: int,
    kv_budget_mib: int,
    kv_per_token_mib: float,
    weights_per_gpu_mib: int,
) -> tuple[bool, int, str]:
    """Return whether the context fits, the usable context, and why."""
    if kv_budget_mib <= 0 or kv_per_token_mib <= 0:
        return (
            False,
            current_context,
            (
                f"weights ({weights_per_gpu_mib} MiB) plus runtime overhead and "
                "reserve exceed the usable device budget"
            ),
        )
    max_context_by_kv = math.floor(kv_budget_mib / kv_per_token_mib)
    if requested.max_model_len is not None:
        if max_context_by_kv < current_context:
            return (
                False,
                current_context,
                (
                    f"explicit max_model_len {current_context} needs more KV "
                    f"memory than the {kv_budget_mib} MiB left after weights"
                ),
            )
        return (True, current_context, "explicit max_model_len fits the budget")
    ceiling = min(current_context, max_context_by_kv, model.max_position_embeddings)
    chosen = largest_ladder_value(ceiling, CONTEXT_LADDER)
    if chosen < MIN_CONTEXT:
        return (
            False,
            MIN_CONTEXT,
            (
                "not enough KV memory for the minimum tested context "
                f"({max_context_by_kv} of {MIN_CONTEXT} tokens fit)"
            ),
        )
    if chosen < desired_context:
        return (
            True,
            chosen,
            (
                f"context reduced from {desired_context} to {chosen} tokens to "
                f"fit the {kv_budget_mib} MiB KV budget"
            ),
        )
    return (True, chosen, f"context {chosen} tokens fits the KV budget")
