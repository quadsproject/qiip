"""Node state domain model for inference nodes.

Represents the state of an inference backend node as tracked in etcd.
InferenceEngine, NodeStatus are StrEnums for type-safe values.
Node and NodeCapabilities are Pydantic models for validation.

Per D-15: No serialization methods on the model -- serialization
is a separate concern handled in a future phase.
Per D-16: The ``model`` field is ``str``, not ``list[str]``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

LLAMACPP_CONTEXT_ALIGNMENT = 256
LLAMACPP_MAX_AGGREGATE_CONTEXT = 4_294_967_040
LLAMACPP_MAX_SEQUENCES = 256

# vLLM --dtype values accepted at the API boundary and by the auto-vLLM
# start script. Kept in one place so a dtype override can never smuggle extra
# vLLM argv (e.g. "float16 --seed 0" is rejected, not word-split). This is the
# exact set the pinned vLLM 0.26.0 accepts for --dtype; the float8_* values are
# KV-cache dtype settings and are intentionally excluded.
SUPPORTED_VLLM_DTYPES = frozenset(
    {
        "auto",
        "half",
        "float16",
        "bfloat16",
        "float",
        "float32",
    }
)


class InferenceEngine(StrEnum):
    """Supported inference engine backends."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"


class VllmParams(BaseModel):
    """Optional vLLM serve parameters submitted at setup time.

    The ``dtype`` field is validated against the supported vLLM dtype
    allowlist rather than passed through verbatim, so a value such as
    ``"float16 --seed 0"`` is rejected instead of becoming an additional
    vLLM flag downstream.
    """

    model_config = ConfigDict(frozen=True)

    tensor_parallel_size: int | None = Field(default=None, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    tool_call_parser: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_parser: str | None = Field(default=None, min_length=1, max_length=256)
    dtype: str | None = Field(default=None, min_length=1, max_length=256)
    gpu_devices: tuple[int, ...] | None = Field(default=None, min_length=1)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value not in SUPPORTED_VLLM_DTYPES:
            allowed = ", ".join(sorted(SUPPORTED_VLLM_DTYPES))
            raise ValueError(f"unsupported vLLM dtype {value!r}; allowed: {allowed}")
        return value

    @field_validator("gpu_devices")
    @classmethod
    def validate_gpu_devices(
        cls, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if value is None:
            return value
        if any(device < 0 for device in value):
            raise ValueError("GPU device indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("GPU device indices must be unique")
        return value

    @model_validator(mode="after")
    def validate_tensor_parallel_fits_devices(self) -> VllmParams:
        if (
            self.gpu_devices is not None
            and self.tensor_parallel_size is not None
            and self.tensor_parallel_size > len(self.gpu_devices)
        ):
            raise ValueError(
                "tensor_parallel_size exceeds the selected GPU device count"
            )
        return self


class NodeStatus(StrEnum):
    """Status of an inference node."""

    AVAILABLE = "available"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    RELAUNCHING = "relaunching"
    RELAUNCH_FAILED = "relaunch_failed"
    PROVISIONING = "provisioning"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NodeCapabilities(BaseModel):
    """Hardware and serving capabilities of a node."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 4096
    gpu_memory: str = ""


class LlamaCppSizingMode(StrEnum):
    """Gateway-authorized llama.cpp sizing policy."""

    AUTO = "auto"
    CUSTOM = "custom"
    PROFILE = "profile"


class LlamaCppCacheType(StrEnum):
    """Managed llama.cpp KV-cache precision.

    ``q4_0`` is reachable only through a catalog profile: the VRAM planner
    behind the automatic and custom policies selects between f16 and q8_0.
    """

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


PLANNER_CACHE_TYPES = frozenset({LlamaCppCacheType.F16, LlamaCppCacheType.Q8_0})


class LlamaCppFlashAttention(StrEnum):
    """Managed llama.cpp Flash Attention policy."""

    AUTO = "auto"
    ON = "on"


class LlamaCppSpeculativeType(StrEnum):
    """Speculative decoding implementations a catalog profile may select."""

    DRAFT_MTP = "draft-mtp"
    DRAFT_DFLASH = "draft-dflash"
    # An MTP head shipped as its own GGUF (Gemma 4 "assistant"). llama-server
    # still takes ``--spec-type draft-mtp``; the difference is a separate
    # draft file that has no KV cache of its own and reads the target's.
    DRAFT_MTP_ASSISTANT = "draft-mtp-assistant"


class LlamaCppSampling(BaseModel):
    """Server-side sampling defaults; a request may still override them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0, le=1000)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)


class LlamaCppProfileRuntime(BaseModel):
    """The exact, typed launch contract of one catalog profile.

    Every value is an enumeration, a bounded number or a content-addressed
    artifact id, so a profile can never carry free-form llama-server argv.
    ``required_free_mib`` is compared with ``nvidia-smi`` free memory before
    the launch. It therefore includes the CUDA context of the new process,
    unlike a measured "VRAM needed" figure, which excludes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    profile_version: int = Field(ge=1)
    ubatch: int = Field(ge=32, le=4096)
    required_free_mib: int = Field(ge=1)
    # The GPU product this launch was planned for, as nvidia-smi names it, and
    # the least total memory that product has. Both are checked on the node:
    # an inventory string is not evidence of what a booted host exposes.
    gpu_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,127}$")
    gpu_min_total_mib: int = Field(ge=1)
    speculative_type: LlamaCppSpeculativeType
    speculative_draft_n_max: int = Field(ge=1, le=16)
    draft_cache_type: LlamaCppCacheType | None = None
    draft_artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sampling: LlamaCppSampling
    disable_cuda_graphs: bool = False

    @model_validator(mode="after")
    def validate_speculation(self) -> LlamaCppProfileRuntime:
        if self.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP:
            if self.draft_artifact_id is not None:
                raise ValueError("MTP drafts from the target file, not an artifact")
            if self.draft_cache_type is None:
                raise ValueError("MTP profiles must state the draft cache type")
        elif self.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP_ASSISTANT:
            if self.draft_artifact_id is None:
                raise ValueError("MTP assistant profiles require a draft artifact")
            if self.draft_cache_type is not None:
                raise ValueError("an MTP assistant shares the target's KV cache")
        else:
            if self.draft_artifact_id is None:
                raise ValueError("DFlash profiles require a draft artifact")
            if self.draft_cache_type is not None:
                raise ValueError("DFlash profiles use the default draft cache")
        return self


class LlamaCppRuntimeRequest(BaseModel):
    """Requested sizing policy retained for retries and later relaunches."""

    model_config = ConfigDict(frozen=True)

    sizing: LlamaCppSizingMode
    fit_target_mib: int = Field(ge=1)
    context_per_slot: int | None = Field(default=None, ge=LLAMACPP_CONTEXT_ALIGNMENT)
    slots: int | None = Field(default=None, ge=1, le=LLAMACPP_MAX_SEQUENCES)
    cache_type: LlamaCppCacheType | None = None
    allow_estimator_overrun: bool = False
    profile: LlamaCppProfileRuntime | None = None

    @model_serializer
    def serialize_policy(self) -> dict[str, object]:
        """Keep the established automatic-policy wire shape compact."""
        values: dict[str, object] = {
            "sizing": self.sizing,
            "fit_target_mib": self.fit_target_mib,
        }
        if self.context_per_slot is not None:
            values["context_per_slot"] = self.context_per_slot
        if self.slots is not None:
            values["slots"] = self.slots
        if self.cache_type is not None:
            values["cache_type"] = self.cache_type
        if self.allow_estimator_overrun:
            values["allow_estimator_overrun"] = True
        if self.profile is not None:
            values["profile"] = self.profile.model_dump(mode="json", exclude_none=True)
        return values

    @model_validator(mode="after")
    def validate_sizing_policy(self) -> LlamaCppRuntimeRequest:
        custom_values = (self.context_per_slot, self.slots, self.cache_type)
        if self.sizing is LlamaCppSizingMode.AUTO:
            if (
                any(value is not None for value in custom_values)
                or self.allow_estimator_overrun
                or self.profile is not None
            ):
                raise ValueError(
                    "automatic llama.cpp sizing does not accept custom values"
                )
            return self

        if any(value is None for value in custom_values):
            raise ValueError(
                f"{self.sizing.value} llama.cpp sizing requires context_per_slot, "
                "slots, and cache_type"
            )
        assert self.context_per_slot is not None
        assert self.slots is not None
        if self.sizing is LlamaCppSizingMode.PROFILE:
            if self.profile is None:
                raise ValueError("profile llama.cpp sizing requires a profile")
            if self.slots != 1:
                raise ValueError("profile llama.cpp sizing serves exactly one slot")
            if self.allow_estimator_overrun:
                raise ValueError("profile llama.cpp sizing has no estimator")
        else:
            if self.profile is not None:
                raise ValueError("custom llama.cpp sizing does not accept a profile")
            if self.cache_type not in PLANNER_CACHE_TYPES:
                raise ValueError("custom llama.cpp sizing supports f16 or q8_0 KV")
        if self.context_per_slot % LLAMACPP_CONTEXT_ALIGNMENT:
            raise ValueError("context_per_slot must be aligned to 256-token increments")
        if self.context_per_slot * self.slots > LLAMACPP_MAX_AGGREGATE_CONTEXT:
            raise ValueError(
                "context_per_slot * slots exceeds llama.cpp's aggregate context limit"
            )
        return self


class LlamaCppSpeculativeEffective(BaseModel):
    """Draft-side evidence verified separately from the target model."""

    model_config = ConfigDict(frozen=True)

    type: LlamaCppSpeculativeType
    draft_n_max: int = Field(ge=1)
    # None for MTP, which drafts from the already-offloaded target model.
    draft_gpu_layers: int | None = Field(default=None, ge=1)
    draft_total_layers: int | None = Field(default=None, ge=1)
    # None for an MTP assistant, whose every layer reads the target's cache.
    draft_cache_type_k: LlamaCppCacheType | None = None
    draft_cache_type_v: LlamaCppCacheType | None = None
    draft_shares_target_cache: bool = False


class LlamaCppRuntimeEffective(BaseModel):
    """Effective llama.cpp runtime configuration verified after startup."""

    model_config = ConfigDict(frozen=True)

    train_context: int = Field(ge=1)
    context_per_slot: int = Field(ge=1)
    slot_context_limit: int = Field(ge=1)
    slots: int = Field(ge=1, le=256)
    aggregate_context: int = Field(ge=1, le=4_294_967_040)
    cache_type_k: LlamaCppCacheType
    cache_type_v: LlamaCppCacheType
    flash_attn: LlamaCppFlashAttention
    kv_unified: bool
    gpu_layers: int = Field(ge=1)
    total_layers: int = Field(ge=1)
    estimator_overrun_used: bool = False
    ubatch: int | None = Field(default=None, ge=1)
    speculative: LlamaCppSpeculativeEffective | None = None

    @model_serializer(mode="wrap")
    def serialize_effective(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Keep the planner-policy wire shape unchanged for non-profile nodes."""
        values: dict[str, object] = handler(self)
        for name in ("ubatch", "speculative"):
            if values.get(name) is None:
                values.pop(name, None)
        return values


class LlamaCppGPUState(BaseModel):
    """One GPU's post-load memory telemetry."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    total_mib: int = Field(ge=1)
    used_mib: int = Field(ge=0)
    free_mib: int = Field(ge=0)


class LlamaCppRuntimeState(BaseModel):
    """Verified managed llama.cpp plan and its post-load observation."""

    model_config = ConfigDict(frozen=True)

    requested: LlamaCppRuntimeRequest
    effective: LlamaCppRuntimeEffective
    gpus: tuple[LlamaCppGPUState, ...] = Field(min_length=1)
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value


class NodeGPU(BaseModel):
    """One physical GPU observed on the node with ``nvidia-smi``."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    uuid: str = Field(pattern=r"^GPU-[0-9a-fA-F-]{8,64}$")
    name: str = Field(min_length=1, max_length=128)
    total_mib: int = Field(ge=1)


class NodePlacement(BaseModel):
    """Provenance of a node that automatic placement provisioned.

    Absent on every node a person set up. Automatic placement only ever
    retries, replaces or counts nodes that carry this marker with its own
    claim id.
    """

    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    profile_version: int = Field(ge=1)
    claim_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class Node(BaseModel):
    """An inference node registered in etcd.

    Instances are immutable (``frozen=True``) to prevent external
    mutation of registry entries without acquiring the registry lock.
    Use ``model_copy(update={...})`` to create modified copies.

    Attributes:
        node_id: Unique identifier for the node.
        name: Optional operator-facing display name (e.g. a friendly label for
            an admin-only server). Empty means the node id is used for display.
        endpoint: HTTP endpoint (host:port) for the inference server.
        status: Current health status of the node.
        model: Name of the model being served.
        engine: Inference engine backend (vllm or llama_cpp).
        artifact_id: Exact gateway-discovered GGUF generation, when applicable.
        llamacpp_runtime: Verified managed llama.cpp sizing and GPU telemetry.
        last_heartbeat: Timestamp of the last health check response.
        capabilities: Hardware and serving capabilities.
        active_connections: Number of active inference requests.
        managed: Whether the proxy owns the node lifecycle. Externally
            registered nodes must opt in explicitly.
        self_setup: Whether the node was adopted from an already-running
            OpenAI-compatible server. These nodes are never torn down.
        admin_only: Whether the node is visible only to admin callers.
            Admin-only nodes are adopted running servers (``self_setup``)
            that are never listed on the non-admin fleet page and are
            routable only to admin callers (admin-role bearer tokens or the
            full-access trust list).
        owner: Email of the endpoint owner. Empty means shared (any user
            may route to it); an owner restricts routing to that user's
            tokens and admin full-access tokens.
        gpus: GPU inventory read from the node during provisioning.
        placement: Set only on nodes that automatic placement provisioned.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    node_id: str
    name: str = ""
    endpoint: str
    status: NodeStatus = NodeStatus.UNKNOWN
    model: str = ""
    engine: InferenceEngine = InferenceEngine.VLLM
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    llamacpp_runtime: LlamaCppRuntimeState | None = None
    last_heartbeat: datetime | None = None
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)
    active_connections: int = 0
    managed: bool = False
    self_setup: bool = False
    admin_only: bool = False
    owner: str = ""
    gpus: tuple[NodeGPU, ...] = ()
    placement: NodePlacement | None = None

    @model_validator(mode="after")
    def placement_is_managed(self) -> Node:
        if self.placement is not None and (not self.managed or self.owner):
            raise ValueError("automatic placements are managed and unowned")
        return self

    @model_validator(mode="after")
    def self_setup_is_unmanaged(self) -> Node:
        if self.self_setup and self.managed:
            raise ValueError("self_setup nodes cannot be managed")
        return self
