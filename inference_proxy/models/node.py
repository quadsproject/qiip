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
    field_validator,
    model_serializer,
    model_validator,
)

LLAMACPP_CONTEXT_ALIGNMENT = 256
LLAMACPP_MAX_AGGREGATE_CONTEXT = 4_294_967_040
LLAMACPP_MAX_SEQUENCES = 256


class InferenceEngine(StrEnum):
    """Supported inference engine backends."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"


class NodeStatus(StrEnum):
    """Status of an inference node."""

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


class LlamaCppCacheType(StrEnum):
    """Managed llama.cpp KV-cache precision."""

    F16 = "f16"
    Q8_0 = "q8_0"


class LlamaCppFlashAttention(StrEnum):
    """Managed llama.cpp Flash Attention policy."""

    AUTO = "auto"
    ON = "on"


class LlamaCppRuntimeRequest(BaseModel):
    """Requested sizing policy retained for retries and later relaunches."""

    model_config = ConfigDict(frozen=True)

    sizing: LlamaCppSizingMode
    fit_target_mib: int = Field(ge=1)
    context_per_slot: int | None = Field(default=None, ge=LLAMACPP_CONTEXT_ALIGNMENT)
    slots: int | None = Field(default=None, ge=1, le=LLAMACPP_MAX_SEQUENCES)
    cache_type: LlamaCppCacheType | None = None
    allow_estimator_overrun: bool = False

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
        return values

    @model_validator(mode="after")
    def validate_sizing_policy(self) -> LlamaCppRuntimeRequest:
        custom_values = (self.context_per_slot, self.slots, self.cache_type)
        if self.sizing is LlamaCppSizingMode.AUTO:
            if any(value is not None for value in custom_values) or (
                self.allow_estimator_overrun
            ):
                raise ValueError(
                    "automatic llama.cpp sizing does not accept custom values"
                )
            return self

        if any(value is None for value in custom_values):
            raise ValueError(
                "custom llama.cpp sizing requires context_per_slot, slots, "
                "and cache_type"
            )
        assert self.context_per_slot is not None
        assert self.slots is not None
        if self.context_per_slot % LLAMACPP_CONTEXT_ALIGNMENT:
            raise ValueError("context_per_slot must be aligned to 256-token increments")
        if self.context_per_slot * self.slots > LLAMACPP_MAX_AGGREGATE_CONTEXT:
            raise ValueError(
                "context_per_slot * slots exceeds llama.cpp's aggregate context limit"
            )
        return self


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


class Node(BaseModel):
    """An inference node registered in etcd.

    Instances are immutable (``frozen=True``) to prevent external
    mutation of registry entries without acquiring the registry lock.
    Use ``model_copy(update={...})`` to create modified copies.

    Attributes:
        node_id: Unique identifier for the node.
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
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    node_id: str
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
