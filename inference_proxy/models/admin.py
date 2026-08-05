"""Admin API response models for operational visibility.

Per METR-03: Each node entry includes identity, health status, active
connections, and circuit breaker state for the operations dashboard.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_proxy.huggingface.artifacts import GGUFArtifact, GGUFDownloadSpec
from inference_proxy.models.llmfit import ModelRecommendation, SystemInfo
from inference_proxy.models.node import InferenceEngine, LlamaCppRuntimeState


class AdminNodeResponse(BaseModel):
    """Admin API response for a single registered node.

    Includes node identity, health status, and operational state
    (active connections, circuit breaker).  The ``status`` field is
    ``str`` (not ``NodeStatus`` enum) because the response serializes
    the enum's value.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str
    endpoint: str
    model: str
    status: str
    active_connections: int
    circuit_breaker_state: str
    engine: InferenceEngine | None = None
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    llamacpp_runtime: LlamaCppRuntimeState | None = None
    state: str = ""
    actions: list[str] = []
    gpu_vendor: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    managed: bool = True
    failed_step: str | None = None
    error: str | None = None


class AdminMetricsResponse(BaseModel):
    """Admin API response for aggregate request metrics.

    Serves the ``/admin/metrics`` endpoint with total and per-dimension
    request counts.
    """

    model_config = ConfigDict(frozen=True)

    total_requests: int
    per_model: dict[str, int]
    per_node: dict[str, int]


class SetupRequest(BaseModel):
    """Request body for POST /admin/nodes/setup."""

    model_config = ConfigDict(frozen=True)

    hostname: str
    managed: bool = True
    model: str | None = Field(default=None, max_length=256)
    engine: InferenceEngine = InferenceEngine.VLLM
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("hostname")
    @classmethod
    def validate_hostname(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 253:
            raise ValueError("hostname must be 1-253 characters")
        if not re.fullmatch(r"[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?", v):
            raise ValueError("hostname contains invalid characters")
        return v

    @model_validator(mode="after")
    def validate_engine_selection(self) -> SetupRequest:
        if self.engine == InferenceEngine.LLAMA_CPP:
            if self.artifact_id is None:
                raise ValueError("llama_cpp setup requires artifact_id")
            if self.model is not None:
                raise ValueError("llama_cpp setup uses artifact_id, not model")
        elif self.artifact_id is not None:
            raise ValueError("artifact_id is only valid for llama_cpp setup")
        return self


class SetupResponse(BaseModel):
    """Response body for POST /admin/nodes/setup (202)."""

    model_config = ConfigDict(frozen=True)

    task_id: str


class TeardownResponse(BaseModel):
    """Response body for DELETE /admin/nodes/{id} (202)."""

    model_config = ConfigDict(frozen=True)

    task_id: str


class TaskStatusResponse(BaseModel):
    """Provisioning task status from etcd."""

    model_config = ConfigDict(frozen=True)

    hostname: str
    current_step: str
    started_at: datetime
    updated_at: datetime
    failed_step: str | None = None
    error: str | None = None


class QUADSStatusResponse(BaseModel):
    """QUADS poller staleness data for the dashboard status indicator."""

    model_config = ConfigDict(frozen=True)

    status: str
    last_sync: datetime | None
    consecutive_failures: int


class PowerAction(str, Enum):
    """Redfish power actions matching _ACTION_TARGET_STATE keys in redfish/client.py."""

    On = "On"
    ForceOff = "ForceOff"
    GracefulRestart = "GracefulRestart"
    ForceRestart = "ForceRestart"


class PowerActionRequest(BaseModel):
    """Request body for POST /admin/nodes/{hostname}/power."""

    model_config = ConfigDict(frozen=True)

    action: PowerAction


class PowerStateResponse(BaseModel):
    """Response body for power state endpoints (D-05)."""

    model_config = ConfigDict(frozen=True)

    hostname: str
    power_state: str


class DownloadState(str, Enum):
    """State of a background model download."""

    DOWNLOADING = "downloading"
    COMPLETE = "complete"
    FAILED = "failed"


class DownloadRequest(BaseModel):
    """Request body for POST /admin/models/download."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str
    revision: str | None = Field(default=None, min_length=1, max_length=256)
    engine: InferenceEngine = InferenceEngine.VLLM
    gguf: GGUFDownloadSpec | None = None

    @model_validator(mode="after")
    def validate_engine_download(self) -> DownloadRequest:
        if self.engine == InferenceEngine.LLAMA_CPP and self.gguf is None:
            raise ValueError("llama_cpp downloads require an exact gguf specification")
        if self.engine == InferenceEngine.VLLM and self.gguf is not None:
            raise ValueError("gguf is only valid for llama_cpp downloads")
        return self


class DownloadStatusResponse(BaseModel):
    """Status of a background model download."""

    model_config = ConfigDict(frozen=True)

    download_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo_id: str
    requested_revision: str | None = None
    resolved_revision: str | None = None
    engine: InferenceEngine = InferenceEngine.VLLM
    gguf: GGUFDownloadSpec | None = None
    artifacts: tuple[GGUFArtifact, ...] = ()
    status: DownloadState
    started_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


class RecommendationResponse(BaseModel):
    """Response body for GET /admin/nodes/{hostname}/recommendations."""

    model_config = ConfigDict(frozen=True)

    hostname: str
    system: SystemInfo
    models: list[ModelRecommendation]
