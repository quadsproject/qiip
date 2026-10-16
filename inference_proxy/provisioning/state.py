"""Provisioning step and state types for node setup tracking.

ProvisioningStep (D-06): StrEnum for tracked setup and lifecycle steps.

ProvisioningState (D-07, D-08): Frozen Pydantic model capturing the
current provisioning state of a host.  The ``failed_step`` and ``error``
fields are populated when ``current_step`` is FAILED (D-08). An optional
``resume_state`` identifies required operator action before a setup retry.

ProvisioningCheckpoint (issue #133): durable per-stage completion evidence
bound to a setup plan identity so a compatible retry can resume instead of
repeating successful work.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ProvisioningStep(StrEnum):
    """Steps in the node provisioning sequence (D-06)."""

    PENDING = "pending"
    POWERING_ON = "powering_on"
    PREFLIGHT = "preflight"
    UPLOADING_SCRIPTS = "uploading_scripts"
    SYSTEM_UPDATE = "system_update"
    SYSTEM_PREREQUISITES = "system_prerequisites"
    NVIDIA_DRIVER = "nvidia_driver"
    CUDA_TOOLKIT = "cuda_toolkit"
    VLLM_INSTALL = "vllm_install"
    LLAMACPP_INSTALL = "llamacpp_install"
    GENERATION_ACTIVATE = "generation_activate"
    NFS_MOUNT = "nfs_mount"
    FIREWALL = "firewall"
    LLMFIT_INSTALL = "llmfit_install"
    STARTING_VLLM = "starting_vllm"
    STARTING_LLAMACPP = "starting_llamacpp"
    HEALTH_POLL = "health_poll"
    REGISTERING = "registering"
    RELAUNCH_VALIDATING = "relaunch_validating"
    DRAINING = "draining"
    STOPPING_VLLM = "stopping_vllm"
    STOPPING_LLAMACPP = "stopping_llamacpp"
    DEREGISTERING = "deregistering"
    ROLLING_BACK = "rolling_back"
    TEARDOWN_COMPLETE = "teardown_complete"
    COMPLETE = "complete"
    FAILED = "failed"


class ProvisioningResumeState(StrEnum):
    """Operator action required before a fresh, fully probed setup retry."""

    MAINTENANCE_REQUIRED = "maintenance_required"
    REBOOT_REQUIRED = "reboot_required"


class ProvisioningStageStatus(StrEnum):
    """Completion state of one checkpointed setup stage."""

    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ProvisioningCheckpoint(BaseModel):
    """Durable evidence that one setup stage finished for a compatible plan.

    ``plan_id`` binds the evidence to the exact setup inputs that produced it;
    a retry whose plan inputs changed must not honour the checkpoint. The
    ``attempt_id``/``attempt_number`` fields preserve the attempt relationship
    so recovery history never loses which attempt produced the evidence.
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    status: ProvisioningStageStatus
    plan_id: str | None = None
    attempt_id: str | None = None
    attempt_number: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    evidence: str | None = None


class ProvisioningState(BaseModel):
    """Current provisioning state for a host (D-07, D-08).

    Frozen to prevent external mutation; use ``model_copy(update={...})``
    to create modified copies, matching the Node model pattern.
    """

    model_config = ConfigDict(frozen=True)

    hostname: str
    attempt_id: str | None = None
    current_step: ProvisioningStep
    started_at: datetime
    updated_at: datetime
    failed_step: str | None = None
    error: str | None = None
    resume_state: ProvisioningResumeState | None = None
    # Issue #133: recovery visibility. ``retry_count`` counts automatic
    # transient retries within this attempt; ``progress`` is the bounded
    # checkpoint summary; ``failure_class``/``next_action`` tell an operator
    # what to do before a manual retry.
    attempt_number: int | None = None
    retry_count: int = 0
    progress: dict[str, Any] | None = None
    failure_class: str | None = None
    next_action: str | None = None
