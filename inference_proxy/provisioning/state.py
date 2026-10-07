"""Provisioning step and state types for node setup tracking.

ProvisioningStep (D-06): StrEnum for tracked setup and lifecycle steps.

ProvisioningState (D-07, D-08): Frozen Pydantic model capturing the
current provisioning state of a host.  The ``failed_step`` and ``error``
fields are populated when ``current_step`` is FAILED (D-08). An optional
``resume_state`` identifies required operator action before a setup retry.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

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
