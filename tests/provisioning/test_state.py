"""Unit tests for provisioning state types.

Tests cover:
- ProvisioningStep enum has 13 members matching step sequence (D-06)
- ProvisioningState model is frozen with all 6 fields (D-07)
- ProvisioningState round-trips through model_dump/model_validate
- ProvisioningState rejects mutation (frozen)
- FAILED state includes failed_step and error fields (D-08)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from inference_proxy.provisioning.state import ProvisioningState, ProvisioningStep


class TestProvisioningStepEnum:
    """ProvisioningStep has 13 members matching D-06 step sequence."""

    def test_member_count(self) -> None:
        assert len(ProvisioningStep) == 25

    def test_member_values(self) -> None:
        expected = {
            "PENDING": "pending",
            "PREFLIGHT": "preflight",
            "UPLOADING_SCRIPTS": "uploading_scripts",
            "SYSTEM_UPDATE": "system_update",
            "NVIDIA_DRIVER": "nvidia_driver",
            "VLLM_INSTALL": "vllm_install",
            "NFS_MOUNT": "nfs_mount",
            "FIREWALL": "firewall",
            "LLMFIT_INSTALL": "llmfit_install",
            "STARTING_VLLM": "starting_vllm",
            "HEALTH_POLL": "health_poll",
            "REGISTERING": "registering",
            "RELAUNCH_VALIDATING": "relaunch_validating",
            "ROLLING_BACK": "rolling_back",
            "DRAINING": "draining",
            "STOPPING_VLLM": "stopping_vllm",
            "DEREGISTERING": "deregistering",
            "TEARDOWN_COMPLETE": "teardown_complete",
            "COMPLETE": "complete",
            "FAILED": "failed",
        }
        for name, value in expected.items():
            assert ProvisioningStep[name] == value

    def test_is_str_enum(self) -> None:
        assert isinstance(ProvisioningStep.PENDING, str)


class TestProvisioningStateModel:
    """ProvisioningState is frozen with 6 fields (D-07)."""

    def test_create_minimal(self) -> None:
        now = datetime.now(tz=UTC)
        state = ProvisioningState(
            hostname="gpu-01.example.com",
            current_step=ProvisioningStep.PENDING,
            started_at=now,
            updated_at=now,
        )
        assert state.hostname == "gpu-01.example.com"
        assert state.current_step == ProvisioningStep.PENDING
        assert state.failed_step is None
        assert state.error is None

    def test_round_trip(self) -> None:
        now = datetime.now(tz=UTC)
        state = ProvisioningState(
            hostname="gpu-01.example.com",
            current_step=ProvisioningStep.NVIDIA_DRIVER,
            started_at=now,
            updated_at=now,
        )
        dumped = state.model_dump(mode="json")
        roundtripped = ProvisioningState.model_validate(dumped)
        assert roundtripped.hostname == state.hostname
        assert roundtripped.current_step == state.current_step
        assert roundtripped.started_at == state.started_at
        assert roundtripped.updated_at == state.updated_at

    def test_frozen_rejects_mutation(self) -> None:
        now = datetime.now(tz=UTC)
        state = ProvisioningState(
            hostname="gpu-01.example.com",
            current_step=ProvisioningStep.PENDING,
            started_at=now,
            updated_at=now,
        )
        with pytest.raises(ValidationError):
            state.current_step = ProvisioningStep.PREFLIGHT  # type: ignore[misc]

    def test_failed_state_with_error(self) -> None:
        """D-08: FAILED state includes failed_step and error fields."""
        now = datetime.now(tz=UTC)
        state = ProvisioningState(
            hostname="gpu-01.example.com",
            current_step=ProvisioningStep.FAILED,
            started_at=now,
            updated_at=now,
            failed_step="nvidia_driver",
            error="SSH connection timed out",
        )
        assert state.current_step == ProvisioningStep.FAILED
        assert state.failed_step == "nvidia_driver"
        assert state.error == "SSH connection timed out"
