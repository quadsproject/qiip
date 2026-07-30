"""Unit tests for the QUADSHost domain model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_proxy.models.quads import QUADSHost


class TestQUADSHostCreation:
    def test_all_fields_set_correctly(self) -> None:
        host = QUADSHost(
            hostname="gpu-host01",
            gpu_vendor="NVIDIA",
            gpu_model="A100",
            gpu_count=4,
        )
        assert host.hostname == "gpu-host01"
        assert host.gpu_vendor == "NVIDIA"
        assert host.gpu_model == "A100"
        assert host.gpu_count == 4


class TestQUADSHostFrozen:
    def test_assignment_raises_type_error(self) -> None:
        host = QUADSHost(
            hostname="gpu-host01",
            gpu_vendor="NVIDIA",
            gpu_model="A100",
            gpu_count=4,
        )
        with pytest.raises(ValidationError):
            host.hostname = "other"  # type: ignore[misc]


class TestQUADSHostExtraFieldsIgnored:
    def test_extra_kwarg_does_not_raise(self) -> None:
        host = QUADSHost(
            hostname="gpu-host01",
            gpu_vendor="NVIDIA",
            gpu_model="A100",
            gpu_count=4,
            interfaces=["eth0", "eth1"],  # type: ignore[call-arg]
        )
        assert host.hostname == "gpu-host01"
        assert not hasattr(host, "interfaces")
