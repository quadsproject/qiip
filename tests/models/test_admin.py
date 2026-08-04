"""Unit tests for the AdminNodeResponse and AdminMetricsResponse models.

Tests cover:
- AdminNodeResponse creation with valid fields (including operational fields)
- AdminNodeResponse is frozen (immutable)
- AdminMetricsResponse creation with valid fields
- AdminMetricsResponse is frozen (immutable)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_proxy.models.admin import (
    AdminMetricsResponse,
    AdminNodeResponse,
    DownloadRequest,
    SetupRequest,
)
from inference_proxy.models.node import InferenceEngine


class TestAdminNodeResponse:
    """AdminNodeResponse model validation and behavior."""

    def test_create_with_valid_fields(self) -> None:
        """AdminNodeResponse accepts all six fields."""
        response = AdminNodeResponse(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            model="llama-3",
            status="healthy",
            active_connections=2,
            circuit_breaker_state="closed",
        )
        assert response.node_id == "node-1"
        assert response.endpoint == "10.0.1.100:8000"
        assert response.model == "llama-3"
        assert response.status == "healthy"
        assert response.active_connections == 2
        assert response.circuit_breaker_state == "closed"
        assert response.engine is None
        assert response.artifact_id is None

    def test_registered_engine_and_artifact_are_typed(self) -> None:
        response = AdminNodeResponse(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            model="model.gguf",
            status="healthy",
            active_connections=0,
            circuit_breaker_state="closed",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="a" * 64,
        )

        assert response.engine is InferenceEngine.LLAMA_CPP
        assert response.artifact_id == "a" * 64

    def test_admin_node_response_error_fields(self) -> None:
        """AdminNodeResponse accepts and defaults failed_step and error fields."""
        with_errors = AdminNodeResponse(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            model="llama-3",
            status="failed",
            active_connections=0,
            circuit_breaker_state="closed",
            failed_step="uploading_scripts",
            error="connection refused",
        )
        assert with_errors.failed_step == "uploading_scripts"
        assert with_errors.error == "connection refused"

        without_errors = AdminNodeResponse(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            model="llama-3",
            status="healthy",
            active_connections=0,
            circuit_breaker_state="closed",
        )
        assert without_errors.failed_step is None
        assert without_errors.error is None

    def test_frozen_rejects_mutation(self) -> None:
        """AdminNodeResponse is immutable -- assigning to a field raises ValidationError."""
        response = AdminNodeResponse(
            node_id="node-1",
            endpoint="10.0.1.100:8000",
            model="llama-3",
            status="healthy",
            active_connections=0,
            circuit_breaker_state="closed",
        )
        with pytest.raises(ValidationError):
            response.status = "unhealthy"  # type: ignore[misc]


class TestAdminMetricsResponse:
    """AdminMetricsResponse model validation and behavior."""

    def test_create_with_valid_fields(self) -> None:
        """AdminMetricsResponse accepts total_requests, per_model, per_node."""
        response = AdminMetricsResponse(
            total_requests=42,
            per_model={"llama-3": 30, "mistral-7b": 12},
            per_node={"node-1": 25, "node-2": 17},
        )
        assert response.total_requests == 42
        assert response.per_model == {"llama-3": 30, "mistral-7b": 12}
        assert response.per_node == {"node-1": 25, "node-2": 17}

    def test_frozen_rejects_mutation(self) -> None:
        """AdminMetricsResponse is immutable."""
        response = AdminMetricsResponse(
            total_requests=0,
            per_model={},
            per_node={},
        )
        with pytest.raises(ValidationError):
            response.total_requests = 1  # type: ignore[misc]


class TestSetupRequest:
    """SetupRequest model validation for the model field."""

    def test_model_defaults_to_none(self) -> None:
        req = SetupRequest(hostname="gpu01")
        assert req.model is None

    def test_model_accepts_string(self) -> None:
        req = SetupRequest(hostname="gpu01", model="Qwen/Qwen2.5-72B-Instruct")
        assert req.model == "Qwen/Qwen2.5-72B-Instruct"

    def test_model_rejects_oversized(self) -> None:
        with pytest.raises(ValidationError):
            SetupRequest(hostname="gpu01", model="x" * 257)

    def test_frozen_rejects_mutation(self) -> None:
        req = SetupRequest(hostname="gpu01", model="org/model")
        with pytest.raises(ValidationError):
            req.model = "other"  # type: ignore[misc]

    @pytest.mark.parametrize("hostname", ["", "-gpu01", "gpu_01"])
    def test_hostname_rejects_empty_or_invalid_values(self, hostname: str) -> None:
        with pytest.raises(ValidationError):
            SetupRequest(hostname=hostname)

    def test_llamacpp_requires_only_an_artifact(self) -> None:
        artifact_id = "a" * 64
        request = SetupRequest(
            hostname="gpu01",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id=artifact_id,
        )
        assert request.artifact_id == artifact_id

        with pytest.raises(ValidationError, match="requires artifact_id"):
            SetupRequest(hostname="gpu01", engine=InferenceEngine.LLAMA_CPP)
        with pytest.raises(ValidationError, match="uses artifact_id, not model"):
            SetupRequest(
                hostname="gpu01",
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id=artifact_id,
                model="org/model",
            )

    def test_vllm_rejects_an_artifact(self) -> None:
        with pytest.raises(ValidationError, match="only valid for llama_cpp"):
            SetupRequest(hostname="gpu01", artifact_id="a" * 64)


class TestDownloadRequest:
    def test_llamacpp_requires_exact_gguf_specification(self) -> None:
        with pytest.raises(ValidationError, match="require an exact gguf"):
            DownloadRequest(repo_id="org/model", engine=InferenceEngine.LLAMA_CPP)

    def test_vllm_rejects_gguf_specification(self) -> None:
        from inference_proxy.huggingface.artifacts import GGUFDownloadSpec

        with pytest.raises(ValidationError, match="only valid for llama_cpp"):
            DownloadRequest(
                repo_id="org/model",
                gguf=GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf"),
            )
