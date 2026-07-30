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
    SetupRequest,
)


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
