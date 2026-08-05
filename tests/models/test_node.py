"""Unit tests for the Node state domain model."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppCacheType,
    LlamaCppFlashAttention,
    LlamaCppGPUState,
    LlamaCppRuntimeEffective,
    LlamaCppRuntimeRequest,
    LlamaCppRuntimeState,
    LlamaCppSizingMode,
    Node,
    NodeCapabilities,
    NodeStatus,
)


def _runtime_state() -> LlamaCppRuntimeState:
    return LlamaCppRuntimeState(
        requested=LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.AUTO,
            fit_target_mib=512,
        ),
        effective=LlamaCppRuntimeEffective(
            train_context=262144,
            context_per_slot=12544,
            slot_context_limit=12544,
            slots=1,
            aggregate_context=12544,
            cache_type_k=LlamaCppCacheType.Q8_0,
            cache_type_v=LlamaCppCacheType.Q8_0,
            flash_attn=LlamaCppFlashAttention.ON,
            kv_unified=True,
            gpu_layers=31,
            total_layers=31,
        ),
        gpus=(
            LlamaCppGPUState(
                index=0,
                total_mib=14911,
                used_mib=14089,
                free_mib=822,
            ),
        ),
        observed_at=datetime(2026, 8, 5, 20, 54, 7, tzinfo=UTC),
    )


class TestNodeStatusEnumValues:
    def test_node_status_enum_values(self) -> None:
        assert NodeStatus.HEALTHY.value == "healthy"
        assert NodeStatus.UNHEALTHY.value == "unhealthy"
        assert NodeStatus.DRAINING.value == "draining"
        assert NodeStatus.UNKNOWN.value == "unknown"
        assert NodeStatus.PROVISIONING.value == "provisioning"
        assert NodeStatus.FAILED.value == "failed"
        assert len(NodeStatus) == 6


class TestNodeStatusIsStrEnum:
    def test_node_status_is_str_enum(self) -> None:
        assert isinstance(NodeStatus.HEALTHY, str)
        assert isinstance(NodeStatus.UNHEALTHY, str)
        assert isinstance(NodeStatus.DRAINING, str)
        assert isinstance(NodeStatus.UNKNOWN, str)


class TestNodeMinimalCreation:
    def test_node_minimal_creation(self) -> None:
        node = Node(node_id="node-1", endpoint="http://10.0.1.100:8000")
        assert node.node_id == "node-1"
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.status == NodeStatus.UNKNOWN
        assert node.model == ""
        assert node.last_heartbeat is None
        assert node.active_connections == 0
        assert node.managed is False
        assert node.engine is InferenceEngine.VLLM
        assert node.artifact_id is None
        assert node.llamacpp_runtime is None


class TestNodeFullCreation:
    def test_node_full_creation(self) -> None:
        now = datetime.now(tz=UTC)
        node = Node(
            node_id="gpu-node-42",
            endpoint="http://10.0.1.200:8000",
            status=NodeStatus.HEALTHY,
            model="meta-llama/Llama-3-70B",
            last_heartbeat=now,
            capabilities=NodeCapabilities(max_tokens=8192, gpu_memory="80GB"),
            active_connections=5,
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="a" * 64,
            llamacpp_runtime=_runtime_state(),
        )
        dumped = node.model_dump()
        roundtripped = Node.model_validate(dumped)
        assert roundtripped.node_id == node.node_id
        assert roundtripped.endpoint == node.endpoint
        assert roundtripped.status == node.status
        assert roundtripped.model == node.model
        assert roundtripped.last_heartbeat == node.last_heartbeat
        assert roundtripped.capabilities.max_tokens == 8192
        assert roundtripped.capabilities.gpu_memory == "80GB"
        assert roundtripped.active_connections == 5
        assert roundtripped.engine is InferenceEngine.LLAMA_CPP
        assert roundtripped.artifact_id == "a" * 64
        assert roundtripped.llamacpp_runtime == _runtime_state()

    def test_rejects_non_sha256_artifact_id(self) -> None:
        with pytest.raises(ValidationError):
            Node(node_id="node-1", endpoint="host:8000", artifact_id="latest")

    def test_runtime_observation_requires_timezone(self) -> None:
        payload = _runtime_state().model_dump()
        payload["observed_at"] = datetime(2026, 8, 5, 20, 54, 7)

        with pytest.raises(ValidationError, match="must include a timezone"):
            LlamaCppRuntimeState.model_validate(payload)


class TestNodeCapabilitiesDefaults:
    def test_node_capabilities_defaults(self) -> None:
        caps = NodeCapabilities()
        assert caps.max_tokens == 4096
        assert caps.gpu_memory == ""


class TestNodeDefaultCapabilities:
    def test_node_default_capabilities(self) -> None:
        node = Node(node_id="node-1", endpoint="http://10.0.1.100:8000")
        assert isinstance(node.capabilities, NodeCapabilities)
        assert node.capabilities.max_tokens == 4096


class TestNodeModelIsStr:
    def test_node_model_is_str(self) -> None:
        node = Node(node_id="node-1", endpoint="http://10.0.1.100:8000")
        assert isinstance(node.model, str)


class TestNodeRejectsInvalidStatus:
    def test_node_rejects_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            Node(node_id="x", endpoint="y", status="invalid")
