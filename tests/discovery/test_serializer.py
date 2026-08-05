"""Unit tests for the etcd-to-Node serializer.

Tests cover valid JSON parsing, malformed JSON handling, empty input,
missing required fields, bytes/str key handling, and roundtrip consistency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

from structlog.testing import capture_logs

from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy
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


_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=[],
    allowed_networks=["10.0.1.0/24"],
    allowed_ports=[8000],
)


class TestNodeFromEtcdValidFullJson:
    """node_from_etcd parses valid JSON with all fields into a Node."""

    def test_parses_all_fields(self) -> None:
        key = "/nodes/node-abc"
        value = json.dumps(
            {
                "endpoint": "http://10.0.1.100:8000",
                "status": "healthy",
                "model": "llama-2-7b",
                "last_heartbeat": "2025-09-12T10:30:00Z",
                "capabilities": {"max_tokens": 4096, "gpu_memory": "24GB"},
                "active_connections": 3,
            }
        ).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix, endpoint_policy=_ENDPOINT_POLICY)

        assert node is not None
        assert node.node_id == "node-abc"
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.status == NodeStatus.HEALTHY
        assert node.model == "llama-2-7b"
        assert node.last_heartbeat is not None
        assert node.capabilities.max_tokens == 4096
        assert node.capabilities.gpu_memory == "24GB"
        assert node.active_connections == 3
        assert node.artifact_id is None


class TestNodeFromEtcdMinimalJson:
    """node_from_etcd parses minimal JSON (only endpoint) with defaults."""

    def test_parses_minimal_with_defaults(self) -> None:
        key = "/nodes/node-min"
        value = json.dumps({"endpoint": "http://10.0.1.200:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix, endpoint_policy=_ENDPOINT_POLICY)

        assert node is not None
        assert node.node_id == "node-min"
        assert node.endpoint == "http://10.0.1.200:8000"
        assert node.status == NodeStatus.UNKNOWN
        assert node.model == ""
        assert node.last_heartbeat is None
        assert node.capabilities.max_tokens == 4096
        assert node.capabilities.gpu_memory == ""
        assert node.active_connections == 0
        assert node.managed is False


class TestNodeFromEtcdOwnership:
    """Ownership is opt-in for external etcd registrations."""

    def test_etcd_node_without_managed_defaults_unmanaged(self) -> None:
        value = json.dumps({"endpoint": "10.0.1.100:8000"}).encode()

        node = node_from_etcd(
            "/nodes/external",
            value,
            "/nodes/",
            endpoint_policy=_ENDPOINT_POLICY,
        )

        assert node is not None
        assert node.managed is False

    def test_etcd_explicit_managed_value_is_preserved(self) -> None:
        for managed in (True, False):
            original = Node(
                node_id=f"managed-{managed}",
                endpoint="10.0.1.100:8000",
                managed=managed,
            )

            key, value = node_to_etcd(original, "/nodes/")
            restored = node_from_etcd(
                key,
                value,
                "/nodes/",
                endpoint_policy=_ENDPOINT_POLICY,
            )

            assert restored is not None
            assert restored.managed is managed


def test_node_from_etcd_normalizes_schemeless_endpoint() -> None:
    value = json.dumps({"endpoint": "10.0.1.200:8000"}).encode()

    node = node_from_etcd(
        "/nodes/node-min",
        value,
        "/nodes/",
        endpoint_policy=_ENDPOINT_POLICY,
    )

    assert node is not None
    assert node.endpoint == "http://10.0.1.200:8000"


def test_node_from_etcd_rejects_disallowed_endpoint_with_reason() -> None:
    value = json.dumps({"endpoint": "169.254.169.254:8000"}).encode()

    with capture_logs() as logs:
        node = node_from_etcd(
            "/nodes/metadata",
            value,
            "/nodes/",
            endpoint_policy=_ENDPOINT_POLICY,
        )

    assert node is None
    assert logs == [
        {
            "event": "skipping malformed node",
            "key": "/nodes/metadata",
            "endpoint": "169.254.169.254:8000",
            "error": ("backend endpoint host is not allowed: '169.254.169.254:8000'"),
            "log_level": "warning",
        }
    ]


class TestNodeFromEtcdMalformedJson:
    """node_from_etcd returns None and logs warning for malformed JSON."""

    def test_returns_none_for_malformed_json(self) -> None:
        key = "/nodes/bad-node"
        value = b"not valid json {{"
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(
                key, value, prefix, endpoint_policy=_ENDPOINT_POLICY
            )

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdEmptyBytes:
    """node_from_etcd returns None and logs warning for empty bytes."""

    def test_returns_none_for_empty_bytes(self) -> None:
        key = "/nodes/empty-node"
        value = b""
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(
                key, value, prefix, endpoint_policy=_ENDPOINT_POLICY
            )

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdMissingEndpoint:
    """node_from_etcd returns None for JSON missing required field (endpoint)."""

    def test_returns_none_when_endpoint_missing(self) -> None:
        key = "/nodes/no-endpoint"
        value = json.dumps({"model": "llama-2-7b"}).encode("utf-8")
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(
                key, value, prefix, endpoint_policy=_ENDPOINT_POLICY
            )

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdBytesAndStrKey:
    """node_from_etcd handles both bytes and str key input defensively."""

    def test_handles_bytes_key(self) -> None:
        key = b"/nodes/bytes-node"
        value = json.dumps({"endpoint": "http://10.0.1.100:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix, endpoint_policy=_ENDPOINT_POLICY)

        assert node is not None
        assert node.node_id == "bytes-node"

    def test_handles_str_key(self) -> None:
        key = "/nodes/str-node"
        value = json.dumps({"endpoint": "http://10.0.1.100:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix, endpoint_policy=_ENDPOINT_POLICY)

        assert node is not None
        assert node.node_id == "str-node"


class TestNodeToEtcd:
    """node_to_etcd converts a Node to (key_string, json_bytes)."""

    def test_converts_node_to_key_and_json(self) -> None:
        node = Node(
            node_id="node-xyz",
            endpoint="http://10.0.1.100:8000",
            status=NodeStatus.HEALTHY,
            model="llama-2-7b",
        )
        prefix = "/nodes/"

        key, value_bytes = node_to_etcd(node, prefix)

        assert key == "/nodes/node-xyz"
        assert isinstance(value_bytes, bytes)

        data = json.loads(value_bytes)
        assert "node_id" not in data
        assert data["endpoint"] == "http://10.0.1.100:8000"
        assert data["status"] == "healthy"
        assert data["model"] == "llama-2-7b"


class TestNodeToEtcdRoundtrip:
    """node_to_etcd roundtrips with node_from_etcd preserving all fields."""

    def test_roundtrip_preserves_fields(self) -> None:
        now = datetime.now(tz=UTC)
        original = Node(
            node_id="roundtrip-node",
            endpoint="http://10.0.1.200:8000",
            status=NodeStatus.HEALTHY,
            model="meta-llama/Llama-3-70B",
            last_heartbeat=now,
            capabilities=NodeCapabilities(max_tokens=8192, gpu_memory="80GB"),
            active_connections=5,
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="b" * 64,
            llamacpp_runtime=_runtime_state(),
        )
        prefix = "/nodes/"

        key, value_bytes = node_to_etcd(original, prefix)
        restored = node_from_etcd(
            key, value_bytes, prefix, endpoint_policy=_ENDPOINT_POLICY
        )

        assert restored is not None
        assert restored.node_id == original.node_id
        assert restored.endpoint == original.endpoint
        assert restored.status == original.status
        assert restored.model == original.model
        assert restored.last_heartbeat == original.last_heartbeat
        assert restored.capabilities.max_tokens == original.capabilities.max_tokens
        assert restored.capabilities.gpu_memory == original.capabilities.gpu_memory
        assert restored.active_connections == original.active_connections
        assert restored.engine is InferenceEngine.LLAMA_CPP
        assert restored.artifact_id == "b" * 64
        assert restored.llamacpp_runtime == _runtime_state()


def test_endpoint_roundtrip_preserves_canonical_form() -> None:
    original = Node(node_id="node-1", endpoint="10.0.1.100:8000")

    key, value = node_to_etcd(original, "/nodes/")
    restored = node_from_etcd(
        key,
        value,
        "/nodes/",
        endpoint_policy=_ENDPOINT_POLICY,
    )

    assert json.loads(value)["endpoint"] == "http://10.0.1.100:8000"
    assert restored is not None
    assert restored.endpoint == "http://10.0.1.100:8000"
