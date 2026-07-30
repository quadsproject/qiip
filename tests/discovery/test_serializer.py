"""Unit tests for the etcd-to-Node serializer.

Tests cover valid JSON parsing, malformed JSON handling, empty input,
missing required fields, bytes/str key handling, and roundtrip consistency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.models.node import Node, NodeCapabilities, NodeStatus


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

        node = node_from_etcd(key, value, prefix)

        assert node is not None
        assert node.node_id == "node-abc"
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.status == NodeStatus.HEALTHY
        assert node.model == "llama-2-7b"
        assert node.last_heartbeat is not None
        assert node.capabilities.max_tokens == 4096
        assert node.capabilities.gpu_memory == "24GB"
        assert node.active_connections == 3


class TestNodeFromEtcdMinimalJson:
    """node_from_etcd parses minimal JSON (only endpoint) with defaults."""

    def test_parses_minimal_with_defaults(self) -> None:
        key = "/nodes/node-min"
        value = json.dumps({"endpoint": "http://10.0.1.200:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix)

        assert node is not None
        assert node.node_id == "node-min"
        assert node.endpoint == "http://10.0.1.200:8000"
        assert node.status == NodeStatus.UNKNOWN
        assert node.model == ""
        assert node.last_heartbeat is None
        assert node.capabilities.max_tokens == 4096
        assert node.capabilities.gpu_memory == ""
        assert node.active_connections == 0


class TestNodeFromEtcdMalformedJson:
    """node_from_etcd returns None and logs warning for malformed JSON."""

    def test_returns_none_for_malformed_json(self) -> None:
        key = "/nodes/bad-node"
        value = b"not valid json {{"
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(key, value, prefix)

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdEmptyBytes:
    """node_from_etcd returns None and logs warning for empty bytes."""

    def test_returns_none_for_empty_bytes(self) -> None:
        key = "/nodes/empty-node"
        value = b""
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(key, value, prefix)

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdMissingEndpoint:
    """node_from_etcd returns None for JSON missing required field (endpoint)."""

    def test_returns_none_when_endpoint_missing(self) -> None:
        key = "/nodes/no-endpoint"
        value = json.dumps({"model": "llama-2-7b"}).encode("utf-8")
        prefix = "/nodes/"

        with patch("inference_proxy.discovery.serializer.logger") as mock_logger:
            result = node_from_etcd(key, value, prefix)

        assert result is None
        mock_logger.warning.assert_called_once()


class TestNodeFromEtcdBytesAndStrKey:
    """node_from_etcd handles both bytes and str key input defensively."""

    def test_handles_bytes_key(self) -> None:
        key = b"/nodes/bytes-node"  # type: ignore[assignment]
        value = json.dumps({"endpoint": "http://10.0.1.100:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix)  # type: ignore[arg-type]

        assert node is not None
        assert node.node_id == "bytes-node"

    def test_handles_str_key(self) -> None:
        key = "/nodes/str-node"
        value = json.dumps({"endpoint": "http://10.0.1.100:8000"}).encode("utf-8")
        prefix = "/nodes/"

        node = node_from_etcd(key, value, prefix)

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
        )
        prefix = "/nodes/"

        key, value_bytes = node_to_etcd(original, prefix)
        restored = node_from_etcd(key, value_bytes, prefix)

        assert restored is not None
        assert restored.node_id == original.node_id
        assert restored.endpoint == original.endpoint
        assert restored.status == original.status
        assert restored.model == original.model
        assert restored.last_heartbeat == original.last_heartbeat
        assert restored.capabilities.max_tokens == original.capabilities.max_tokens
        assert restored.capabilities.gpu_memory == original.capabilities.gpu_memory
        assert restored.active_connections == original.active_connections
