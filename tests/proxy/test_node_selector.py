"""Unit tests for the node selection function.

Tests cover empty registry, single/multiple healthy nodes, filtering
of unhealthy nodes, and all-unhealthy edge case.
"""

from __future__ import annotations

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.proxy.node_selector import select_node


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "http://10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
) -> Node:
    """Create a minimal Node for testing."""
    return Node(node_id=node_id, endpoint=endpoint, status=status)


class TestSelectNode:
    """select_node returns the first healthy node or None."""

    def test_empty_registry_returns_none(self) -> None:
        registry = NodeRegistry()

        result = select_node(registry)

        assert result is None

    def test_single_healthy_node_returns_it(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)

        result = select_node(registry)

        assert result is not None
        assert result.node_id == "node-1"

    def test_multiple_healthy_nodes_returns_first(self) -> None:
        registry = NodeRegistry()
        registry.add(_make_node("node-1", "http://10.0.1.100:8000"))
        registry.add(_make_node("node-2", "http://10.0.1.200:8000"))

        result = select_node(registry)

        assert result is not None
        assert result.node_id in {"node-1", "node-2"}

    def test_skips_unhealthy_nodes(self) -> None:
        registry = NodeRegistry()
        registry.add(
            _make_node("unhealthy-1", "http://10.0.1.100:8000", NodeStatus.UNHEALTHY)
        )
        registry.add(
            _make_node("healthy-1", "http://10.0.1.200:8000", NodeStatus.HEALTHY)
        )

        result = select_node(registry)

        assert result is not None
        assert result.node_id == "healthy-1"

    def test_all_unhealthy_returns_none(self) -> None:
        registry = NodeRegistry()
        registry.add(
            _make_node("node-1", "http://10.0.1.100:8000", NodeStatus.UNHEALTHY)
        )
        registry.add(
            _make_node("node-2", "http://10.0.1.200:8000", NodeStatus.DRAINING)
        )
        registry.add(_make_node("node-3", "http://10.0.1.300:8000", NodeStatus.UNKNOWN))

        result = select_node(registry)

        assert result is None
