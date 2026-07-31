"""Unit tests for the NodeSelector strategy class.

Tests cover empty registry, single/multiple healthy nodes, least-connections
selection, random tie-breaking, model-aware filtering, status filtering,
and the has_model helper method.
"""

from __future__ import annotations

from unittest.mock import patch

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "http://10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
) -> Node:
    """Create a minimal Node for testing."""
    return Node(node_id=node_id, endpoint=endpoint, status=status, model=model)


def _make_selector(
    nodes: list[Node] | None = None,
) -> tuple[NodeSelector, NodeRegistry, ConnectionTracker]:
    """Create a NodeSelector with a registry pre-populated with nodes."""
    registry = NodeRegistry()
    tracker = ConnectionTracker()
    if nodes:
        for node in nodes:
            registry.add(node)
    selector = NodeSelector(registry=registry, tracker=tracker)
    return selector, registry, tracker


class TestSelectEmpty:
    """select() returns None when the registry is empty."""

    def test_empty_registry_returns_none(self) -> None:
        selector, _, _ = _make_selector()

        result = selector.select()

        assert result is None


class TestSelectSingleNode:
    """select() returns the single healthy node."""

    def test_single_healthy_node_returns_it(self) -> None:
        node = _make_node()
        selector, _, _ = _make_selector([node])

        result = selector.select()

        assert result is not None
        assert result.node_id == "node-1"


class TestSelectLeastConnections:
    """select() returns the node with fewest active connections (LBAL-01)."""

    def test_selects_node_with_fewer_connections(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        node_b = _make_node("node-b", "http://10.0.1.200:8000")
        selector, _, tracker = _make_selector([node_a, node_b])
        tracker.increment("node-a")
        tracker.increment("node-a")
        tracker.increment("node-b")

        result = selector.select()

        assert result is not None
        assert result.node_id == "node-b"


class TestSelectTieBreaking:
    """select() breaks ties randomly among nodes with equal connection counts (D-03)."""

    def test_tie_break_uses_random_choice_for_all_tied_nodes(self) -> None:
        nodes = [
            _make_node("node-a", "http://10.0.1.100:8000"),
            _make_node("node-b", "http://10.0.1.200:8000"),
            _make_node("node-c", "http://10.0.1.300:8000"),
        ]
        selector, _, _ = _make_selector(nodes)

        with patch(
            "inference_proxy.routing.node_selector.random.choice",
            return_value=nodes[1],
        ) as choose:
            result = selector.select()

        tied = choose.call_args.args[0]
        assert [node.node_id for node in tied] == ["node-a", "node-b", "node-c"]
        assert result is nodes[1]


class TestSelectModelFiltering:
    """select() filters nodes by model name (DISC-03, D-05)."""

    def test_model_filter_returns_matching_node(self) -> None:
        node_llama = _make_node("node-llama", model="llama-3")
        node_gpt = _make_node(
            "node-gpt", endpoint="http://10.0.1.200:8000", model="gpt-4"
        )
        selector, _, _ = _make_selector([node_llama, node_gpt])

        result = selector.select(model="llama-3")

        assert result is not None
        assert result.node_id == "node-llama"

    def test_nonexistent_model_returns_none(self) -> None:
        node = _make_node(model="llama-3")
        selector, _, _ = _make_selector([node])

        result = selector.select(model="nonexistent")

        assert result is None

    def test_model_none_considers_all_healthy_nodes(self) -> None:
        node_llama = _make_node("node-llama", model="llama-3")
        node_gpt = _make_node(
            "node-gpt", endpoint="http://10.0.1.200:8000", model="gpt-4"
        )
        selector, _, _ = _make_selector([node_llama, node_gpt])

        result = selector.select(model=None)

        assert result is not None
        assert result.node_id in {"node-llama", "node-gpt"}


class TestSelectSkipsDraining:
    """select() skips nodes with DRAINING status."""

    def test_draining_node_is_skipped(self) -> None:
        draining = _make_node("draining-1", status=NodeStatus.DRAINING)
        healthy = _make_node("healthy-1", endpoint="http://10.0.1.200:8000")
        selector, _, _ = _make_selector([draining, healthy])

        result = selector.select()

        assert result is not None
        assert result.node_id == "healthy-1"


class TestSelectSkipsUnhealthy:
    """select() skips nodes with UNHEALTHY and UNKNOWN status."""

    def test_unhealthy_node_is_skipped(self) -> None:
        unhealthy = _make_node("unhealthy-1", status=NodeStatus.UNHEALTHY)
        healthy = _make_node("healthy-1", endpoint="http://10.0.1.200:8000")
        selector, _, _ = _make_selector([unhealthy, healthy])

        result = selector.select()

        assert result is not None
        assert result.node_id == "healthy-1"

    def test_unknown_node_is_skipped(self) -> None:
        unknown = _make_node("unknown-1", status=NodeStatus.UNKNOWN)
        healthy = _make_node("healthy-1", endpoint="http://10.0.1.200:8000")
        selector, _, _ = _make_selector([unknown, healthy])

        result = selector.select()

        assert result is not None
        assert result.node_id == "healthy-1"


class TestSelectExcludeNodeIds:
    """select() supports excluding specific node_ids from selection."""

    def test_exclude_single_node(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        node_b = _make_node("node-b", "http://10.0.1.200:8000")
        selector, _, _ = _make_selector([node_a, node_b])

        result = selector.select(exclude_node_ids={"node-a"})

        assert result is not None
        assert result.node_id == "node-b"

    def test_exclude_all_nodes_returns_none(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        node_b = _make_node("node-b", "http://10.0.1.200:8000")
        selector, _, _ = _make_selector([node_a, node_b])

        result = selector.select(exclude_node_ids={"node-a", "node-b"})

        assert result is None

    def test_exclude_none_no_filtering(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        selector, _, _ = _make_selector([node_a])

        result = selector.select(exclude_node_ids=None)

        assert result is not None
        assert result.node_id == "node-a"

    def test_exclude_empty_set_no_filtering(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        selector, _, _ = _make_selector([node_a])

        result = selector.select(exclude_node_ids=set())

        assert result is not None
        assert result.node_id == "node-a"

    def test_exclude_with_model_filter(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000", model="llama-3")
        node_b = _make_node("node-b", "http://10.0.1.200:8000", model="llama-3")
        selector, _, _ = _make_selector([node_a, node_b])

        result = selector.select(model="llama-3", exclude_node_ids={"node-a"})

        assert result is not None
        assert result.node_id == "node-b"

    def test_exclude_nonexistent_node_id_has_no_effect(self) -> None:
        node_a = _make_node("node-a", "http://10.0.1.100:8000")
        selector, _, _ = _make_selector([node_a])

        result = selector.select(exclude_node_ids={"nonexistent"})

        assert result is not None
        assert result.node_id == "node-a"


class TestHasModel:
    """has_model() checks if any node (any status) serves the given model."""

    def test_has_model_returns_true_when_model_exists(self) -> None:
        node = _make_node(model="llama-3")
        selector, _, _ = _make_selector([node])

        assert selector.has_model("llama-3") is True

    def test_has_model_returns_false_when_model_absent(self) -> None:
        node = _make_node(model="llama-3")
        selector, _, _ = _make_selector([node])

        assert selector.has_model("nonexistent") is False

    def test_has_model_includes_unhealthy_nodes(self) -> None:
        node = _make_node(model="llama-3", status=NodeStatus.UNHEALTHY)
        selector, _, _ = _make_selector([node])

        assert selector.has_model("llama-3") is True
