"""Unit tests for the thread-safe NodeRegistry.

Tests cover add, upsert, remove, get, get_all operations, copy-on-read
semantics, and concurrent thread safety.
"""

from __future__ import annotations

import threading

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus


def _make_node(
    node_id: str = "node-1", endpoint: str = "http://10.0.1.100:8000"
) -> Node:
    """Create a minimal Node for testing."""
    return Node(node_id=node_id, endpoint=endpoint)


class TestRegistryAdd:
    """add() stores a node retrievable by get(node_id)."""

    def test_add_stores_node(self) -> None:
        registry = NodeRegistry()
        node = _make_node()

        registry.add(node)

        assert registry.get("node-1") is not None
        assert registry.get("node-1") == node


class TestRegistryAddUpsert:
    """add() with same node_id replaces existing node (upsert behavior)."""

    def test_add_replaces_existing(self) -> None:
        registry = NodeRegistry()
        original = _make_node(endpoint="http://10.0.1.100:8000")
        updated = _make_node(endpoint="http://10.0.1.200:9000")

        registry.add(original)
        registry.add(updated)

        result = registry.get("node-1")
        assert result is not None
        assert result.endpoint == "http://10.0.1.200:9000"


class TestRegistryRemove:
    """remove() removes an existing node; get() returns None after removal."""

    def test_remove_existing_node(self) -> None:
        registry = NodeRegistry()
        registry.add(_make_node())

        registry.remove("node-1")

        assert registry.get("node-1") is None


class TestRegistryRemoveNonExistent:
    """remove() on non-existent node_id does not raise (silent no-op)."""

    def test_remove_nonexistent_is_silent(self) -> None:
        registry = NodeRegistry()

        registry.remove("nonexistent")  # should not raise


class TestRegistryGetUnknown:
    """get() returns None for unknown node_id."""

    def test_get_unknown_returns_none(self) -> None:
        registry = NodeRegistry()

        assert registry.get("unknown") is None


class TestRegistryGetAll:
    """get_all() returns list of all stored nodes (empty list when empty)."""

    def test_get_all_empty(self) -> None:
        registry = NodeRegistry()

        assert registry.get_all() == []

    def test_get_all_returns_all_nodes(self) -> None:
        registry = NodeRegistry()
        node1 = _make_node("node-1", "http://10.0.1.100:8000")
        node2 = _make_node("node-2", "http://10.0.1.200:8000")

        registry.add(node1)
        registry.add(node2)

        result = registry.get_all()
        assert len(result) == 2
        ids = {n.node_id for n in result}
        assert ids == {"node-1", "node-2"}


class TestRegistryGetAllReturnsCopy:
    """get_all() returns a copy -- mutating returned list does not affect internal state."""

    def test_mutation_does_not_affect_registry(self) -> None:
        registry = NodeRegistry()
        registry.add(_make_node())

        result = registry.get_all()
        result.clear()

        assert len(registry.get_all()) == 1


class TestDrain:
    """drain() sets a node's status to DRAINING."""

    def test_drain_existing_node_returns_true(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.HEALTHY,
            )
        )

        result = registry.drain("node-1")

        assert result is True

    def test_drain_sets_status_to_draining(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.HEALTHY,
            )
        )

        registry.drain("node-1")

        node = registry.get("node-1")
        assert node is not None
        assert node.status == NodeStatus.DRAINING

    def test_drain_nonexistent_returns_false(self) -> None:
        registry = NodeRegistry()

        result = registry.drain("nonexistent")

        assert result is False

    def test_drain_preserves_other_fields(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.HEALTHY,
                model="llama-3",
            )
        )

        registry.drain("node-1")

        node = registry.get("node-1")
        assert node is not None
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.model == "llama-3"


class TestRegistryConcurrentAccess:
    """Concurrent add/remove from multiple threads does not corrupt state."""

    def test_thread_safety(self) -> None:
        registry = NodeRegistry()
        barrier = threading.Barrier(10)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                node_id = f"node-{thread_id}"
                node = _make_node(node_id, f"http://10.0.1.{thread_id}:8000")
                registry.add(node)
                registry.get(node_id)
                registry.get_all()
                registry.remove(node_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # After all threads remove their nodes, registry should be empty
        assert registry.get_all() == []
