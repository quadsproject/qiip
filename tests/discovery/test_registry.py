"""Unit tests for the thread-safe NodeRegistry.

Tests cover add, upsert, remove, get, get_all operations, copy-on-read
semantics, and concurrent thread safety.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import structlog

from inference_proxy.discovery.registry import NodeRegistry, node_with_status
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

    def test_remove_notifies_listener_only_for_existing_node(self) -> None:
        registry = NodeRegistry()
        removed: list[str] = []
        registry.register_remove_listener(removed.append)
        registry.add(_make_node())

        registry.remove("missing")
        registry.remove("node-1")

        assert removed == ["node-1"]

    def test_remove_listener_can_be_unregistered(self) -> None:
        registry = NodeRegistry()
        removed: list[str] = []
        unregister = registry.register_remove_listener(removed.append)
        unregister()
        unregister()
        registry.add(_make_node())

        registry.remove("node-1")

        assert removed == []

    def test_listener_failure_does_not_escape_or_skip_later_listeners(
        self,
    ) -> None:
        registry = NodeRegistry()
        notified: list[str] = []

        def broken_listener(_node_id: str) -> None:
            raise RuntimeError("cleanup failed")

        registry.register_remove_listener(broken_listener)
        registry.register_remove_listener(notified.append)
        registry.add(_make_node())

        with structlog.testing.capture_logs() as captured:
            registry.remove("node-1")

        assert registry.get("node-1") is None
        assert notified == ["node-1"]
        failure = next(
            event
            for event in captured
            if event.get("event") == "node removal listener failed"
        )
        assert failure["node_id"] == "node-1"
        assert failure["log_level"] == "warning"


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


class TestUpdateStatus:
    """update_status() applies conditional transitions under the registry lock."""

    def test_missing_node_returns_false(self) -> None:
        registry = NodeRegistry()

        updated = registry.update_status(
            "missing",
            NodeStatus.UNHEALTHY,
            allowed_from={NodeStatus.HEALTHY},
        )

        assert updated is False

    def test_disallowed_source_status_returns_false(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.DRAINING,
            )
        )

        updated = registry.update_status(
            "node-1",
            NodeStatus.UNHEALTHY,
            allowed_from={NodeStatus.HEALTHY},
        )

        assert updated is False
        node = registry.get("node-1")
        assert node is not None
        assert node.status == NodeStatus.DRAINING

    def test_permitted_transition_returns_true(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.UNKNOWN,
            )
        )

        updated = registry.update_status(
            "node-1",
            NodeStatus.HEALTHY,
            allowed_from={NodeStatus.UNKNOWN},
        )

        assert updated is True
        node = registry.get("node-1")
        assert node is not None
        assert node.status == NodeStatus.HEALTHY

    def test_transition_preserves_non_status_fields(self) -> None:
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.UNHEALTHY,
                model="llama-3",
                managed=False,
            )
        )

        registry.update_status(
            "node-1",
            NodeStatus.HEALTHY,
            allowed_from={NodeStatus.UNHEALTHY},
        )

        node = registry.get("node-1")
        assert node is not None
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.model == "llama-3"
        assert node.managed is False


def test_status_transitions_use_registry_primitive() -> None:
    """No production caller may copy a status and write it back with add()."""
    source_root = Path(__file__).parents[2] / "inference_proxy"
    registry_path = source_root / "discovery" / "registry.py"
    offenders: list[str] = []

    # This deliberately detects literal model_copy(update={"status": ...})
    # transitions. Direct Node(...) construction followed by add() is outside
    # this AST pattern and remains a code-review responsibility.
    for path in source_root.rglob("*.py"):
        if path == registry_path:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not (
                isinstance(call.func, ast.Attribute) and call.func.attr == "model_copy"
            ):
                continue
            for keyword in call.keywords:
                if keyword.arg != "update" or not isinstance(keyword.value, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in keyword.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "status" in keys:
                    relative = path.relative_to(source_root.parent)
                    offenders.append(f"{relative}:{call.lineno}")

    assert offenders == []


def test_node_with_status_preserves_fields_and_applies_owned_changes() -> None:
    node = Node(
        node_id="gpu01",
        endpoint="http://gpu01:8000",
        status=NodeStatus.HEALTHY,
        model="original",
    )

    transitioned = node_with_status(
        node,
        NodeStatus.RELAUNCH_FAILED,
        model="replacement",
    )

    assert transitioned.status is NodeStatus.RELAUNCH_FAILED
    assert transitioned.model == "replacement"
    assert transitioned.endpoint == node.endpoint
    assert node.status is NodeStatus.HEALTHY


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
