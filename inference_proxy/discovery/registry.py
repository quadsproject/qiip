"""Thread-safe in-memory registry of discovered vLLM nodes.

Provides add/remove/get/get_all operations protected by a
``threading.RLock``.  The lock is required because the watch thread
(an OS thread, not a coroutine) mutates the registry while async
handlers read from it.

Per D-06: Nodes held in a ``dict[str, Node]`` protected by a lock.
Per D-08: Thread-safe methods ``add``, ``remove``, ``get``, ``get_all``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress

import structlog

from inference_proxy.models.node import Node, NodeStatus

logger = structlog.get_logger()

_LIVENESS_STATUSES = {
    NodeStatus.HEALTHY,
    NodeStatus.UNHEALTHY,
    NodeStatus.UNKNOWN,
}

RemoveListener = Callable[[str], None]


def node_with_status(
    node: Node,
    status: NodeStatus,
    **changes: object,
) -> Node:
    """Return one node transition while preserving all unspecified fields."""
    changes["status"] = status
    return node.model_copy(update=changes)


class NodeRegistry:
    """Thread-safe registry of discovered vLLM nodes.

    All public methods acquire ``self._lock`` before accessing the
    internal dictionary.  ``get_all`` returns a shallow copy so that
    callers cannot mutate internal state.  Coordinated operations may
    hold ``locked()`` while calling other registry methods, so the lock
    is re-entrant.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._remove_listeners: list[RemoveListener] = []
        self._lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the registry coordination lock across related operations.

        Callers use this when an operation must remain atomic with registry
        mutations such as ``drain()``.  The lock is re-entrant, so ordinary
        registry methods remain safe to call inside this context.
        """
        with self._lock:
            yield

    def add(self, node: Node) -> None:
        """Store or replace a node by its ``node_id``."""
        with self._lock:
            self._nodes[node.node_id] = node

    def register_remove_listener(
        self,
        listener: RemoveListener,
    ) -> Callable[[], None]:
        """Run *listener* whenever a registered node is removed.

        Listeners run synchronously while the registry coordination lock is
        held. This orders per-node cleanup before another thread can register
        the same node ID. The resulting global lock order is registry lock,
        then each listener's own state lock; listeners must be non-blocking and
        must not acquire unrelated locks, mutate the registry, or raise.

        Returns a callback that unregisters the listener. Calling it more than
        once is harmless.
        """
        with self._lock:
            self._remove_listeners.append(listener)

        def unregister() -> None:
            with self._lock, suppress(ValueError):
                self._remove_listeners.remove(listener)

        return unregister

    def add_discovered(self, node: Node) -> None:
        """Merge an etcd node without overwriting a local liveness decision.

        HEALTHY, UNHEALTHY, and UNKNOWN are locally probed liveness states.
        When both the current and discovered state are in that set, retain the
        current state while refreshing etcd-owned fields. Lifecycle
        transitions involving PROVISIONING, FAILED, DRAINING, RELAUNCHING, or
        RELAUNCH_FAILED remain authoritative so lifecycle operations can
        complete and fresh registrations can replace drained nodes.
        """
        with self._lock:
            current = self._nodes.get(node.node_id)
            if (
                current is not None
                and current.status in _LIVENESS_STATUSES
                and node.status in _LIVENESS_STATUSES
            ):
                node = node.model_copy(update={"status": current.status})
            self._nodes[node.node_id] = node

    def reconcile_discovered(
        self,
        nodes: dict[str, Node],
        present_node_ids: set[str],
    ) -> set[str]:
        """Converge etcd-owned node data while draining keys absent in etcd.

        ``present_node_ids`` includes malformed entries that could not be
        deserialized. Such entries are left untouched instead of being
        mistaken for deletions. Missing nodes transition to DRAINING rather
        than being removed so existing request reservations remain valid.
        Returns the missing node IDs while still under the same atomic view.
        """
        with self._lock:
            for node in nodes.values():
                self.add_discovered(node)
            missing_node_ids = self._nodes.keys() - present_node_ids
            for node_id in missing_node_ids:
                self.drain(node_id)
            return missing_node_ids

    def remove(self, node_id: str) -> None:
        """Remove a node by its ``node_id``.  No-op if absent."""
        with self._lock:
            removed = self._nodes.pop(node_id, None)
            if removed is None:
                return
            for listener in tuple(self._remove_listeners):
                try:
                    listener(node_id)
                except Exception:
                    logger.warning(
                        "node removal listener failed",
                        node_id=node_id,
                        listener=repr(listener),
                        exc_info=True,
                    )

    def get(self, node_id: str) -> Node | None:
        """Return the node with the given ``node_id``, or ``None``."""
        with self._lock:
            return self._nodes.get(node_id)

    def update_status(
        self,
        node_id: str,
        new: NodeStatus,
        allowed_from: set[NodeStatus],
    ) -> bool:
        """Conditionally update a registered node's status.

        The current node is read and replaced while holding the same lock,
        so a stale caller cannot resurrect a removed node, overwrite a
        concurrent status transition, or revert other node fields.

        Returns ``True`` when the current status was allowed and the update
        was applied, otherwise ``False``.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None or node.status not in allowed_from:
                return False
            self._nodes[node_id] = node.model_copy(update={"status": new})
            return True

    def drain(self, node_id: str) -> bool:
        """Mark a node as DRAINING.

        Returns ``True`` if the node was found and transitioned,
        ``False`` if the node was not in the registry.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return False
            self._nodes[node_id] = node.model_copy(
                update={"status": NodeStatus.DRAINING}
            )
            return True

    def get_all(self) -> list[Node]:
        """Return a copy of all registered nodes."""
        with self._lock:
            return list(self._nodes.values())
