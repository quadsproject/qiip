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
from collections.abc import Iterator
from contextlib import contextmanager

from inference_proxy.models.node import Node, NodeStatus


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

    def remove(self, node_id: str) -> None:
        """Remove a node by its ``node_id``.  No-op if absent."""
        with self._lock:
            self._nodes.pop(node_id, None)

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
