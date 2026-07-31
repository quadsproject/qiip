"""Model-aware least-connections node selection strategy.

Selects the healthiest vLLM node with the fewest active connections,
optionally filtering by model name.  This replaces the simple
"first healthy node" selector from Phase 3.

Per D-03: Ties among nodes with equal connection counts break randomly.
Per D-05: Model filtering uses exact string match only.
Per D-07: NodeSelector is a strategy class injected via FastAPI Depends.
Per D-08: Injected into route handlers following existing DI patterns.
Per D-09: ``model=None`` selects among all healthy nodes.
"""

from __future__ import annotations

import random
import threading

import structlog

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.routing.connection_tracker import ConnectionTracker

logger = structlog.get_logger()


class NodeReservation:
    """A selected node with exactly one active-connection reservation."""

    def __init__(self, selector: NodeSelector, node: Node) -> None:
        self.node = node
        self._selector = selector
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Release the reservation once, even from competing finalizers."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._selector._release(self.node.node_id)


class NodeSelector:
    """Least-connections node selector with model-aware filtering.

    Constructor takes ``registry`` and ``tracker`` via dependency
    injection (D-07).  Reservations own counter changes for request
    routing; the ``tracker`` property remains available for status
    reporting and drain coordination.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        tracker: ConnectionTracker,
    ) -> None:
        self._registry = registry
        self._tracker = tracker

    @property
    def tracker(self) -> ConnectionTracker:
        """Return the connection tracker for status and drain consumers."""
        return self._tracker

    def select(
        self,
        model: str | None = None,
        exclude_node_ids: set[str] | None = None,
    ) -> Node | None:
        """Select the optimal node for a request.

        Args:
            model: If provided, only consider nodes serving this exact
                model name (D-05).  If ``None``, consider all healthy
                nodes regardless of model (D-09).
            exclude_node_ids: If provided, skip nodes whose ``node_id``
                is in this set.  Used by retry logic to avoid re-selecting
                a node that already failed for the current request.

        Returns:
            The healthy ``Node`` with the fewest active connections,
            or ``None`` if no suitable nodes are available.
        """
        with self._registry.locked():
            return self._select_locked(model, exclude_node_ids)

    def select_and_reserve(
        self,
        model: str | None = None,
        exclude_node_ids: set[str] | None = None,
    ) -> NodeReservation | None:
        """Atomically select a healthy node and reserve one connection.

        Selection and the counter increment share the registry lock with
        status transitions such as ``drain()``.  A drain therefore either
        wins before selection, excluding the node, or observes the completed
        reservation and waits for it.
        """
        with self._registry.locked():
            node = self._select_locked(model, exclude_node_ids)
            if node is None:
                return None
            self._tracker.increment(node.node_id)
            return NodeReservation(self, node)

    def _select_locked(
        self,
        model: str | None,
        exclude_node_ids: set[str] | None,
    ) -> Node | None:
        """Select a node while the caller holds the registry lock."""
        nodes = self._registry.get_all()

        # Filter to HEALTHY nodes only -- skip DRAINING, UNHEALTHY, UNKNOWN
        healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]

        if not healthy:
            logger.warning("no healthy nodes available", total_nodes=len(nodes))
            return None

        # Exclude specific nodes (used for retry failover)
        if exclude_node_ids:
            healthy = [n for n in healthy if n.node_id not in exclude_node_ids]
            if not healthy:
                logger.debug(
                    "all healthy nodes excluded",
                    excluded=len(exclude_node_ids),
                )
                return None

        # Apply model filter if specified (D-05: exact string match)
        if model is not None:
            healthy = [n for n in healthy if n.model == model]
            if not healthy:
                logger.debug(
                    "no healthy nodes for model",
                    model=model,
                    total_nodes=len(nodes),
                )
                return None

        # Sort by active connection count (ascending)
        healthy.sort(key=lambda n: self._tracker.get(n.node_id))

        # Collect all nodes tied at the minimum connection count
        min_connections = self._tracker.get(healthy[0].node_id)
        tied = [n for n in healthy if self._tracker.get(n.node_id) == min_connections]

        # Random tie-break (D-03)
        selected = random.choice(tied)

        logger.debug(
            "selected node",
            node_id=selected.node_id,
            endpoint=selected.endpoint,
            model=selected.model,
            connections=self._tracker.get(selected.node_id),
            healthy_count=len(healthy),
            tied_count=len(tied),
        )
        return selected

    def _release(self, node_id: str) -> None:
        """Release a connection and atomically remove fully drained nodes."""
        removed: list[str] = []
        with self._registry.locked():
            self._tracker.decrement(node_id)
            for node in self._registry.get_all():
                if (
                    node.status == NodeStatus.DRAINING
                    and self._tracker.get(node.node_id) == 0
                ):
                    self._registry.remove(node.node_id)
                    self._tracker.remove(node.node_id)
                    removed.append(node.node_id)

        for removed_node_id in removed:
            logger.info("drained node removed", node_id=removed_node_id)

    def has_model(self, model: str) -> bool:
        """Check whether any registered node serves the given model.

        Considers all nodes regardless of status -- a DRAINING or
        UNHEALTHY node still counts as "model exists" for the purpose
        of distinguishing 404 (model not found) from 503 (model
        temporarily unavailable).

        Args:
            model: The model name to check for.

        Returns:
            ``True`` if at least one node serves the model.
        """
        nodes = self._registry.get_all()
        return any(n.model == model for n in nodes)
