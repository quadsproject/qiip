"""Atomic cleanup shared by request finalization and periodic health cycles."""

from __future__ import annotations

import structlog

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import NodeStatus
from inference_proxy.routing.connection_tracker import ConnectionTracker

logger = structlog.get_logger()


def sweep_drained_nodes(
    registry: NodeRegistry,
    tracker: ConnectionTracker,
) -> tuple[str, ...]:
    """Remove zero-connection DRAINING nodes under one coordinated lock.

    Both callers use this exact path so the status/count check and removal
    cannot drift back into the observe-then-remove race closed by PR 4.
    """
    removed: list[str] = []
    with registry.locked():
        for node in registry.get_all():
            if node.status == NodeStatus.DRAINING and tracker.get(node.node_id) == 0:
                registry.remove(node.node_id)
                tracker.remove(node.node_id)
                removed.append(node.node_id)

    for node_id in removed:
        logger.info("drained node removed", node_id=node_id)
    return tuple(removed)
