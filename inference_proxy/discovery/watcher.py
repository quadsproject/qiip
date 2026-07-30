"""Watch thread with reconnection loop for etcd node events.

Runs in a dedicated ``threading.Thread`` (per D-03) started during FastAPI
lifespan startup.  Watches for node additions and removals under the
configured etcd prefix and dispatches events to the ``NodeRegistry``.

**Reconnection necessity** (per D-10): etcd3gw's watcher has **no built-in
reconnection logic**.  When the HTTP stream breaks (network error, etcd
restart, timeout), the events iterator silently terminates.  This module
wraps ``watch_prefix`` in a reconnection loop so the gateway continues
receiving node updates after transient failures.

Usage::

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_watcher,
        args=(etcd_client, registry, stop_event),
        daemon=True,
    )
    thread.start()

    # On shutdown:
    stop_event.set()
    thread.join(timeout=10)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import structlog

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_from_etcd

if TYPE_CHECKING:
    from etcd3gw.types import Event

logger = structlog.get_logger()


def run_watcher(
    etcd_client: EtcdClient,
    registry: NodeRegistry,
    stop_event: threading.Event,
    retry_delay: float = 5.0,
) -> None:
    """Watch for node changes, reconnecting on failure.

    Runs in a dedicated thread.  Stops when *stop_event* is set.

    Args:
        etcd_client: The etcd client wrapper providing ``watch_prefix``.
        registry: The node registry to update on events.
        stop_event: A ``threading.Event`` signalling graceful shutdown.
        retry_delay: Seconds to wait before reconnecting after failure.
    """
    while not stop_event.is_set():
        try:
            events_iter, cancel = etcd_client.watch_prefix()
            try:
                for event in events_iter:
                    if stop_event.is_set():
                        break
                    try:
                        _handle_event(event, registry, etcd_client.prefix)
                    except Exception:
                        logger.warning(
                            "failed to handle watch event, skipping",
                            event=event,
                            exc_info=True,
                        )
            finally:
                cancel()
        except Exception:
            logger.warning(
                "etcd watch disconnected, reconnecting",
                retry_delay=retry_delay,
                exc_info=True,
            )
            if stop_event.wait(timeout=retry_delay):
                # stop_event was set during the wait -- exit loop
                break


def _handle_event(event: Event, registry: NodeRegistry, prefix: str) -> None:
    """Dispatch a single watch event to the appropriate registry operation.

    Per Pitfall 3 (proto3 JSON): PUT events have no ``type`` field (the
    default enum value of 0 is omitted).  DELETE events have
    ``type: "DELETE"``.

    Args:
        event: The raw event dict from etcd3gw.
        registry: The node registry to update.
        prefix: The configured node key prefix (e.g., ``/nodes/``).
    """
    kv = event.get("kv")
    if kv is None:
        logger.debug("skipping event without kv", event_type=event.get("type"))
        return
    raw_key: bytes | str | None = kv.get("key")
    if raw_key is None:
        logger.warning("skipping event with missing key")
        return

    # Handle both bytes and str keys (Pitfall 2)
    key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key

    event_type = event.get("type", "PUT")

    if event_type == "DELETE":
        node_id = key.removeprefix(prefix)
        if registry.drain(node_id):
            logger.info("node draining", node_id=node_id)
        else:
            logger.debug("delete event for unknown node, skipping", node_id=node_id)
    else:
        value = kv.get("value", b"")
        # Handle str values: encode to bytes for serializer
        if isinstance(value, str):
            value = value.encode("utf-8")
        node = node_from_etcd(key, value, prefix)
        if node is not None:
            registry.add(node)
            logger.info(
                "node added",
                node_id=node.node_id,
                endpoint=node.endpoint,
            )
