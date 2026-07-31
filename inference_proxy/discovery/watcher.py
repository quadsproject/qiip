"""Revision-aware etcd watch recovery for the in-memory node registry.

The etcd3gw public watch iterator cannot be used safely here: its queue read
has no timeout, it flattens watch-response batches, and it ignores compaction
cancellation. The adapter therefore exposes a raw stream with a finite read
timeout, while ``EtcdWatcher`` owns the active response so shutdown can
interrupt its socket read immediately.
"""

from __future__ import annotations

import threading

import structlog

from inference_proxy.discovery.etcd_client import (
    EtcdClient,
    EtcdEvent,
    EtcdSnapshot,
    EtcdWatchBatch,
    EtcdWatchStream,
    WatchCompactedError,
    WatchReadTimeoutError,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_from_etcd
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node

logger = structlog.get_logger()


class EtcdWatcher:
    """Own the active watch stream so shutdown can interrupt its blocking read."""

    def __init__(
        self,
        etcd_client: EtcdClient,
        registry: NodeRegistry,
        stop_event: threading.Event,
        endpoint_policy: EndpointPolicy,
        retry_delay: float = 5.0,
    ) -> None:
        self._etcd_client = etcd_client
        self._registry = registry
        self._stop_event = stop_event
        self._endpoint_policy = endpoint_policy
        self._retry_delay = retry_delay
        self._stream_lock = threading.Lock()
        self._active_stream: EtcdWatchStream | None = None

    def run(self) -> None:
        """Watch node changes, resuming gaps and snapshotting after compaction."""
        resume_revision: int | None = None
        key_revisions: dict[str, int] = {}

        while not self._stop_event.is_set():
            try:
                if resume_revision is None:
                    snapshot = self._etcd_client.get_snapshot()
                    key_revisions = _reconcile_snapshot(
                        snapshot,
                        self._registry,
                        self._etcd_client.prefix,
                        self._endpoint_policy,
                    )
                    resume_revision = snapshot.revision
                    logger.info(
                        "etcd node snapshot reconciled",
                        node_count=len(snapshot.records),
                        revision=resume_revision,
                    )

                stream = self._etcd_client.watch_prefix(
                    start_revision=resume_revision + 1,
                )
                if not self._activate(stream):
                    stream.close()
                    break
                try:
                    for batch in stream:
                        if self._stop_event.is_set():
                            break
                        _apply_batch(
                            batch,
                            self._registry,
                            self._etcd_client.prefix,
                            key_revisions,
                            self._endpoint_policy,
                        )
                        # Advance only after the whole batch succeeds. Per-key
                        # gates make replay safe if a later event fails.
                        resume_revision = max(resume_revision, batch.revision)
                finally:
                    self._deactivate(stream)
                    stream.close()

                if self._stop_event.is_set():
                    break
                logger.warning(
                    "etcd watch stream ended, reconnecting",
                    retry_delay=self._retry_delay,
                )
                if self._stop_event.wait(timeout=self._retry_delay):
                    break
            except WatchReadTimeoutError:
                if self._stop_event.is_set():
                    break
                logger.debug(
                    "etcd watch read timed out, resuming",
                    revision=resume_revision,
                )
            except WatchCompactedError as exc:
                if self._stop_event.is_set():
                    break
                # A fresh snapshot resets the resume point and per-key gates.
                logger.warning(
                    "etcd watch revision compacted, taking fresh snapshot",
                    compact_revision=exc.compact_revision,
                    reason=exc.reason,
                )
                resume_revision = None
                key_revisions = {}
            except Exception:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "etcd watch disconnected, reconnecting",
                    retry_delay=self._retry_delay,
                    exc_info=True,
                )
                if self._stop_event.wait(timeout=self._retry_delay):
                    break

    def stop(self) -> None:
        """Signal shutdown and close the active response to unblock iteration."""
        self._stop_event.set()
        with self._stream_lock:
            stream = self._active_stream
        if stream is not None:
            stream.close()

    def _activate(self, stream: EtcdWatchStream) -> bool:
        with self._stream_lock:
            if self._stop_event.is_set():
                return False
            self._active_stream = stream
            return True

    def _deactivate(self, stream: EtcdWatchStream) -> None:
        with self._stream_lock:
            if self._active_stream is stream:
                self._active_stream = None


def run_watcher(
    etcd_client: EtcdClient,
    registry: NodeRegistry,
    stop_event: threading.Event,
    endpoint_policy: EndpointPolicy,
    retry_delay: float = 5.0,
) -> None:
    """Compatibility entry point for callers that do not need direct stop()."""
    EtcdWatcher(
        etcd_client,
        registry,
        stop_event,
        endpoint_policy,
        retry_delay,
    ).run()


def _reconcile_snapshot(
    snapshot: EtcdSnapshot,
    registry: NodeRegistry,
    prefix: str,
    endpoint_policy: EndpointPolicy,
) -> dict[str, int]:
    """Apply one prefix snapshot and seed per-key revision gates."""
    nodes: dict[str, Node] = {}
    present_node_ids: set[str] = set()
    key_revisions: dict[str, int] = {}

    for record in snapshot.records:
        node_id = _node_id(record.key, prefix)
        if node_id is None:
            continue
        present_node_ids.add(node_id)
        key_revisions[node_id] = record.mod_revision
        node = node_from_etcd(
            record.key,
            record.value,
            prefix,
            endpoint_policy=endpoint_policy,
        )
        if node is not None:
            nodes[node_id] = node

    missing_node_ids = registry.reconcile_discovered(nodes, present_node_ids)

    # Missing keys have no mod_revision in a range response. Retain a
    # tombstone at the authoritative snapshot revision so replayed events,
    # especially late DELETEs, cannot affect the reconciled state.
    for node_id in missing_node_ids:
        key_revisions[node_id] = snapshot.revision
    return key_revisions


def _apply_batch(
    batch: EtcdWatchBatch,
    registry: NodeRegistry,
    prefix: str,
    key_revisions: dict[str, int],
    endpoint_policy: EndpointPolicy,
) -> None:
    """Apply every event in a response using independent per-key gates."""
    for event in batch.events:
        node_id = _node_id(event.key, prefix)
        if node_id is None:
            continue
        previous_revision = key_revisions.get(node_id, 0)
        if event.mod_revision <= previous_revision:
            logger.debug(
                "skipping stale etcd watch event",
                node_id=node_id,
                event_revision=event.mod_revision,
                applied_revision=previous_revision,
            )
            continue
        _apply_event(event, node_id, registry, prefix, endpoint_policy)
        key_revisions[node_id] = event.mod_revision


def _apply_event(
    event: EtcdEvent,
    node_id: str,
    registry: NodeRegistry,
    prefix: str,
    endpoint_policy: EndpointPolicy,
) -> None:
    if event.is_delete:
        if registry.drain(node_id):
            logger.info("node draining", node_id=node_id)
        else:
            logger.debug("delete event for unknown node, skipping", node_id=node_id)
        return

    if event.value is None:
        logger.warning("skipping put event with missing value", node_id=node_id)
        return
    node = node_from_etcd(
        event.key,
        event.value,
        prefix,
        endpoint_policy=endpoint_policy,
    )
    if node is not None:
        registry.add_discovered(node)
        logger.info(
            "node added",
            node_id=node.node_id,
            endpoint=node.endpoint,
        )


def _node_id(key: bytes, prefix: str) -> str | None:
    try:
        decoded_key = key.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping etcd key that is not utf-8")
        return None
    if not decoded_key.startswith(prefix):
        logger.warning(
            "skipping etcd key outside configured prefix",
            key=decoded_key,
            prefix=prefix,
        )
        return None
    node_id = decoded_key.removeprefix(prefix)
    if not node_id:
        logger.warning("skipping etcd key without node id", key=decoded_key)
        return None
    return node_id
