"""Behavioral tests for revision-aware etcd watch recovery."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.node import Node, NodeStatus

try:
    from inference_proxy.discovery.etcd_client import (
        WatchCompactedError as _WatchCompactedError,
    )
    from inference_proxy.discovery.etcd_client import (
        WatchReadTimeoutError as _WatchReadTimeoutError,
    )
except ImportError:
    # Unfixed main does not define the adapter exception. Keeping a compatible
    # fallback lets behavioral tests reach assertions instead of failing
    # collection merely because the implementation seam is new.
    class _WatchCompactedError(RuntimeError):
        def __init__(self, compact_revision: int, reason: str = "") -> None:
            super().__init__(reason)
            self.compact_revision = compact_revision
            self.reason = reason

    class _WatchReadTimeoutError(TimeoutError):
        pass


try:
    from inference_proxy.discovery.watcher import EtcdWatcher as _EtcdWatcher
except ImportError:
    from inference_proxy.discovery.watcher import run_watcher as _legacy_run_watcher

    class _EtcdWatcher:
        """Old-main adapter that keeps behavioral comparisons importable."""

        def __init__(
            self,
            client: EtcdClient,
            registry: NodeRegistry,
            stop_event: threading.Event,
            retry_delay: float,
        ) -> None:
            self._args = (client, registry, stop_event, retry_delay)
            self._stop_event = stop_event

        def run(self) -> None:
            _legacy_run_watcher(*self._args)

        def stop(self) -> None:
            self._stop_event.set()


_TIMEOUT = 1.0


@dataclass(frozen=True)
class _Record:
    key: bytes
    value: bytes
    mod_revision: int


@dataclass(frozen=True)
class _Snapshot:
    records: tuple[_Record, ...]
    revision: int


@dataclass(frozen=True)
class _Event:
    key: bytes
    value: bytes | None
    mod_revision: int
    is_delete: bool


@dataclass(frozen=True)
class _Batch:
    events: tuple[_Event, ...]
    revision: int


def _node(
    node_id: str,
    endpoint: str | None = None,
    *,
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "model-a",
    managed: bool = True,
) -> Node:
    return Node(
        node_id=node_id,
        endpoint=endpoint or f"{node_id}:8000",
        status=status,
        model=model,
        managed=managed,
    )


def _record(node: Node, revision: int) -> _Record:
    key, value = node_to_etcd(node, "/nodes/")
    return _Record(key.encode(), value, revision)


def _snapshot(revision: int, *nodes: tuple[Node, int]) -> _Snapshot:
    return _Snapshot(
        tuple(_record(node, mod_revision) for node, mod_revision in nodes),
        revision,
    )


def _put(node: Node, revision: int) -> _Event:
    key, value = node_to_etcd(node, "/nodes/")
    return _Event(key.encode(), value, revision, is_delete=False)


def _delete(node_id: str, revision: int) -> _Event:
    return _Event(
        f"/nodes/{node_id}".encode(),
        None,
        revision,
        is_delete=True,
    )


def _legacy_event(event: _Event) -> dict[str, object]:
    kv: dict[str, object] = {"key": event.key}
    if event.value is not None:
        kv["value"] = event.value
    raw: dict[str, object] = {"kv": kv}
    if event.is_delete:
        raw["type"] = "DELETE"
    return raw


class _FakeStream:
    """A closeable stream that can remain blocked after finite batches."""

    def __init__(
        self,
        batches: tuple[_Batch, ...] = (),
        *,
        error: Exception | None = None,
        block_after: bool = False,
    ) -> None:
        self._batches = batches
        self._error = error
        self._block_after = block_after
        self.started = threading.Event()
        self.closed = threading.Event()
        self.release = threading.Event()

    def __iter__(self) -> Iterator[_Batch]:
        self.started.set()
        yield from self._batches
        if self._error is not None:
            raise self._error
        if self._block_after:
            self.release.wait(timeout=5)

    def close(self) -> None:
        self.closed.set()
        self.release.set()


class _LegacyStream:
    """Old iterator behavior used to verify failures against unfixed main."""

    def __init__(
        self,
        events: tuple[dict[str, object], ...] = (),
        *,
        block_after: bool = False,
    ) -> None:
        self._events = events
        self._block_after = block_after
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def __iter__(self) -> Iterator[dict[str, object]]:
        self.started.set()
        yield from self._events
        if self._block_after:
            self.release.wait(timeout=5)

    def cancel(self) -> None:
        self.cancelled.set()


class _FakeEtcdClient:
    """Supports both the fixed and old watch signatures for mutation checks."""

    prefix = "/nodes/"

    def __init__(
        self,
        snapshots: tuple[_Snapshot, ...],
        watches: tuple[_FakeStream, ...],
        *,
        legacy_watches: tuple[_LegacyStream, ...] = (),
    ) -> None:
        self._snapshots = deque(snapshots)
        self._watches = deque(watches)
        self._legacy_watches = deque(legacy_watches)
        self.snapshot_calls = 0
        self.start_revisions: list[int | None] = []

    def get_snapshot(self) -> _Snapshot:
        self.snapshot_calls += 1
        return self._snapshots.popleft()

    def watch_prefix(
        self,
        *,
        start_revision: int | None = None,
    ) -> object | tuple[Iterator[dict[str, object]], object]:
        self.start_revisions.append(start_revision)
        if start_revision is None:
            stream = self._legacy_watches.popleft()
            return iter(stream), stream.cancel
        return self._watches.popleft()


class _CountingRegistry(NodeRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_writes = 0

    def add_discovered(self, node: Node) -> None:
        super().add_discovered(node)
        self.discovery_writes += 1


def _start_watcher(
    client: _FakeEtcdClient,
    registry: NodeRegistry,
    *,
    retry_delay: float = 0.001,
) -> tuple[_EtcdWatcher, threading.Thread]:
    stop_event = threading.Event()
    watcher = _EtcdWatcher(
        cast(EtcdClient, client),
        registry,
        stop_event,
        retry_delay=retry_delay,
    )
    thread = threading.Thread(
        target=watcher.run,
        daemon=True,
    )
    thread.start()
    return watcher, thread


def _stop_watcher(
    watcher: _EtcdWatcher,
    thread: threading.Thread,
    *streams: _FakeStream | _LegacyStream,
    release_before_join: bool = True,
) -> None:
    watcher.stop()
    if release_before_join:
        for stream in streams:
            stream.release.set()
    thread.join(timeout=_TIMEOUT)
    try:
        assert not thread.is_alive()
    finally:
        for stream in streams:
            stream.release.set()


def _wait_for_stream(
    stream: _FakeStream,
    legacy_stream: _LegacyStream,
) -> bool:
    """Bound waiting across fixed and unfixed entry-point shapes."""
    return stream.started.wait(timeout=_TIMEOUT / 2) or legacy_stream.started.wait(
        timeout=_TIMEOUT / 2
    )


def test_idle_watch_stops_promptly() -> None:
    """A blocked transport read cannot hold the watcher thread open (R1)."""
    stream = _FakeStream(block_after=True)
    legacy = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(1),),
        (stream,),
        legacy_watches=(legacy,),
    )
    watcher, thread = _start_watcher(client, NodeRegistry())
    assert _wait_for_stream(stream, legacy)

    _stop_watcher(
        watcher,
        thread,
        stream,
        legacy,
        release_before_join=False,
    )

    assert stream.closed.is_set()


def test_silent_watch_disconnect_reconnects() -> None:
    """Clean EOF starts a replacement watch from the saved revision."""
    disconnected = _FakeStream()
    replacement = _FakeStream(block_after=True)
    legacy_disconnected = _LegacyStream()
    legacy_replacement = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(4),),
        (disconnected, replacement),
        legacy_watches=(legacy_disconnected, legacy_replacement),
    )
    watcher, thread = _start_watcher(client, NodeRegistry())

    assert _wait_for_stream(replacement, legacy_replacement)
    _stop_watcher(
        watcher,
        thread,
        disconnected,
        replacement,
        legacy_disconnected,
        legacy_replacement,
    )

    assert client.start_revisions[:2] == [5, 5]


def test_watch_read_timeout_reconnects_from_saved_revision() -> None:
    """An idle or silently dead connection cannot freeze discovery."""
    timed_out = _FakeStream(error=_WatchReadTimeoutError("watch read timed out"))
    replacement = _FakeStream(block_after=True)
    legacy_timed_out = _LegacyStream()
    legacy_replacement = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(4),),
        (timed_out, replacement),
        legacy_watches=(legacy_timed_out, legacy_replacement),
    )
    with capture_logs() as logs:
        # A large backoff makes the assertion falsify routing the idle timeout
        # through the generic disconnect handler.
        watcher, thread = _start_watcher(
            client,
            NodeRegistry(),
            retry_delay=60,
        )

        try:
            reconnected = _wait_for_stream(replacement, legacy_replacement)
        finally:
            _stop_watcher(
                watcher,
                thread,
                timed_out,
                replacement,
                legacy_timed_out,
                legacy_replacement,
            )

    assert reconnected
    assert client.start_revisions[:2] == [5, 5]
    timeout_logs = [
        event
        for event in logs
        if event["event"] == "etcd watch read timed out, resuming"
    ]
    assert timeout_logs == [
        {
            "event": "etcd watch read timed out, resuming",
            "log_level": "debug",
            "revision": 4,
        }
    ]


def test_registry_reconciles_after_watch_gap() -> None:
    """Historical events converge changed, added, and deleted nodes (B5/T14)."""
    old = _node("old", "old:8000")
    removed = _node("removed")
    changed = _node("old", "new:9000", model="model-b", managed=False)
    added = _node("added")
    gap_batch = _Batch(
        (
            _put(changed, 11),
            _put(added, 12),
            _delete("removed", 13),
        ),
        revision=13,
    )
    disconnected = _FakeStream()
    replay = _FakeStream((gap_batch,))
    settled = _FakeStream(block_after=True)
    # Unfixed code reconnects at "now" and receives no historical gap events.
    legacy_disconnected = _LegacyStream()
    legacy_settled = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(10, (old, 8), (removed, 9)),),
        (disconnected, replay, settled),
        legacy_watches=(legacy_disconnected, legacy_settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled, legacy_settled)
    _stop_watcher(
        watcher,
        thread,
        disconnected,
        replay,
        settled,
        legacy_disconnected,
        legacy_settled,
    )

    current = registry.get("old")
    assert current is not None
    assert current.endpoint == "new:9000"
    assert current.model == "model-b"
    assert current.managed is False
    assert registry.get("added") is not None
    drained = registry.get("removed")
    assert drained is not None
    assert drained.status == NodeStatus.DRAINING


def test_watch_resumes_after_snapshot_revision() -> None:
    """Snapshot and response header revisions define each watch start."""
    progress = _FakeStream((_Batch((), revision=12),))
    replacement = _FakeStream(block_after=True)
    legacy_progress = _LegacyStream()
    legacy_replacement = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(8),),
        (progress, replacement),
        legacy_watches=(legacy_progress, legacy_replacement),
    )
    watcher, thread = _start_watcher(client, NodeRegistry())

    assert _wait_for_stream(replacement, legacy_replacement)
    _stop_watcher(
        watcher,
        thread,
        progress,
        replacement,
        legacy_progress,
        legacy_replacement,
    )

    assert client.start_revisions[:2] == [9, 13]


@pytest.mark.parametrize("stale_kind", ["put", "delete"])
def test_stale_watch_event_does_not_override_reconciled_node(
    stale_kind: str,
) -> None:
    """A late PUT or DELETE cannot mutate a fresher snapshot registration."""
    fresh = _node("node-1", "fresh:8000")
    stale_event = (
        _put(_node("node-1", "stale:8000"), 9)
        if stale_kind == "put"
        else _delete("node-1", 9)
    )
    stale = _FakeStream((_Batch((stale_event,), revision=10),))
    settled = _FakeStream(block_after=True)
    legacy_event = _LegacyStream(
        (
            _legacy_event(_put(fresh, 10)),
            _legacy_event(stale_event),
        )
    )
    legacy_settled = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(10, (fresh, 10)),),
        (stale, settled),
        legacy_watches=(legacy_event, legacy_settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled, legacy_settled)
    _stop_watcher(
        watcher,
        thread,
        stale,
        settled,
        legacy_event,
        legacy_settled,
    )

    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "fresh:8000"
    assert current.status == NodeStatus.HEALTHY


def test_event_between_snapshot_and_watch_is_applied() -> None:
    """The snapshot-to-watch handoff neither misses nor duplicates an event."""
    between = _put(_node("between"), 11)
    first = _FakeStream((_Batch((between,), revision=11),))
    settled = _FakeStream(block_after=True)
    legacy_first = _LegacyStream()
    legacy_settled = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(10),),
        (first, settled),
        legacy_watches=(legacy_first, legacy_settled),
    )
    registry = _CountingRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled, legacy_settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
        legacy_first,
        legacy_settled,
    )

    assert registry.get("between") is not None
    assert registry.discovery_writes == 1
    assert client.start_revisions[:2] == [11, 12]


def test_reconcile_preserves_local_liveness_status() -> None:
    """Snapshot data refresh does not resurrect a locally unhealthy node."""
    registry = NodeRegistry()
    registry.add(
        _node(
            "node-1",
            "old:8000",
            status=NodeStatus.UNHEALTHY,
            model="old-model",
        )
    )
    discovered = _node(
        "node-1",
        "new:9000",
        status=NodeStatus.HEALTHY,
        model="new-model",
        managed=False,
    )
    stream = _FakeStream(block_after=True)
    legacy_stream = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(20, (discovered, 19)),),
        (stream,),
        legacy_watches=(legacy_stream,),
    )
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(stream, legacy_stream)
    _stop_watcher(watcher, thread, stream, legacy_stream)

    current = registry.get("node-1")
    assert current is not None
    assert current.status == NodeStatus.UNHEALTHY
    assert current.endpoint == "new:9000"
    assert current.model == "new-model"
    assert current.managed is False


@pytest.mark.parametrize(
    ("current_status", "incoming_status"),
    [
        (NodeStatus.PROVISIONING, NodeStatus.HEALTHY),
        (NodeStatus.PROVISIONING, NodeStatus.FAILED),
        (NodeStatus.DRAINING, NodeStatus.HEALTHY),
    ],
)
def test_reconcile_applies_etcd_lifecycle_transition(
    current_status: NodeStatus,
    incoming_status: NodeStatus,
) -> None:
    """Lifecycle state from etcd is not frozen by liveness preservation."""
    registry = NodeRegistry()
    registry.add(_node("node-1", status=current_status))
    discovered = _node("node-1", status=incoming_status)
    stream = _FakeStream(block_after=True)
    legacy_stream = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(20, (discovered, 20)),),
        (stream,),
        legacy_watches=(legacy_stream,),
    )
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(stream, legacy_stream)
    _stop_watcher(watcher, thread, stream, legacy_stream)

    current = registry.get("node-1")
    assert current is not None
    assert current.status == incoming_status


def test_compacted_watch_takes_fresh_snapshot_and_recovers() -> None:
    """Compaction discards the resume point and establishes a fresh baseline."""
    old = _node("node-1", "old:8000")
    removed = _node("removed")
    fresh = _node("node-1", "fresh:9000")
    compacted = _FakeStream(
        error=_WatchCompactedError(15, "required revision has been compacted")
    )
    recovered = _FakeStream(block_after=True)
    legacy_compacted = _LegacyStream()
    legacy_recovered = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (
            _snapshot(5, (old, 4), (removed, 5)),
            _snapshot(20, (fresh, 20)),
        ),
        (compacted, recovered),
        legacy_watches=(legacy_compacted, legacy_recovered),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(recovered, legacy_recovered)
    _stop_watcher(
        watcher,
        thread,
        compacted,
        recovered,
        legacy_compacted,
        legacy_recovered,
    )

    assert client.snapshot_calls == 2
    assert client.start_revisions[:2] == [6, 21]
    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "fresh:9000"
    missing = registry.get("removed")
    assert missing is not None
    assert missing.status == NodeStatus.DRAINING


def test_same_revision_batch_applies_every_event() -> None:
    """A global per-event gate cannot drop peers from one etcd transaction."""
    batch = _Batch(
        (
            _put(_node("node-1"), 10),
            _put(_node("node-2"), 10),
        ),
        revision=10,
    )
    first = _FakeStream((batch,))
    settled = _FakeStream(block_after=True)
    legacy_batch = _LegacyStream(tuple(_legacy_event(event) for event in batch.events))
    legacy_settled = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(9),),
        (first, settled),
        legacy_watches=(legacy_batch, legacy_settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled, legacy_settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
        legacy_batch,
        legacy_settled,
    )

    assert {node.node_id for node in registry.get_all()} == {"node-1", "node-2"}


def test_numeric_revision_order_is_not_lexicographic() -> None:
    """Revision 10 must be newer than revision 9 after proto3 normalization."""
    original = _node("node-1", "old:8000")
    updated = _node("node-1", "new:9000")
    first = _FakeStream((_Batch((_put(updated, 10),), revision=10),))
    settled = _FakeStream(block_after=True)
    legacy_first = _LegacyStream((_legacy_event(_put(updated, 10)),))
    legacy_settled = _LegacyStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(9, (original, 9)),),
        (first, settled),
        legacy_watches=(legacy_first, legacy_settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled, legacy_settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
        legacy_first,
        legacy_settled,
    )

    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "new:9000"
