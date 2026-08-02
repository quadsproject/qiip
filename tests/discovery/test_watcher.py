"""Behavioral tests for revision-aware etcd watch recovery."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import pytest
from structlog.testing import capture_logs

from inference_proxy.discovery.etcd_client import (
    EtcdClient,
    WatchCompactedError,
    WatchReadTimeoutError,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.discovery.watcher import EtcdWatcher
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import Node, NodeStatus

_TIMEOUT = 1.0
_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=[
        "added",
        "between",
        "fresh",
        "gpu01",
        "new",
        "node-1",
        "node-2",
        "old",
        "removed",
        "stale",
    ],
    allowed_networks=[],
    allowed_ports=[8000, 9000],
)


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


class _FakeEtcdClient:
    """Provide revision-aware snapshots and watch streams to the watcher."""

    prefix = "/nodes/"

    def __init__(
        self,
        snapshots: tuple[_Snapshot, ...],
        watches: tuple[_FakeStream, ...],
    ) -> None:
        self._snapshots = deque(snapshots)
        self._watches = deque(watches)
        self.snapshot_calls = 0
        self.start_revisions: list[int | None] = []

    def get_snapshot(self) -> _Snapshot:
        self.snapshot_calls += 1
        return self._snapshots.popleft()

    def watch_prefix(
        self,
        *,
        start_revision: int | None = None,
    ) -> _FakeStream:
        self.start_revisions.append(start_revision)
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
) -> tuple[EtcdWatcher, threading.Thread]:
    stop_event = threading.Event()
    watcher = EtcdWatcher(
        cast(EtcdClient, client),
        registry,
        stop_event,
        endpoint_policy=_ENDPOINT_POLICY,
        retry_delay=retry_delay,
    )
    thread = threading.Thread(
        target=watcher.run,
        daemon=True,
    )
    thread.start()
    return watcher, thread


def _stop_watcher(
    watcher: EtcdWatcher,
    thread: threading.Thread,
    *streams: _FakeStream,
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


def _wait_for_stream(stream: _FakeStream) -> bool:
    """Bound waiting for a revision-aware watch stream to start."""
    return stream.started.wait(timeout=_TIMEOUT)


def test_start_watcher_does_not_hide_constructor_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation guard: a reintroduced legacy fallback must not hide a bug."""

    class _DummyWatcher:
        def run(self) -> None:
            pass

        def stop(self) -> None:
            pass

    def exploding_watcher(*_args: object, **kwargs: object) -> _DummyWatcher:
        if "endpoint_policy" in kwargs:
            raise TypeError("modern signature rejected")
        return _DummyWatcher()

    monkeypatch.setitem(globals(), "EtcdWatcher", exploding_watcher)

    with pytest.raises(TypeError, match="modern signature rejected"):
        _start_watcher(_FakeEtcdClient((), ()), NodeRegistry())


def test_idle_watch_stops_promptly() -> None:
    """A blocked transport read cannot hold the watcher thread open (R1)."""
    stream = _FakeStream(block_after=True)
    client = _FakeEtcdClient((_snapshot(1),), (stream,))
    watcher, thread = _start_watcher(client, NodeRegistry())
    assert _wait_for_stream(stream)

    _stop_watcher(
        watcher,
        thread,
        stream,
        release_before_join=False,
    )

    assert stream.closed.is_set()


def test_silent_watch_disconnect_reconnects() -> None:
    """Clean EOF starts a replacement watch from the saved revision."""
    disconnected = _FakeStream()
    replacement = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(4),),
        (disconnected, replacement),
    )
    watcher, thread = _start_watcher(client, NodeRegistry())

    assert _wait_for_stream(replacement)
    _stop_watcher(
        watcher,
        thread,
        disconnected,
        replacement,
    )

    assert client.start_revisions[:2] == [5, 5]


def test_watch_read_timeout_reconnects_from_saved_revision() -> None:
    """An idle or silently dead connection cannot freeze discovery."""
    timed_out = _FakeStream(error=WatchReadTimeoutError("watch read timed out"))
    replacement = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(4),),
        (timed_out, replacement),
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
            reconnected = _wait_for_stream(replacement)
        finally:
            _stop_watcher(
                watcher,
                thread,
                timed_out,
                replacement,
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
    client = _FakeEtcdClient(
        (_snapshot(10, (old, 8), (removed, 9)),),
        (disconnected, replay, settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        disconnected,
        replay,
        settled,
    )

    current = registry.get("old")
    assert current is not None
    assert current.endpoint == "http://new:9000"
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
    client = _FakeEtcdClient(
        (_snapshot(8),),
        (progress, replacement),
    )
    watcher, thread = _start_watcher(client, NodeRegistry())

    assert _wait_for_stream(replacement)
    _stop_watcher(
        watcher,
        thread,
        progress,
        replacement,
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
    client = _FakeEtcdClient(
        (_snapshot(10, (fresh, 10)),),
        (stale, settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        stale,
        settled,
    )

    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "http://fresh:8000"
    assert current.status == NodeStatus.HEALTHY


def test_event_between_snapshot_and_watch_is_applied() -> None:
    """The snapshot-to-watch handoff neither misses nor duplicates an event."""
    between = _put(_node("between"), 11)
    first = _FakeStream((_Batch((between,), revision=11),))
    settled = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(10),),
        (first, settled),
    )
    registry = _CountingRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
    )

    assert registry.get("between") is not None
    assert registry.discovery_writes == 1
    assert client.start_revisions[:2] == [11, 12]


def test_late_cleanup_delete_does_not_drain_new_registration() -> None:
    """A retry's delayed cleanup DELETE cannot drain its newer HEALTHY PUT."""
    failed = _node("gpu01", status=NodeStatus.FAILED)
    healthy = _node("gpu01", "gpu01:8000", status=NodeStatus.HEALTHY)
    registration_then_late_delete = _Batch(
        (
            _put(healthy, 3),
            _delete("gpu01", 2),
        ),
        revision=3,
    )
    events = _FakeStream((registration_then_late_delete,))
    settled = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(1, (failed, 1)),),
        (events, settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        events,
        settled,
    )

    current = registry.get("gpu01")
    assert current is not None
    assert current.status == NodeStatus.HEALTHY
    assert current.endpoint == "http://gpu01:8000"


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
    client = _FakeEtcdClient(
        (_snapshot(20, (discovered, 19)),),
        (stream,),
    )
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(stream)
    _stop_watcher(watcher, thread, stream)

    current = registry.get("node-1")
    assert current is not None
    assert current.status == NodeStatus.UNHEALTHY
    assert current.endpoint == "http://new:9000"
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
    client = _FakeEtcdClient(
        (_snapshot(20, (discovered, 20)),),
        (stream,),
    )
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(stream)
    _stop_watcher(watcher, thread, stream)

    current = registry.get("node-1")
    assert current is not None
    assert current.status == incoming_status


def test_compacted_watch_takes_fresh_snapshot_and_recovers() -> None:
    """Compaction discards the resume point and establishes a fresh baseline."""
    old = _node("node-1", "old:8000")
    removed = _node("removed")
    fresh = _node("node-1", "fresh:9000")
    compacted = _FakeStream(
        error=WatchCompactedError(15, "required revision has been compacted")
    )
    recovered = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (
            _snapshot(5, (old, 4), (removed, 5)),
            _snapshot(20, (fresh, 20)),
        ),
        (compacted, recovered),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(recovered)
    _stop_watcher(
        watcher,
        thread,
        compacted,
        recovered,
    )

    assert client.snapshot_calls == 2
    assert client.start_revisions[:2] == [6, 21]
    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "http://fresh:9000"
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
    client = _FakeEtcdClient(
        (_snapshot(9),),
        (first, settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
    )

    assert {node.node_id for node in registry.get_all()} == {"node-1", "node-2"}


def test_numeric_revision_order_is_not_lexicographic() -> None:
    """Revision 10 must be newer than revision 9 after proto3 normalization."""
    original = _node("node-1", "old:8000")
    updated = _node("node-1", "new:9000")
    first = _FakeStream((_Batch((_put(updated, 10),), revision=10),))
    settled = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(9, (original, 9)),),
        (first, settled),
    )
    registry = NodeRegistry()
    watcher, thread = _start_watcher(client, registry)

    assert _wait_for_stream(settled)
    _stop_watcher(
        watcher,
        thread,
        first,
        settled,
    )

    current = registry.get("node-1")
    assert current is not None
    assert current.endpoint == "http://new:9000"


def test_watch_rejects_disallowed_endpoint() -> None:
    """A live etcd PUT cannot add a backend outside the endpoint policy."""
    disallowed = _node("metadata", "169.254.169.254:8000")
    events = _FakeStream((_Batch((_put(disallowed, 2),), revision=2),))
    settled = _FakeStream(block_after=True)
    client = _FakeEtcdClient(
        (_snapshot(1),),
        (events, settled),
    )
    registry = NodeRegistry()

    with capture_logs() as logs:
        watcher, thread = _start_watcher(client, registry)
        assert _wait_for_stream(settled)
        _stop_watcher(
            watcher,
            thread,
            events,
            settled,
        )

    assert registry.get("metadata") is None
    rejection = [log for log in logs if log["event"] == "skipping malformed node"]
    assert len(rejection) == 1
    assert rejection[0]["log_level"] == "warning"
    assert rejection[0]["endpoint"] == "http://169.254.169.254:8000"
    assert "host is not allowed" in rejection[0]["error"]
