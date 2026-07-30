"""Unit tests for the watch thread with reconnection loop and event dispatch.

Tests cover:
- PUT event dispatch (type field absent per proto3 JSON)
- DELETE event dispatch
- Malformed value handling
- Bytes key/value decoding
- Reconnection after exception
- Stop event terminates loop
- cancel() always called via finally
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.watcher import _handle_event, run_watcher
from inference_proxy.models.node import Node, NodeStatus


class TestPutEventAddsNode:
    """PUT event (type field absent) calls node_from_etcd and registry.add."""

    def test_put_event_adds_node(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"
        event = {
            "kv": {
                "key": "/nodes/node-1",
                "value": b'{"endpoint": "http://10.0.1.100:8000"}',
            }
        }

        _handle_event(event, registry, prefix)

        node = registry.get("node-1")
        assert node is not None
        assert node.node_id == "node-1"
        assert node.endpoint == "http://10.0.1.100:8000"


class TestDeleteEventDrainsNode:
    """DELETE event (type='DELETE') sets node to DRAINING instead of removing (D-10)."""

    def test_delete_event_sets_draining(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"
        # Pre-populate registry with HEALTHY node
        registry.add(
            Node(
                node_id="node-1",
                endpoint="http://10.0.1.100:8000",
                status=NodeStatus.HEALTHY,
            )
        )

        event = {
            "kv": {"key": "/nodes/node-1"},
            "type": "DELETE",
        }

        _handle_event(event, registry, prefix)

        node = registry.get("node-1")
        assert node is not None
        assert node.status == NodeStatus.DRAINING

    def test_delete_nonexistent_node_is_noop(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"

        event = {
            "kv": {"key": "/nodes/nonexistent"},
            "type": "DELETE",
        }

        # Should not raise
        _handle_event(event, registry, prefix)


class TestPutMalformedValueSkipped:
    """PUT event with malformed value (node_from_etcd returns None) does not add."""

    def test_put_malformed_value_skipped(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"
        event = {
            "kv": {
                "key": "/nodes/node-bad",
                "value": b"not valid json!!!",
            }
        }

        _handle_event(event, registry, prefix)

        assert registry.get("node-bad") is None
        assert registry.get_all() == []


class TestBytesKeyDecoded:
    """Event with bytes key is decoded to str before processing."""

    def test_bytes_key_decoded(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"
        event = {
            "kv": {
                "key": b"/nodes/node-bytes",
                "value": b'{"endpoint": "http://10.0.1.200:8000"}',
            }
        }

        _handle_event(event, registry, prefix)

        node = registry.get("node-bytes")
        assert node is not None
        assert node.endpoint == "http://10.0.1.200:8000"


class TestWatchPrefixExceptionReconnects:
    """When watch_prefix raises an exception, watcher reconnects after delay."""

    def test_watch_prefix_exception_reconnects(self) -> None:
        mock_client = MagicMock(spec=EtcdClient)
        mock_client.prefix = "/nodes/"

        call_count = 0

        def watch_side_effect() -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("etcd unavailable")
            # Second call: return empty iterator that stops
            return iter([]), MagicMock()

        mock_client.watch_prefix.side_effect = watch_side_effect

        registry = NodeRegistry()
        stop_event = threading.Event()

        def stop_after_reconnect() -> None:
            """Stop after watcher has had a chance to reconnect."""
            while call_count < 2:
                pass
            stop_event.set()

        stopper = threading.Thread(target=stop_after_reconnect)
        stopper.start()

        run_watcher(mock_client, registry, stop_event, retry_delay=0.01)

        stopper.join(timeout=2)
        assert mock_client.watch_prefix.call_count >= 2


class TestStopEventTerminatesLoop:
    """When stop_event is set, watcher exits the reconnection loop."""

    def test_stop_event_terminates_loop(self) -> None:
        mock_client = MagicMock(spec=EtcdClient)
        mock_client.prefix = "/nodes/"

        stop_event = threading.Event()
        stop_event.set()  # Pre-set: watcher should exit immediately

        registry = NodeRegistry()

        run_watcher(mock_client, registry, stop_event, retry_delay=0.01)

        # Should have exited without calling watch_prefix
        mock_client.watch_prefix.assert_not_called()


class TestCancelCalledInFinally:
    """cancel() is called in finally block when watch_prefix returns."""

    def test_cancel_called_on_normal_exit(self) -> None:
        mock_client = MagicMock(spec=EtcdClient)
        mock_client.prefix = "/nodes/"

        cancel_fn = MagicMock()

        # Return empty iterator once, then set stop on second call
        stop_event = threading.Event()

        call_count = 0

        def watch_side_effect() -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([]), cancel_fn
            # Second call: signal stop and return another empty iter
            stop_event.set()
            return iter([]), MagicMock()

        mock_client.watch_prefix.side_effect = watch_side_effect

        registry = NodeRegistry()

        run_watcher(mock_client, registry, stop_event, retry_delay=0.01)

        cancel_fn.assert_called_once()

    def test_cancel_called_on_iteration_error(self) -> None:
        mock_client = MagicMock(spec=EtcdClient)
        mock_client.prefix = "/nodes/"

        cancel_fn = MagicMock()

        def make_failing_iter():
            """Iterator that raises during iteration."""
            yield {"kv": {"key": "/nodes/n1", "value": b'{"endpoint": "http://x:1"}'}}
            raise ValueError("stream error mid-iteration")

        call_count = 0

        def watch_side_effect() -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_failing_iter(), cancel_fn
            return iter([]), MagicMock()

        mock_client.watch_prefix.side_effect = watch_side_effect

        registry = NodeRegistry()
        stop_event = threading.Event()

        def stop_delayed() -> None:
            while call_count < 2:
                pass
            stop_event.set()

        stopper = threading.Thread(target=stop_delayed)
        stopper.start()

        run_watcher(mock_client, registry, stop_event, retry_delay=0.01)

        stopper.join(timeout=2)
        cancel_fn.assert_called_once()


class TestBytesValueHandled:
    """Event with bytes value is handled (passed to node_from_etcd as-is)."""

    def test_bytes_value_handled(self) -> None:
        registry = NodeRegistry()
        prefix = "/nodes/"
        event = {
            "kv": {
                "key": "/nodes/node-bv",
                "value": b'{"endpoint": "http://10.0.1.300:8000"}',
            }
        }

        _handle_event(event, registry, prefix)

        node = registry.get("node-bv")
        assert node is not None

    def test_str_value_encoded_to_bytes(self) -> None:
        """String values are encoded to bytes before passing to node_from_etcd."""
        registry = NodeRegistry()
        prefix = "/nodes/"
        event = {
            "kv": {
                "key": "/nodes/node-sv",
                "value": '{"endpoint": "http://10.0.1.400:8000"}',
            }
        }

        _handle_event(event, registry, prefix)

        node = registry.get("node-sv")
        assert node is not None
        assert node.endpoint == "http://10.0.1.400:8000"
