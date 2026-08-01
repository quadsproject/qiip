"""Contract tests for the etcd3gw adapter boundary."""

from __future__ import annotations

import base64
import json
import socket
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

from inference_proxy.config.settings import EtcdSettings
from inference_proxy.discovery.etcd_client import (
    EtcdClient,
    EtcdWatchStream,
    WatchCompactedError,
    WatchReadTimeoutError,
)


def _settings(
    endpoint: str = "http://localhost:2379",
    prefix: str = "/nodes/",
) -> EtcdSettings:
    return EtcdSettings(endpoints=[endpoint], node_prefix=prefix)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


class _StreamingResponse:
    def __init__(
        self,
        payloads: list[dict[str, object]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._lines = [json.dumps(payload).encode() for payload in payloads]
        self._error = error
        self.closed = False
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1

    def iter_lines(self, *, decode_unicode: bool = False) -> Iterator[bytes]:
        assert decode_unicode is False
        yield from self._lines
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


class TestEtcdClientInit:
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_parses_endpoint_url(self, mock_etcd3_cls: MagicMock) -> None:
        EtcdClient(_settings("http://etcd.internal:2379"))

        mock_etcd3_cls.assert_called_once_with(
            host="etcd.internal",
            port=2379,
            protocol="http",
            timeout=5,
        )

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_default_port(self, mock_etcd3_cls: MagicMock) -> None:
        EtcdClient(_settings("http://etcd.internal"))

        mock_etcd3_cls.assert_called_once_with(
            host="etcd.internal",
            port=2379,
            protocol="http",
            timeout=5,
        )

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_https_protocol(self, mock_etcd3_cls: MagicMock) -> None:
        EtcdClient(_settings("https://secure-etcd.internal:2380"))

        mock_etcd3_cls.assert_called_once_with(
            host="secure-etcd.internal",
            port=2380,
            protocol="https",
            timeout=5,
        )

    @pytest.mark.parametrize("endpoint", ["etcd.internal:2379", "etcd.internal"])
    def test_schemeless_endpoint_is_rejected(self, endpoint: str) -> None:
        with pytest.raises(ValueError, match="Invalid etcd endpoint URL"):
            EtcdClient(_settings(endpoint))

    @patch("inference_proxy.discovery.etcd_client.logger")
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_multiple_endpoints_warns(
        self,
        mock_etcd3_cls: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        settings = EtcdSettings(
            endpoints=["http://etcd1:2379", "http://etcd2:2379"],
            node_prefix="/nodes/",
        )

        EtcdClient(settings)

        mock_etcd3_cls.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "multiple etcd endpoints configured but only the first is used",
            endpoint="http://etcd1:2379",
            ignored=["http://etcd2:2379"],
        )

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_close_releases_underlying_http_session(
        self, mock_etcd3_cls: MagicMock
    ) -> None:
        client = EtcdClient(_settings())

        client.close()

        mock_etcd3_cls.return_value.session.close.assert_called_once_with()


class TestEtcdSnapshot:
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_snapshot_revision_comes_from_range_header(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.get_url.return_value = "http://etcd/v3/kv/range"
        mock_instance.post.return_value = {
            "header": {"revision": "4711"},
            "kvs": [
                {
                    "key": _b64(b"/nodes/node-1"),
                    "value": _b64(b'{"endpoint":"node-1:8000"}'),
                    # Deliberately not the store revision. Taking max KV
                    # revision would lose deletions and activity elsewhere.
                    "mod_revision": "9",
                }
            ],
        }
        client = EtcdClient(_settings())

        snapshot = client.get_snapshot()

        assert snapshot.revision == 4711
        assert isinstance(snapshot.revision, int)
        assert snapshot.records[0].mod_revision == 9
        assert isinstance(snapshot.records[0].mod_revision, int)
        assert snapshot.records[0].key == b"/nodes/node-1"
        assert snapshot.records[0].value == b'{"endpoint":"node-1:8000"}'
        mock_instance.post.assert_called_once_with(
            "http://etcd/v3/kv/range",
            json={
                "key": _b64(b"/nodes/"),
                "range_end": _b64(b"/nodes0"),
            },
        )

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_custom_snapshot_prefix_is_encoded(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.get_url.return_value = "http://etcd/v3/kv/range"
        mock_instance.post.return_value = {
            "header": {"revision": "2"},
            "kvs": [],
        }

        EtcdClient(_settings()).get_snapshot("/provisioning/")

        mock_instance.post.assert_called_once_with(
            "http://etcd/v3/kv/range",
            json={
                "key": _b64(b"/provisioning/"),
                "range_end": _b64(b"/provisioning0"),
            },
        )


class TestEtcdWrites:
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_replace_delegates_compare_and_swap(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.replace.return_value = True
        client = EtcdClient(_settings())

        replaced = client.replace("/nodes/gpu01", b"old", b"new")

        assert replaced is True
        mock_instance.replace.assert_called_once_with(
            "/nodes/gpu01",
            b"old",
            b"new",
        )


class TestRawWatch:
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_watch_forwards_start_revision_and_preserves_batch(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        response = _StreamingResponse(
            [
                {"result": {"created": True, "header": {"revision": "9"}}},
                {
                    "result": {
                        "header": {"revision": "10"},
                        "events": [
                            {
                                "kv": {
                                    "key": _b64(b"/nodes/node-1"),
                                    "value": _b64(b'{"endpoint":"one:8000"}'),
                                    "mod_revision": "10",
                                }
                            },
                            {
                                "type": "DELETE",
                                "kv": {
                                    "key": _b64(b"/nodes/node-2"),
                                    "mod_revision": "10",
                                },
                            },
                        ],
                    }
                },
            ]
        )
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.timeout = 5
        mock_instance.get_url.return_value = "http://etcd/v3/watch"
        mock_instance.session.post.return_value = response
        client = EtcdClient(_settings())

        stream = client.watch_prefix(start_revision=10)
        batches = list(stream)

        assert len(batches) == 1
        assert batches[0].revision == 10
        assert isinstance(batches[0].revision, int)
        assert [event.mod_revision for event in batches[0].events] == [10, 10]
        assert all(isinstance(event.mod_revision, int) for event in batches[0].events)
        assert batches[0].events[0].value == b'{"endpoint":"one:8000"}'
        assert batches[0].events[1].is_delete is True
        assert response.raise_calls == 1
        mock_instance.session.post.assert_called_once_with(
            "http://etcd/v3/watch",
            json={
                "create_request": {
                    "key": _b64(b"/nodes/"),
                    "range_end": _b64(b"/nodes0"),
                    "start_revision": 10,
                    "progress_notify": True,
                }
            },
            stream=True,
            timeout=(5, 30.0),
        )

    def test_compaction_cancellation_is_not_silently_dropped(self) -> None:
        response = _StreamingResponse(
            [
                {
                    "result": {
                        "canceled": True,
                        "compact_revision": "4711",
                        "cancel_reason": "required revision has been compacted",
                    }
                }
            ]
        )

        with pytest.raises(WatchCompactedError) as error:
            list(EtcdWatchStream(response))

        assert error.value.compact_revision == 4711
        assert isinstance(error.value.compact_revision, int)

    def test_streamed_read_timeout_is_normalized(self) -> None:
        wrapped_timeout = RequestsConnectionError(
            Urllib3ReadTimeoutError(None, None, "read timed out")
        )
        response = _StreamingResponse([], error=wrapped_timeout)

        with pytest.raises(WatchReadTimeoutError):
            list(EtcdWatchStream(response))

    def test_close_is_idempotent(self) -> None:
        response = MagicMock()
        response.raw._fp.fileno.return_value = 42
        transport = MagicMock()
        stream = EtcdWatchStream(response)

        with patch(
            "inference_proxy.discovery.etcd_client.socket.fromfd",
            return_value=transport,
        ) as mock_fromfd:
            stream.close()
            stream.close()

        mock_fromfd.assert_called_once_with(42, socket.AF_INET, socket.SOCK_STREAM)
        transport.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        transport.close.assert_called_once()
        response.close.assert_called_once()


class TestLegacyOperations:
    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_get_prefix_remains_available_for_startup_load(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.get_prefix.return_value = [
            (b'{"endpoint":"node-1:8000"}', {"key": b"/nodes/node-1"})
        ]
        client = EtcdClient(_settings(prefix="/test-nodes/"))

        result = client.get_prefix()

        assert len(result) == 1
        mock_instance.get_prefix.assert_called_once_with("/test-nodes/")

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_get_prefix_accepts_custom_prefix(
        self,
        mock_etcd3_cls: MagicMock,
    ) -> None:
        client = EtcdClient(_settings())

        client.get_prefix("/provisioning/")

        mock_etcd3_cls.return_value.get_prefix.assert_called_once_with("/provisioning/")

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_put_delegates(self, mock_etcd3_cls: MagicMock) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.put.return_value = True
        client = EtcdClient(_settings())

        result = client.put("/nodes/host-1", b'{"endpoint":"host-1:8000"}')

        assert result is True
        mock_instance.put.assert_called_once_with(
            "/nodes/host-1",
            b'{"endpoint":"host-1:8000"}',
        )

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_delete_delegates(self, mock_etcd3_cls: MagicMock) -> None:
        mock_instance = mock_etcd3_cls.return_value
        mock_instance.delete.return_value = True
        client = EtcdClient(_settings())

        result = client.delete("/nodes/host-1")

        assert result is True
        mock_instance.delete.assert_called_once_with("/nodes/host-1")

    @patch("inference_proxy.discovery.etcd_client.Etcd3Client")
    def test_prefix_property(self, mock_etcd3_cls: MagicMock) -> None:
        client = EtcdClient(_settings(prefix="/custom-prefix/"))

        assert client.prefix == "/custom-prefix/"
