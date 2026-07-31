"""Thin wrapper around etcd3gw providing typed node operations.

This module is the **sole consumer** of ``etcd3gw`` in the codebase,
following the Dependency Inversion Principle (DIP): all other modules
depend on this wrapper rather than importing ``etcd3gw`` directly.

Per D-13: Encapsulates connection configuration and provides typed
methods for node operations.
Per D-14: Created from ``EtcdSettings`` (endpoints, node_prefix).
"""

from __future__ import annotations

import base64
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse

import structlog
from etcd3gw.client import Etcd3Client
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout as RequestsReadTimeout
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

from inference_proxy.config.settings import EtcdSettings

if TYPE_CHECKING:
    from etcd3gw.types import KeyValue

logger = structlog.get_logger()

# Keep connection establishment fast while bounding an otherwise silent,
# half-open watch independently.
_WATCH_READ_TIMEOUT_SECONDS = 30.0


class EtcdProtocolError(RuntimeError):
    """The etcd gateway returned a response that violates its JSON contract."""


class WatchCompactedError(RuntimeError):
    """The requested watch revision is no longer available."""

    def __init__(self, compact_revision: int, reason: str = "") -> None:
        detail = f"watch revision compacted at {compact_revision}"
        if reason:
            detail = f"{detail}: {reason}"
        super().__init__(detail)
        self.compact_revision = compact_revision
        self.reason = reason


class WatchCanceledError(RuntimeError):
    """The etcd server canceled a watch for a non-compaction reason."""


class WatchReadTimeoutError(TimeoutError):
    """The watch stream produced no data before its idle read deadline."""


@dataclass(frozen=True)
class EtcdRecord:
    """A key/value record returned by a revisioned prefix snapshot."""

    key: bytes
    value: bytes
    mod_revision: int


@dataclass(frozen=True)
class EtcdSnapshot:
    """A consistent prefix snapshot and its authoritative store revision."""

    records: tuple[EtcdRecord, ...]
    revision: int


@dataclass(frozen=True)
class EtcdEvent:
    """A normalized PUT or DELETE event from an etcd watch response."""

    key: bytes
    value: bytes | None
    mod_revision: int
    is_delete: bool


@dataclass(frozen=True)
class EtcdWatchBatch:
    """All events delivered in one etcd watch response."""

    events: tuple[EtcdEvent, ...]
    revision: int


class _StreamingResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def iter_lines(self, *, decode_unicode: bool = False) -> Iterator[bytes]: ...

    def close(self) -> None: ...


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: object, field: str) -> bytes:
    if not isinstance(value, str):
        raise EtcdProtocolError(f"{field} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise EtcdProtocolError(f"{field} is not valid base64") from exc


def _revision(value: object, field: str) -> int:
    """Normalize proto3 JSON int64 values, which are encoded as strings."""
    if isinstance(value, bool):
        raise EtcdProtocolError(f"{field} must be an integer")
    if not isinstance(value, (int, str)):
        raise EtcdProtocolError(f"{field} must be an integer")
    try:
        revision = int(value)
    except (TypeError, ValueError) as exc:
        raise EtcdProtocolError(f"{field} must be an integer") from exc
    if revision < 0:
        raise EtcdProtocolError(f"{field} must not be negative")
    return revision


def _prefix_range_end(prefix: bytes) -> bytes:
    """Return etcd's exclusive range end for all keys under *prefix*."""
    end = bytearray(prefix)
    for index in range(len(end) - 1, -1, -1):
        if end[index] < 0xFF:
            end[index] += 1
            return bytes(end[: index + 1])
    return b"\0"


def _header_revision(response: dict[str, Any], context: str) -> int:
    header = response.get("header")
    if not isinstance(header, dict):
        raise EtcdProtocolError(f"{context} response is missing header")
    return _revision(header.get("revision"), f"{context} header.revision")


def _parse_kv(
    raw_kv: object,
    context: str,
    *,
    value_required: bool,
) -> tuple[bytes, bytes | None, int]:
    if not isinstance(raw_kv, dict):
        raise EtcdProtocolError(f"{context} kv must be an object")
    value = None
    if value_required:
        value = _decode(raw_kv.get("value", ""), f"{context} kv.value")
    return (
        _decode(raw_kv.get("key"), f"{context} kv.key"),
        value,
        _revision(raw_kv.get("mod_revision"), f"{context} kv.mod_revision"),
    )


class EtcdWatchStream:
    """Raw, closeable etcd watch stream retaining response boundaries."""

    def __init__(self, response: _StreamingResponse) -> None:
        self._response = response
        self._close_lock = threading.Lock()
        self._closed = False

    def __iter__(self) -> Iterator[EtcdWatchBatch]:
        lines = self._response.iter_lines(decode_unicode=False)
        while True:
            try:
                line = next(lines)
            except StopIteration:
                return
            except RequestsReadTimeout as exc:
                raise WatchReadTimeoutError("etcd watch read timed out") from exc
            except RequestsConnectionError as exc:
                if any(
                    isinstance(reason, Urllib3ReadTimeoutError) for reason in exc.args
                ):
                    raise WatchReadTimeoutError("etcd watch read timed out") from exc
                raise
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EtcdProtocolError("malformed etcd watch response") from exc
            if not isinstance(envelope, dict) or not isinstance(
                envelope.get("result"), dict
            ):
                raise EtcdProtocolError("watch response is missing result")
            result = cast(dict[str, Any], envelope["result"])

            if result.get("canceled"):
                compact_revision = _revision(
                    result.get("compact_revision", 0),
                    "watch compact_revision",
                )
                reason = str(result.get("cancel_reason", ""))
                if compact_revision:
                    raise WatchCompactedError(compact_revision, reason)
                raise WatchCanceledError(reason or "etcd watch canceled")

            # The create acknowledgement is not a progress guarantee. Advancing
            # the resume point here could skip historical events still queued.
            if "created" in result:
                if result["created"] is True:
                    continue
                raise WatchCanceledError("etcd could not create watch")

            response_revision = _header_revision(result, "watch")

            events: list[EtcdEvent] = []
            raw_events = result.get("events", [])
            if not isinstance(raw_events, list):
                raise EtcdProtocolError("watch events must be a list")
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    raise EtcdProtocolError("watch event must be an object")
                is_delete = raw_event.get("type", "PUT") == "DELETE"
                key, value, mod_revision = _parse_kv(
                    raw_event.get("kv"),
                    "watch",
                    value_required=not is_delete,
                )
                events.append(
                    EtcdEvent(
                        key=key,
                        value=value,
                        mod_revision=mod_revision,
                        is_delete=is_delete,
                    )
                )
            if any(event.mod_revision > response_revision for event in events):
                raise EtcdProtocolError(
                    "watch event revision exceeds response revision"
                )
            yield EtcdWatchBatch(tuple(events), response_revision)

    def close(self) -> None:
        """Interrupt the socket read and close the HTTP response exactly once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                raw_response = cast(Any, self._response).raw
                file_pointer = raw_response._fp
                file_descriptor = file_pointer.fileno()
                # fromfd() duplicates the descriptor. shutdown() wakes the
                # original blocked recv; transport.close() closes only the
                # duplicate, then response.close() below closes the original.
                transport = socket.fromfd(
                    file_descriptor,
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                finally:
                    transport.close()
            except (AttributeError, OSError, TypeError, ValueError):
                # Test doubles and already-closed responses may not expose a
                # live urllib3 socket. response.close() remains the fallback.
                pass
            self._response.close()


class EtcdClient:
    """Wrapper around ``etcd3gw.Etcd3Client`` for node discovery.

    Parses the first endpoint URL from ``EtcdSettings`` to extract
    host, port, and protocol for the underlying etcd HTTP gateway
    client.

    Attributes:
        prefix: The configured node key prefix (e.g., ``/nodes/``).
    """

    def __init__(self, settings: EtcdSettings) -> None:
        if len(settings.endpoints) > 1:
            logger.warning(
                "multiple etcd endpoints configured but only the first is used",
                endpoint=settings.endpoints[0],
                ignored=settings.endpoints[1:],
            )
        endpoint = settings.endpoints[0]
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"Invalid etcd endpoint URL: '{endpoint}'. "
                f"Must include scheme (e.g., 'http://etcd.internal:2379')"
            )
        self._client = Etcd3Client(
            host=parsed.hostname,
            port=parsed.port or 2379,
            protocol=parsed.scheme,
            timeout=5,
        )
        self._prefix = settings.node_prefix

    @property
    def prefix(self) -> str:
        """Return the configured node key prefix."""
        return self._prefix

    def get_prefix(self, prefix: str | None = None) -> list[tuple[bytes, KeyValue]]:
        """Fetch all key-value pairs under a prefix.

        Args:
            prefix: The key prefix to scan.  Defaults to the configured
                node prefix when ``None``.

        Returns:
            A list of ``(value_bytes, metadata_dict)`` tuples where
            ``metadata_dict`` contains the key under ``metadata["key"]``.
        """
        return self._client.get_prefix(prefix or self._prefix)

    def get_snapshot(self, prefix: str | None = None) -> EtcdSnapshot:
        """Fetch a prefix snapshot with the range response's store revision."""
        raw_prefix = (prefix or self._prefix).encode("utf-8")
        response = self._client.post(
            self._client.get_url("/kv/range"),
            json={
                "key": _encode(raw_prefix),
                "range_end": _encode(_prefix_range_end(raw_prefix)),
            },
        )
        if not isinstance(response, dict):
            raise EtcdProtocolError("range response must be an object")
        response = cast(dict[str, Any], response)
        snapshot_revision = _header_revision(response, "range")

        raw_records = response.get("kvs", [])
        if not isinstance(raw_records, list):
            raise EtcdProtocolError("range kvs must be a list")
        records: list[EtcdRecord] = []
        for raw_record in raw_records:
            key, value, mod_revision = _parse_kv(
                raw_record,
                "range",
                value_required=True,
            )
            if value is None:
                raise EtcdProtocolError("range kv is missing value")
            records.append(
                EtcdRecord(
                    key=key,
                    value=value,
                    mod_revision=mod_revision,
                )
            )
        if any(record.mod_revision > snapshot_revision for record in records):
            raise EtcdProtocolError("range kv revision exceeds snapshot revision")
        return EtcdSnapshot(tuple(records), snapshot_revision)

    def put(self, key: str, value: str | bytes) -> bool:
        """Put a key-value pair into etcd.

        Args:
            key: The full key (e.g., ``/nodes/hostname``).
            value: The value to store (typically JSON-encoded).

        Returns:
            True on success.
        """
        return self._client.put(key, value)

    def delete(self, key: str) -> bool:
        """Delete a key from etcd.

        Args:
            key: The full key to delete (e.g., ``/nodes/hostname``).

        Returns:
            True if the key was deleted.
        """
        return self._client.delete(key)

    def watch_prefix(self, *, start_revision: int) -> EtcdWatchStream:
        """Open a raw prefix watch beginning at *start_revision*.

        The raw stream is consumed here instead of through etcd3gw's public
        iterator because that iterator has an untimed queue read, discards
        watch-response batch boundaries, and ignores compaction cancellation.
        """
        raw_prefix = self._prefix.encode("utf-8")
        response = cast(
            _StreamingResponse,
            self._client.session.post(
                self._client.get_url("/watch"),
                json={
                    "create_request": {
                        "key": _encode(raw_prefix),
                        "range_end": _encode(_prefix_range_end(raw_prefix)),
                        "start_revision": start_revision,
                        "progress_notify": True,
                    }
                },
                stream=True,
                timeout=(self._client.timeout, _WATCH_READ_TIMEOUT_SECONDS),
            ),
        )
        response.raise_for_status()
        return EtcdWatchStream(response)
