"""Thin wrapper around etcd3gw providing typed node operations.

This module is the **sole consumer** of ``etcd3gw`` in the codebase,
following the Dependency Inversion Principle (DIP): all other modules
depend on this wrapper rather than importing ``etcd3gw`` directly.

Per D-13: Encapsulates connection configuration and provides typed
methods for node operations.
Per D-14: Created from ``EtcdSettings`` (endpoints, node_prefix).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import structlog
from etcd3gw.client import Etcd3Client

from inference_proxy.config.settings import EtcdSettings

if TYPE_CHECKING:
    from etcd3gw.types import Event, KeyValue

logger = structlog.get_logger()


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

    def watch_prefix(self) -> tuple[Iterator[Event], Callable[[], None]]:
        """Start watching for changes under the configured node prefix.

        Returns:
            A tuple of ``(events_iterator, cancel_fn)``.  The iterator
            blocks on an internal ``queue.Queue`` and yields event dicts.
            Call ``cancel_fn()`` to stop the watch.
        """
        return self._client.watch_prefix(self._prefix)
