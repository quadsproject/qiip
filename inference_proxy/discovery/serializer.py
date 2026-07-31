"""Serialize and deserialize Node objects for etcd storage.

Provides pure functions for converting between etcd key-value pairs and
Node domain objects.

Per D-11: Separate serializer module with ``node_from_etcd`` and
``node_to_etcd`` conversion functions.
Per D-12: Handles missing/malformed JSON gracefully -- logs warning and
returns None rather than crashing the gateway.
Per D-02: Node ID is derived from the etcd key (last segment after prefix),
not stored in the JSON value.
"""

from __future__ import annotations

import json

import structlog
from pydantic import ValidationError

from inference_proxy.models.endpoint import EndpointPolicy, parse_endpoint
from inference_proxy.models.node import Node

logger = structlog.get_logger()


def node_from_etcd(
    key: str | bytes,
    value: bytes,
    prefix: str,
    *,
    endpoint_policy: EndpointPolicy,
) -> Node | None:
    """Parse an etcd key-value pair into a Node.

    Handles both ``bytes`` and ``str`` keys defensively (Pitfall 2).

    Args:
        key: The etcd key (e.g., ``/nodes/node-abc``).
        value: The raw JSON bytes from etcd.
        prefix: The configured node prefix (e.g., ``/nodes/``).

    Returns:
        A ``Node`` instance, or ``None`` if parsing fails.
    """
    raw_endpoint: object = None
    try:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        node_id = key.removeprefix(prefix)
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("node payload must be a JSON object")
        raw_endpoint = data.get("endpoint")
        if not isinstance(raw_endpoint, str):
            raise ValueError("node endpoint must be a string")
        data["endpoint"] = endpoint_policy.normalize(raw_endpoint)
        return Node(node_id=node_id, **data)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        logger.warning(
            "skipping malformed node",
            key=key,
            endpoint=raw_endpoint,
            error=str(exc),
        )
        return None


def node_to_etcd(node: Node, prefix: str) -> tuple[str, bytes]:
    """Convert a Node to an etcd key-value pair.

    The node_id is excluded from the JSON value (per D-02) and used
    to compose the key as ``prefix + node_id`` (per D-01).

    Args:
        node: The Node to serialize.
        prefix: The configured node prefix (e.g., ``/nodes/``).

    Returns:
        A tuple of ``(key_string, json_bytes)``.
    """
    key = prefix + node.node_id
    data = node.model_dump(exclude={"node_id"}, mode="json")
    data["endpoint"] = parse_endpoint(node.endpoint).origin
    value_bytes = json.dumps(data).encode("utf-8")
    return key, value_bytes
