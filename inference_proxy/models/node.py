"""Node state domain model for vLLM inference nodes.

Represents the state of a vLLM backend node as tracked in etcd.
NodeStatus is a StrEnum for type-safe status values.
Node and NodeCapabilities are Pydantic models for validation.

Per D-15: No serialization methods on the model -- serialization
is a separate concern handled in a future phase.
Per D-16: The ``model`` field is ``str``, not ``list[str]``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class NodeStatus(StrEnum):
    """Status of a vLLM inference node."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    PROVISIONING = "provisioning"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NodeCapabilities(BaseModel):
    """Hardware and serving capabilities of a node."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = 4096
    gpu_memory: str = ""


class Node(BaseModel):
    """A vLLM inference node registered in etcd.

    Instances are immutable (``frozen=True``) to prevent external
    mutation of registry entries without acquiring the registry lock.
    Use ``model_copy(update={...})`` to create modified copies.

    Attributes:
        node_id: Unique identifier for the node.
        endpoint: HTTP endpoint (host:port) for the vLLM server.
        status: Current health status of the node.
        model: Name of the model being served.
        last_heartbeat: Timestamp of the last health check response.
        capabilities: Hardware and serving capabilities.
        active_connections: Number of active inference requests.
        managed: Whether the proxy owns the node lifecycle. Externally
            registered nodes must opt in explicitly.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str
    endpoint: str
    status: NodeStatus = NodeStatus.UNKNOWN
    model: str = ""
    last_heartbeat: datetime | None = None
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)
    active_connections: int = 0
    managed: bool = False
