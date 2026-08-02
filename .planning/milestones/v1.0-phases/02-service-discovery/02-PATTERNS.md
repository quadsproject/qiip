# Phase 2: Service Discovery - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 11 files (7 new, 4 modified)
**Analogs found:** 11 / 11

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `inference_proxy/discovery/etcd_client.py` | service | request-response | `inference_proxy/config/settings.py` | role-match |
| `inference_proxy/discovery/registry.py` | service | CRUD | `inference_proxy/models/node.py` | role-match |
| `inference_proxy/discovery/serializer.py` | utility | transform | `inference_proxy/models/node.py` | role-match |
| `inference_proxy/discovery/watcher.py` | service | event-driven | `inference_proxy/config/logging.py` | partial-match |
| `inference_proxy/config/dependencies.py` | provider | request-response | `inference_proxy/config/dependencies.py` | exact |
| `inference_proxy/main.py` | config | request-response | `inference_proxy/main.py` | exact |
| `tests/discovery/__init__.py` | test | -- | `tests/models/__init__.py` | exact |
| `tests/discovery/test_etcd_client.py` | test | unit | `tests/config/test_settings.py` | role-match |
| `tests/discovery/test_registry.py` | test | unit | `tests/models/test_node.py` | role-match |
| `tests/discovery/test_serializer.py` | test | unit | `tests/models/test_node.py` | role-match |
| `tests/discovery/test_watcher.py` | test | unit | `tests/config/test_settings.py` | role-match |

## Pattern Assignments

### `inference_proxy/discovery/etcd_client.py` (service, request-response)

**Analog:** `inference_proxy/config/settings.py`

**Purpose:** Thin wrapper around etcd3gw providing typed node operations. Follows Dependency Inversion Principle from CLAUDE.md SOLID requirements.

**Imports pattern** (lines 1-10):
```python
"""Application settings via pydantic-settings.

Sub-models inherit from BaseModel (not BaseSettings) to ensure
nested env var resolution works correctly through the root Settings class.
Only the root Settings class inherits from BaseSettings.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
```

**Pattern for etcd_client.py:**
- Docstring at module level explaining purpose
- Import external dependencies first (etcd3gw)
- Import project modules second (config.settings)
- Class encapsulates single responsibility (etcd communication)

**Core pattern** (lines 19-24):
```python
class EtcdSettings(BaseModel):
    """etcd service discovery configuration."""

    endpoints: list[str] = ["http://localhost:2379"]
    node_prefix: str = "/nodes/"
```

**Pattern for etcd_client.py:**
- Clean class structure with typed attributes
- Docstring on class
- Default values for configuration
- Use BaseModel for value objects
- Initialize from Settings object in constructor

**Type hints and validation:**
```python
# From settings.py - strict typing pattern
class GatewaySettings(BaseModel):
    """Gateway server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
```

**Pattern for etcd_client.py:**
- All attributes have explicit type hints
- No Optional unless truly optional
- Use list[str] not List[str] (modern Python 3.12 syntax)

---

### `inference_proxy/discovery/registry.py` (service, CRUD)

**Analog:** `inference_proxy/models/node.py`

**Purpose:** Thread-safe registry holding discovered nodes. Provides add/remove/get/get_all interface.

**Imports pattern** (lines 1-18):
```python
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

from pydantic import BaseModel, Field
```

**Pattern for registry.py:**
- Comprehensive docstring referencing design decisions
- `from __future__ import annotations` for forward references
- Standard library imports first (threading)
- Third-party imports second
- Project imports last
- Import specific symbols, not modules

**Core domain model pattern** (lines 36-56):
```python
class Node(BaseModel):
    """A vLLM inference node registered in etcd.

    Attributes:
        node_id: Unique identifier for the node.
        endpoint: HTTP endpoint (host:port) for the vLLM server.
        status: Current health status of the node.
        model: Name of the model being served.
        last_heartbeat: Timestamp of the last health check response.
        capabilities: Hardware and serving capabilities.
        active_connections: Number of active inference requests.
    """

    node_id: str
    endpoint: str
    status: NodeStatus = NodeStatus.UNKNOWN
    model: str = ""
    last_heartbeat: datetime | None = None
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)
    active_connections: int = 0
```

**Pattern for registry.py:**
- Detailed docstring with Attributes section
- All fields explicitly typed
- Use `| None` not `Optional[...]`
- Default values for optional fields
- Field(default_factory=...) for mutable defaults
- Clean single-responsibility class

**Type safety pattern** (lines 20-27):
```python
class NodeStatus(StrEnum):
    """Status of a vLLM inference node."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    UNKNOWN = "unknown"
```

**Pattern for registry.py:**
- Use StrEnum for status/state values
- Docstring on enum
- All caps for enum members
- String values match lowercase names

---

### `inference_proxy/discovery/serializer.py` (utility, transform)

**Analog:** `inference_proxy/models/node.py`

**Purpose:** Pure functions converting between etcd key/value pairs and Node domain objects. Handles malformed JSON gracefully.

**Imports pattern** (lines 1-18):
```python
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

from pydantic import BaseModel, Field
```

**Pattern for serializer.py:**
- Module docstring explaining purpose and referencing design decisions
- Import json from stdlib
- Import structlog for error logging
- Import Node from project models
- Pure functions, no classes

**Pydantic validation pattern** (lines 36-56):
```python
class Node(BaseModel):
    """A vLLM inference node registered in etcd.

    Attributes:
        node_id: Unique identifier for the node.
        endpoint: HTTP endpoint (host:port) for the vLLM server.
        status: Current health status of the node.
        model: Name of the model being served.
        last_heartbeat: Timestamp of the last health check response.
        capabilities: Hardware and serving capabilities.
        active_connections: Number of active inference requests.
    """

    node_id: str
    endpoint: str
    status: NodeStatus = NodeStatus.UNKNOWN
    model: str = ""
    last_heartbeat: datetime | None = None
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)
    active_connections: int = 0
```

**Pattern for serializer.py:**
- Use Pydantic's model_validate for parsing
- Rely on Pydantic's validation errors
- Catch json.JSONDecodeError, TypeError, ValueError
- Return None on parse failure (don't raise)
- Use structlog.warning to log skipped nodes

---

### `inference_proxy/discovery/watcher.py` (service, event-driven)

**Analog:** `inference_proxy/config/logging.py`

**Purpose:** Watch thread with reconnection loop. Runs in background, stops on shutdown signal.

**Imports pattern** (lines 1-11):
```python
"""Structured logging configuration using structlog.

Provides JSON output for production and colored console output for
development.  Call ``configure_logging()`` once during application
startup (inside the FastAPI lifespan context manager).
"""

import logging

import structlog
```

**Pattern for watcher.py:**
- Module docstring explaining when/how to use
- Standard library imports first (threading, time)
- Third-party imports second (structlog)
- Project imports last (etcd_client, registry, serializer)

**Function definition pattern** (lines 14-24):
```python
def configure_logging(
    *,
    json_output: bool = False,
    log_level: int = logging.INFO,
) -> None:
    """Configure structlog with the appropriate renderer and log level.

    Args:
        json_output: Use JSON renderer when ``True``, console renderer
            when ``False``.
        log_level: Minimum log level to emit (default ``logging.INFO``).
    """
```

**Pattern for watcher.py:**
- Keyword-only args after `*`
- Default values for all optional params
- Comprehensive docstring with Args section
- Explicit return type annotation (-> None)

**Logging pattern** (lines 26-44):
```python
    renderer: structlog.types.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=json_output,
    )
```

**Pattern for watcher.py:**
- Get logger with `structlog.get_logger()`
- Use `logger.warning()`, `logger.error()`, `logger.info()`
- Include context in log calls: `logger.warning("message", key=value, error=str(exc))`
- Use structured key-value pairs, not f-strings

---

### `inference_proxy/config/dependencies.py` (provider, request-response) - MODIFY

**Analog:** `inference_proxy/config/dependencies.py` (self)

**Purpose:** Add get_registry() dependency following the same pattern as get_settings().

**Existing pattern** (lines 1-18):
```python
"""Dependency injection providers for application configuration.

Settings are provided via ``@lru_cache`` so the same instance is reused
across requests.  In tests, use ``app.dependency_overrides[get_settings]``
to inject test-specific settings -- never call ``get_settings()`` directly
in application code.
"""

from functools import lru_cache

from .settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
```

**Pattern to copy for get_registry():**
- Add import for NodeRegistry from discovery.registry
- Create function with `@lru_cache` decorator
- Return type annotation
- Simple docstring
- One-line implementation
- Add note in module docstring about registry dependency

---

### `inference_proxy/main.py` (config, request-response) - MODIFY

**Analog:** `inference_proxy/main.py` (self)

**Purpose:** Extend lifespan to initialize registry and manage watch thread.

**Existing lifespan pattern** (lines 22-31):
```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown hooks.  Future phases will add
    service discovery initialization, health-check tasks, and
    graceful connection draining here.
    """
    configure_logging()
    yield
```

**Pattern to extend:**
- Import threading module
- Import new discovery modules (etcd_client, registry, watcher)
- Import get_settings from config.dependencies
- Add initialization code before yield
- Store registry in app.state
- Add shutdown code after yield
- Keep configure_logging() first
- Update docstring to remove "Future phases will add service discovery"

**Existing imports pattern** (lines 13-19):
```python
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from inference_proxy.config.logging import configure_logging
```

**Pattern for new imports:**
- Add threading to standard library imports section
- Add new project imports after existing ones
- Keep alphabetical order within sections

---

### `tests/discovery/__init__.py` (test, --) 

**Analog:** `tests/models/__init__.py`

**Expected content:** Empty file (package marker)

---

### `tests/discovery/test_etcd_client.py` (test, unit)

**Analog:** `tests/config/test_settings.py`

**Purpose:** Test etcd client wrapper with mocked etcd3gw.

**Imports pattern** (lines 1-13):
```python
"""Unit tests for configuration settings loading and env var overrides."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from inference_proxy.config.settings import (
    EtcdSettings,
    GatewaySettings,
    RoutingSettings,
    Settings,
)
```

**Pattern for test_etcd_client.py:**
- Module docstring
- `from __future__ import annotations`
- Standard library imports (unittest.mock)
- Third-party test imports (pytest)
- Project imports last
- Import symbols being tested explicitly

**Test class structure** (lines 16-20, 23-27):
```python
class TestDefaultGatewaySettings:
    def test_default_gateway_settings(self) -> None:
        settings = Settings()
        assert settings.gateway.host == "0.0.0.0"
        assert settings.gateway.port == 8080

class TestDefaultEtcdSettings:
    def test_default_etcd_settings(self) -> None:
        settings = Settings()
        assert settings.etcd.endpoints == ["http://localhost:2379"]
        assert settings.etcd.node_prefix == "/nodes/"
```

**Pattern for test_etcd_client.py:**
- One test class per behavior being tested
- Class name: `Test<BehaviorUnderTest>`
- Method name: `test_<specific_behavior>`
- Explicit return type `-> None`
- Arrange-Act-Assert structure
- Direct assertions, not helper methods

**Env var mocking pattern** (lines 39-43):
```python
class TestEnvVarOverrideGatewayPort:
    def test_env_var_override_gateway_port(self, monkeypatch: object) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_GATEWAY__PORT", "9090")  # type: ignore[attr-defined]
        settings = Settings()
        assert settings.gateway.port == 9090
```

**Pattern for test_etcd_client.py:**
- Use pytest fixtures (monkeypatch, mocker)
- Type hint fixtures as object with # type: ignore[attr-defined]
- Mock at the boundary (etcd3gw.Etcd3Client)
- Return mock data matching real format

---

### `tests/discovery/test_registry.py` (test, unit)

**Analog:** `tests/models/test_node.py`

**Purpose:** Test thread-safe registry operations.

**Imports pattern** (lines 1-10):
```python
"""Unit tests for the Node state domain model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inference_proxy.models.node import Node, NodeCapabilities, NodeStatus
```

**Pattern for test_registry.py:**
- Module docstring
- `from __future__ import annotations`
- Import threading for concurrent tests
- Import pytest
- Import NodeRegistry and Node

**Test class structure** (lines 13-19, 30-38):
```python
class TestNodeStatusEnumValues:
    def test_node_status_enum_values(self) -> None:
        assert NodeStatus.HEALTHY == "healthy"
        assert NodeStatus.UNHEALTHY == "unhealthy"
        assert NodeStatus.DRAINING == "draining"
        assert NodeStatus.UNKNOWN == "unknown"
        assert len(NodeStatus) == 4

class TestNodeMinimalCreation:
    def test_node_minimal_creation(self) -> None:
        node = Node(node_id="node-1", endpoint="http://10.0.1.100:8000")
        assert node.node_id == "node-1"
        assert node.endpoint == "http://10.0.1.100:8000"
        assert node.status == NodeStatus.UNKNOWN
        assert node.model == ""
        assert node.last_heartbeat is None
        assert node.active_connections == 0
```

**Pattern for test_registry.py:**
- Test class per behavior
- Minimal test data
- Explicit assertions on all fields
- Test defaults
- Test thread-safety with concurrent access

**Full object test pattern** (lines 41-62):
```python
class TestNodeFullCreation:
    def test_node_full_creation(self) -> None:
        now = datetime.now(tz=timezone.utc)
        node = Node(
            node_id="gpu-node-42",
            endpoint="http://10.0.1.200:8000",
            status=NodeStatus.HEALTHY,
            model="meta-llama/Llama-3-70B",
            last_heartbeat=now,
            capabilities=NodeCapabilities(max_tokens=8192, gpu_memory="80GB"),
            active_connections=5,
        )
        dumped = node.model_dump()
        roundtripped = Node.model_validate(dumped)
        assert roundtripped.node_id == node.node_id
        assert roundtripped.endpoint == node.endpoint
        assert roundtripped.status == node.status
        assert roundtripped.model == node.model
        assert roundtripped.last_heartbeat == node.last_heartbeat
        assert roundtripped.capabilities.max_tokens == 8192
        assert roundtripped.capabilities.gpu_memory == "80GB"
        assert roundtripped.active_connections == 5
```

**Pattern for test_registry.py:**
- Create complete object with all fields
- Test serialization roundtrip if applicable
- Assert equality on all fields

---

### `tests/discovery/test_serializer.py` (test, unit)

**Analog:** `tests/models/test_node.py`

**Purpose:** Test JSON parsing edge cases and error handling.

**Test error handling pattern** (lines 85-89):
```python
class TestNodeRejectsInvalidStatus:
    def test_node_rejects_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            Node(node_id="x", endpoint="y", status="invalid")
```

**Pattern for test_serializer.py:**
- Test valid JSON parsing
- Test malformed JSON returns None
- Test missing fields
- Test invalid field types
- Use pytest.raises for expected errors
- Mock structlog to verify warning logs

---

### `tests/discovery/test_watcher.py` (test, unit)

**Analog:** `tests/config/test_settings.py`

**Purpose:** Test watch event dispatch and reconnection logic.

**Pattern:** Follow test_settings.py structure but:
- Mock etcd3gw watch_prefix to return controlled events
- Test PUT events add nodes
- Test DELETE events remove nodes
- Test reconnection on exception
- Test stop_event stops the loop
- Use threading.Event and timeouts

---

## Shared Patterns

### Dependency Injection
**Source:** `inference_proxy/config/dependencies.py` (lines 1-18)
**Apply to:** All new dependencies (get_registry)

```python
"""Dependency injection providers for application configuration.

Settings are provided via ``@lru_cache`` so the same instance is reused
across requests.  In tests, use ``app.dependency_overrides[get_settings]``
to inject test-specific settings -- never call ``get_settings()`` directly
in application code.
"""

from functools import lru_cache

from .settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
```

**Key elements:**
- `@lru_cache` decorator for singleton behavior
- Module docstring explains override pattern for tests
- Simple function returning instance
- Type annotations on return

### Module Documentation
**Source:** All existing modules
**Apply to:** All new modules

```python
"""Short one-line summary.

Optional longer explanation providing context, usage notes,
and references to design decisions.
"""
```

**Key elements:**
- Triple-quoted docstring at module top
- First line is standalone summary
- Blank line before extended description
- Reference design decisions with "Per D-XX: ..."

### Import Organization
**Source:** `inference_proxy/models/node.py` (lines 12-17)
**Apply to:** All new modules

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field
```

**Order:**
1. `from __future__ import annotations` (if using forward refs)
2. Standard library imports (alphabetical)
3. Blank line
4. Third-party imports (alphabetical)
5. Blank line
6. Project imports (alphabetical, relative imports)

### Test Fixtures
**Source:** `tests/conftest.py` (lines 22-28)
**Apply to:** All new test fixtures

```python
@pytest.fixture
def test_settings() -> Settings:
    """Return a Settings instance with test-safe defaults."""
    return Settings(
        gateway=GatewaySettings(host="127.0.0.1", port=9999),
        etcd=EtcdSettings(endpoints=["http://localhost:2379"], node_prefix="/test-nodes/"),
        routing=RoutingSettings(strategy="least_connections", max_retries=1, timeout=5),
    )
```

**Key elements:**
- `@pytest.fixture` decorator
- Return type annotation
- Docstring
- Test-safe values (non-default ports, test prefixes)

### Logging with structlog
**Source:** `inference_proxy/config/logging.py` (lines 8-10)
**Apply to:** All modules that log

```python
import structlog

logger = structlog.get_logger()
```

**Usage pattern from RESEARCH.md:**
```python
logger.warning("etcd watch disconnected, reconnecting", retry_delay=retry_delay)
logger.error("failed to parse node", key=key, error=str(exc))
logger.info("node added", node_id=node.node_id, endpoint=node.endpoint)
```

**Key elements:**
- Get logger at module level
- Use structured key-value pairs
- Include context (keys, errors, values)
- Don't use f-strings, use key=value

### Type Hints (Python 3.12 Style)
**Source:** All existing modules
**Apply to:** All new code

```python
# Use built-in types, not typing module
def get_all(self) -> list[Node]:  # NOT List[Node]
def get(self, node_id: str) -> Node | None:  # NOT Optional[Node]
```

**Key rules:**
- Use `list[T]` not `List[T]`
- Use `dict[K, V]` not `Dict[K, V]`
- Use `X | None` not `Optional[X]`
- Use `X | Y` not `Union[X, Y]`
- Always annotate function signatures

### SOLID Principles (from CLAUDE.md)
**Apply to:** All new code

**Single Responsibility:**
- EtcdClient: only etcd communication
- NodeRegistry: only node storage
- Serializer: only JSON parsing
- Watcher: only watch loop

**Dependency Inversion:**
- Depend on abstractions (NodeRegistry interface), not concrete etcd3gw
- Only etcd_client.py imports etcd3gw
- All other modules depend on the wrapper

**Interface Segregation:**
- NodeRegistry has focused interface: add/remove/get/get_all
- No unused methods

## No Analog Found

No files lack analogs. All patterns found in existing codebase.

## Metadata

**Analog search scope:** 
- `/home/developer/Sources/inference-proxy/inference_proxy/`
- `/home/developer/Sources/inference-proxy/tests/`

**Files scanned:** 20 Python files

**Pattern extraction date:** 2026-06-11

**Key patterns identified:**
1. **Dependency Injection:** `@lru_cache` pattern for singletons, override in tests
2. **Pydantic Models:** BaseModel for data, Field(default_factory=...) for mutable defaults
3. **Module Structure:** Docstring, imports (stdlib → third-party → project), implementation
4. **Test Structure:** One class per behavior, explicit assertions, use fixtures
5. **Threading Pattern:** Use threading.Lock for OS thread safety (not asyncio.Lock)
6. **Logging:** structlog with structured key-value pairs
7. **Type Hints:** Modern Python 3.12 syntax (list[T], X | None)
8. **SOLID:** Single responsibility classes, dependency inversion via wrappers
