# Phase 1: Foundation - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 22
**Analogs found:** 0 / 22 (greenfield project -- no existing source code)

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `pyproject.toml` | config | -- | none (greenfield) | no-analog |
| `.python-version` | config | -- | none (greenfield) | no-analog |
| `.env.example` | config | -- | none (greenfield) | no-analog |
| `.gitignore` | config | -- | none (greenfield) | no-analog |
| `inference_proxy/__init__.py` | module-init | -- | none (greenfield) | no-analog |
| `inference_proxy/main.py` | controller | request-response | none (greenfield) | no-analog |
| `inference_proxy/config/__init__.py` | module-init | -- | none (greenfield) | no-analog |
| `inference_proxy/config/settings.py` | config | -- | none (greenfield) | no-analog |
| `inference_proxy/config/dependencies.py` | provider | request-response | none (greenfield) | no-analog |
| `inference_proxy/models/__init__.py` | module-init | -- | none (greenfield) | no-analog |
| `inference_proxy/models/openai.py` | model | transform | none (greenfield) | no-analog |
| `inference_proxy/models/node.py` | model | transform | none (greenfield) | no-analog |
| `inference_proxy/api/__init__.py` | module-init (stub) | -- | none (greenfield) | no-analog |
| `inference_proxy/discovery/__init__.py` | module-init (stub) | -- | none (greenfield) | no-analog |
| `inference_proxy/routing/__init__.py` | module-init (stub) | -- | none (greenfield) | no-analog |
| `inference_proxy/resilience/__init__.py` | module-init (stub) | -- | none (greenfield) | no-analog |
| `tests/__init__.py` | module-init | -- | none (greenfield) | no-analog |
| `tests/conftest.py` | test-fixture | -- | none (greenfield) | no-analog |
| `tests/test_app.py` | test | request-response | none (greenfield) | no-analog |
| `tests/models/test_openai.py` | test | transform | none (greenfield) | no-analog |
| `tests/models/test_node.py` | test | transform | none (greenfield) | no-analog |
| `tests/config/test_settings.py` | test | -- | none (greenfield) | no-analog |

## Pattern Assignments

Since this is a greenfield project with zero existing source code, all patterns are sourced from RESEARCH.md (01-RESEARCH.md) which provides verified, concrete code examples from official documentation. Each pattern below includes the canonical reference and the code to replicate.

---

### `pyproject.toml` (config)

**Source:** RESEARCH.md "Code Examples > pyproject.toml Configuration"
**Canonical refs:** [uv Project Configuration](https://docs.astral.sh/uv/concepts/projects/config/)

**Full pattern:**
```toml
[project]
name = "inference-proxy"
version = "0.1.0"
description = "QUADS LLM Inference Proxy - routes requests to vLLM nodes"
readme = "README.md"
requires-python = ">=3.12,<3.14"
dependencies = [
    "fastapi>=0.135,<1.0",
    "uvicorn[standard]>=0.45",
    "pydantic>=2.10,<3.0",
    "pydantic-settings>=2.14",
    "structlog>=26.1.0",
]

[dependency-groups]
dev = [
    "ruff>=0.15",
    "mypy>=2.1",
    "pytest>=8.0",
    "pytest-asyncio>=1.4",
    "pytest-httpx>=0.36",
    "coverage>=7.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]
```

**Open question from RESEARCH.md:** A `[build-system]` section (e.g., hatchling) may be needed so `uv run uvicorn inference_proxy.main:app` can import the package. The planner should resolve this -- recommendation is to add hatchling build-system.

---

### `inference_proxy/main.py` (controller, request-response)

**Source:** RESEARCH.md "Pattern 1: App Factory with Lifespan"
**Canonical refs:** [FastAPI Settings docs](https://fastapi.tiangolo.com/advanced/settings/), [FastAPI Lifespan docs](https://fastapi.tiangolo.com/advanced/events/)

**App factory pattern:**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize structlog, etc.
    yield
    # Shutdown: cleanup

def create_app() -> FastAPI:
    app = FastAPI(
        title="QUADS LLM Inference Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Register routers here as phases add them
    return app

# Module-level for `uv run uvicorn inference_proxy.main:app`
app = create_app()
```

**Key decisions:**
- D-08: Settings via DI, not global state
- Factory pattern enables test isolation (Claude's Discretion -- recommended by RESEARCH.md)
- `lifespan` context manager, NOT deprecated `@app.on_event()` decorators

---

### `inference_proxy/config/settings.py` (config)

**Source:** RESEARCH.md "Pattern 2: Settings with Dependency Injection"
**Canonical refs:** [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

**Settings composition pattern:**
```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class GatewaySettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080

class EtcdSettings(BaseModel):
    endpoints: list[str] = ["http://localhost:2379"]
    node_prefix: str = "/nodes/"

class RoutingSettings(BaseModel):
    strategy: str = "least_connections"
    health_check_interval: int = 30
    max_retries: int = 3
    timeout: int = 30

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_PROXY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    gateway: GatewaySettings = GatewaySettings()
    etcd: EtcdSettings = EtcdSettings()
    routing: RoutingSettings = RoutingSettings()
```

**Critical anti-pattern (from RESEARCH.md Pitfall 2):**
- Sub-models (`GatewaySettings`, `EtcdSettings`, `RoutingSettings`) MUST inherit from `pydantic.BaseModel`, NOT from `BaseSettings`
- Only the root `Settings` class inherits from `BaseSettings`
- Violating this causes nested env var resolution to break silently

**Key decisions:**
- D-05: Split by domain, composed into root `Settings`
- D-06: Env prefix `INFERENCE_PROXY_`, nested delimiter `__`
- D-07: Defaults match `.env.example` values

---

### `inference_proxy/config/dependencies.py` (provider, request-response)

**Source:** RESEARCH.md "Pattern 2: Settings with Dependency Injection"
**Canonical refs:** [FastAPI Settings docs](https://fastapi.tiangolo.com/advanced/settings/)

**DI provider pattern:**
```python
from functools import lru_cache
from .settings import Settings

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Critical anti-pattern (from RESEARCH.md Pitfall 3):**
- Never call `get_settings()` directly in application code -- always access via `Depends(get_settings)`
- In tests, clear cache between tests: `get_settings.cache_clear()`
- `dependency_overrides` bypasses the function, but direct imports use the cached value

**Key decisions:**
- D-08: `@lru_cache` + `Depends()` pattern

---

### `inference_proxy/models/openai.py` (model, transform)

**Source:** RESEARCH.md "Pattern 3: OpenAI Request Models" + "Pattern 5: OpenAI Response Models"
**Canonical refs:** [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [OpenAI Streaming Events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)

**Request model pattern (extra='allow'):**
```python
from pydantic import BaseModel, ConfigDict, Field

class ChatMessage(BaseModel):
    role: str
    content: str | None = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
```

**Response model pattern:**
```python
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
```

**Streaming chunk pattern:**
```python
class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
```

**Error schema pattern:**
```python
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | int | None = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
```

**Key decisions:**
- D-09: vLLM-relevant subset only (no tools/function calling)
- D-10: `extra='allow'` on request models for forward compatibility
- D-11: Both request AND response models in Phase 1
- D-12: Both `/v1/chat/completions` AND `/v1/completions` (text completion has `prompt: str` instead of `messages`, and `text` instead of `message` in choices)
- RESEARCH.md recommends keeping chat and text completion models fully separate (no shared base class) per SRP

---

### `inference_proxy/models/node.py` (model, transform)

**Source:** RESEARCH.md "Pattern 4: Node State Model with StrEnum"
**Canonical refs:** PLAN.md "etcd Service Registry > Data Schema"

**Node model pattern:**
```python
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field

class NodeStatus(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    UNKNOWN = "unknown"

class NodeCapabilities(BaseModel):
    max_tokens: int = 4096
    gpu_memory: str = ""

class Node(BaseModel):
    node_id: str
    endpoint: str
    status: NodeStatus = NodeStatus.UNKNOWN
    model: str = ""
    last_heartbeat: datetime | None = None
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)
    active_connections: int = 0
```

**Key decisions:**
- D-13: `StrEnum` with four values
- D-14: PLAN.md fields + routing metadata (`active_connections`, `node_id`). `NodeCapabilities` as nested model.
- D-15: Serialization (etcd JSON <-> Node) is a SEPARATE module (not in this file). SRP.
- D-16: `model` is `str`, not `list[str]`. One model per node.

**PLAN.md etcd data schema (the source of truth for field names):**
```json
{
  "endpoint": "http://10.0.1.100:8000",
  "status": "healthy",
  "model": "llama-2-7b",
  "last_heartbeat": "2025-09-12T10:30:00Z",
  "capabilities": {
    "max_tokens": 4096,
    "gpu_memory": "24GB"
  }
}
```

---

### `tests/conftest.py` (test-fixture)

**Source:** RESEARCH.md "Code Examples > conftest.py with App and Settings Fixtures"
**Canonical refs:** [FastAPI Testing Dependencies](https://fastapi.tiangolo.com/advanced/testing-dependencies/)

**Test fixture pattern:**
```python
import pytest
from fastapi.testclient import TestClient

from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings, GatewaySettings
from inference_proxy.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        gateway=GatewaySettings(host="127.0.0.1", port=9999),
    )


@pytest.fixture
def app(test_settings: Settings):
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: test_settings
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)
```

**Key patterns:**
- `dependency_overrides` for settings injection in tests
- `yield` + cleanup in `app` fixture
- Separate `test_settings` fixture for override customization
- `TestClient` wraps the app (synchronous test client for non-streaming tests)

---

### `tests/test_app.py` (test, request-response)

**Source:** RESEARCH.md validation architecture (SC-1, SC-2)

**Smoke test pattern:**
```python
def test_app_starts(client):
    """SC-2: uvicorn starts FastAPI app on a port."""
    response = client.get("/")  # or /health
    assert response.status_code in (200, 404)  # app is responsive
```

**Key patterns:**
- Uses `client` fixture from conftest.py
- Tests that the app is reachable and responsive
- Health endpoint pattern: `GET /health` returning 200

---

### `tests/models/test_openai.py` (test, transform)

**Source:** RESEARCH.md validation architecture (SC-3a)

**Model validation test pattern:**
```python
from inference_proxy.models.openai import ChatCompletionRequest, ChatMessage

def test_valid_chat_completion_request():
    req = ChatCompletionRequest(
        model="llama-2-7b",
        messages=[ChatMessage(role="user", content="Hello!")],
    )
    assert req.model == "llama-2-7b"
    assert len(req.messages) == 1

def test_extra_fields_allowed():
    """D-10: unknown fields pass through to vLLM."""
    req = ChatCompletionRequest(
        model="llama-2-7b",
        messages=[ChatMessage(role="user", content="Hello!")],
        custom_vllm_param=42,
    )
    assert req.model_extra == {"custom_vllm_param": 42}
```

---

### `tests/models/test_node.py` (test, transform)

**Source:** RESEARCH.md validation architecture (SC-3b)

**Node model test pattern:**
```python
from inference_proxy.models.node import Node, NodeStatus, NodeCapabilities

def test_node_default_status():
    node = Node(node_id="node-1", endpoint="http://10.0.1.100:8000")
    assert node.status == NodeStatus.UNKNOWN

def test_node_status_enum_values():
    assert NodeStatus.HEALTHY == "healthy"
    assert NodeStatus.DRAINING == "draining"
```

---

### `tests/config/test_settings.py` (test)

**Source:** RESEARCH.md validation architecture (SC-3c)

**Settings test pattern:**
```python
from inference_proxy.config.settings import Settings, GatewaySettings

def test_default_settings():
    settings = Settings()
    assert settings.gateway.host == "0.0.0.0"
    assert settings.gateway.port == 8080

def test_env_var_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROXY_GATEWAY__PORT", "9090")
    settings = Settings()
    assert settings.gateway.port == 9090
```

---

### Stub `__init__.py` files (module-init)

**Files:** `inference_proxy/__init__.py`, `inference_proxy/config/__init__.py`, `inference_proxy/models/__init__.py`, `inference_proxy/api/__init__.py`, `inference_proxy/discovery/__init__.py`, `inference_proxy/routing/__init__.py`, `inference_proxy/resilience/__init__.py`, `tests/__init__.py`, `tests/models/__init__.py`, `tests/config/__init__.py`

**Pattern:** Empty files or minimal docstrings. Required by D-04 for import resolution.

**Anti-pattern (from RESEARCH.md Pitfall 4):** Missing `__init__.py` causes `ModuleNotFoundError`. All subdirectories must have them.

---

## Shared Patterns

### structlog Configuration
**Source:** RESEARCH.md "Code Examples > structlog Configuration"
**Apply to:** `inference_proxy/main.py` (called during lifespan startup)

```python
import logging
import structlog

def configure_logging(*, json_output: bool = False, log_level: int = logging.INFO) -> None:
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

**Note:** This may warrant its own module (e.g., `inference_proxy/config/logging.py`) per SRP. The planner should decide whether to put it in `main.py` lifespan or extract it.

### Pydantic v2 Model Configuration
**Apply to:** All model files (`openai.py`, `node.py`)
**Convention:** Use `model_config = ConfigDict(...)` class variable, NOT Pydantic v1 `class Config:` inner class.

### pytest-asyncio Auto Mode
**Apply to:** All async test files
**Convention:** No `@pytest.mark.asyncio` markers needed. Set in `pyproject.toml` `[tool.pytest.ini_options]` as `asyncio_mode = "auto"`.

### Import Conventions
**Apply to:** All source files
**Convention:** Python 3.12 union types (`str | None` not `Optional[str]`), `from __future__ import annotations` not needed.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| ALL 22 files | various | various | Greenfield project -- zero existing source code. All patterns sourced from RESEARCH.md reference examples, official documentation, and PLAN.md architecture. |

**Planner guidance:** Since there are no codebase analogs, the RESEARCH.md code examples serve as the canonical patterns. Each pattern above includes the concrete code excerpt and the decision IDs (D-01 through D-16) that constrain the implementation. The planner should copy these patterns directly into plan action steps.

---

## PLAN.md Field Mapping (Node Model)

The Node model must match the etcd data schema from PLAN.md. This table maps PLAN.md fields to Pydantic model fields:

| PLAN.md etcd field | Node model field | Type | Notes |
|--------------------|-----------------|------|-------|
| `endpoint` | `endpoint` | `str` | URL like `http://10.0.1.100:8000` |
| `status` | `status` | `NodeStatus` (StrEnum) | D-13: healthy/unhealthy/draining/unknown |
| `model` | `model` | `str` | D-16: single model per node |
| `last_heartbeat` | `last_heartbeat` | `datetime \| None` | ISO 8601 format |
| `capabilities.max_tokens` | `capabilities.max_tokens` | `int` | Nested in NodeCapabilities |
| `capabilities.gpu_memory` | `capabilities.gpu_memory` | `str` | e.g., "24GB" |
| (not in PLAN.md) | `node_id` | `str` | D-14: routing metadata, corresponds to etcd key |
| (not in PLAN.md) | `active_connections` | `int` | D-14: routing metadata for load balancing |

## Gateway Config Field Mapping

The Settings model must cover the gateway.yaml fields from PLAN.md Appendix:

| PLAN.md gateway.yaml field | Settings path | Type | Default |
|---------------------------|--------------|------|---------|
| `gateway.host` | `settings.gateway.host` | `str` | `"0.0.0.0"` |
| `gateway.port` | `settings.gateway.port` | `int` | `8080` |
| `etcd.endpoints` | `settings.etcd.endpoints` | `list[str]` | `["http://localhost:2379"]` |
| `routing.strategy` | `settings.routing.strategy` | `str` | `"least_connections"` |
| `routing.health_check_interval` | `settings.routing.health_check_interval` | `int` | `30` |
| `routing.max_retries` | `settings.routing.max_retries` | `int` | `3` |
| `routing.timeout` | `settings.routing.timeout` | `int` | `30` |

## Metadata

**Analog search scope:** `/home/developer/Sources/inference-proxy/` (excluding `.git/`, `.planning/`)
**Files scanned:** 2 (PLAN.md, CLAUDE.md -- the only non-planning files in the repo)
**Python source files found:** 0
**Pattern extraction date:** 2026-06-10
**Pattern source:** RESEARCH.md code examples (verified against official documentation)
