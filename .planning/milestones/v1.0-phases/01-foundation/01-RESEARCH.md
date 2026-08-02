# Phase 1: Foundation - Research

**Researched:** 2026-06-10
**Domain:** Python project scaffolding, FastAPI application setup, Pydantic data modeling, test infrastructure
**Confidence:** HIGH

## Summary

Phase 1 is a greenfield scaffolding phase that establishes the buildable, runnable, testable project skeleton for the QUADS LLM Inference Proxy. The deliverables are: a `pyproject.toml`-managed uv project targeting Python 3.12, a FastAPI application reachable via `uv run uvicorn`, a comprehensive set of Pydantic models (OpenAI request/response schemas, node state, gateway config), and a passing pytest suite with async support.

The technology stack is fully locked in CLAUDE.md and CONTEXT.md -- no library selection decisions remain. Research focused on verifying current package versions on PyPI, documenting the correct API patterns for each library, and identifying pitfalls specific to the combination of uv + FastAPI + Pydantic v2 + pytest-asyncio. All packages passed slopcheck verification and registry checks.

**Primary recommendation:** Use the app factory pattern (`create_app()`) for FastAPI to maximize testability. Configure pydantic-settings with `env_nested_delimiter="__"` and `env_prefix="INFERENCE_PROXY_"` per D-06. Use pytest-asyncio in `auto` mode to eliminate marker boilerplate.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Domain-grouped source tree under `inference_proxy/` with subdirectories per concern: `config/`, `models/`, `api/`, `discovery/`, `routing/`, `resilience/`
- **D-02:** Root Python package named `inference_proxy` (matches repo name)
- **D-03:** Separate `tests/` tree at repo root mirroring source structure (e.g., `tests/models/test_openai.py`)
- **D-04:** Full skeleton with stub `__init__.py` files for all future phase modules (discovery/, routing/) -- gives a complete architecture view upfront
- **D-05:** Split pydantic-settings classes by domain: `GatewaySettings`, `EtcdSettings`, `RoutingSettings` composed into a root `Settings` class
- **D-06:** Env var prefix `INFERENCE_PROXY_` (nested vars use double-underscore: `INFERENCE_PROXY_GATEWAY__HOST`)
- **D-07:** Ship `.env.example` with all config keys and sensible defaults for local development
- **D-08:** Settings provided via FastAPI dependency injection -- `get_settings()` function with `@lru_cache`, injected via `Depends()`. Tests can override the dependency.
- **D-09:** Model the vLLM-relevant subset of the OpenAI API (messages, model, temperature, max_tokens, top_p, stream, stop) -- skip OpenAI-only features like tools/function calling
- **D-10:** Use Pydantic `extra='allow'` on request models so unknown fields pass through to vLLM untouched -- future-proof against new vLLM parameters
- **D-11:** Define both request AND response Pydantic models in Phase 1 (ChatCompletionResponse, CompletionResponse, streaming chunk models, OpenAI error schema)
- **D-12:** Include models for both `/v1/chat/completions` AND `/v1/completions` (text completion) endpoints
- **D-13:** Node status as Python `StrEnum` with values: `healthy`, `unhealthy`, `draining`, `unknown`
- **D-14:** Node model includes PLAN.md fields (endpoint, status, model, last_heartbeat, capabilities) PLUS routing metadata (`active_connections: int`, `node_id: str`). Capabilities as a nested `NodeCapabilities` model.
- **D-15:** Separate serializer module for etcd JSON to/from Node conversion (not built into the model). Keeps domain model testable without etcd dependency.
- **D-16:** One model per node -- `model` field is a `str`, not `list[str]`. Multiple models means multiple containers on different ports.

### Claude's Discretion
- pyproject.toml structure and uv configuration details
- pytest fixture design and conftest.py organization
- structlog configuration specifics
- Ruff and mypy configuration settings
- FastAPI app factory pattern vs direct instantiation

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

## Project Constraints (from CLAUDE.md)

- **SOLID Principles:** All code must apply SOLID. Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion are mandatory. Violations must be called out before proceeding.
- **Tech stack locked:** Python, FastAPI, httpx, etcd3gw. No alternatives.
- **Network:** Internal network only, no external-facing endpoints in v1.
- **Compatibility:** Must implement OpenAI API contract for standard SDK usage.
- **Scope:** Code complete and tested locally; deployment is separate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Project scaffolding (pyproject.toml, uv) | Build tooling | -- | Build system config, no runtime tier |
| Configuration management (pydantic-settings) | API / Backend | -- | Settings loaded at app startup, injected into handlers |
| OpenAI request/response models | API / Backend | -- | Pydantic models validate incoming HTTP bodies and shape outgoing responses |
| Node state model | API / Backend | Database / Storage | Domain model lives in backend; serialized to/from etcd (storage tier) in Phase 2 |
| FastAPI app skeleton | API / Backend | -- | HTTP entry point, route registration, middleware |
| Test infrastructure | Build tooling | -- | Developer tooling, not a runtime concern |
| Structured logging setup | API / Backend | -- | Cross-cutting concern initialized at app startup |

## Standard Stack

### Core (Phase 1 active)

| Library | Version (verified) | Purpose | Why Standard |
|---------|-------------------|---------|--------------|
| Python | 3.12.13 | Runtime | Available locally at `/usr/bin/python3.12`. Matches team target. System default is 3.14 but project pins 3.12. [VERIFIED: `uv python list`] |
| uv | 0.6.5 | Package/project manager | Installed at `/home/developer/.local/bin/uv`. Handles venv, deps, lockfile. [VERIFIED: local install] |
| FastAPI | 0.136.3 | HTTP framework | Latest stable. Native SSE (>=0.135), Pydantic v2 integration. [VERIFIED: PyPI registry] |
| Uvicorn | 0.49.0 | ASGI server | Latest stable. Install with `[standard]` for uvloop. [VERIFIED: PyPI registry] |
| Pydantic | 2.13.4 | Data validation | Latest stable. Rust-backed v2 validation. Core FastAPI dependency. [VERIFIED: PyPI registry] |
| pydantic-settings | 2.14.1 | Configuration | Latest stable. Type-safe env var loading, nested model support. [VERIFIED: PyPI registry] |
| structlog | 26.1.0 | Structured logging | Latest stable. JSON for prod, console for dev. [VERIFIED: PyPI registry] |

### Development & Quality (Phase 1 active)

| Library | Version (verified) | Purpose | When to Use |
|---------|-------------------|---------|-------------|
| ruff | 0.15.16 | Linter + formatter | Every commit. Already installed globally. [VERIFIED: PyPI + local] |
| mypy | 2.1.0 | Type checking | Every commit. `--strict` mode. [VERIFIED: PyPI registry] |
| pytest | 9.0.3 | Test framework | Latest stable (major bump from 8.x). [VERIFIED: PyPI registry] |
| pytest-asyncio | 1.4.0 | Async test support | Auto mode for all async tests. [VERIFIED: PyPI registry] |
| pytest-httpx | 0.36.2 | HTTP mocking | Mock httpx requests in proxy tests (Phase 3+). [VERIFIED: PyPI registry] |
| coverage | 7.x+ | Code coverage | With pytest-cov plugin. [ASSUMED] |

### Future phases (stub only in Phase 1)

| Library | Version (verified) | Purpose | Phase |
|---------|-------------------|---------|-------|
| httpx | 0.28.1 | Async HTTP client | Phase 3 (proxy engine) |
| httpx-sse | 0.4.3 | SSE client consumption | Phase 3 (streaming) |
| etcd3gw | 2.7.0 (not 2.5.0 as in CLAUDE.md) | etcd client | Phase 2 (discovery) |
| tenacity | 9.x+ | Retry logic | Phase 5 (resilience) |

### Note on etcd3gw version

CLAUDE.md references etcd3gw >=2.5.0. The actual latest on PyPI is 2.7.0, released after the original research. The minimum of >=2.5.0 is still correct but the planner should use `>=2.5.0` to get the latest. [VERIFIED: PyPI registry]

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| App factory `create_app()` | Direct `app = FastAPI()` at module level | Direct instantiation is simpler but harder to test with different configs. Factory pattern enables test isolation. Recommend factory. |
| pytest-asyncio auto mode | Strict mode with explicit markers | Auto mode eliminates `@pytest.mark.asyncio` on every test. Since this project is asyncio-only, auto mode is correct. |
| Flat package layout | `src/` layout | D-01 locks `inference_proxy/` at repo root (not `src/inference_proxy/`). Flat layout is simpler for a single-package project. No `[build-system]` needed unless we want `uv run` to install the package. |

**Installation (Phase 1 only):**
```bash
uv init --python 3.12
uv add "fastapi>=0.135,<1.0" "uvicorn[standard]>=0.45" "pydantic>=2.10,<3.0" "pydantic-settings>=2.14" "structlog>=26.1.0"
uv add --dev "ruff>=0.15" "mypy>=2.1" "pytest>=8.0" "pytest-asyncio>=1.4" "pytest-httpx>=0.36" "coverage>=7.0"
```

**Version verification:** All versions confirmed against PyPI on 2026-06-10.

## Package Legitimacy Audit

> All 16 packages passed slopcheck v0.6.1 verification on 2026-06-10.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| fastapi | PyPI | 8+ yrs | Very high | github.com/fastapi/fastapi | [OK] | Approved |
| uvicorn | PyPI | 8+ yrs | Very high | github.com/encode/uvicorn | [OK] | Approved |
| pydantic | PyPI | 8+ yrs | Very high | github.com/pydantic/pydantic | [OK] | Approved |
| pydantic-settings | PyPI | 3+ yrs | Very high | github.com/pydantic/pydantic-settings | [OK] | Approved |
| structlog | PyPI | 11+ yrs | High | github.com/hynek/structlog | [OK] | Approved |
| ruff | PyPI | 3+ yrs | Very high | github.com/astral-sh/ruff | [OK] | Approved |
| mypy | PyPI | 10+ yrs | Very high | github.com/python/mypy | [OK] | Approved |
| pytest | PyPI | 15+ yrs | Very high | github.com/pytest-dev/pytest | [OK] | Approved |
| pytest-asyncio | PyPI | 7+ yrs | Very high | github.com/pytest-dev/pytest-asyncio | [OK] | Approved |
| pytest-httpx | PyPI | 5+ yrs | High | github.com/colin-b/pytest_httpx | [OK] | Approved |
| coverage | PyPI | 15+ yrs | Very high | github.com/nedbat/coveragepy | [OK] | Approved |
| httpx | PyPI | 5+ yrs | Very high | github.com/encode/httpx | [OK] | Approved |
| httpx-sse | PyPI | 2+ yrs | Very high | github.com/florimondmanca/httpx-sse | [OK] | Approved |
| etcd3gw | PyPI | 7+ yrs | Moderate | opendev.org/openstack/etcd3gw | [OK] | Approved |
| tenacity | PyPI | 8+ yrs | Very high | github.com/jd/tenacity | [OK] | Approved |
| anyio | PyPI | 6+ yrs | Very high | github.com/agronholm/anyio | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram (Phase 1 scope)

```
pyproject.toml (uv project root)
    |
    v
inference_proxy/          <-- Python package
    |
    +-- main.py           <-- create_app() factory, FastAPI instance
    |       |
    |       +-- lifespan manager (startup/shutdown hooks)
    |       +-- registers API router (placeholder for Phase 3)
    |
    +-- config/
    |       |
    |       +-- settings.py  <-- GatewaySettings, EtcdSettings, RoutingSettings, root Settings
    |       +-- dependencies.py  <-- get_settings() with @lru_cache, for Depends()
    |
    +-- models/
    |       |
    |       +-- openai.py    <-- ChatCompletionRequest/Response, CompletionRequest/Response,
    |       |                    streaming chunks, error schema
    |       +-- node.py      <-- NodeStatus (StrEnum), NodeCapabilities, Node
    |
    +-- api/                 <-- stub __init__.py only (Phase 3)
    +-- discovery/           <-- stub __init__.py only (Phase 2)
    +-- routing/             <-- stub __init__.py only (Phase 4)
    +-- resilience/          <-- stub __init__.py only (Phase 5)

tests/
    +-- conftest.py          <-- shared fixtures (app, client, settings override)
    +-- models/
    |       +-- test_openai.py
    |       +-- test_node.py
    +-- config/
            +-- test_settings.py
```

### Recommended Project Structure

```
inference-proxy/
├── pyproject.toml
├── uv.lock
├── .python-version           # pins 3.12
├── .env.example              # all config keys with defaults (D-07)
├── .gitignore
├── PLAN.md
├── CLAUDE.md
├── inference_proxy/
│   ├── __init__.py
│   ├── main.py               # create_app() factory
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py       # pydantic-settings classes
│   │   └── dependencies.py   # get_settings() DI provider
│   ├── models/
│   │   ├── __init__.py
│   │   ├── openai.py         # OpenAI request/response/chunk/error models
│   │   └── node.py           # NodeStatus enum, NodeCapabilities, Node
│   ├── api/
│   │   └── __init__.py       # stub for Phase 3
│   ├── discovery/
│   │   └── __init__.py       # stub for Phase 2
│   ├── routing/
│   │   └── __init__.py       # stub for Phase 4
│   └── resilience/
│       └── __init__.py       # stub for Phase 5
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # app fixture, settings override fixture
│   ├── test_app.py           # app starts, health endpoint
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_openai.py
│   │   └── test_node.py
│   └── config/
│       ├── __init__.py
│       └── test_settings.py
└── .planning/                # existing planning artifacts
```

### Pattern 1: App Factory with Lifespan

**What:** Create the FastAPI app via a `create_app()` function rather than a module-level global. Use the lifespan context manager for startup/shutdown hooks.

**When to use:** Always -- enables test isolation, different configs per test.

**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/settings/
# Source: https://fastapi.tiangolo.com/advanced/events/
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

### Pattern 2: Settings with Dependency Injection

**What:** Split settings into domain-specific classes composed into a root `Settings`. Provide via `@lru_cache` + `Depends()`. Override in tests via `app.dependency_overrides`.

**When to use:** All settings access throughout the app.

**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/settings/
# Source: https://pydantic.dev/docs/validation/latest/concepts/models/ (pydantic-settings nested)
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

```python
# dependencies.py
from functools import lru_cache
from .settings import Settings

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
# test override pattern
# Source: https://fastapi.tiangolo.com/advanced/testing-dependencies/
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings

def get_test_settings() -> Settings:
    return Settings(gateway=GatewaySettings(port=9999))

app.dependency_overrides[get_settings] = get_test_settings
```

### Pattern 3: OpenAI Request Models with `extra='allow'`

**What:** Define vLLM-relevant fields explicitly. Use `extra='allow'` so unknown fields pass through to the backend untouched.

**When to use:** All request models (ChatCompletionRequest, CompletionRequest).

**Example:**
```python
# Source: https://pydantic.dev/docs/validation/latest/concepts/models/
# Source: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
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

### Pattern 4: Node State Model with StrEnum

**What:** Use Python 3.11+ `StrEnum` for node status. Nest capabilities in a sub-model. Keep serialization separate (D-15, SRP).

**Example:**
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

### Pattern 5: OpenAI Response Models

**What:** Response models for non-streaming and streaming responses, plus error schema.

**Example:**
```python
# Source: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
# Source: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events

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

# Streaming chunk
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

# Error response (OpenAI-compatible)
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | int | None = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
```

### Anti-Patterns to Avoid

- **Global mutable state for settings:** Do not use module-level `settings = Settings()`. This prevents test overrides. Use the `@lru_cache` + `Depends()` pattern from D-08.
- **Building the model into serializers:** D-15 requires keeping Node serialization (etcd JSON conversion) in a separate module. Mixing domain model with persistence logic violates SRP.
- **Strict validation on requests:** D-10 requires `extra='allow'`. Using `extra='forbid'` would break when vLLM adds new parameters.
- **Inheriting settings sub-classes from BaseSettings:** Sub-models (`GatewaySettings`, `EtcdSettings`) should inherit from `pydantic.BaseModel`, not `BaseSettings`. Only the root `Settings` class inherits from `BaseSettings`. Otherwise, pydantic-settings tries to collect env vars separately for each sub-model, causing unexpected behavior.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Configuration from env vars | Custom env parser | pydantic-settings `BaseSettings` | Type validation, nested model support, `.env` file loading, test overrideability |
| Request/response validation | Manual JSON parsing + validation | Pydantic v2 `BaseModel` | Rust-backed validation, automatic OpenAPI schema generation, `extra='allow'` support |
| Async test infrastructure | Custom event loop setup | pytest-asyncio auto mode | Handles event loop lifecycle, fixture scoping, auto-detection of async tests |
| Code formatting + linting | Multiple tools (black, isort, flake8) | ruff (single tool) | 10-100x faster, single config, replaces three tools |
| Project dependency management | pip + venv + requirements.txt | uv | Lockfile, fast resolution, Python version management |

**Key insight:** Phase 1 is infrastructure -- every "build" problem here has a well-tested library solution. Hand-rolling any of these creates technical debt that compounds across all six phases.

## Common Pitfalls

### Pitfall 1: System Python vs Project Python

**What goes wrong:** System Python is 3.14.4 but the project targets 3.12. Running `python` or `pytest` directly uses the wrong version.
**Why it happens:** Fedora ships latest Python as system default. The `uv python list` output confirms 3.12.13 is available at `/usr/bin/python3.12` but is not the default.
**How to avoid:** Pin `.python-version` to `3.12` during `uv init --python 3.12`. Always use `uv run pytest`, `uv run uvicorn`, `uv run mypy` -- never bare `python` or `pytest`. uv resolves the correct interpreter from `.python-version`.
**Warning signs:** Import errors for packages only installed in the uv venv. Unexpected syntax features (3.14 has new syntax not in 3.12).

### Pitfall 2: pydantic-settings Sub-Model Inheritance

**What goes wrong:** If `GatewaySettings` inherits from `BaseSettings` instead of `BaseModel`, pydantic-settings independently collects env vars for it, ignoring the nested delimiter and prefix from the parent `Settings` class.
**Why it happens:** `BaseSettings` triggers its own env var source for each sub-model. With double-underscore nesting, only the root `Settings` class should be `BaseSettings`.
**How to avoid:** Sub-models (`GatewaySettings`, `EtcdSettings`, `RoutingSettings`) inherit from `pydantic.BaseModel`. Only the root `Settings` inherits from `pydantic_settings.BaseSettings`.
**Warning signs:** Config values not loading despite correct env var names. Tests pass but runtime fails.

### Pitfall 3: `@lru_cache` Prevents Test Overrides

**What goes wrong:** If `get_settings()` is called before `app.dependency_overrides` is set, the cached result persists. Subsequent tests get stale settings.
**Why it happens:** `@lru_cache` caches the first call forever. In tests, `dependency_overrides` bypasses the function entirely, but direct imports of `get_settings()` (outside Depends) use the cached value.
**How to avoid:** Never call `get_settings()` directly in application code. Always access via `Depends(get_settings)`. In conftest.py, clear the cache between tests: `get_settings.cache_clear()`.
**Warning signs:** Tests pass individually but fail when run together. Settings changes in one test leak into another.

### Pitfall 4: Missing `__init__.py` Breaks Imports

**What goes wrong:** Python cannot resolve relative imports if `__init__.py` files are missing from subdirectories.
**Why it happens:** D-04 requires stub `__init__.py` in all subdirectories including future-phase ones. Easy to forget one.
**How to avoid:** Create all `__init__.py` files during scaffolding (D-04). Test by importing every sub-package in a smoke test.
**Warning signs:** `ModuleNotFoundError` when importing from subdirectories.

### Pitfall 5: pytest-asyncio Mode Not Set

**What goes wrong:** Without `asyncio_mode = "auto"` in pyproject.toml, async test functions are silently skipped or require explicit `@pytest.mark.asyncio` markers.
**Why it happens:** Default mode is `strict`, which ignores unmarked async tests.
**How to avoid:** Set `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`. Verify by writing an async test without the marker and confirming it runs.
**Warning signs:** Async tests show as "passed" with 0 assertions (they were skipped/ignored).

### Pitfall 6: Pydantic v2 `model` Field Name Collision

**What goes wrong:** The OpenAI API uses `model` as a field name. In Pydantic v2, `model_` prefix is reserved for Pydantic methods (`model_dump`, `model_validate`, etc.). The field name `model` itself is allowed but triggers a warning in some versions.
**Why it happens:** Pydantic v2 uses `model_*` namespace for its methods.
**How to avoid:** Using `model: str` as a field name works in Pydantic v2 without issues -- it is NOT a reserved name. Only `model_` prefix methods are reserved. Verify with a simple test that `ChatCompletionRequest(model="llama-2-7b", messages=[...])` works.
**Warning signs:** Pydantic deprecation warnings about `model_` namespace.

## Code Examples

### pyproject.toml Configuration

```toml
# Source: https://docs.astral.sh/uv/concepts/projects/config/
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

### conftest.py with App and Settings Fixtures

```python
# Source: https://fastapi.tiangolo.com/advanced/testing-dependencies/
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

### structlog Configuration

```python
# Source: https://www.structlog.org/en/stable/getting-started.html
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

### .env.example

```bash
# Gateway
INFERENCE_PROXY_GATEWAY__HOST=0.0.0.0
INFERENCE_PROXY_GATEWAY__PORT=8080

# etcd
INFERENCE_PROXY_ETCD__ENDPOINTS=["http://localhost:2379"]
INFERENCE_PROXY_ETCD__NODE_PREFIX=/nodes/

# Routing
INFERENCE_PROXY_ROUTING__STRATEGY=least_connections
INFERENCE_PROXY_ROUTING__HEALTH_CHECK_INTERVAL=30
INFERENCE_PROXY_ROUTING__MAX_RETRIES=3
INFERENCE_PROXY_ROUTING__TIMEOUT=30
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pytest-asyncio strict mode (explicit markers) | Auto mode (v0.21+, recommended v1.0+) | 2023+ | No `@pytest.mark.asyncio` boilerplate |
| Pydantic v1 `Config` inner class | Pydantic v2 `model_config = ConfigDict(...)` | 2023 (v2.0) | Must use v2 syntax. v1 is deprecated in FastAPI. |
| `@app.on_event("startup")` / `@app.on_event("shutdown")` | `lifespan` context manager | FastAPI 0.95+ (2023) | Old decorators are deprecated. Use `lifespan` parameter. |
| sse-starlette for SSE | FastAPI built-in `EventSourceResponse` | FastAPI 0.135 (2025) | No extra dependency needed for server-side SSE |
| pip + venv + requirements.txt | uv + pyproject.toml + uv.lock | 2024+ | 10-100x faster, single tool, lockfile |
| black + isort + flake8 | ruff (single tool) | 2023+ | One config, one tool, much faster |
| pytest 8.x | pytest 9.0 | 2025 | New major version. Minor breaking changes in fixture scoping. |

**Deprecated/outdated:**
- `@app.on_event("startup")`: Replaced by `lifespan` context manager. Do not use.
- Pydantic v1 syntax (`class Config:`): Replaced by `model_config = ConfigDict(...)`. FastAPI requires v2.
- `sse-starlette`: Redundant with FastAPI >=0.135 built-in SSE.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | coverage >=7.0 version constraint is current | Standard Stack | Low -- coverage is extremely stable; any 7.x+ works |
| A2 | pytest 9.0 has no breaking changes affecting this project | Standard Stack | Low -- major version bump may have minor fixture scoping changes; tests would surface issues immediately |
| A3 | `requires-python = ">=3.12,<3.14"` is the correct upper bound | Code Examples | Medium -- if team wants 3.13 support too, the bound is fine; <3.14 excludes the system Python which is desired |

## Open Questions (RESOLVED)

1. **RESOLVED: Package build system for `uv run uvicorn`**
   - What we know: `uv run uvicorn inference_proxy.main:app` requires the package to be importable. Without a `[build-system]` in pyproject.toml, uv does not install the project into the venv.
   - What's unclear: Whether uv's `--no-project` or adding a build system is needed.
   - Recommendation: Add a `[build-system]` with hatchling so `uv run` installs the package in editable mode. This makes `inference_proxy` importable. Alternative: use `PYTHONPATH=. uv run uvicorn ...` but this is fragile. The build-system approach is cleaner.
   - **Resolution:** Plan 01-01 Task 1 Step 2 adds `[build-system]` with hatchling backend.

2. **RESOLVED: Text completion response models vs chat completion**
   - What we know: D-12 requires models for both `/v1/completions` and `/v1/chat/completions`. The text completion response uses `text` field in choices instead of `message`.
   - What's unclear: Whether to share base classes or keep them fully separate.
   - Recommendation: Keep them separate. The schemas are similar but differ in key fields (`text` vs `message`, `prompt` vs `messages`). Sharing a base class would create awkward optionality. Two clean, explicit model sets are better per SRP.
   - **Resolution:** Plan 01-03 Task 1 keeps chat and text completion models fully separate per SRP.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | Runtime target | Yes | 3.12.13 | -- (required, uv will use it via .python-version) |
| uv | Package management | Yes | 0.6.5 | -- (required) |
| ruff | Linting/formatting | Yes | 0.15.12 | -- (also installed as dev dep) |
| mypy | Type checking | Not globally | -- | Installed via `uv add --dev` |
| pytest | Testing | Not globally | -- | Installed via `uv add --dev` |
| etcd | Service discovery (Phase 2) | Not checked | -- | Not needed in Phase 1 |

**Missing dependencies with no fallback:** None for Phase 1.

**Missing dependencies with fallback:** mypy, pytest -- installed as dev dependencies via uv.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 + pytest-asyncio 1.4.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] -- created in Wave 0 |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v --tb=short` |

### Phase Requirements to Test Map

Phase 1 has no direct requirement IDs but has three success criteria:

| Criterion | Behavior | Test Type | Automated Command | File Exists? |
|-----------|----------|-----------|-------------------|-------------|
| SC-1 | `uv run pytest` executes and passes | smoke | `uv run pytest tests/ -x` | Wave 0 |
| SC-2 | `uv run uvicorn` starts FastAPI app on a port | smoke | `uv run pytest tests/test_app.py -x` (TestClient) | Wave 0 |
| SC-3a | Pydantic OpenAI models validate input | unit | `uv run pytest tests/models/test_openai.py -x` | Wave 0 |
| SC-3b | Pydantic node state model validates input | unit | `uv run pytest tests/models/test_node.py -x` | Wave 0 |
| SC-3c | Pydantic config model validates input | unit | `uv run pytest tests/config/test_settings.py -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `pyproject.toml` -- project config with pytest, ruff, mypy sections
- [ ] `tests/conftest.py` -- app fixture, settings override fixture, TestClient fixture
- [ ] `tests/test_app.py` -- smoke test: app starts, responds
- [ ] `tests/models/test_openai.py` -- unit tests for OpenAI request/response models
- [ ] `tests/models/test_node.py` -- unit tests for Node, NodeStatus, NodeCapabilities
- [ ] `tests/config/test_settings.py` -- unit tests for Settings with env var overrides
- [ ] Framework install: `uv add --dev pytest pytest-asyncio`

## Security Domain

> Phase 1 is internal infrastructure with no external endpoints, no authentication, no user input processing. Security requirements are minimal.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Internal network only (out of scope per REQUIREMENTS.md) |
| V3 Session Management | No | Stateless proxy, no sessions |
| V4 Access Control | No | No authz in v1 |
| V5 Input Validation | Yes (partial) | Pydantic model validation on all request bodies |
| V6 Cryptography | No | No secrets handling in Phase 1 |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed JSON request body | Tampering | Pydantic validation with `extra='allow'` rejects invalid types while passing through unknown fields |
| Env var injection via `.env` file | Information Disclosure | `.env` file excluded from git via `.gitignore`. `.env.example` committed with safe defaults only |
| Dependency confusion | Tampering | All packages verified via slopcheck. Lockfile (`uv.lock`) pinned. |

## Sources

### Primary (HIGH confidence)
- [FastAPI Settings docs](https://fastapi.tiangolo.com/advanced/settings/) -- Settings + lru_cache + Depends() pattern
- [FastAPI Testing Dependencies docs](https://fastapi.tiangolo.com/advanced/testing-dependencies/) -- dependency_overrides pattern
- [Pydantic v2 Models docs](https://pydantic.dev/docs/validation/latest/concepts/models/) -- ConfigDict, extra='allow', nested models, Field
- [structlog Getting Started](https://www.structlog.org/en/stable/getting-started.html) -- Processor chain, JSON/console rendering
- [pytest-asyncio Concepts](https://pytest-asyncio.readthedocs.io/en/latest/concepts.html) -- Auto mode vs strict mode
- [uv Project Configuration](https://docs.astral.sh/uv/concepts/projects/config/) -- pyproject.toml, dependency groups
- [uv Working on Projects](https://docs.astral.sh/uv/guides/projects/) -- uv init, uv add, uv run
- [OpenAI Chat Completions API Reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) -- Request/response schema
- [OpenAI Streaming Events Reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) -- chunk format, delta, data: [DONE]
- [OpenAI Streaming Guide](https://developers.openai.com/api/docs/guides/streaming-responses) -- SSE lifecycle
- PyPI registry -- all package versions verified 2026-06-10
- slopcheck v0.6.1 -- all 16 packages passed legitimacy check

### Secondary (MEDIUM confidence)
- [pydantic-settings env_nested_delimiter docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) -- nested model configuration
- [vLLM OpenAI-Compatible Server docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) -- supported fields, vLLM-specific extensions

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified on PyPI with slopcheck, versions confirmed
- Architecture: HIGH -- patterns sourced from official FastAPI and Pydantic docs, locked by CONTEXT.md decisions
- Pitfalls: HIGH -- documented from official docs and well-known community patterns

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (30 days -- stable stack, no fast-moving components)
