# QIIP (QUADS Idle Inference Proxy)

[![CI](https://github.com/quadsproject/qiip/actions/workflows/ci.yml/badge.svg)](https://github.com/quadsproject/qiip/actions/workflows/ci.yml)

A QUADS-native inference abstraction framework that fully automates installation, drivers, setup and presentation of disparate, free/idle GPU-equipped systems into a single unified usage interface and inference API.

QIIP provides a gateway service that proxies OpenAI-compatible requests to [vLLM](https://docs.vllm.ai/) inference nodes running on idle QUADS lab servers or standalone, free GPU-equipped hardware.  It dynamically discovers backends via etcd, health-checks them, and routes requests with automatic failover — so clients see a single, reliable endpoint.

```
Clients ──► NGINX ──► Inference Proxy  ──► vLLM Node A
                           │           ──► vLLM Node B
                           │           ──► vLLM Node C
                           ▼
                          etcd
                     (service registry)
```

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
- **Streaming support** — Server-Sent Events (SSE) for real-time token generation
- **Chat playground** — browser-based chat UI at `/chat` with markdown rendering and model selection
- **Service discovery** — watches etcd for node registration/deregistration in real time
- **Least-connections load balancing** — routes to the node with the fewest in-flight requests
- **Automatic failover** — retries failed requests on alternate healthy nodes (configurable, default 3 attempts)
- **Circuit breakers** — per-node circuit breakers trip after consecutive failures, preventing cascade
- **Health checking** — background thread probes each node's `/health` endpoint; marks nodes unhealthy after repeated failures and recovers them automatically
- **Graceful shutdown** — drains in-flight requests before stopping, with configurable timeout
- **Structured logging** — JSON or pretty console output via structlog
- **Operations dashboard** — interactive web UI at `/dashboard` with real-time node table, detail pages, and provisioning status
- **QUADS integration** — background polling of QUADS inventory and availability; unified view merging QUADS hosts with etcd-registered nodes
- **QUADS schedule enforcement** — automated teardown of managed nodes when QUADS reports an upcoming scheduling conflict
- **End-to-end node provisioning** — SSH-based pipeline: BMC power-on, NVIDIA driver and CUDA toolkit install, vLLM setup, NFS mount, firewall, health poll, and etcd registration
- **Node teardown** — graceful shutdown with connection draining, force teardown option, and provisioning task cancellation
- **Provisioning log streaming** — live SSE stream of provisioning and vLLM logs viewable in the dashboard
- **BMC power management (Redfish)** — query and control node power state; supports On, ForceOff, GracefulRestart, and ForceRestart
- **Model catalog** — scans shared NFS-mounted HuggingFace cache, verifies model completeness via tree manifests, exposed via `/admin/models/catalog`
- **Background model downloads** — concurrent HuggingFace downloads with status tracking; duplicate-safe and re-downloadable after completion or failure
- **Hardware-aware model recommendations** — runs llmfit via SSH on a target host to produce ranked recommendations with fit levels, throughput, and memory estimates; auto-installs the binary on first use
- **Request metrics** — per-model and per-node counters exposed via `/admin/metrics`
- **Admin authentication** — HTTP Basic required on all `/admin/*` endpoints and `/dashboard*` pages; inference API remains public
- **Backend endpoint allowlist** — configurable hostname wildcard, CIDR network, and port allowlists; rejects non-matching registrations with loopback-only defaults
- **Client config downloads** — one-click download of OpenCode CLI and Pi coding agent configuration files from the dashboard and node detail pages; dashboard configs point at the proxy for load-balanced access, node detail configs point at individual vLLM endpoints

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Verify it's running](#verify-its-running)
  - [Send a request](#send-a-request)
  - [Use with the OpenAI Python SDK](#use-with-the-openai-python-sdk)
- [API Endpoints](#api-endpoints)
  - [Administrative access](#administrative-access)
  - [Error responses](#error-responses)
- [Configuration](#configuration)
  - [Upgrade requirements](#upgrade-requirements)
  - [Gateway](#gateway)
  - [Admin authentication](#admin-authentication)
  - [etcd](#etcd)
  - [Routing](#routing)
  - [SSH and provisioning commands](#ssh-and-provisioning-commands)
  - [Proxy (HTTP client)](#proxy-http-client)
  - [Resilience](#resilience)
  - [Logging](#logging)
  - [Redfish BMC](#redfish-bmc)
- [Architecture](#architecture)
  - [Request flow](#request-flow)
  - [Background threads](#background-threads)
- [Development](#development)
  - [Setup](#setup)
  - [Run tests](#run-tests)
  - [Lint and format](#lint-and-format)
  - [Type check](#type-check)
- [Technology Stack](#technology-stack)
- [License](#license)

## Requirements

- Python 3.12 or 3.13
- [uv](https://github.com/astral-sh/uv) (package manager)
- A running etcd cluster (v3 API)
- One or more vLLM nodes registered in etcd

## Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd inference-proxy

# Install dependencies
uv sync

# Copy and edit configuration
cp .env.example .env
# Set the required INFERENCE_PROXY_ADMIN__USERNAME and
# INFERENCE_PROXY_ADMIN__PASSWORD values in .env

# Run the gateway
uv run uvicorn inference_proxy.main:create_app --factory --host 0.0.0.0 --port 8080
```

The gateway starts, connects to etcd, discovers available vLLM nodes, and begins accepting requests.

The administrative API and dashboard use HTTP Basic authentication, which sends
base64-encoded credentials—not encryption—on every request. A trusted work LAN
may use HTTP; use a TLS terminator whenever that network path is not trusted.

### Verify it's running

```bash
curl http://localhost:8080/health
# {"status": "ok", "nodes_registered": 2}
```

### Send a request

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # no auth in v1
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Explain QUADS in one sentence."}],
)
print(response.choices[0].message.content)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Gateway health check (returns node count) |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST` | `/v1/completions` | Text completion (OpenAI-compatible) |
| `GET` | `/v1/models` | List models available across healthy nodes |
| `GET` | `/chat` | Browser-based chat playground |
| `GET` | `/dashboard` | Operations dashboard (HTTP Basic) |
| `GET` | `/dashboard/nodes/{node_id}` | Node detail page (HTTP Basic) |
| `GET` | `/admin/nodes` | Registered nodes and their status (HTTP Basic) |
| `DELETE` | `/admin/nodes/{node_id}` | Tear down and deregister a node (HTTP Basic) |
| `GET` | `/admin/metrics` | Per-model and per-node request counters (HTTP Basic) |
| `GET` | `/admin/models/catalog` | Available models on shared NFS cache (HTTP Basic) |
| `POST` | `/admin/models/download` | Start a background model download (HTTP Basic) |
| `GET` | `/admin/models/downloads` | Status of background downloads (HTTP Basic) |
| `POST` | `/admin/nodes/setup` | Provision a new inference node (HTTP Basic) |
| `GET` | `/admin/provisioning/tasks` | Active provisioning/teardown tasks (HTTP Basic) |
| `GET` | `/admin/provisioning/{hostname}/logs` | SSE stream of provisioning logs (HTTP Basic) |
| `GET` | `/admin/quads/status` | QUADS poller status (HTTP Basic) |
| `GET` | `/admin/nodes/{hostname}/power` | Query BMC power state via Redfish (HTTP Basic) |
| `POST` | `/admin/nodes/{hostname}/power` | Set BMC power state via Redfish (HTTP Basic) |
| `GET` | `/admin/nodes/{hostname}/recommendations` | Hardware-aware model recommendations (HTTP Basic) |

### Administrative access

All `/admin/*` API endpoints and `/dashboard*` pages require the shared HTTP
Basic credentials configured below. The inference API, chat page, and health
endpoint remain public. For example:

```bash
curl -u "$INFERENCE_PROXY_ADMIN__USERNAME:$INFERENCE_PROXY_ADMIN__PASSWORD" \
  http://gateway.example.com/admin/nodes
```

On a trusted work LAN, the administrative surface may run over HTTP. Anyone able
to observe that traffic can recover the reusable credential, so deploy a
TLS-terminating reverse proxy whenever the network path is not trusted. HTTP
Basic is used deliberately: the browser's `EventSource` API cannot set a Bearer
header, while browser-cached Basic credentials apply to the provisioning SSE
stream without exposing a token to JavaScript.

State-changing admin endpoints accept JSON only. This is part of the CSRF
boundary: cross-origin JSON requests and all DELETE requests require a browser
preflight. Do not add form-encoded, multipart, or plain-text state-changing
admin endpoints without adding explicit CSRF protection. Authentication also
does not protect an already-authenticated browser from same-origin XSS.

### Error responses

All errors follow the OpenAI error format:

| Code | Meaning |
|------|---------|
| 404 | Model not found — no node serves the requested model |
| 502 | Backend connection failed |
| 503 | No healthy nodes available, or model temporarily unavailable |
| 504 | Backend request timed out |

## Configuration

All settings are loaded from environment variables with the prefix `INFERENCE_PROXY_` and double-underscore nesting for nested groups. A `.env` file is also supported.

### Upgrade requirements

Deployments upgrading from releases before the reliability campaign must apply
these changes together:

- Launch with `uvicorn inference_proxy.main:create_app --factory`; the old
  `inference_proxy.main:app` target no longer exists.
- Move any `GATEWAY__GRACEFUL_SHUTDOWN_TIMEOUT` setting to Uvicorn's
  `--timeout-graceful-shutdown` launcher option. Uvicorn owns request draining;
  the retired gateway setting is ignored with a startup warning.
- Configure `ROUTING__ALLOWED_ENDPOINT_HOSTS`,
  `ROUTING__ALLOWED_ENDPOINT_NETWORKS`, and
  `ROUTING__ALLOWED_ENDPOINT_PORTS` for every backend the gateway may contact.
- Configure `ADMIN__USERNAME` and `ADMIN__PASSWORD`. HTTP is permitted on a
  trusted work LAN; use TLS when the network path is not trusted.
- Add `"managed": true` to externally written etcd node records only when the
  proxy should own and enforce their lifecycle. Missing values now default to
  unmanaged.
- When QUADS is enabled, configure `QUADS__SERVER_TIMEZONE` to the IANA timezone
  used by the QUADS server's local clock.

Enabling QUADS requires both `INFERENCE_PROXY_QUADS__BASE_URL` and
`INFERENCE_PROXY_QUADS__SERVER_TIMEZONE`. Set the latter to the IANA timezone
used by the QUADS server's local clock, for example `America/New_York`. The
QUADS availability endpoint accepts timezone-naive `YYYY-MM-DDTHH:MM` values,
so the proxy converts its UTC scheduling deadline into that configured server
timezone before querying availability.

### Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_GATEWAY__HOST` | `0.0.0.0` | Bind address |
| `INFERENCE_PROXY_GATEWAY__PORT` | `8080` | Bind port |

Uvicorn owns graceful request draining. Configure its
`--timeout-graceful-shutdown <seconds>` launcher option when the default does
not fit the deployment. The retired
`INFERENCE_PROXY_GATEWAY__GRACEFUL_SHUTDOWN_TIMEOUT` setting is ignored and
emits a migration warning.

### Admin authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_ADMIN__USERNAME` | required | Shared username for `/admin/*` and `/dashboard*` |
| `INFERENCE_PROXY_ADMIN__PASSWORD` | required | Shared password, stored as a masked secret |

Both values are required at startup. Existing deployments must configure them
before upgrading. Credentials are accepted only through HTTP Basic and must be
protected by TLS whenever clients do not reach the gateway over a trusted
network.

### etcd

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_ETCD__ENDPOINTS` | `["http://localhost:2379"]` | etcd cluster endpoints (JSON array) |
| `INFERENCE_PROXY_ETCD__NODE_PREFIX` | `/nodes/` | etcd key prefix for node registration |

### Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_ROUTING__STRATEGY` | `least_connections` | Load balancing strategy |
| `INFERENCE_PROXY_ROUTING__MAX_RETRIES` | `3` | Max retry attempts on failure |
| `INFERENCE_PROXY_ROUTING__TIMEOUT` | `30` | General routing timeout (seconds) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_HOSTS` | `["localhost"]` | Exact backend DNS names or `*.suffix` rules (JSON array) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_NETWORKS` | `["127.0.0.0/8","::1/128"]` | Backend IP CIDR allowlist (JSON array) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_PORTS` | `[8000]` | Backend TCP port allowlist (JSON array) |

The endpoint allowlist is intentionally loopback-only by default. Configure the
GPU host suffixes or IP networks before upgrading an existing deployment;
otherwise non-loopback etcd node registrations are rejected with warning logs
and do not appear in `/admin/nodes`. CIDR rules apply to IP-literal endpoints;
DNS endpoints must match an exact hostname or `*.suffix` rule. The configured
provisioning vLLM port must also appear in the endpoint port allowlist. Setup
requests whose generated backend endpoint is not allowed fail before any
power, SSH, or installation work and name the allowlist setting to update.

### SSH and provisioning commands

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_SSH__KEY_PATH` | `~/.ssh/id_rsa` | Private key used for node access; `~` is expanded after environment loading |
| `INFERENCE_PROXY_SSH__USERNAME` | `root` | Remote provisioning user |
| `INFERENCE_PROXY_SSH__CONNECT_TIMEOUT` | `10` | SSH connection timeout (seconds) |
| `INFERENCE_PROXY_SSH__STREAMING_COMMAND_TIMEOUT` | `3600` | Total wall-clock deadline for a streaming remote command (seconds) |
| `INFERENCE_PROXY_SSH__STREAMING_INACTIVITY_TIMEOUT` | `900` | Maximum interval without stdout or stderr from a streaming command (seconds) |

LLMFit has one version setting: `INFERENCE_PROXY_LLMFIT__VERSION`. The retired
`INFERENCE_PROXY_PROVISIONING__LLMFIT_VERSION` variable is ignored and emits a
startup warning so an existing pin cannot disappear silently. Remove the old
variable and move its value to the LLMFit setting when upgrading.

The default NVIDIA driver and LLMFit versions each ship with a verified
SHA-256. Changing either version requires configuring its matching digest via
`INFERENCE_PROXY_PROVISIONING__NVIDIA_DRIVER_SHA256` or
`INFERENCE_PROXY_LLMFIT__SHA256`; provisioning fails before SSH or installation
when a custom version has no explicit digest.

The gateway cache path and node cache mount may differ, but provisioning uses
one declared backing export: `INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT`. It is
optional for proxy-only deployments and required before a node setup operation
can acquire a host lease or start SSH work. The retired
`INFERENCE_PROXY_PROVISIONING__NFS_SERVER` variable is ignored with a startup
warning; move its value to the HuggingFace setting when upgrading.

Node-side launch tuning uses the `AUTOVLLM_*` namespace. The retired
`VLLM_TENSOR_PARALLEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`,
`VLLM_MAX_BATCHED_TOKENS`, and `VLLM_EXTRA_ARGS` names are ignored by
`start-vllm.sh` and produce a warning naming their replacements. This keeps
script inputs out of the environment namespace reserved by vLLM itself.

### Proxy (HTTP client)

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PROXY__CONNECT_TIMEOUT` | `5.0` | TCP connect timeout (seconds) |
| `INFERENCE_PROXY_PROXY__READ_TIMEOUT` | `120.0` | Read timeout — high for LLM first-token latency |
| `INFERENCE_PROXY_PROXY__WRITE_TIMEOUT` | `10.0` | Write timeout (seconds) |
| `INFERENCE_PROXY_PROXY__POOL_TIMEOUT` | `10.0` | Connection pool acquisition timeout |
| `INFERENCE_PROXY_PROXY__MAX_CONNECTIONS` | `100` | Max total connections in pool |
| `INFERENCE_PROXY_PROXY__MAX_KEEPALIVE_CONNECTIONS` | `20` | Max idle keepalive connections |
| `INFERENCE_PROXY_PROXY__KEEPALIVE_EXPIRY` | `30` | Keepalive connection TTL (seconds) |

### Resilience

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_RESILIENCE__CIRCUIT_BREAKER_THRESHOLD` | `3` | Consecutive failures before tripping circuit breaker |
| `INFERENCE_PROXY_RESILIENCE__HEALTH_CHECK_FAILURE_THRESHOLD` | `3` | Consecutive probe failures before marking node unhealthy |
| `INFERENCE_PROXY_RESILIENCE__HEALTH_CHECK_INTERVAL` | `30` | Seconds between health probe cycles |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_LOGGING__JSON_OUTPUT` | `false` | `true` for JSON logs (production), `false` for pretty console |
| `INFERENCE_PROXY_LOGGING__LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Redfish BMC

Redfish power management is enabled only when both
`INFERENCE_PROXY_REDFISH__BMC_USERNAME` and
`INFERENCE_PROXY_REDFISH__BMC_PASSWORD` are set. Partial credentials fail
configuration validation. Caller-supplied node names must be DNS names allowed
by `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_HOSTS`; IP literals are not
accepted for BMC template expansion. The host template must contain exactly one
plain `{hostname}` field and may not contain a scheme, port, path, query, or
fragment.

`INFERENCE_PROXY_REDFISH__VERIFY_SSL` remains `false` by default for BMCs with
self-signed certificates. Credentials are attached per request only after the
node hostname passes the allowlist and the validated template resolves the BMC
destination.

## Architecture

```
inference_proxy/
├── main.py                    # App factory, lifespan (startup/shutdown)
├── api/
│   ├── routes.py              # OpenAI-compatible proxy endpoints
│   ├── admin.py               # Admin API endpoints
│   ├── chat.py                # Chat playground page
│   ├── dashboard.py           # Operations dashboard and node detail pages
│   ├── errors.py              # Error response mapping
│   └── middleware.py          # Request logging middleware
├── config/
│   ├── settings.py            # Pydantic settings (env vars)
│   ├── dependencies.py        # FastAPI dependency injection
│   └── logging.py             # structlog configuration
├── discovery/
│   ├── registry.py            # Thread-safe in-memory node registry
│   ├── etcd_client.py         # etcd3gw wrapper
│   ├── watcher.py             # Background thread watching etcd for changes
│   ├── node_leases.py         # etcd lease reconciliation
│   └── serializer.py          # etcd value to Node model deserialization
├── huggingface/
│   ├── catalog.py             # NFS model cache scanner
│   └── downloader.py          # Background model download service
├── llmfit/
│   ├── runner.py              # SSH-based llmfit execution and auto-install
│   └── errors.py              # LLMFit error types
├── models/
│   ├── node.py                # Node, NodeStatus, NodeCapabilities
│   ├── openai.py              # OpenAI API request/response models
│   ├── admin.py               # Admin API response models
│   ├── endpoint.py            # Endpoint parsing and allowlist policy
│   ├── llmfit.py              # LLMFit data models
│   └── quads.py               # QUADS data models
├── provisioning/
│   ├── provisioner.py         # End-to-end node setup pipeline
│   ├── ssh_client.py          # Async SSH command execution
│   ├── log_buffer.py          # Provisioning log ring buffer and SSE stream
│   ├── host_lifecycle.py      # Per-host mutual exclusion leases
│   └── state.py               # Provisioning step and state models
├── proxy/
│   └── client.py              # httpx async client for forwarding requests
├── quads/
│   ├── client.py              # QUADS REST API client
│   ├── poller.py              # Background QUADS inventory polling
│   └── schedule_enforcer.py   # Teardown on scheduling conflicts
├── redfish/
│   ├── client.py              # Redfish BMC power management
│   └── errors.py              # Redfish error types
├── resilience/
│   ├── health_checker.py      # Background health probe thread
│   └── circuit_breaker.py     # Per-node circuit breaker
├── routing/
│   ├── node_selector.py       # Least-connections node selection
│   ├── connection_tracker.py  # Per-node in-flight request counter
│   ├── request_metrics.py     # Per-model and per-node counters
│   └── drain_cleanup.py       # Automatic DRAINING node removal
├── services/
│   └── unified_nodes.py       # Merged QUADS + etcd node view
├── static/                    # CSS, JS, vendored client libraries
└── templates/                 # Jinja2 HTML (dashboard, node detail, chat)
```

### Request flow

1. Client sends an OpenAI-compatible request to the gateway
2. `NodeSelector` picks the healthiest node with the fewest active connections
3. `ProxyClient` forwards the request to the vLLM backend via httpx
4. On success, the response (or SSE stream) is relayed back to the client
5. On failure, the circuit breaker records the failure and the request retries on a different node
6. Once retries are exhausted, an OpenAI-format error is returned

### Background threads

- **etcd watcher** — watches the configured key prefix for node PUT/DELETE events; updates the registry in real time
- **Health checker** — probes each registered node's `/health` endpoint at a configurable interval; transitions nodes between HEALTHY and UNHEALTHY states

## Development

### Setup

```bash
# Install all dependencies (including dev)
uv sync

# Activate the virtual environment (optional — uv run handles this)
source .venv/bin/activate
```

### Run tests

```bash
# All tests
uv run pytest

# With coverage
uv run coverage run -m pytest
uv run coverage report

# Specific module
uv run pytest tests/api/test_routes.py -v
```

### Lint and format

```bash
# Check lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type check

```bash
uv run mypy inference_proxy
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI >=0.135 | HTTP framework with native SSE |
| Server | Uvicorn | ASGI server with uvloop |
| Validation | Pydantic v2 | Request/response models |
| Config | pydantic-settings | Type-safe env var loading |
| HTTP Client | httpx + httpx-sse | Async proxy engine with SSE support |
| Service Discovery | etcd3gw | etcd v3 HTTP gateway client |
| Logging | structlog | Structured JSON/console logging |
| Linter/Formatter | Ruff | Replaces flake8 + black + isort |
| Type Checker | mypy (strict) | Static type safety |
| Templates | Jinja2 | Dashboard, node detail, and chat HTML |
| SSH | asyncssh | Async SSH for node provisioning |
| Model Hub | huggingface-hub | Model catalog and background downloads |
| Testing | pytest + pytest-asyncio + pytest-httpx | Async tests with HTTP mocking |

## License

Open Source, crafted with :heart: via [GPLv3](LICENSE)
