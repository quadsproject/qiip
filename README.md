# QUADS LLM Inference Proxy

A gateway service that proxies OpenAI-compatible requests to [vLLM](https://docs.vllm.ai/) inference nodes running on idle QUADS lab servers. It dynamically discovers backends via etcd, health-checks them, and routes requests with automatic failover — so clients see a single, reliable endpoint.

```
Clients ──► NGINX ──► Inference Proxy ──► vLLM Node A
                           │           ──► vLLM Node B
                           │           ──► vLLM Node C
                           ▼
                          etcd
                     (service registry)
```

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
- **Streaming support** — Server-Sent Events (SSE) for real-time token generation
- **Service discovery** — watches etcd for node registration/deregistration in real time
- **Least-connections load balancing** — routes to the node with the fewest in-flight requests
- **Automatic failover** — retries failed requests on alternate healthy nodes (configurable, default 3 attempts)
- **Circuit breakers** — per-node circuit breakers trip after consecutive failures, preventing cascade
- **Health checking** — background thread probes each node's `/health` endpoint; marks nodes unhealthy after repeated failures and recovers them automatically
- **Graceful shutdown** — drains in-flight requests before stopping, with configurable timeout
- **Structured logging** — JSON or pretty console output via structlog

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

# Run the gateway
uv run uvicorn inference_proxy.main:app --host 0.0.0.0 --port 8080
```

The gateway starts, connects to etcd, discovers available vLLM nodes, and begins accepting requests.

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
| `GET` | `/admin/nodes` | Inspect all registered nodes and their status |

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

### Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_GATEWAY__HOST` | `0.0.0.0` | Bind address |
| `INFERENCE_PROXY_GATEWAY__PORT` | `8080` | Bind port |
| `INFERENCE_PROXY_GATEWAY__GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | Seconds to drain requests on shutdown |

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

## Architecture

```
inference_proxy/
├── main.py                 # App factory, lifespan (startup/shutdown)
├── api/
│   ├── routes.py           # OpenAI-compatible proxy endpoints
│   ├── admin.py            # Admin inspection endpoints
│   ├── middleware.py        # Request logging middleware
│   └── errors.py           # Error response mapping
├── config/
│   ├── settings.py         # Pydantic settings (env vars)
│   ├── dependencies.py     # FastAPI dependency injection
│   └── logging.py          # structlog configuration
├── discovery/
│   ├── registry.py         # Thread-safe in-memory node registry
│   ├── etcd_client.py      # etcd3gw wrapper
│   ├── watcher.py          # Background thread watching etcd for changes
│   └── serializer.py       # etcd value → Node model deserialization
├── models/
│   ├── node.py             # Node, NodeStatus, NodeCapabilities
│   ├── openai.py           # OpenAI API request/response models
│   └── admin.py            # Admin API response models
├── proxy/
│   └── client.py           # httpx async client for forwarding requests
├── resilience/
│   ├── health_checker.py   # Background health probe thread
│   ├── circuit_breaker.py  # Per-node circuit breaker
│   └── shutdown.py         # Graceful shutdown middleware
└── routing/
    ├── node_selector.py    # Least-connections node selection
    └── connection_tracker.py  # Per-node in-flight request counter
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
uv run pytest --cov=inference_proxy

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
| Testing | pytest + pytest-asyncio + pytest-httpx | Async tests with HTTP mocking |

## License

Internal use — Perf/Scale DevOps Team.
