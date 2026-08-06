# QIIP (QUADS Idle Inference Proxy)

[![CI](https://github.com/quadsproject/qiip/actions/workflows/ci.yml/badge.svg)](https://github.com/quadsproject/qiip/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/sadsfae/188b760b19592c8913101f598f7cb382/raw/qiip-coverage.json)](https://github.com/quadsproject/qiip/actions/workflows/ci.yml)
[![vLLM](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/sadsfae/188b760b19592c8913101f598f7cb382/raw/qiip-vllm.json)](https://docs.vllm.ai/)
[![llama.cpp](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/sadsfae/188b760b19592c8913101f598f7cb382/raw/qiip-llamacpp.json)](https://github.com/ggml-org/llama.cpp)

A QUADS-native inference abstraction framework that automates installation,
drivers, setup, and presentation of disparate, free or idle NVIDIA GPU systems
through one inference API. Setup requests explicitly choose either
[vLLM](https://docs.vllm.ai/) or
[llama.cpp](https://github.com/ggml-org/llama.cpp); QIIP then applies the
engine-specific provisioning path.

QIIP provides a gateway service that proxies OpenAI-compatible requests to inference nodes on idle QUADS lab servers or standalone, free GPU-equipped hardware. It dynamically discovers backends via etcd, health-checks them, and routes requests with automatic failover so clients see a single, reliable endpoint. Both engines expose OpenAI-compatible HTTP APIs; the proxy layer is engine-agnostic and the engine choice is invisible to API consumers. See [auto-vllm/](auto-vllm/README.md) and [auto-llamacpp/](auto-llamacpp/README.md) for engine-specific provisioning details.

```
Clients ──► NGINX ──► Inference Proxy  ──► vLLM Node A
                           │           ──► vLLM Node B
                           │           ──► llama.cpp Node C
                           ▼
                          etcd
                     (service registry)
```

## Features

- **OpenAI-compatible API** -- drop-in replacement for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
- **Streaming support** -- Server-Sent Events (SSE) for real-time token generation
- **Chat playground** -- browser-based chat UI at `/chat` with markdown rendering and model selection
- **Service discovery** -- watches etcd for node registration/deregistration in real time
- **Least-connections load balancing** -- routes to the node with the fewest in-flight requests
- **Automatic failover** -- retries transport, timeout, and 5xx failures on alternate healthy nodes before a response begins (configurable, default 3 attempts)
- **Circuit breakers** -- per-node circuit breakers trip after consecutive failures, preventing cascade
- **Health checking** -- background thread probes each node's `/health` endpoint; marks nodes unhealthy after repeated failures and recovers them automatically
- **Graceful shutdown** -- Uvicorn drains in-flight requests before application resources close; its server timeout remains configurable
- **Structured logging** -- JSON or pretty console output via structlog
- **Operations dashboard** -- interactive web UI at `/dashboard` with real-time node and engine identity, catalog-backed setup controls, detail pages, and provisioning status
- **QUADS integration** -- background polling of QUADS inventory and availability; unified view merging QUADS hosts with etcd-registered nodes
- **QUADS schedule enforcement** -- automated teardown of managed nodes when QUADS reports an upcoming scheduling conflict
- **End-to-end node provisioning** -- SSH-based pipeline: BMC power-on, NVIDIA GPU verification, driver and CUDA toolkit install, inference engine setup (vLLM or llama.cpp), NFS mount, firewall, health poll, and etcd registration
- **Node teardown** -- graceful shutdown with connection draining, force teardown option, and provisioning task cancellation
- **Provisioning log streaming** -- live SSE stream of provisioning and inference engine logs viewable in the dashboard
- **BMC power management (Redfish)** -- query and control node power state; supports On, ForceOff, GracefulRestart, and ForceRestart
- **Model catalog** -- scans shared NFS-mounted HuggingFace cache, verifies model completeness via tree manifests, exposed via `/admin/models/catalog`
- **Background model downloads** -- concurrent HuggingFace downloads with status tracking; duplicate-safe and re-downloadable after completion or failure
- **Hardware-aware model recommendations** -- runs llmfit via SSH on a target host to produce ranked, runtime-normalized recommendations with fit levels, throughput, memory estimates, and typed GGUF sources; auto-installs the binary on first use
- **Request metrics** -- per-model and per-node counters exposed via `/admin/metrics`
- **Admin authentication** -- HTTP Basic required on all `/admin/*` endpoints and `/dashboard*` pages; inference API remains public
- **Backend endpoint allowlist** -- configurable hostname wildcard, CIDR network, and port allowlists; rejects non-matching registrations with loopback-only defaults
- **Client config downloads** -- one-click download of OpenCode CLI and Pi coding agent configuration files from the dashboard and node detail pages; dashboard configs point at the proxy for load-balanced access, node detail configs point at individual backend endpoints

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Verify it's running](#verify-its-running)
  - [Send a request](#send-a-request)
  - [Use with the OpenAI Python SDK](#use-with-the-openai-python-sdk)
  - [Chat playground](#chat-playground)
- [API Endpoints](#api-endpoints)
  - [Administrative access](#administrative-access)
  - [Node inventory identity](#node-inventory-identity)
  - [Relaunch managed llama.cpp sizing](#relaunch-managed-llamacpp-sizing)
  - [Force-recover an unregistered engine](#force-recover-an-unregistered-engine)
  - [Error responses](#error-responses)
- [Configuration](#configuration)
  - [Upgrade requirements](#upgrade-requirements)
  - [Server launch](#server-launch)
  - [Admin authentication](#admin-authentication)
  - [etcd](#etcd)
  - [Routing](#routing)
  - [SSH and provisioning commands](#ssh-and-provisioning-commands)
  - [HuggingFace model downloads](#huggingface-model-downloads)
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
- Node.js for the frontend behavioral tests (CI uses version 24; not required
  at runtime)

An etcd v3 service is required for persistent discovery, registration, and
provisioning state, but a temporary outage does not prevent the gateway from
starting. At least one healthy registered inference node (vLLM or llama.cpp) is required to serve
inference; health, discovery, dashboard, and provisioning functionality can
start with an empty registry.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/quadsproject/qiip.git && cd qiip

# Install dependencies
uv sync

# Copy and edit configuration
cp .env.example .env
# Set the required INFERENCE_PROXY_ADMIN__USERNAME and
# INFERENCE_PROXY_ADMIN__PASSWORD values in .env

# Run the gateway
uv run uvicorn inference_proxy.main:create_app --factory --host 0.0.0.0 --port 8080
```

The gateway starts even when etcd or inference nodes are temporarily
unavailable. Its discovery workers reconnect to etcd in the background, and
inference requests become routable after a healthy node is registered.

The administrative API and dashboard use HTTP Basic authentication, which sends
base64-encoded credentials --not encryption --on every request. A trusted work LAN
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

### Chat playground

The `/chat` playground saves its optional System Prompt in browser local
storage and sends it to the backend as an OpenAI `system` message. Some model
chat templates enforce strict user/assistant alternation and reject that role.
If such a model reports that conversation roles must alternate, clear the
System Prompt and retry. Failed turns are not retained in the next request's
history; partial assistant text already shown after a connection failure is
retained so the visible transcript and future context stay aligned.

## API Endpoints

Public endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Gateway health check (returns node count) |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST` | `/v1/completions` | Text completion (OpenAI-compatible) |
| `GET` | `/v1/models` | List models available across healthy nodes |
| `GET` | `/chat` | Browser chat playground |

HTTP Basic-protected administrative endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/nodes` | Unified registered and QUADS node inventory, including engine and artifact identity when known |
| `GET` | `/admin/metrics` | Request counters by model and node |
| `GET` | `/admin/models/catalog` | Verified models in the shared HuggingFace cache |
| `POST` | `/admin/models/download` | Start or inspect a duplicate-safe model download |
| `GET` | `/admin/models/downloads` | List tracked model-download states |
| `POST` | `/admin/nodes/setup` | Start background node provisioning |
| `POST` | `/admin/nodes/{hostname}/llamacpp/relaunch` | Drain and relaunch a healthy managed llama.cpp node with a typed sizing policy |
| `DELETE` | `/admin/nodes/{node_id}` | Drain and tear down a node; supports force and the scoped recovery procedure below |
| `GET` | `/admin/provisioning/tasks` | List provisioning task states |
| `GET` | `/admin/provisioning/{hostname}/logs` | Stream provisioning logs over SSE |
| `GET` | `/admin/quads/status` | QUADS integration and cache status |
| `GET` | `/admin/nodes/{hostname}/power` | Read Redfish power state |
| `POST` | `/admin/nodes/{hostname}/power` | Execute an allowed Redfish power action |
| `GET` | `/admin/nodes/{hostname}/recommendations` | Run hardware-aware model recommendations |
| `GET` | `/dashboard` | Authenticated operations dashboard |
| `GET` | `/dashboard/nodes/{node_id}` | Authenticated node detail page |

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

### Node inventory identity

Each `/admin/nodes` item reports the inference `engine` and immutable
`artifact_id` when QIIP knows them. Registered nodes report `engine` as
`"vllm"` or `"llama_cpp"`. A llama.cpp node provisioned from the managed GGUF
catalog also reports the selected 64-character artifact ID; vLLM and manually
registered nodes normally report `artifact_id: null`.

Successful managed llama.cpp setup also reports `llamacpp_runtime`, containing
the requested sizing policy, verified effective plan, device-indexed post-load
GPU memory, and an ISO-8601 UTC observation time. Other nodes and older records
report `llamacpp_runtime: null`; the memory values are a snapshot, not live
telemetry. Automatic policy records contain `sizing` and `fit_target_mib`.
Gateway-authorized custom records additionally contain exact
`context_per_slot`, `slots`, and `cache_type` values.

A host present only in QUADS has not been provisioned and therefore reports
both `engine: null` and `artifact_id: null`. Do not interpret a null engine as
vLLM. It means QIIP has no registered serving identity for that host.

### Relaunch managed llama.cpp sizing

The relaunch endpoint accepts the same typed automatic or custom policy stored
in `llamacpp_runtime.requested`. For example, this requests two simultaneous
24,576-token slots with matching Q8_0 K/V caches:

```bash
curl -X POST \
  -u "$INFERENCE_PROXY_ADMIN__USERNAME:$INFERENCE_PROXY_ADMIN__PASSWORD" \
  -H 'Content-Type: application/json' \
  http://gateway.example.com/admin/nodes/gpu01/llamacpp/relaunch \
  -d '{
    "sizing": "custom",
    "fit_target_mib": 512,
    "context_per_slot": 24576,
    "slots": 2,
    "cache_type": "q8_0"
  }'
```

Only a healthy managed llama.cpp node with an exact artifact and verified
runtime record is eligible. Unknown request fields are rejected. Custom
context must be 256-token aligned and no larger than the model training
context; slots are limited to 1-256; the aggregate must fit llama.cpp's
32-bit ceiling; and the reserve must be smaller than every observed GPU.

A 202 response queues a capacity-counted background operation. QIIP removes
the node from routing, waits for tracked requests to drain, stops the server,
estimates and launches the requested policy, and verifies the effective
runtime before restoring healthy status. A drain timeout restores the original
registration without stopping the server. A failed launch attempts the prior
requested policy: automatic sizing is recomputed, while custom sizing is
replayed exactly. If rollback also fails, the node enters `relaunch_failed`,
clears stale runtime telemetry, and permits teardown only. Follow progress at
`/admin/provisioning/{hostname}/logs`.

### Force-recover an unregistered engine

Normal teardown obtains the engine from an active provisioning operation or
the node's etcd registration and fails closed when neither exists. If etcd lost
a node record while a known inference process remained on the host, an operator
can supply the missing engine explicitly:

```bash
curl -X DELETE \
  -u "$INFERENCE_PROXY_ADMIN__USERNAME:$INFERENCE_PROXY_ADMIN__PASSWORD" \
  "http://gateway.example.com/admin/nodes/gpu01?force=true&recovery_engine=llama_cpp"
```

Use this recovery path only after verifying which engine is actually running.
It is accepted only when `force=true`, the node is unregistered, the hostname
passes the configured backend endpoint allowlist, and no host lifecycle
operation holds the lease. QIIP rechecks registration after acquiring the
lease and never cancels active provisioning on this path. A wrong
`recovery_engine` selects the wrong stop script; it is not treated as an engine
autodetection hint. A successful request returns 202 and runs teardown in the
background, with progress available from the normal provisioning log stream.

### Error responses

Inference-proxy errors follow the OpenAI error format. Upstream 4xx responses
are passed through without changing their JSON shape.

| Code | Meaning |
|------|---------|
| 404 | Model not found -- no node serves the requested model |
| 502 | Backend connection failed |
| 503 | No healthy nodes available, or model temporarily unavailable |
| 504 | Backend request timed out |

When an attempt loop ends after at least one retryable backend failure, the
error code is `failover_exhausted` and the response includes
`X-Inference-Proxy-Failover: exhausted` and
`X-Inference-Proxy-Attempts: <n>`. This means the configured attempt budget or
eligible-node set ended; it does not claim that every fleet node was tried.
See [Client-visible compatibility changes](UPGRADING.md#client-visible-compatibility-changes)
before upgrading inference clients.

## Configuration

All settings are loaded from environment variables with the prefix
`INFERENCE_PROXY_` and double-underscore nesting for nested groups. A `.env`
file is also supported. The checked-in [.env.example](.env.example) is the
exhaustive environment-variable reference; this section explains the settings
whose interactions or security properties need more context.

### Upgrade requirements

Deployments upgrading from before the reliability campaign must follow the
complete [upgrade and compatibility guide](UPGRADING.md). It covers startup
requirements, silent behavior changes, node-package and mirror policy,
lease-expiry recovery, and client-visible API changes.

The easiest changes to miss are that `ROUTING__MAX_ATTEMPTS` counts the first
request, missing etcd `managed` values now mean externally owned,
proxy-managed keys expire after their lease TTL without successful health
evidence, and streaming requests can return a non-200 response before SSE
begins.

Enabling QUADS requires both `INFERENCE_PROXY_QUADS__BASE_URL` and
`INFERENCE_PROXY_QUADS__SERVER_TIMEZONE`. Set the latter to the IANA timezone
used by the QUADS server's local clock, for example `America/New_York`. The
QUADS availability endpoint accepts timezone-naive `YYYY-MM-DDTHH:MM` values,
so the proxy converts its UTC scheduling deadline into that configured server
timezone before querying availability.

### Server launch

Uvicorn owns the listening socket and graceful request draining. Configure the
bind address and port with its `--host` and `--port` launcher options; there are
no `INFERENCE_PROXY_GATEWAY__*` bind settings. Configure
`--timeout-graceful-shutdown <seconds>` when Uvicorn's default drain timeout
does not fit the deployment.

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
| `INFERENCE_PROXY_ETCD__NODE_LEASE_TTL` | `600` | Lease TTL for healthy proxy-managed node keys; must exceed 300 seconds and three health cycles |

Endpoint values must include an HTTP(S) scheme. The current client uses the
first configured endpoint and warns when additional list entries are ignored;
multiple values do not currently provide client-side etcd failover. See the
[lease maintenance runbook](UPGRADING.md#8-plan-for-lease-backed-managed-registrations)
before a gateway outage longer than the active managed-node TTL.

### Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_ROUTING__MAX_ATTEMPTS` | `3` | Maximum total backend attempts, including the first request |
| `INFERENCE_PROXY_ROUTING__TIMEOUT` | `30` | Total pre-response streaming handshake budget across all attempts (seconds) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_HOSTS` | `["localhost"]` | Exact backend DNS names or `*.suffix` rules (JSON array) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_NETWORKS` | `["127.0.0.0/8","::1/128"]` | Backend IP CIDR allowlist (JSON array) |
| `INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_PORTS` | `[8000]` | Backend TCP port allowlist (JSON array) |

QIIP currently implements least-connections routing only. There is no strategy
setting; adding another algorithm requires an implementation rather than a
configuration-only change.

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

Provisioning resource and retention controls:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PROVISIONING__MAX_CONCURRENT_PROVISIONS` | `32` | Concurrent setup-task limit; excess setup requests return 429 while teardown remains available |
| `INFERENCE_PROXY_PROVISIONING__LOG_MAX_ENTRIES_PER_HOST` | `1000` | Retained log entries per host operation |
| `INFERENCE_PROXY_PROVISIONING__LOG_MAX_BYTES_PER_HOST` | `1048576` | Retained message bytes per host operation |
| `INFERENCE_PROXY_PROVISIONING__LOG_MAX_ENTRY_BYTES` | `16384` | Maximum bytes in one retained log message |
| `INFERENCE_PROXY_PROVISIONING__LOG_MAX_COMPLETED_HOSTS` | `64` | Completed host-operation buffers retained, oldest first |

Managed llama.cpp provisioning builds a verified source tag with CUDA enabled
for the NVIDIA GPU attached to the node. It has five gateway settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PROVISIONING__LLAMACPP_VERSION` | `b10242` | Pinned llama.cpp build tag |
| `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SHA256` | committed digest | SHA-256 of the source archive selected by the version |
| `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SOURCE_URL` | GitHub tag archive | Validated HTTP(S) URL template containing exactly one `{version}` placeholder |
| `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SETUP_TIMEOUT` | `7200` | Total wall-clock deadline for the llama.cpp setup command, including the CUDA source build (seconds) |
| `INFERENCE_PROXY_PROVISIONING__LLAMACPP_FIT_TARGET_MIB` | `512` | Free VRAM margin per GPU enforced by the llama.cpp capacity planner (MiB) |

Changing `LLAMACPP_VERSION` requires an explicitly configured matching digest.
The node verifies the archive before extracting it, builds `llama-server`,
`llama-fit-params`, and `llama-quantize`, and atomically publishes a versioned
installation under `/opt/llama.cpp`. CUDA kernels target the attached GPU; the
supporting CPU backend uses a portable non-native profile so host assembler
support cannot invalidate a CUDA build. The source build requires a working
NVIDIA driver and CUDA compiler; QIIP-managed llama.cpp nodes do not fall back
to CPU inference.

Managed launch uses llama.cpp's own memory estimator to maximize the guaranteed
context per request up to the model's trained length, then maximize concurrency
up to llama.cpp's 256-sequence limit. It sizes one unified KV pool to at least
`context_per_slot * slots`, so every selected slot has capacity for the reported
context instead of sharing one model-length pool across a fixed four slots.
The planner prefers F16 K/V cache storage. If F16 cannot fully offload even one
request at the 4,096-token floor while retaining the configured reserve, it
replans with Q8_0 for both K and V and enables Flash Attention. Managed mode
never falls below Q8_0 automatically.

Fresh managed setup uses that automatic policy and the globally configured
free-VRAM target. The internal provisioning contract can also carry a complete,
typed custom policy consisting of context per slot, slots, matching F16 or Q8_0
K/V cache, and a per-GPU free-VRAM target. Custom launch still runs the pinned
estimator once and fails before server startup unless the exact configuration
fully offloads and preserves its target. Host-ambient sizing overrides remain
unsupported; the custom contract is gateway-owned and is not a direct shell
configuration surface.

After `/health` succeeds, QIIP requires the runtime to match that plan, keep the
configured free-VRAM margin, use the selected KV types and unified cache, and
offload every model layer to GPU before registering the node healthy. The
provisioning log records the simultaneous per-slot guarantee, llama.cpp
per-request ceiling, aggregate context, slot count, KV types, layer offload,
configured margin, and post-load GPU memory. A successful managed setup also
persists that verified runtime state, including model training context and a
timestamped per-GPU memory snapshot, and exposes it on the node-detail
dashboard. See
[auto-llamacpp](auto-llamacpp/README.md) for the direct script contract and
build details.

LLMFit has one version setting: `INFERENCE_PROXY_LLMFIT__VERSION`.

The default NVIDIA driver and LLMFit versions each ship with a verified
SHA-256. Changing either version requires configuring its matching digest via
`INFERENCE_PROXY_PROVISIONING__NVIDIA_DRIVER_SHA256` or
`INFERENCE_PROXY_LLMFIT__SHA256`; provisioning fails before SSH or installation
when a custom version has no explicit digest.

The gateway cache path and node cache mount may differ. Provisioning uses one
declared backing export and, for llama.cpp, the gateway path that corresponds
to the root of that export:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR` | required | Gateway-local Hugging Face Hub cache directory |
| `INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT` | none | NFS export mounted on provisioned nodes |
| `INFERENCE_PROXY_HUGGINGFACE__SHARED_ROOT` | none | Gateway-local root of that same export; required for llama.cpp setup and must contain `CACHE_DIR` |
| `INFERENCE_PROXY_HUGGINGFACE__API_TOKEN` | none | Optional token for gated Hugging Face repositories |

For example, a gateway with shared root `/mnt/scratch`, Hub cache
`/mnt/scratch/hub`, and node mount `/srv/hf-cache` sends a native snapshot path
beginning with `hub/` to the node. `NFS_EXPORT` remains optional for proxy-only
deployments and is required before any node setup can acquire a host lease or
start SSH work. `SHARED_ROOT` is needed only for llama.cpp setup; catalog
browsing and vLLM setup do not need that path translation.

Node-side launch tuning uses the `AUTOVLLM_*` namespace. The retired
`VLLM_TENSOR_PARALLEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`,
`VLLM_MAX_BATCHED_TOKENS`, and `VLLM_EXTRA_ARGS` names are ignored by
`start-vllm.sh`. All seven reserved legacy inputs, including `VLLM_MODEL` and
`VLLM_PORT`, are removed from the child environment so script inputs cannot
leak into the namespace reserved by vLLM itself.

### HuggingFace model downloads

Every completed download records the immutable commit SHA returned by
HuggingFace. A request may supply a branch, tag, or commit through `revision`;
when it is omitted, the repository's default revision is resolved at download
time and the resulting SHA is still preserved in the status response.

Full vLLM snapshots use the default `engine: "vllm"`. A llama.cpp download
must name the exact files and load entrypoint:

```json
{
  "repo_id": "org/model-GGUF",
  "revision": "main",
  "engine": "llama_cpp",
  "gguf": {
    "files": ["model-Q4_K_M.gguf"],
    "entrypoint": "model-Q4_K_M.gguf"
  }
}
```

QIIP discovers GGUF generations directly from native Hugging Face snapshot
directories. It writes no parallel `gguf/` tree: every standalone `.gguf` is
one generation, while a complete llama.cpp split family is one generation with
shard 1 as its entrypoint. Existing snapshots downloaded outside QIIP therefore
become selectable without copying model data or publishing a manifest.

`/admin/models/catalog` keeps full vLLM models in `models` and returns exact
llama.cpp generations separately in `gguf_artifacts`. A GGUF can be discoverable
even when its repository contributes to `incomplete_count` or
`unverifiable_count`; those counters describe Hugging Face cache metadata and
are not suppressed merely because a valid GGUF exists.

The dashboard setup controls select either a full vLLM model or one exact GGUF
`artifact_id`; the two engine-specific values are never sent together. No GGUF
artifact is selected by default, so the operator must choose the intended
entrypoint or quantization. Node retry requests omit both values so the server
can retain the latest registered engine, model, and artifact identity under the
host lifecycle lease. A llama.cpp retry also retains its requested sizing
policy: automatic sizing is recomputed against current free VRAM, while custom
sizing replays the complete exact request and must pass estimation again.

LLMFit recommendation runtimes are normalized to `vllm`, `llama_cpp`, `mlx`,
or `unknown`. A llama.cpp recommendation can list typed `gguf_sources`, and the
node-detail dashboard reports a discovered generation as available only when a
source repository exactly matches an artifact's `repo_id`. LLMFit does not
identify an exact file set, shard group, or entrypoint, so the browser never
guesses a GGUF download request: it shows `No GGUF source`, `Not downloaded`,
`Catalog unavailable`, or the number of matching generations instead.

Re-downloading a mutable branch after it advances creates a distinct artifact
generation because identity uses the resolved SHA. QIIP never automatically
deletes snapshots. A persisted artifact ID resolves only while its native
snapshot and exact GGUF file family remain in the shared cache, so coordinate
retention with every running or restartable node that records that ID.

### Proxy (HTTP client)

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PROXY__CONNECT_TIMEOUT` | `5.0` | TCP connect timeout (seconds) |
| `INFERENCE_PROXY_PROXY__READ_TIMEOUT` | `120.0` | Read timeout -- high for LLM first-token latency |
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
| `INFERENCE_PROXY_LOGGING__LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`); invalid values fail startup |

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
│   ├── node_leases.py         # Managed-node lease observation and keepalive
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
│   └── drain_cleanup.py       # Atomic traffic-independent drain removal
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
5. Upstream 4xx responses are returned verbatim and neither increment nor reset the circuit breaker
6. Other exceptions record a circuit-breaker failure; non-retryable exceptions return a mapped error immediately
7. Transport, timeout, and 5xx exceptions are retryable on another eligible node while the attempt budget remains
8. Once an SSE response begins, stream failures are returned as an in-band error event followed by `[DONE]` without failover
9. When pre-response retries are exhausted, an OpenAI-format error and failover headers are returned

### Background threads

- **etcd watcher** -- watches the configured key prefix for node PUT/DELETE events; updates the registry in real time
- **Health checker** -- probes each registered node's `/health` endpoint, transitions liveness state, maintains managed-node leases after valid evidence, and removes idle draining ghosts
- **QUADS poller and schedule enforcer** -- refresh QUADS inventory and tear down managed nodes before scheduling conflicts, with bounded retry backoff

## Development

### Setup

```bash
# Install all dependencies (including dev)
uv sync --locked --all-groups

# Activate the virtual environment (optional -- uv run handles this)
source .venv/bin/activate
```

### Run tests

```bash
# All tests
uv run --frozen pytest

# With branch coverage (the same gate used by CI)
uv run --frozen coverage run -m pytest
uv run --frozen coverage report

# Specific module
uv run --frozen pytest tests/api/test_routes.py -v
```

Coverage is measured over `inference_proxy` with branch tracking enabled. CI
enforces a 92% combined statement-and-branch floor, raised from 91.5% when the
exact-artifact work brought the measured total to 92.08%. The total may move as
code is added or removed. The floor prevents new untested code from materially
reducing coverage; it does not prove that covered behavior is asserted
correctly.

### CI badges

After `Quality` passes on `main`, a separate non-blocking job publishes the
coverage, vLLM, and llama.cpp badges to the configured Gist. `GIST_SECRET` must
be a fine-grained personal access token with only the **Gists: write** user
permission. Prefer a service identity, record the token's expiration, and
replace the repository secret before it expires. To change the publishing
identity, create and pre-seed a Gist under the new owner, then update the Gist
ID in `.github/workflows/ci.yml` and all three README badge URLs together.
Verify that the three raw JSON URLs return HTTP 200 before merging that change.

### Lint and format

```bash
# Check lint
uv run --frozen ruff check .

# Auto-fix lint issues
uv run --frozen ruff check --fix .

# Format code
uv run --frozen ruff format .
```

### Type check

```bash
uv run --frozen mypy inference_proxy tests
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
