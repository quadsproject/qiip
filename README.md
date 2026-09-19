<p>
  <img src="assets/qiip.svg" width="15%" />
</p>

# QUADS Idle Inference Proxy

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
- **Claude Code and Codex** -- the Anthropic Messages API (`/v1/messages`) and the OpenAI Responses API (`/v1/responses`) are forwarded to vLLM and llama.cpp nodes, which implement them natively, with the same token auth, routing, failover, and usage tracking as chat completions
- **Streaming support** -- Server-Sent Events (SSE) for real-time token generation
- **Chat playground** -- browser-based chat UI at `/chat` with markdown rendering and model selection
- **Service discovery** -- watches etcd for node registration/deregistration in real time
- **Least-connections load balancing** -- routes to the node with the fewest in-flight requests
- **Automatic failover** -- retries transport, timeout, and 5xx failures on alternate healthy nodes before a response begins (configurable, default 3 attempts)
- **Circuit breakers** -- per-node circuit breakers trip after consecutive failures, preventing cascade
- **Health checking** -- background thread probes each node's `/health` endpoint; marks nodes unhealthy after repeated failures and recovers them automatically. Self-setup nodes fall back to `/v1/models` when `/health` is missing (HTTP 404/405/501)
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
- **Admin authentication** -- HTTP Basic credentials or a signed-in admin-role session (local-admin form or Google OAuth) on all `/admin/*` endpoints; browser pages gate with a sign-in page instead of 401ing
- **Fleet sign-in gate** -- anonymous visitors to the fleet dashboard get a sign-in page with two choices: **Local Admin** (a collapsible option that expands to an in-page username/password form establishing a signed session cookie — no browser Basic challenge popup; HTTP Basic still works for scripts and SSE) and **Google Auth** (same flow as the profile page)
- **Admin roles** -- the HTTP Basic admin user (bootstrap authority) can grant or revoke the admin role to Google-authenticated users on the token dashboard (`/dashboard/tokens`); role admins then reach the admin surface through their session and see admin-only servers
- **Admin-only inference servers** -- admin-defined adopted OpenAI-compatible servers (URL-based, self-setup semantics, no provisioning steps). At `/v1` they are routable only to bearer tokens of admin-role users or the full-access trust list (HTTP Basic covers UI surfaces only; `/v1` is Bearer-only), never listed on the non-admin fleet page or public `/v1/models`, and appear bold with an `admin_only` badge in the admin fleet view. Token usage from admin-only servers is tracked on the token summary pages exactly like any other node
- **Google OAuth (SSO)** -- open `/profile` to sign in with a Google account (optional hosted-domain allowlist); sessions ride a signed cookie
- **Self-service onboarding** -- signed-in normal users land on `/start`: one question per screen (name a token, pick a coding tool, pick models) ending in a short-lived `curl ... | bash` line that writes the tool's config; returning users see their single token and its models. See [Self-service onboarding](#self-service-onboarding-start)
- **Per-token model scope** -- a token minted by the onboarding flow may only request the models chosen for it; other models are refused on `/v1` with `403 model_not_permitted`, and `/v1/models` lists only the token's models
- **User API tokens** -- admin-role users can mint `qiip_...` bearer tokens on their profile page to call the `/v1` inference API; normal users own exactly one token, managed on `/start`. Tokens are stored as SHA-256 digests and can be revoked at any time
- **Stable agent-config token** -- one derived per-user key (`agent-config`) is shared by every config download across servers and browsers; its raw value is derived from `auth.session_secret` + user + generation and never stored, so revoking it rotates the key embedded in already-downloaded configs (configuration downloads for admin-only servers require the Google session that can mint it)
- **Config-gated inference auth** -- a valid `qiip_...` bearer token is always accepted on `/v1`; requiring a token for every `/v1` request (`auth.enforce_api_tokens`) is optional and off by default, so existing public deployments keep serving anonymous requests unchanged
- **Token usage tracking** -- token-authenticated requests record token usage per token/model for reporting on the profile page
- **Backend endpoint allowlist** -- configurable hostname wildcard, CIDR network, and port allowlists; rejects non-matching registrations with loopback-only defaults
- **Client config downloads** -- one-click download of OpenCode CLI and Pi coding agent configuration files from the dashboard and node detail pages; dashboard configs point at the proxy for load-balanced access, node detail configs point at individual backend endpoints

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Running etcd](#running-etcd)
- [Quick Start](#quick-start)
  - [Verify it's running](#verify-its-running)
  - [Send a request](#send-a-request)
  - [Use with the OpenAI Python SDK](#use-with-the-openai-python-sdk)
  - [Chat playground](#chat-playground)
- [API Endpoints](#api-endpoints)
  - [Claude Code and Codex](#claude-code-and-codex)
  - [Administrative access](#administrative-access)
  - [Node inventory identity](#node-inventory-identity)
  - [Relaunch managed llama.cpp sizing](#relaunch-managed-llamacpp-sizing)
  - [Force-recover an unregistered engine](#force-recover-an-unregistered-engine)
  - [Error responses](#error-responses)
- [Configuration](#configuration)
  - [Upgrade requirements](#upgrade-requirements)
  - [Server launch](#server-launch)
  - [Admin authentication](#admin-authentication)
  - [User authentication (Google OAuth)](#user-authentication-google-oauth)
  - [Self-service onboarding (`/start`)](#self-service-onboarding-start)
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
- [Durable provisioning evidence](#durable-provisioning-evidence)
- [Troubleshooting](#troubleshooting)
  - [Reading provisioning logs offline](#reading-provisioning-logs-offline)
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

### Running etcd

The gateway expects etcd on `localhost:2379` by default. Run a single-node
instance with Podman:

```bash
podman run -d --name etcd -p 2379:2379 \
  -v etcd-data:/etcd-data \
  quay.io/coreos/etcd:v3.5.21 \
  /usr/local/bin/etcd \
  --data-dir /etcd-data \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-client-urls http://0.0.0.0:2379
```

Verify it is healthy:

```bash
curl -s http://localhost:2379/health
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/quadsproject/qiip.git && cd qiip

# Install dependencies
uv sync

# Copy and edit configuration. YAML is the primary configuration format;
# create a conf/ directory in the gateway's working directory (or set
# INFERENCE_PROXY_CONF_DIR to a directory such as /etc/qiip/conf).
mkdir -p conf
cp conf/qiip.yml.example conf/qiip.yml
cp conf/auth.yml.example conf/auth.yml
cp conf/plugins.yml.example conf/plugins.yml
# Set the required admin.username and admin.password values in
# conf/qiip.yml, and huggingface.cache_dir (required).
# Environment variables with the INFERENCE_PROXY_ prefix override any
# value in the YAML files, so secrets can stay in the environment or .env.

# Run the gateway
uv run uvicorn inference_proxy.main:create_app --factory --host 0.0.0.0 --port 5000
```

The gateway starts even when etcd or inference nodes are temporarily
unavailable. Its discovery workers reconnect to etcd in the background, and
inference requests become routable after a healthy node is registered.

The administrative JSON API accepts HTTP Basic credentials or a signed-in
admin-role session; browser pages use a signed session cookie (local-admin
form or Google OAuth) and never show a native Basic challenge. HTTP Basic
sends base64-encoded credentials --not encryption --on every request. A trusted
work LAN may use HTTP; use a TLS terminator whenever that network path is not
trusted.

Optional TLS termination (rootless Podman container or RPM nginx, self-signed
certificate bootstrap): see [nginx/nginx.md](nginx/nginx.md).

### Verify it's running

```bash
curl http://localhost:5000/health
# {"status": "ok", "nodes_registered": 2}
```

### Send a request

```bash
# Non-streaming
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:5000/v1/chat/completions \
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
    base_url="http://localhost:5000/v1",
    api_key="not-needed",  # no auth in v1
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Explain QUADS in one sentence."}],
)
print(response.choices[0].message.content)
```

### Deploy with systemd

A production deployment ships a systemd unit (`systemd/inference-proxy.service`)
matching the stage/dev convention: repo checkout at `/opt/inference-proxy`
(uv-synced), settings in `/opt/inference-proxy/.env`, the service listening on
port **5000**, and nginx terminating TLS and proxying to it
(`nginx/nginx.conf`). Install it with:

```bash
sudo cp systemd/inference-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now inference-proxy
```

The unit runs as `root` because provisioning stores host SSH keys under
`/root/.ssh`; tighten it if the deployment does not provision nodes. Use the
same port (`5000`) for the gateway and the nginx upstream — a deployer mixing
the quick-start port with the shipped unit behind nginx gets 502s.

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
| `POST` | `/v1/messages` | Anthropic Messages API (Claude Code) |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting |
| `POST` | `/v1/responses` | OpenAI Responses API (Codex) |
| `GET` | `/v1/models` | List models available across healthy nodes |
| `GET` | `/chat` | Browser chat playground |
| `GET` | `/profile` | Profile page: Google sign-in, API-token manager, and per-token usage (signed-in normal users are redirected to `/start`) |
| `GET` | `/start` | Onboarding wizard and token home for signed-in normal users; anonymous visitors get the sign-in page, admins are redirected to `/dashboard` |
| `GET` | `/s/{id}` | Setup script for a live onboarding link. The id is the credential: 15 minute life, not logged. A dead link returns a script that explains and exits 1 |
| `GET` | `/auth/login` | Start Google OAuth sign-in (302 to Google) |
| `GET` | `/auth/callback` | Google redirect target; signs the session cookie |
| `GET` | `/auth/local-admin` | Local admin login page entry (302 to `/dashboard`; the sign-in form POSTs here) |
| `POST` | `/auth/local-admin` | Sign in as the local admin via the form; sets the session cookie and 302s to `/dashboard` |
| `POST` | `/auth/logout` | Clear the session cookie |
| `GET` | `/auth/me` | JSON identity of the signed-in user (401 when anonymous) |

Fleet (any signed-in user, or HTTP Basic local admin):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/fleet/nodes` | Fleet JSON for non-admin viewers: registered nodes without admin-only servers or operational actions (no page renders it for them any more; normal users live on `/start`) |

User-session-protected profile endpoints (require `auth.session_secret` and a
signed-in session):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/profile/me` | Public identity of the signed-in user |
| `GET` | `/profile/tokens` | List the user's API tokens (prefix only) |
| `POST` | `/profile/tokens` | Admin-role users only (403 for normal users, who use `/start`). Mint a token; accepts an optional `endpoints` pin (hostnames); returns the raw secret exactly once (except `name: agent-config`, the reusable derived config key) |
| `DELETE` | `/profile/tokens/{id}` | Revoke a token |
| `GET` | `/profile/usage` | Aggregated usage per token/model plus headline totals |
| `GET` | `/profile/endpoints` | Registered nodes the user may pin (unowned nodes plus nodes they own) |

Onboarding endpoints (signed-in normal users only; admin-role sessions get 403):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/onboarding/state` | Identity, the user's current token (never the raw secret), models online right now, and the supported coding tools |
| `POST` | `/onboarding/token` | Mint the user's single token with a name and a model list; revokes every other token the user has |
| `PUT` | `/onboarding/token/models` | Replace the token's model list (409 for tokens minted before this flow) |
| `POST` | `/onboarding/setup-link` | Create a `/s/{id}` link for one coding tool; returns the ready-to-paste command and its expiry |

Admin-authenticated endpoints (HTTP Basic or admin-role session):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/nodes` | Unified registered and QUADS node inventory, including engine and artifact identity when known |
| `GET` | `/admin/metrics` | Request counters by model and node |
| `GET` | `/admin/models/catalog` | Verified models in the shared HuggingFace cache |
| `POST` | `/admin/models/download` | Start or inspect a duplicate-safe model download |
| `GET` | `/admin/models/downloads` | List tracked model-download states |
| `POST` | `/admin/nodes/setup` | Start background node provisioning |
| `POST` | `/admin/nodes/pool` | Add a node to the available pool, or adopt an already-running OpenAI-compatible server; `"admin_only": true` registers an admin-only server (implies `self_setup`) |
| `POST` | `/admin/nodes/{hostname}/llamacpp/relaunch` | Drain and relaunch a healthy managed llama.cpp node with a typed sizing policy |
| `DELETE` | `/admin/nodes/{node_id}` | Drain and tear down a node; supports force and the scoped recovery procedure below |
| `PATCH` | `/admin/nodes/{node_id}/owner` | Set or clear a node's owner email (`"owner": ""` clears it) |
| `GET` | `/admin/users` | List Google users with token usage and `is_admin` |
| `POST` | `/admin/users/{user_id}/admin` | Grant the admin role (204) |
| `DELETE` | `/admin/users/{user_id}/admin` | Revoke the admin role (204) |
| `GET` | `/admin/provisioning/tasks` | List provisioning task states |
| `GET` | `/admin/provisioning/{hostname}/logs` | Stream provisioning logs over SSE |
| `GET` | `/admin/quads/status` | QUADS integration and cache status |
| `GET` | `/admin/nodes/{hostname}/power` | Read Redfish power state |
| `POST` | `/admin/nodes/{hostname}/power` | Execute an allowed Redfish power action |
| `GET` | `/admin/nodes/{hostname}/recommendations` | Run hardware-aware model recommendations |
| `GET` | `/dashboard` | Authenticated operations dashboard; anonymous visitors get the sign-in page |
| `GET` | `/dashboard/nodes/{node_id}` | Authenticated node detail page |
| `GET` | `/dashboard/admin` | Admin page: manage admin-only inference servers (admin-role management lives on `/dashboard/tokens`) |

### Claude Code and Codex

Claude Code speaks the Anthropic Messages API and Codex speaks the OpenAI
Responses API. vLLM and llama.cpp both implement these APIs natively, so qiip
forwards requests without converting between formats. The one change it makes
is the system-message rewrite described below. Point the tools at the gateway
with a `qiip_...` token and a served model name:

```bash
# Claude Code (ANTHROPIC_API_KEY, sent as x-api-key, also works)
export ANTHROPIC_BASE_URL=https://inference-proxy.example.com
export ANTHROPIC_AUTH_TOKEN=qiip_...
export ANTHROPIC_MODEL=Qwen/Qwen3-14B-AWQ
export ANTHROPIC_DEFAULT_OPUS_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_SONNET_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$ANTHROPIC_MODEL
```

```toml
# Codex (~/.codex/config.toml), with `export QIIP_API_KEY=qiip_...`
model = "Qwen/Qwen3-14B-AWQ"
model_provider = "qiip"

[model_providers.qiip]
name = "qiip"
base_url = "https://inference-proxy.example.com/v1"
wire_api = "responses"
env_key = "QIIP_API_KEY"
```

- **Same gateway behavior.** Requests go through the same token
  authentication, node selection, failover, and per-token usage tracking as
  chat completions. Anthropic reports cached prompt tokens separately; qiip
  counts them as prompt tokens.
- **Streams are relayed event by event** with their SSE event names, which
  the Anthropic SDK requires. These formats do not end with `[DONE]`.
- **System messages inside the conversation.** Claude Code sends system
  messages between turns, and Codex sends a developer message. Some chat
  templates (Qwen 3.x, for example) accept a system message only as the first
  message. qiip therefore merges system and developer messages that come before
  the conversation into the system prompt (Codex: `instructions`), and turns
  every later one into a user message at the same position, wrapped in
  `<system-reminder>` tags. Moving the later ones to the front instead would
  change the start of the prompt every turn and defeat the backend's prompt
  cache. This applies on every node. Templates that require strict
  user/assistant alternation (Gemma-style) are still not supported, because
  the rewrite can place two user messages in a row.
- **Stateless Responses API.** Send the whole conversation each turn, as Codex
  does with `store: false`. `previous_response_id` and stored-response
  retrieval are not supported.
- **Context size.** Neither tool knows a local model's context window. Claude
  Code assumes 200,000 tokens for unknown models; set
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` to the
  served model's limits. For Codex, set `model_context_window` and
  `model_auto_compact_token_limit`.
- **Long prompts on llama.cpp.** llama.cpp sends no response headers until
  the first token, so processing a long uncached prompt counts against the
  streaming handshake deadline, `routing.timeout` (default 30 seconds). If
  agents resume large contexts on llama.cpp nodes, raise it above the longest
  prompt processing time you expect.
- **Backend support.** Managed nodes at the pinned vLLM and llama.cpp versions
  serve both APIs. An adopted server that does not implement them returns its
  own 404, which qiip passes through.

### Administrative access

All `/admin/*` API endpoints and `/dashboard` admin pages accept either the
shared HTTP Basic credentials configured below or a signed-in Google user
carrying the **admin role** (granted by the HTTP Basic admin user on the
token dashboard at `/dashboard/tokens`). The inference API, chat page, profile page,
and health endpoint are public; the inference API may additionally require a
user API token (see [User authentication (Google OAuth)](#user-authentication-google-oauth)).
For example:

```bash
curl -u "$INFERENCE_PROXY_ADMIN__USERNAME:$INFERENCE_PROXY_ADMIN__PASSWORD" \
  http://gateway.example.com/admin/nodes
```

The fleet page (`/dashboard`) is available to every authenticated viewer:
anonymous visitors receive a sign-in page with **Local Admin**
(a collapsible choice that expands to an in-page username/password form
creating a signed admin session; the browser native Basic prompt is no longer
used, though HTTP Basic requests and SSE still pass through unchanged) and
**Google Auth** (the same flow as the profile page).
Signed-in non-admin users do not see the fleet pages at all: every operations
page redirects them to `/start` (see
[Self-service onboarding](#self-service-onboarding-start)). The
`/fleet/nodes` JSON API still answers for them with admin-only servers
removed, no operational actions, and nodes owned by another user excluded
(ownership is private: `/v1/models` and the endpoint picker treat it the same
way).

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

### Admin-only inference servers

`POST /admin/nodes/pool` with `"admin_only": true` (which implies `"self_setup":
true`) registers an already-running OpenAI-compatible server as an
**admin-only server**: no provisioning steps are performed, QIIP never owns its
lifecycle, and its node id is the server hostname. Admin-only servers:

- are routable only to bearer tokens of admin-role users or the
  `admin_only_tokens_full_access` trust list — node selection, retries, and
  `/v1/models` all enforce this (HTTP Basic covers UI surfaces only; `/v1`
  accepts Bearer tokens, and anonymous/Basic callers are treated as unowned
  and rejected by selection);
- are never listed on the non-admin fleet page or in the public `/v1/models`
  catalog; admins see them in `/admin/nodes` with `"admin_only": true`,
  displayed bold with an `admin_only` badge;
- track per-token usage on the profile and admin token summary pages exactly
  like any other node;
- may carry an operator-facing display `name` (e.g. `"DeepSeek-V4-Flash-Vision-Exp (qiip)"`)
  that is shown in place of the raw short hostname in the admin fleet view;
- are removed with the normal pool removal endpoint
  (`DELETE /admin/nodes/{node_id}/pool`) — the server itself keeps running.

The server URL host and port must satisfy the configured endpoint allowlist
before registration is accepted.

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

### Adopt an existing OpenAI-compatible server

`POST /admin/nodes/pool` with `"self_setup": true` adopts a server that is
already running on the host and exposing the OpenAI-compatible API. QIIP
registers it without provisioning it and never owns its lifecycle. The optional
`port` selects a non-default listening port:

```bash
curl -X POST \
  -u "$INFERENCE_PROXY_ADMIN__USERNAME:$INFERENCE_PROXY_ADMIN__PASSWORD" \
  -H 'Content-Type: application/json' \
  http://gateway.example.com/admin/nodes/pool \
  -d '{"hostname": "gpu01", "self_setup": true, "port": 9000}'
```

Adoption is keyed on the OpenAI-compatible contract (`GET /v1/models`).
`/health` is probed best-effort, but only a missing endpoint (HTTP 404/405/501)
is treated as optional; an authoritative unhealthy response (such as `/health`
503) refuses adoption because the model list does not establish inference
readiness. QIIP registers the first model the server reports and never tears
the node down.

Requirements and limits:

- **Plain HTTP only**: a backend that requires credentials on `/v1/models` is
  not supported.
- **Allowed port**: `port` must satisfy `routing.allowed_endpoint_ports`.
  Omitting it uses the configured `provisioning.vllm_port`. Re-adopting a node
  without a `port` currently selects that configured default rather than the
  port recorded on the previous registration.
- **Single model**: a self-setup node tracks exactly one model — the primary id
  the server reports first on `/v1/models`. Aliases and LoRA entries are
  ignored by design.
- **Breaker recovery**: opening the breaker recovers by probing the registered
  model on `/v1/completions`, so the server must expose an OpenAI-compatible
  completions endpoint.

A custom `port` is rejected for a plain pool registration (without
`self_setup`), because that node is provisioned later on the configured default
`provisioning.vllm_port` and a stored custom port would be silently replaced at
launch. The dashboard's manual-setup form shows the port field only when the
"Existing OpenAI-compatible server" option is enabled.

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

The node-detail dashboard exposes the same contract as a typed editor. It
seeds every control from `llamacpp_runtime.requested`, including a non-default
automatic reserve. Automatic sizing leaves the effective context, slots, and
cache visible but disabled; custom sizing enables exact context, slot, and
F16/Q8_0 cache controls. The aggregate preview is arithmetic only. It does not
predict whether the requested configuration will fit; the gateway runs the
authoritative estimator after the node drains.

Applying a policy requires confirmation and follows the per-host provisioning
task and log stream. The form remains disabled during an ambiguous network
outcome until the new task generation is observed. If no new task appears,
reload the page to reconcile the current node and task state before retrying;
the gateway lifecycle lease rejects a concurrent duplicate. Polling silently
adopts a new verified policy while the form is pristine. If another browser
changes the runtime while local edits exist, the editor marks them stale and
requires a reset instead of submitting against an obsolete observation.

A 202 response queues a capacity-counted background operation. QIIP removes
the node from routing, waits for tracked requests to drain, stops the server,
estimates and launches the requested policy, and verifies the effective
runtime before restoring healthy status. A drain timeout restores the original
registration without stopping the server. A failed launch attempts the prior
requested policy: automatic sizing is recomputed, while custom sizing is
replayed exactly. If rollback also fails, the node enters `relaunch_failed`,
clears stale runtime telemetry, and permits teardown only. Follow progress at
`/admin/provisioning/{hostname}/logs`. After an interrupted relaunch, startup
reconciliation also marks the stale provisioning task failed and records the
step where the gateway stopped.

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

Inference-proxy errors follow the OpenAI error format, except on
`/v1/messages` and `/v1/messages/count_tokens`, where they use Anthropic's
`{"type": "error", "error": {"type": ..., "message": ...}}` envelope with
the same qiip code in `error.code`. Upstream 4xx responses are passed through
without changing their JSON shape.

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

Configuration is loaded from modular YAML files in a `conf/` directory
(`conf/qiip.yml`, `conf/auth.yml`, `conf/plugins.yml`), mirroring the
[QUADS conf/ layout](https://github.com/quadsproject/quads/tree/development/conf).
The directory is taken from `INFERENCE_PROXY_CONF_DIR` and defaults to `conf/`
relative to the working directory. Copy the checked-in examples
(`conf/*.yml.example`) and edit them. Load precedence, highest first:

1. Settings passed to the app constructor
2. `INFERENCE_PROXY_*` environment variables
3. YAML files in `INFERENCE_PROXY_CONF_DIR` (merged in filename order)
4. `.env` file
5. Built-in defaults

This keeps existing deployments working unchanged: a host that only sets
environment variables is unaffected, secrets can stay in exported environment
variables or `.env`, and YAML always wins over a stale `.env` for values it
actually sets. A `null` in YAML means unset, so the environment, `.env`, or
the built-in default still applies (the shipped examples use `null` for
secrets, so copying them cannot clobber a secret a host keeps in `.env`; the
admin password is an empty string on purpose and must be set in
`conf/qiip.yml`). Unrecognized section names fail startup, while unrecognized
keys inside a section are ignored (matching today's handling of unknown
environment variables). The checked-in
[.env.example](.env.example) remains the exhaustive environment-variable
reference. The rest of this section explains the settings whose interactions
or security properties need more context.

Existing `.env`-based hosts can migrate in one shot with a tested one-time
script shipped with this feature (see the pull request for
`qiip-env-to-conf.py`): it converts every well-formed
`INFERENCE_PROXY_GROUP__FIELD` value into the matching YAML file(s), skips
retired groups with a warning, writes the conf directory `0700` and files
`0600`, and leaves the `.env` untouched. Environment variables still win after
the migration, so exported or unit-managed values continue to apply; remove
migrated keys from `.env` once the YAML files are trusted, but keep the file
present: the packaged `systemd/inference-proxy.service` reads
`EnvironmentFile=/opt/inference-proxy/.env`, so an empty or comment-only
`.env` is the safe end state. Restart the gateway afterwards
(`sudo systemctl restart inference-proxy` with the packaged unit, otherwise
restart whatever supervises the process).

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

### User authentication (Google OAuth)

User accounts are optional and off by default. When enabled, users sign in with
their Google account on `/profile`, mint personal `qiip_...` bearer tokens, and
track per-token inference usage. User data and token digests live in a SQLite
database (`auth.db_path`); nothing is ever stored in the session cookie except
the signed user id and expiry.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_OAUTH__CLIENT_ID` | required (to enable) | Google OAuth 2.0 client id |
| `INFERENCE_PROXY_OAUTH__CLIENT_SECRET` | required (to enable) | Google OAuth 2.0 client secret, stored as a masked secret |
| `INFERENCE_PROXY_OAUTH__REDIRECT_URI` | required (to enable) | Absolute `http(s)://` callback URI, e.g. `https://gateway.example.com/auth/callback` |
| `INFERENCE_PROXY_OAUTH__ALLOWED_DOMAINS` | `[]` | JSON array of hosted domains allowed to sign in; empty allows any Google account |
| `INFERENCE_PROXY_OAUTH__ALLOWED_REDIRECT_HOSTS` | `[]` | JSON array of extra hostnames that may start an OAuth flow (multi-name deployments behind one wildcard cert, e.g. `["inference-proxy.scalelab.example.com"]`); the callback returns to the hostname used to sign in. Hosts outside the list fall back to `REDIRECT_URI`, so single-name deployments are unchanged |
| `INFERENCE_PROXY_AUTH__DB_PATH` | `data/qiip.db` | SQLite file holding users, token digests, and usage |
| `INFERENCE_PROXY_AUTH__SESSION_SECRET` | required for browser sign-in | Long random secret signing the session cookie (local-admin form and Google OAuth) |
| `INFERENCE_PROXY_AUTH__SESSION_COOKIE` | `qiip_session` | Session cookie name (alphanumeric plus `_` and `-`) |
| `INFERENCE_PROXY_AUTH__SESSION_TTL_SECONDS` | `43200` | Session lifetime (300 to 7 days) |
| `INFERENCE_PROXY_AUTH__ENFORCE_API_TOKENS` | `false` | Require a valid bearer token for every `/v1` inference request |
| `INFERENCE_PROXY_AUTH__REQUIRE_EMAIL_VERIFICATION` | `true` | Reject Google accounts whose email is not verified |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_URL` | unset | HTTPS URL returning a per-domain JSON whitelist, e.g. `{"example.com": ["alice", "bob"]}` |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_DEFAULT_DOMAIN` | unset | Optional domain (e.g. `example.com`) that resolves bare usernames in a flat JSON whitelist list to `user@domain` |
| `INFERENCE_PROXY_AUTH__ENFORCE_SSO_WHITELIST` | `false` | Gate SSO users on the per-domain username whitelist |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_POLL_INTERVAL` | `hourly` | Refresh cadence: `hourly` (top of the hour) or `daily` |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_POLL_TIME` | unset | `HH:MM` wall-clock refresh time; required with `daily` (server local time) |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_CACHE_FILE` | unset | Optional flat-file cache of the last successful document (warm start + inspection, atomically replaced) |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_EXTRA_USERS` | `[]` | Extra emails always allowed, merged over the fetched document |
| `INFERENCE_PROXY_AUTH__SSO_WHITELIST_EXTRA_DOMAINS` | `[]` | Extra domains where any username is allowed, merged over the fetched document |
| `INFERENCE_PROXY_AUTH__ADMIN_ONLY_TOKENS_FULL_ACCESS` | `[]` | Emails whose tokens get unrestricted access — no owner isolation, no endpoint pin, no SSO whitelist gate — but only when unpinned: stored pins are still honored; they may pin tokens to any endpoint |

Enablement and guardrails:

- Enable OAuth by configuring all three `OAUTH__*` credential variables; they
  are validated all-or-none. `auth.session_secret` is then required, and
  `auth.enforce_api_tokens` is only allowed while OAuth is enabled (users need a
  way to mint tokens).
- Google user accounts are keyed by their stable `sub` claim, so a renamed email
  still resolves to the same account.
- `/v1` behavior (AUTH-03): a valid `qiip_...` bearer token is always
  accepted and attributes usage. What happens without a usable token is set
  by `auth.enforce_api_tokens`. With it `false` (the default), an absent,
  invalid, or unknown bearer token simply means an anonymous request — no
  token is ever *required*. With it `true`, absent or invalid tokens are
  rejected with an OpenAI-shaped `401 invalid_api_key`. Keep the default to
  preserve fully public `/v1` deployments; set
  `INFERENCE_PROXY_AUTH__ENFORCE_API_TOKENS=true` once you want to require a
  token.
- `/v1/models`, `/health`, and the chat playground stay public in both modes;
  `/v1/models` lists only models served by unowned nodes (owner-private models
  are never enumerated).
- Anonymously reached `/v1` requests are proxied but not attributed; only
  token-authenticated calls record per-token usage (AUTH-04).

SSO whitelist (per-domain user filtering):

- `sso_whitelist_url` and `enforce_sso_whitelist` must be set together;
  either one without the other fails startup (a silently dead allowlist is
  worse than none). The URL must be HTTPS, is fetched with a 5s/10s timeout,
  does not follow redirects, and is capped at 1 MiB. The document maps
  domains to username lists:
  `{"example.com": ["alice", "bob"], "lab.example.com": ["carol"]}`.
  Matching is case-insensitive on both the domain and the username.
  Alternatively the document may be a flat list of usernames when
  `sso_whitelist_default_domain` is set (a domain like `example.com`):
  `["alice", "bob"]` is treated as `alice@example.com`, `bob@example.com`;
  entries containing `@` are used as-is. A flat list without a configured
  default domain fails closed. The payload can live anywhere reachable over
  HTTPS: a static file, an S3
  object, a config repo, or output from an LDAP/group export; the guard
  rejects literal non-global IP addresses and `localhost` names, and DNS
  names are operator-trusted (TLS still verified). Local grants do not
  require the remote feed at all (`sso_whitelist_extra_users` /
  `sso_whitelist_extra_domains`).
- Domain-level control is the existing `oauth.allowed_domains` gate
  (`domain_not_allowed` at sign-in); the whitelist adds username-level
  filtering inside allowed domains.
- Caching: the fetched document is held in memory and refreshed at the
  configured cadence (hourly at the top of the hour, or daily at
  `sso_whitelist_poll_time` in the server's local time) on the first
  check after the window. The optional `sso_whitelist_cache_file` persists
  the last successful document: it seeds a cold start and is atomically
  replaced on refresh, and is only authoritative inside the current refresh
  window. `sso_whitelist_extra_users` (emails) and
  `sso_whitelist_extra_domains` (any username in the domain) are granted
  locally on top of the fetched document, so specific accounts or domains
  can be opened without touching the remote payload.
- Enforcement points when the flag is on:
  1. `/auth/callback`: a user not on the list is redirected with
     `error=not_whitelisted` before any user row is created or session is
     signed.
  2. `POST /profile/tokens`: minting is denied with 403.
  3. `/v1` token resolution: an already-minted token whose owner is no
     longer listed is rejected with 401, so removals take effect on the
     next request (within the refresh window).
- Failure semantics are fail closed: if the URL is unreachable, returns a
  non-200, or the document is invalid/oversized, minting returns 503, the
  callback redirects with `error=allowlist_unavailable`, and presented
  tokens are rejected. Stale cached data is never served past the refresh
  window; a fetch failure enters a 30s cooldown so an outage does not queue
  a fetch per request.
- The whitelist governs the user identity, not anonymous traffic: while
  `enforce_api_tokens` is `false`, `/v1` still accepts requests without a
  token. Combine both flags to fully gate inference.

Endpoint scoping (per-token pins and owner isolation):

- A token may be pinned at creation to one or more endpoint hostnames
  (`POST /profile/tokens` with `endpoints: ["host1.example.com"]`, selected
  via the profile page). A pinned token routes only to those nodes — node
  selection and retry/failover stay inside the pin — and requests whose
  model exists only off-pin get the normal 404/503 error mapping. The pin is
  enforced for every token, admin-role and full-access tokens included
  (admins may pin any registered node, but the pin still binds them).
- Nodes may carry an `owner` (email) set at registration
  (`POST /admin/nodes/pool`, `POST /admin/nodes/setup`) or later with
  `PATCH /admin/nodes/{node_id}/owner` (empty string clears it). An owned
  node is reachable only by that owner's tokens and admin full-access
  tokens; unowned nodes stay shared. Node selection and `/v1` routing are
  filtered accordingly for every caller, including anonymous requests, so
  owned endpoints are never reached by other users' tokens or by
  anonymous traffic.
- `GET /profile/endpoints` lists the registered nodes the signed-in user may
  pin (unowned nodes plus nodes they own); admins see everything. Minting
  rejects unknown hostnames (400) and nodes owned by someone else (403), and
  rejects an empty pin.
- `/v1/models` never lists models served by owner-private nodes, so ownership
  stays private even on the public catalog.
- `admin_only_tokens_full_access` is a small static trust list of emails.
  Unpinned tokens of those users are unrestricted — no owner isolation, no
  endpoint pin, no SSO whitelist gate (login, mint, and use time) — and they
  may pin tokens to any endpoint; a stored pin is still enforced. Tokens are
  still required and OAuth sign-in still applies.
- Scoping is enforced at the gateway. Node detail pages and dashboards are
  admin-only (HTTP Basic), but backend origins are operator-visible
  surface: keep backends of owned nodes off untrusted networks, because a
  direct backend URL bypasses the gateway entirely.

Upgrading an existing deployment: with user auth disabled (the default) nothing
changes. To roll out tokens without waking an oversight surface, first deploy
with OAuth enabled but `enforce_api_tokens` left `false`, then flip enforcement
once users have minted tokens. Rotate `auth.session_secret` to log every session
out at once.

Token management dashboards (admin and per-user):

- The admin token dashboard lives at `/dashboard/tokens` (same HTTP Basic
  gate as the ops dashboard). It lists every generated token by user with
  creation time, last-used time, revoked state, endpoint scope, request
  count, prompt/completion/total tokens, and an estimated premium-equivalent
  cost. A per-user view at `/dashboard/users/{id}` adds the usage breakdown
  by token/model/endpoint, a daily timeline (last 30 days, UTC days), and
  revoke buttons for each token.
- The per-user profile page (`/profile`) keeps its existing token list and
  revocation, and adds the same premium-equivalent cost estimate next to the
  usage totals. The admin dashboard front page shows the grand total
  "token budget saved" for all token-attributed requests.
- The estimate is `prompt_tokens/1M * input_rate + completion_tokens/1M *
  output_rate`: what the recorded mix would have cost at the configured
  premium frontier-model rate. It is an equivalence figure for self-hosted
  usage (near-zero marginal cost), not a cash ledger, and it ignores
  prompt-cache discounts because the usage store does not record cache-token
  splits. Only token-authenticated requests are recorded (AUTH-04), so the
  global figure covers token-attributed traffic; anonymous requests in
  enforcement-off deployments are not included.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PRICING__INPUT_RATE_PER_MTOK` | `5.0` | USD per million input (prompt) tokens |
| `INFERENCE_PROXY_PRICING__OUTPUT_RATE_PER_MTOK` | `25.0` | USD per million output (completion) tokens |
| `INFERENCE_PROXY_PRICING__MODEL_LABEL` | `claude-opus-4.8` | Reference model shown next to the estimate |

The defaults track published Claude Opus-class 1M-context pricing; override
them when the reference model or your accounting changes.

### Self-service onboarding (`/start`)

Normal (non-admin) users get one page. After Google sign-in they land on
`/start`, and every operations page (`/dashboard`, node detail, `/models`,
`/chat`, `/profile`, the token and admin pages) redirects them back to it.
Admins are unaffected and are redirected away from `/start`.

A user without a token walks through one question per screen:

1. a name for the token,
2. the coding tool they use,
3. the models they want (one model for single-model tools, any number otherwise).

The last screen shows one line to paste into a terminal:

```bash
curl -sSL https://gateway.example.com/s/k7m2x9qd4tpa | bash
```

The script only writes that tool's config file, already pointed at the gateway
with the user's token and models. It backs up an existing file first (the first
backup is kept across reruns), merges into existing JSON and TOML configs
instead of replacing them, and leaves the file readable only by the user. It
needs `bash`; `python3` is used for JSON merges when present.

A user who already has a token sees only that token, the models it may use
(tap to change), and buttons to set up another tool or replace the token.

| Coding tool | Config written | Models | Needs |
|-------------|----------------|--------|-------|
| OpenCode | `~/.config/opencode/opencode.json` | many | `/v1/chat/completions` |
| Pi | `~/.pi/agent/models.json` | many | `/v1/chat/completions` |
| Oh My Pi | `~/.omp/agent/models.yml` (replaced, not merged) | many | `/v1/chat/completions` |
| Claude Code | `~/.claude/settings.json` | one | `/v1/messages` |
| Codex | `~/.codex/config.toml` | one | `/v1/responses` |

A tool is offered only when the gateway serves the API route it speaks, so
Claude Code and Codex are available through `/v1/messages` and
`/v1/responses`. The models offered are the models healthy nodes are serving
at that moment, limited to nodes the user may reach (never admin-only servers or nodes
owned by someone else).

Rules and guardrails:

- **One token per user.** Minting on `/start` revokes every other token the
  user has. `POST /profile/tokens` is admin-only so the rule cannot be bypassed.
- **Model scope.** The token may only request its chosen models. Anything else
  is refused with `403 model_not_permitted`. Asking for a setup command that
  includes a new model adds it to the token. Scope binds tokens, so it has teeth
  only with `auth.enforce_api_tokens=true`; with the default `false`, requests
  without a token are anonymous and unrestricted.
- **Setup links.** `/s/{id}` needs no session because the id is the credential.
  It lives for 15 minutes, can be fetched again inside that window (a failed
  first run can simply be retried), and dies when the token is replaced. Only a
  SHA-256 of the id is stored, the gateway logs the path as `/s/[redacted]`,
  and responses are `no-store`, `noindex`, and `no-referrer`. A reverse proxy in
  front still logs request paths unless its log format is changed, so treat
  proxy access logs as sensitive for 15 minutes after a link is made.
- **No stored secret.** The token is derived from `auth.session_secret`, the
  user id, and a random per-token value, and only its digest is stored. That is
  what lets a user set up a second tool later. It also means the database plus
  the session secret is enough to recover tokens: protect the secret like a
  credential. Rotating it leaves existing tokens working on `/v1` but makes
  them impossible to export again; the page then asks the user for a new token.
- **Public origin.** The command and the written configs use the origin of
  `oauth.redirect_uri` (or an `oauth.allowed_redirect_hosts` name when the
  request came in on one), never a bare `Host` header, and never downgrade a
  configured `https` origin to `http` behind a proxy hop uvicorn does not trust.

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

### Plugins

QIIP uses a QUADS-style plugin architecture: category interfaces (currently
`auth`), built-in implementations, and an optional external plugin directory.
Plugins are configured in `conf/plugins.yml` (see the checked-in example); the
`plugins:` section maps 1:1 onto the settings model, and environment variables
with the `INFERENCE_PROXY_` prefix still override it for legacy setups.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROXY_PLUGINS__EXTERNAL_DIR` | unset | Optional directory scanned for external (downstream/third-party) plugins; module code executes at startup as the service user, so the directory must be root/user-owned and not group/world-writable |
| `INFERENCE_PROXY_PLUGINS__DISABLED` | `[]` | Fully-qualified plugin names that are never loaded, e.g. `["auth.google"]` (JSON array) |
| `INFERENCE_PROXY_PLUGINS__CONFIG` | `{}` | Per-plugin config keyed by plugin name, e.g. `{"myplugin":{"api_key":"..."}}` (JSON object) |

External plugins are trusted code: discovery imports and executes them with
the service user's privileges. They can never share a registered name with a
built-in plugin, so to make an external plugin the active implementation of a
category, give it a distinct name (e.g. `auth.okta`) and disable the built-in
(`INFERENCE_PROXY_PLUGINS__DISABLED=["auth.google"]`); the built-in keeps its
name and wins on any collision.

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
| `INFERENCE_PROXY_PROVISIONING__LOG_DB_PATH` | `data/provisioning-logs.sqlite3` | Durable gateway attempt database; use persistent local storage |
| `INFERENCE_PROXY_PROVISIONING__LOG_RETENTION_DAYS` | `30` | Retention of gateway attempt history |
| `INFERENCE_PROXY_PROVISIONING__LOG_STORAGE_MAX_BYTES` | `268435456` | Gateway retained record payload budget |
| `INFERENCE_PROXY_PROVISIONING__LOG_ATTEMPT_MAX_BYTES` | `33554432` | Gateway record payload budget per attempt |
| `INFERENCE_PROXY_PROVISIONING__LOG_MAX_ATTEMPTS` | `1000` | Gateway attempt manifests retained |
| `INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_ROOT` | `/var/lib/qiip/provisioning-logs` | Node database and bounded engine tails |
| `INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_RETENTION_DAYS` | `7` | Node attempt retention |
| `INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_MAX_BYTES` | `134217728` | Node payload budget, half for records and half for raw tails |
| `INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_ATTEMPT_MAX_BYTES` | `16777216` | Node record and raw-tail limit per attempt, subject to total budgets |
| `INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_MAX_ATTEMPTS` | `32` | Node attempt manifests and raw tails retained |
| `INFERENCE_PROXY_PROVISIONING__LOG_RECONNECT_ATTEMPTS` | `3` | Consecutive automatic retrieval retries after SSH errors |
| `INFERENCE_PROXY_PROVISIONING__LOG_POLL_INTERVAL` | `1` | Seconds between node log retrieval requests |

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
- **Health checker** -- probes each registered node's `/health` endpoint, transitions liveness state, maintains managed-node leases after valid evidence, and removes idle draining ghosts. A self-setup node with a missing `/health` endpoint (HTTP 404/405/501) falls back to `/v1/models`; managed nodes always require a healthy `/health`
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

## Durable provisioning evidence

Each setup, relaunch, and teardown receives a UUID and a SHA-256 identity of its
setup bundle. Setup stdout/stderr, launch stdout/stderr, engine startup output,
and available `vllm`, `llamacpp`, and NVIDIA Fabric Manager journal records are
stored on the node and retrieved into the gateway database. Every record carries
the hostname, attempt, engine, model selection (null until known for automatic
selection), bundle version, stage, source, timestamp, and sequence. Node timestamps
represent capture time; journal JSON also contains the original journal timestamp
and cursor. Gateway messages use gateway time.

Configure the durable-log settings under `provisioning:` in `conf/qiip.yml`
(see `conf/qiip.yml.example`). The gateway requires a writable persistent
`log_db_path`; startup fails if this database cannot be opened, rather than
silently losing durable history. The gateway payload budget must cover one
attempt. Half the node budget is reserved for SQLite records, so the node total
must be at least twice its per-attempt record budget.

The node recorder requires Python 3.9+ with SQLite and write access to the remote
log root. It is uploaded with the setup bundle. Recording survives loss of the
SSH connection; reconnects retrieve by sequence and commit the retrieval cursor
with each record. A lost launch acknowledgement never causes a second setup or
engine launch. Explicit teardown cancellation signals the detached command group
and retains its final output. Gateway shutdown stops retrieval and leaves the
node command and recorder running; the attempt is marked interrupted locally.
If completion cannot be established within the deadline, the
attempt fails with an explicit collection warning. Recorded commands retain the
configured SSH total and inactivity deadlines; llama.cpp setup retains its longer
setup timeout. This feature retrieves evidence;
it does not reconcile or resume a provisioning process after a gateway restart.

On the node detail page, **Provisioning history** lists attempts independently of
the current node state. Select an attempt to search all retained messages, filter
by source, retrieve missed node output, or download a gzip-compressed JSONL bundle
containing its manifest, records, and an export summary that reports concurrent
rotation during download. Failed stages and their output appear in the
summary; unavailable sources, sequence gaps, and retention losses remain visible.
The live stream also resumes by attempt and sequence. The engine pipe consumer
batches output for up to 100 ms or 64 KiB before committing. Catchable recorder
failures and SIGINT/SIGTERM stop recording and drain the pipe. SIGKILL, an OOM
kill, or node loss cannot run that drain; those events can interrupt the engine
and require operator recovery.

Administrative API (existing admin authentication and JSON request requirements):

- `GET /admin/provisioning/{hostname}/attempts?limit=100&offset=0`
- `GET /admin/provisioning/{hostname}/attempts/{id}/logs?q=error&source=setup.stderr&after=0&limit=500`
- `POST /admin/provisioning/{hostname}/attempts/{id}/collect` with JSON `{}`
- `GET /admin/provisioning/{hostname}/attempts/{id}/bundle`
- `GET /admin/provisioning/{hostname}/logs?attempt_id={id}&after=0` (SSE; supports `Last-Event-ID: {id}:{seq}`)

Offsets are inclusive sequence positions; use `next_offset` for the next page.
Search is a case-insensitive literal substring match, including `%` and `_`.
Retrieval reports unavailable sources in the returned manifest while preserving
previously collected data. Restarted gateway attempts are marked `interrupted`;
retrieving their evidence does not claim that provisioning succeeded.

Byte limits bound UTF-8 JSON record payloads, plus bounded raw tails on nodes;
allow extra filesystem space for SQLite pages, indexes, manifests, and the
SQLite WAL (long-lived readers may delay checkpointing). New stores use full
auto-vacuum. A pre-existing SQLite file with auto-vacuum disabled requires an
explicit rebuild to enable page reclamation; this upgrade does not rebuild it
during startup. Prefix rotation retains monotonic sequence numbers
and dropped-record counts. Age/count retention runs at store initialization,
attempt creation, and before history snapshots when expired/excess rows exist.
History snapshots use read transactions; WAL allows writers to proceed while
readers inspect a snapshot. Active attempts are protected from manifest
eviction and do not consume the completed-attempt count budget; their record
payloads still rotate. Expired manifests are counted in
`evicted_attempts` (gateway-wide), and requesting an evicted attempt returns 404.
Raw-tail capacity is divided across the configured node attempt count, keeping
runtime output bounded after startup collection ends. Existing pre-upgrade logs
are not imported. Keep the gateway database on one persistent local volume for
its owning gateway process; separate gateway replicas do not share this history.

Controlled verification lives in `tests/provisioning/test_attempt_logs.py`: it
runs the uploaded recorder and shipped setup/launch boundaries with fixture
installers and a fake engine. It exercises lost acknowledgements, stream
interruption, restart, concurrent readers, retries, record/raw-file rotation, and
missing remote/journal sources. These checks do not establish success rates or
failure causes on real fleet hardware.

## Troubleshooting

### Reading provisioning logs offline

When the gateway is down, use the SQLite CLI to read its persistent database.
The default path is `data/provisioning-logs.sqlite3`; substitute your configured
`INFERENCE_PROXY_PROVISIONING__LOG_DB_PATH` if different. Open it read-only to
avoid accidentally creating or modifying a database. First list attempts:

```bash
sqlite3 -readonly -header -column data/provisioning-logs.sqlite3 \
  "SELECT id, hostname, json_extract(metadata,'$.started_at') AS started_at,
          json_extract(metadata,'$.status') AS status,
          json_extract(metadata,'$.failure_summary') AS failure_summary,
          dropped AS dropped_records
   FROM attempts ORDER BY created DESC;"
```

Replace `ATTEMPT_ID` below with an ID from that list to read one attempt in
sequence order. Sequence numbers are local to each attempt:

```bash
sqlite3 -readonly data/provisioning-logs.sqlite3 \
  "SELECT json_extract(payload,'$.ts') || ' [' ||
          json_extract(payload,'$.source') || '] ' || json_extract(payload,'$.msg')
   FROM records WHERE attempt='ATTEMPT_ID' ORDER BY seq;"
```

For a JSONL export preserving all record metadata, use `SELECT payload` (each
row is already JSON). The companion manifest includes source availability,
issues, and retention counters, which distinguish missing evidence from an
empty log:

```bash
sqlite3 -readonly data/provisioning-logs.sqlite3 \
  "SELECT payload FROM records WHERE attempt='ATTEMPT_ID' ORDER BY seq;" \
  | jq -c . > provisioning-attempt.jsonl
sqlite3 -readonly data/provisioning-logs.sqlite3 \
  "SELECT json_object('metadata',json(metadata),'next_seq',next_seq,
                      'remote_cursor',remote_cursor,'dropped_records',dropped,
                      'retained_bytes',bytes)
   FROM attempts WHERE id='ATTEMPT_ID';" \
  | jq . > provisioning-attempt-manifest.json
```

`sqlite3 -readonly data/provisioning-logs.sqlite3 .dump` produces a SQL backup,
not JSONL. These reads do not require the gateway or network access. If only a
node's evidence is available, run the same queries on
`/var/lib/qiip/provisioning-logs/attempts.sqlite3` (or the configured
`INFERENCE_PROXY_PROVISIONING__LOG_REMOTE_ROOT` plus `/attempts.sqlite3`). The
node also keeps bounded `<attempt-id>.engine.log` raw tails alongside that
database. Copy the database while its writers are stopped, or use SQLite's
`.backup` command for a consistent snapshot of a live database.

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
