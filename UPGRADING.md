# Upgrading QIIP After the Reliability Campaign

This guide is for deployments upgrading from a build that predates the reliability and security campaign. The campaign changed gateway startup, routing semantics, the administrative security boundary, node provisioning, etcd registration lifetime, and several client-visible API behaviors. Read this guide before replacing a running gateway.

The guide separates three kinds of change:

- **Operator migrations** require configuration, launcher, node, or runbook changes.
- **Client compatibility changes** affect callers of the inference or administrative APIs.
- **Correctness fixes** restore intended behavior, but may still affect clients that implemented workarounds for the old behavior.

## Contents

- [Recommended upgrade sequence](#recommended-upgrade-sequence)
- [Required operator migrations](#required-operator-migrations)
- [Artifact sources and mirror policy](#artifact-sources-and-mirror-policy)
- [Client-visible compatibility changes](#client-visible-compatibility-changes)
- [Operational runbooks](#operational-runbooks)
- [Verification checklist](#verification-checklist)

## Recommended Upgrade Sequence

1. Save the current gateway environment, launcher command, and service definition.
2. Record every current `/nodes/` etcd value and whether the proxy should manage that node's lifecycle.
3. Update the launcher, required credentials, endpoint allowlist, and any QUADS configuration before starting the new gateway.
4. Review the node package versions, artifact digests, NFS export, and retired node environment variables before provisioning another host.
5. Review the client-visible changes, especially streaming HTTP statuses, `failover_exhausted`, verbatim 4xx responses, and the `max_attempts` budget.
6. Start the gateway and complete the verification checklist at the end of this guide.

Do not treat a successful process start as sufficient verification. A loopback-only endpoint allowlist can produce a healthy gateway with an empty registry, and an expired managed-node lease can leave vLLM running on a host that the gateway no longer knows about.

## Required Operator Migrations

### 1. Launch the FastAPI application as a factory

The import-time `inference_proxy.main:app` object no longer exists. Launch Uvicorn with the application factory:

```bash
uv run uvicorn inference_proxy.main:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8080
```

Update systemd units, containers, shell wrappers, probes, and development commands that still name `inference_proxy.main:app`.

### 2. Move server ownership to Uvicorn

`INFERENCE_PROXY_GATEWAY__HOST` and `INFERENCE_PROXY_GATEWAY__PORT` never
controlled the listening socket and have been removed. They are silently
ignored. Configure the bind address and port with Uvicorn's `--host` and
`--port` launcher options shown above, then remove the obsolete environment
variables.

`INFERENCE_PROXY_GATEWAY__GRACEFUL_SHUTDOWN_TIMEOUT` has been removed and is silently ignored. Uvicorn stops accepting requests and drains in-flight work before application lifespan cleanup begins, so configure its server-owned option instead:

```bash
uv run uvicorn inference_proxy.main:create_app \
  --factory \
  --timeout-graceful-shutdown 60
```

Remove the obsolete environment variable after migrating the launcher. Leaving it set does not configure either QIIP or Uvicorn.

### 3. Configure administrative credentials

The gateway now requires both of these values at startup:

```dotenv
INFERENCE_PROXY_ADMIN__USERNAME=operator
INFERENCE_PROXY_ADMIN__PASSWORD=replace-me
```

HTTP Basic authentication protects `/admin/*` and `/dashboard*`. The inference endpoints, chat page, and `/health` remain public. HTTP Basic sends a reusable base64-encoded credential on every request. Plain HTTP is an accepted deployment choice on a trusted work LAN; use a TLS terminator whenever the network path is not trusted.

State-changing admin endpoints accept JSON only. Cross-origin JSON and DELETE requests require a browser preflight, which is part of the current CSRF boundary. Do not add form-encoded, multipart, or plain-text state-changing admin endpoints without adding explicit CSRF protection.

### 4. Configure the backend endpoint allowlist before startup

The secure default permits only `localhost`, loopback networks, and port 8000. Existing non-loopback nodes are rejected during discovery and omitted from `/admin/nodes` until the deployment declares its trust boundary:

```dotenv
INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_HOSTS=["*.inference.example.com"]
INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_NETWORKS=["10.0.1.0/24"]
INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_PORTS=[8000]
```

DNS endpoints must match an exact name or `*.suffix` rule. IP literals match CIDRs, not hostname rules. Backend origins must use HTTP or HTTPS and include an explicit port. Bracket IPv6 literals, for example `http://[2001:db8::10]:8000`.

The configured provisioning vLLM port must be in the allowed-port list. Setup validates the hostname-derived endpoint before taking a host lease, opening SSH, or changing power state.

### 5. Rename `max_retries` and recalculate the attempt budget

`INFERENCE_PROXY_ROUTING__MAX_RETRIES` has been removed and is silently ignored. Replace it with `INFERENCE_PROXY_ROUTING__MAX_ATTEMPTS`, whose value is **total attempts including the first request**. The minimum is 1.

| Previous `MAX_RETRIES` value | Previous maximum attempts | Same `MAX_ATTEMPTS` value | `MAX_ATTEMPTS` value preserving the old budget |
|---:|---:|---:|---:|
| 1 | 2 | 1 | 2 |
| 2 | 3 | 2 | 3 |
| 3 | 4 | 3 | 4 |

Copying an existing value of 3 therefore loses one backend attempt unless it is changed to 4. The same total-attempt budget applies to non-streaming requests and the pre-response streaming handshake.

`INFERENCE_PROXY_ROUTING__STRATEGY` was also accepted but never changed
routing; QIIP has always used least-connections selection. The inert setting
has been removed and is silently ignored. There is no replacement unless
another routing strategy is implemented.

### 6. Mark externally registered nodes as managed explicitly

An etcd node record with no `managed` field now defaults to `false`. This is the safe ownership direction: the proxy will route to the node but will not assume permission to tear it down through QUADS schedule enforcement.

External registrants must write `"managed": true` only when QIIP should own the node's lifecycle. `SetupRequest.managed` still defaults to true, so nodes created through the administrative setup workflow remain proxy-managed unless the caller opts out.

### 7. Use fully qualified etcd endpoint URLs

Every value in `INFERENCE_PROXY_ETCD__ENDPOINTS` must include an `http://` or `https://` scheme and a hostname. A schemeless value such as `etcd.internal:2379` now fails startup instead of being parsed ambiguously.

The setting remains a JSON list for compatibility, but the current client uses only the first endpoint and logs the ignored values. Do not mistake multiple configured entries for etcd-client failover.

### 8. Plan for lease-backed managed registrations

Healthy proxy-managed `/nodes/` keys now carry an etcd lease with a default 600-second TTL. The health checker refreshes a lease only after valid health evidence. Existing managed keys without a lease are adopted after a successful probe. Unmanaged external registrations are never adopted; their registrant remains responsible for their lifetime.

Clean gateway shutdown deliberately does not revoke node leases. If no gateway refreshes a proxy-managed lease before its TTL expires, etcd deletes the key. The discovery watcher drains the node and the health-cycle sweeper removes the idle registry entry.

This is an operational contract, not merely a setting:

- A gateway outage longer than 600 seconds can expire every proxy-managed registration even while vLLM continues running on the nodes.
- Recovery after expiry is reprovisioning or another explicit registration workflow; restarting the gateway cannot rediscover a key that etcd deleted.
- Before a planned maintenance window longer than the active leases' TTL, either keep a gateway lease maintainer running or plan to reprovision the managed nodes afterward.
- `INFERENCE_PROXY_ETCD__NODE_LEASE_TTL` must be greater than 300 seconds and greater than three complete health-check intervals. Changing the configured value affects newly granted leases; do not assume it retroactively changes every existing etcd lease.

Configure the active probe cadence with `INFERENCE_PROXY_RESILIENCE__HEALTH_CHECK_INTERVAL`. The removed `INFERENCE_PROXY_ROUTING__HEALTH_CHECK_INTERVAL` name is silently ignored and does not affect lease maintenance.

### 9. Configure the QUADS server timezone

When `INFERENCE_PROXY_QUADS__BASE_URL` is set, `INFERENCE_PROXY_QUADS__SERVER_TIMEZONE` is also required and must be a valid IANA timezone:

```dotenv
INFERENCE_PROXY_QUADS__BASE_URL=http://quads.example.com
INFERENCE_PROXY_QUADS__SERVER_TIMEZONE=America/New_York
```

The QUADS availability API verified during implementation parses timezone-naive `YYYY-MM-DDTHH:MM` values. QIIP therefore converts its UTC deadline to the configured QUADS server timezone before formatting the query. This contract was checked against QUADS commit `bbada78`.

Because the external API is timezone-naive, a deadline in the repeated hour during a daylight-saving fall-back cannot identify which occurrence was intended. An availability window can be ambiguous by one hour during that transition.

Setup now checks the complete `QUADS__SCHEDULE_LOOKAHEAD_HOURS` window, not only whether a host is free at the current instant. A host with an upcoming assignment inside the window returns 400 before provisioning starts. Schedule-enforcer teardown retries use capped exponential backoff and emit `schedule_enforcer_teardown_requires_operator` when repeated failures reach the one-hour ceiling.

### 10. Declare the HuggingFace NFS export for provisioning

`INFERENCE_PROXY_PROVISIONING__NFS_SERVER` has been removed and is silently ignored. Configure the single backing export instead:

```dotenv
INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR=/data/huggingface
INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT=storage.example.com:/exports/huggingface
INFERENCE_PROXY_PROVISIONING__NFS_MOUNT_POINT=/srv/hf-cache
```

The gateway cache path and node mount path may differ; `NFS_EXPORT` is the declared mapping between them. Proxy-only deployments may omit the export. A node setup without it returns 400 before acquiring a lease or doing remote work.

### 11. Rename node launch overrides to `AUTOVLLM_*`

The setup and launch scripts no longer use their private inputs in vLLM's reserved `VLLM_*` namespace. Rename these values:

| Retired name | Replacement |
|---|---|
| `VLLM_TENSOR_PARALLEL` | `AUTOVLLM_TENSOR_PARALLEL` |
| `VLLM_GPU_MEM_UTIL` | `AUTOVLLM_GPU_MEM_UTIL` |
| `VLLM_MAX_MODEL_LEN` | `AUTOVLLM_MAX_MODEL_LEN` |
| `VLLM_MAX_BATCHED_TOKENS` | `AUTOVLLM_MAX_BATCHED_TOKENS` |
| `VLLM_EXTRA_ARGS` | `AUTOVLLM_EXTRA_ARGS` |

The removed names are silently ignored. The launcher also strips all seven old gateway inputs from the child environment so they cannot be reinterpreted by vLLM itself. `VLLM_PORT` is not a supported API-port override; use `AUTOVLLM_API_PORT` when invoking the scripts directly or `INFERENCE_PROXY_PROVISIONING__VLLM_PORT` through the gateway.

### 12. Move the LLMFit version to its single settings group

`INFERENCE_PROXY_PROVISIONING__LLMFIT_VERSION` has been removed and is silently ignored. Move its value to:

```dotenv
INFERENCE_PROXY_LLMFIT__VERSION=1.1.6
INFERENCE_PROXY_LLMFIT__SHA256=1e09232a128455596a2d348ab5893741d04b94aa6d924f1253462dc13304f7c6
```

Changing the version requires explicitly supplying the matching digest. The gateway does not silently retain the built-in digest for a custom version.

### 13. Prepare for strict driver and cache checks

Provisioning no longer reports success merely because `nvidia-smi` exists. If the installed driver differs from `INFERENCE_PROXY_PROVISIONING__NVIDIA_DRIVER_VERSION`, setup fails and prints both versions. Align the configured version with each fleet or update the node deliberately; setup does not hot-swap a live kernel driver.

The committed default pair is NVIDIA driver 580.126.09 with SHA-256 `4cac53e48f8adff661d47c8788ed24059a248c9fd8098ceafd088a498986ec26`. Changing that version requires an explicit `INFERENCE_PROXY_PROVISIONING__NVIDIA_DRIVER_SHA256`. Downloaded driver and LLMFit artifacts are verified before any privileged installer consumes them.

The node HuggingFace cache target must be a mount point or a replaceable symlink. Setup refuses to replace an existing real cache directory because doing so could destroy locally downloaded models. Move or reconcile that directory before provisioning.

### 14. Move node Python installation to the locked `uv` bundle

Node provisioning no longer invokes pip. It bootstraps a pinned, checksum-verified `uv`, then synchronizes a frozen, wheel-only Python 3.12 environment from `auto-vllm/uv.lock`. Reprovisioning prunes packages absent from the lock instead of accumulating environment drift.

The current node bundle installs:

- vLLM 0.26.0
- FlashInfer Python and AOT cubins 0.6.14
- uv 0.12.1
- CPython 3.12 on Linux x86_64, targeting `manylinux_2_34`

The upgrade therefore changes the node runtime versions as well as the package tool. Review vLLM release compatibility and model behavior before rolling the bundle across the fleet.

`FLASHINFER_INDEX_URL` is retired and causes setup to fail. Package sources belong to the frozen project and lock; a custom FlashInfer or Python index requires regenerating and reviewing the complete `auto-vllm` bundle.

### 15. Account for model catalog verification

The model catalog lists only snapshots that HuggingFace's local-only resolution verifies as complete and that carry a tree manifest. Two counts explain hidden entries:

- `incomplete_count`: a manifest exists but required files are missing.
- `unverifiable_count`: the legacy cache entry has no tree manifest, so completeness cannot be proven.

When either count is nonzero, `/admin/models/catalog` returns `X-Inference-Proxy-Data-Degraded: model-catalog`, and the dashboard explains that some cache entries are hidden. A pre-manifest cache can therefore appear empty even when model files exist. Re-download or otherwise migrate those snapshots so the current HuggingFace tooling creates verifiable metadata.

Duplicate download requests now return 200 with the existing status; a newly started download returns 202. The download worker limit remains two concurrent downloads.

### 16. Correct invalid log levels before upgrading

Logging levels are validated at startup. Supported values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`, case-insensitively. An unknown value now fails configuration instead of silently falling back to INFO.

### 17. Account for provisioning admission and log retention

At most `INFERENCE_PROXY_PROVISIONING__MAX_CONCURRENT_PROVISIONS` setup tasks run at once; the default is 32. Further setup requests return 429 with the active count and limit. Teardown remains admissible while provisioning capacity is full.

Provisioning logs are bounded by four settings:

| Setting | Default |
|---|---:|
| `PROVISIONING__LOG_MAX_ENTRIES_PER_HOST` | 1,000 |
| `PROVISIONING__LOG_MAX_BYTES_PER_HOST` | 1,048,576 |
| `PROVISIONING__LOG_MAX_ENTRY_BYTES` | 16,384 |
| `PROVISIONING__LOG_MAX_COMPLETED_HOSTS` | 64 |

Oversized messages carry a visible truncation suffix. A slow SSE reader receives a synthetic warning identifying how many earlier entries were evicted. Completed-host buffers are evicted oldest first; active buffers are not removed to make room.

### 18. Account for bounded SSH command execution

Streaming SSH commands now have both a one-hour total deadline and a 15-minute inactivity deadline:

```dotenv
INFERENCE_PROXY_SSH__STREAMING_COMMAND_TIMEOUT=3600
INFERENCE_PROXY_SSH__STREAMING_INACTIVITY_TIMEOUT=900
```

Any stdout or stderr resets the inactivity deadline. A command that exceeds either bound fails with structured provisioning state instead of holding the per-host lifecycle lease forever.

### 19. Validate Redfish configuration and destinations

Redfish is disabled only when both BMC credentials are absent. Configuring only one credential now fails startup. The BMC host template must contain exactly one plain `{hostname}` placeholder and no scheme, port, path, query, fragment, conversion, or format specifier.

Power endpoints accept DNS hostnames that pass the routing hostname allowlist; IP literals are rejected for BMC template expansion. Credentials are attached per request only after both the input hostname and rendered BMC destination validate. `REDFISH__VERIFY_SSL=false` remains the default for self-signed BMC certificates.

### 20. Prepare managed llama.cpp nodes for verified CUDA source builds

QIIP no longer attempts to download a Linux CUDA archive that upstream does not
publish. Managed llama.cpp setup downloads the pinned tag source, verifies its
SHA-256 before extraction, and builds CUDA-enabled `llama-server` and
`llama-quantize` binaries on the node. The default setup deadline for this path
is two hours:

```dotenv
INFERENCE_PROXY_PROVISIONING__LLAMACPP_VERSION=b10242
INFERENCE_PROXY_PROVISIONING__LLAMACPP_SHA256=b5c2b0d09d2af9988e47570f7f96e8473b4e07fad2c99f6e2e0745e5b3935fe3
INFERENCE_PROXY_PROVISIONING__LLAMACPP_SETUP_TIMEOUT=7200
```

Changing the version requires an explicitly configured matching digest. A
custom source mirror is selected through the validated
`INFERENCE_PROXY_PROVISIONING__LLAMACPP_SOURCE_URL` template. Ensure managed
nodes can reach the selected source and NVIDIA package repositories and have
enough local space for a CUDA build.

Managed setup and launch now require a verified NVIDIA GPU and never fall back
to CPU inference. Direct use of the scripts retains the existing standalone CPU
branch, but it is outside QIIP's managed-node support boundary. Validate a
disposable node from each GPU family before fleet rollout because the build is
specialized for the attached CUDA architecture.

### 21. Use exact immutable GGUF artifacts

The short-lived `allow_patterns` field on `POST /admin/models/download` has
been removed. It could select multiple unrelated files and could not identify
one loadable split GGUF generation. Unknown request fields are now rejected,
so callers still sending `allow_patterns` receive 422 rather than a silently
different download.

For llama.cpp, send `engine: "llama_cpp"` plus an exact `gguf` object containing
the ordered repository-relative files and the entrypoint. Download status now
includes a stable `download_id`, the requested revision, the resolved commit
SHA, and the published artifact. Setup requires that artifact's 64-character
`artifact_id`; a missing, unknown, or invalid artifact is rejected before a
host lease, QUADS lookup, SSH, driver installation, or CUDA source build.

`AUTOLLAMACPP_MODEL` and `AUTOLLAMACPP_QUANTIZATION` no longer select files.
Standalone launchers must provide `AUTOLLAMACPP_GGUF_PATH` and
`AUTOLLAMACPP_MODEL_ALIAS`. The path is cache-relative and must resolve beneath
the configured NFS mount.

Artifact identity uses the resolved commit SHA, exact ordered file set,
entrypoint, and alias. Re-downloading a moving branch therefore creates a new
immutable generation instead of replacing the old one. QIIP deliberately does
not prune generations or backing HuggingFace snapshots; coordinate retention
only after confirming no running or restartable node depends on them.

## Artifact Sources and Mirror Policy

There is no single global mirror switch. Each source has a different trust and configuration boundary.

| Artifact source | How it is selected | Integrity boundary | Mirror procedure |
|---|---|---|---|
| NVIDIA `.run` installer used by `setup.sh` | Node-side `NVIDIA_DRIVER_URL`; the gateway does not forward a URL setting | `AUTOVLLM_NVIDIA_DRIVER_SHA256`, normally populated from `PROVISIONING__NVIDIA_DRIVER_SHA256` | Point the node-side variable at a mirror serving byte-identical content, or configure the matching custom digest with the custom version. |
| LLMFit archive used by `setup.sh` | Node-side `LLMFIT_URL`; the gateway forwards version and digest but not this URL | `AUTOVLLM_LLMFIT_SHA256`, normally populated from `LLMFIT__SHA256` | Point the node-side variable at byte-identical mirrored content. When invoking setup through QIIP, arrange the variable in the node's SSH environment or customize the shipped bundle. |
| On-demand LLMFit runner install | `INFERENCE_PROXY_LLMFIT__INSTALL_URL` | `INFERENCE_PROXY_LLMFIT__SHA256` | Set the validated HTTP(S) `{version}` URL template and matching digest. Configure this separately from the setup-script URL. |
| llama.cpp source archive | `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SOURCE_URL` with the pinned `LLAMACPP_VERSION` | `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SHA256`; verification happens before extraction or build | Point the validated `{version}` URL template at a mirror serving byte-identical tag archives, or configure the matching digest with a custom version. Managed nodes compile CUDA-enabled binaries locally because the pinned release has no Linux CUDA archive. |
| HuggingFace model snapshots and GGUF files | Repository ID, optional requested revision, and exact GGUF file set in the administrative download request | Download status records the resolved immutable commit; GGUF manifests bind that commit to an exact entrypoint and file set | Pre-seed the standard HuggingFace cache on the declared NFS export. QIIP does not provide a separate model-hub mirror URL. A resolved commit records reproducibility but is not an independently configured content digest. |
| uv bootstrap binary | Pinned GitHub release in `setup.sh` and `auto-vllm/.uv-version` | Vendored checksum from Astral's release assets | Air-gapped nodes may preinstall the exact pinned uv version at the expected path. Changing the download source requires a reviewed bundle change. |
| vLLM, FlashInfer, and Python dependencies | `auto-vllm/pyproject.toml` plus `auto-vllm/uv.lock` | Registry artifact hashes for the full installable closure; source builds are refused | Regenerate and review the node project and lock for a custom index. Runtime `FLASHINFER_INDEX_URL` overrides are rejected. A pre-seeded uv cache may be used only when it satisfies the frozen lock. |
| DNF and CUDA RPM packages | Node repository configuration; setup also enables NVIDIA's RHEL 9 CUDA repository | Repository metadata and RPM signatures | Manage mirrors through node repository policy or a reviewed setup-bundle customization. QIIP has no gateway-level DNF mirror setting. |

The node-only `NVIDIA_DRIVER_URL` and `LLMFIT_URL` variables are intentionally documented as node-local controls rather than application settings. A mirror serving different bytes must never be accepted under the committed default digest.

## Client-Visible Compatibility Changes

### Inference API

| Classification | Change | Client action |
|---|---|---|
| **New surface** | Retry-loop exhaustion sets OpenAI error `code` to `failover_exhausted` and adds `X-Inference-Proxy-Failover: exhausted` plus `X-Inference-Proxy-Attempts: <n>`. The HTTP status is the last attempted upstream status; transport failures without an upstream response map to 502, and timeouts map to 504. | Treat the marker as “the configured attempt loop ended after at least one backend failure,” not as proof every fleet node was tried. Preserve handling for the accompanying HTTP status. |
| **Behavioral break** | `ROUTING__MAX_RETRIES` was replaced by `ROUTING__MAX_ATTEMPTS`, which counts total attempts including the first request. The removed name is silently ignored. | Rename the variable and increase existing values by one when preserving the old maximum attempt budget. |
| **Behavioral break** | Streaming no longer commits HTTP 200 before contacting an upstream. Pre-stream failures can return 502, 503, 504, or a preserved upstream 5xx response. | Ensure streaming callers handle non-200 responses before parsing SSE. |
| **Behavioral break** | Streaming response headers wait for a successful upstream handshake. The complete retry phase is bounded by `ROUTING__TIMEOUT`, so header latency can be as high as that total deadline. | Set client header timeouts above the configured routing timeout plus expected network overhead. |
| **Correctness fix** | Retryable non-streaming 5xx and transport failures now fail over to another eligible node and count against the circuit breaker. | Remove workarounds that manually retried every 5xx without considering the proxy's attempt budget. |
| **Correctness fix** | Upstream 4xx responses pass through verbatim, are not retried, and do not carry the exhaustion marker. A 4xx is neutral circuit-breaker evidence: it neither increments failures nor clears earlier failures. | Expect upstream `error.type`, `error.code`, `param`, and other fields to survive unchanged. Remove wrapper-specific parsing workarounds. |
| **Correctness fix** | Chat messages preserve tool calls, `content: null`, multimodal content parts, and additional OpenAI-compatible fields. Completion prompts preserve string, string-array, token-ID, and nested token-ID forms. | Remove client-side transformations that existed only to prevent the proxy from stripping these fields; retaining them can cause duplicate handling. |
| **Behavioral break** | `/v1/models[].owned_by` now reports the registered node's engine (`vllm` or `llama_cpp`) instead of always reporting `vllm`. | Treat `owned_by` as backend metadata rather than a constant. Do not filter otherwise valid models solely because the value is not `vllm`. |
| **Defined boundary** | After a streaming response has started, an upstream failure is not retried. The proxy emits an OpenAI-format error event followed by `[DONE]`. | Treat a mid-stream error as terminal and do not concatenate a second backend's output onto the partial response. |

No failover marker is added when no backend attempt occurred, such as when no node serves the requested model. A non-retryable error returned after one attempt is also not marked as exhausted.

Node records now accept and ignore unknown additive fields during
deserialization. This permits mixed-version gateways to read records written by
a newer peer without discarding the node. The `engine` value itself remains a
closed enum: an unrecognized engine string makes the record invalid and the
watcher excludes it, so introduce new engine values only with an ordered
gateway rollout.

### Administrative API and browser interfaces

| Classification | Change | Client action |
|---|---|---|
| **Security break** | `/admin/*` and `/dashboard*` require HTTP Basic authentication. | Configure the shared credentials in scripts, monitoring, and browser sessions. |
| **Security break** | State-changing POST, PUT, and PATCH endpoints require `Content-Type: application/json`; unsupported content returns 415. | Send JSON rather than form, multipart, or plain-text bodies. |
| **New surface** | Setup returns 429 when the configurable provisioning limit is full, including the active count and limit in `detail`. | Retry after another setup completes; teardown remains available and should not be blocked behind setup retries. |
| **New surface** | `/admin/nodes` and `/admin/models/catalog` can return `X-Inference-Proxy-Data-Degraded` with `provisioning-tasks` or `model-catalog`. | Surface the degraded state instead of treating missing task fields or an empty catalog as authoritative. |
| **Correctness fix** | A duplicate model-download POST returns 200 with the existing status; a newly accepted download returns 202. | Accept both success codes and inspect the returned download state. |
| **Behavioral break** | Model downloads reject the removed `allow_patterns` field. llama.cpp downloads require an exact `gguf.files` and `gguf.entrypoint`; setup requires the resulting `artifact_id`. | Update direct administrative clients to the exact-artifact contract and handle 422 for obsolete request bodies. |
| **New surface** | Download status records `download_id`, requested and resolved revisions, and published artifacts. The catalog returns validated GGUF generations in `gguf_artifacts`, separate from full vLLM `models`. | Persist the resolved SHA and artifact ID when reproducibility matters. Do not treat intentional partial GGUF snapshots as missing vLLM models. |
| **Correctness fix** | Setup eligibility is status-aware and serialized by a per-host lifecycle lease. Live healthy or unhealthy nodes require explicit teardown; stale provisioning, failed, unknown, or zero-connection draining records can be cleaned before retry. | Handle actionable 400/409 responses rather than assuming every repeated setup request is accepted. |
| **Correctness fix** | Recommendation targets must be registered or currently QUADS-available and must pass the endpoint allowlist. | Do not use the recommendation endpoint as an unrestricted SSH target. |
| **Correctness fix** | `GracefulRestart` and `ForceRestart` always issue a Redfish reset even when the machine is already On. Only `On` and `ForceOff` are idempotent shortcuts. Malformed or unsupported BMC power states return a structured 502. | Do not use a restart action as a state probe; expect it to restart a running machine. |
| **Security fix** | Chat Markdown is rendered through vendored Marked and DOMPurify assets. All HTML attributes are removed, so model output cannot create event handlers, remote-image loads, or navigable URLs. | Do not depend on model-generated raw HTML, image sources, or link targets surviving rendering. Update the vendored assets through their documented review process. |

## Operational Runbooks

### Planned gateway maintenance

1. Check the maintenance duration against the active managed-node lease TTL.
2. If the outage can exceed the TTL, keep another gateway refreshing leases or accept that managed registrations will expire.
3. After restart, inspect `/admin/nodes` and etcd before sending inference traffic.
4. If keys expired while vLLM remained alive, use the normal teardown/reprovision workflow so process ownership, GPU state, and registration converge again.

Do not recreate expired keys manually with `managed: true` unless the external writer also assumes the lifecycle contract. QIIP attaches and maintains leases only after authoritative watcher state and successful health evidence agree.

### Empty or degraded model catalog

1. Request `/admin/models/catalog` with headers and inspect `incomplete_count`, `unverifiable_count`, `invalid_artifact_count`, `cache_warning_count`, and `X-Inference-Proxy-Data-Degraded`.
2. For incomplete snapshots, restart the download and verify the download status reaches completion.
3. For unverifiable legacy snapshots, re-download with the current HuggingFace tooling so tree-manifest metadata is created.
4. For invalid GGUF artifacts, preserve the backing snapshot for diagnosis and re-run the exact download rather than editing a manifest or link manually.
5. Refresh the catalog and verify all degraded counts are zero before provisioning a node with one of those models.

The catalog intentionally hides entries it cannot prove complete. Do not work around this by advertising the directory name directly; vLLM can otherwise start a long re-download or fail during model load while setup reports only a health-poll timeout.

Gateway shutdown cancels the async download wrapper without waiting indefinitely for a blocked worker thread. A shutdown during download can therefore leave a partial snapshot on the shared cache. It remains hidden by the completeness check but still consumes storage. Before deleting partial data, confirm no gateway is downloading that repository and preserve any complete snapshots referenced by the cache metadata.

### Provisioning failure or cancellation

- Setup is bounded by total and inactivity deadlines. Inspect the provisioning task's failed step and streamed log rather than waiting indefinitely.
- A launch that exits immediately prints the vLLM log tail and fails the start step. Setup verifies process identity rather than accepting any process that happens to hold the recorded PID.
- Teardown cancels and awaits an active local provision task, then verifies vLLM termination. Closing the SSH operation cannot guarantee that a remote installer backgrounded by the shell also stopped; inspect package-manager locks and node state after cancelling during driver or package installation.
- Failed teardown retains registration and PID evidence when shutdown cannot be verified. Do not delete those records merely to clear the dashboard; they are the information needed to finish cleanup safely.
- A scheduling enforcer failure backs off rather than retrying every cycle. Alert on `schedule_enforcer_teardown_requires_operator`.

### Rollback

There is no database migration to reverse, but rollback does not restore every previous behavior:

- Node environments synchronized from the new frozen bundle remain on the new vLLM and FlashInfer versions until another reviewed bundle converges them.
- A managed key already deleted by lease expiry is not recreated by installing an older gateway.
- Node records written without `managed: true` remain externally owned.
- Clients changed to understand pre-stream non-200 responses and exhaustion markers should keep that handling; it is backward-compatible with older responses.

## Verification Checklist

Validate the loaded environment before changing traffic:

```bash
uv run python -c 'from inference_proxy.config.settings import Settings; Settings(); print("configuration valid")'
```

Then verify the live deployment:

- `GET /health` succeeds.
- Authenticated `GET /admin/nodes` lists every intended backend; no allowlist warning explains a missing node.
- Every externally written node has an explicit and correct `managed` value.
- Proxy-managed healthy node keys carry leases, and the lease TTL refreshes after successful health evidence.
- `GET /admin/models/catalog` has zero incomplete and unverifiable entries, or the degradation is understood and shown to operators.
- A non-streaming request succeeds through the gateway.
- A streaming request returns valid SSE after the upstream handshake.
- A controlled backend failure produces the expected status, `failover_exhausted` marker, and attempt headers.
- A representative upstream 4xx response reaches the client unchanged.
- The dashboard and provisioning log stream work with HTTP Basic credentials.
- A setup dry run or disposable host confirms NFS, driver, uv, vLLM, FlashInfer, LLMFit, and firewall expectations before fleet rollout.

Retain the previous launcher and environment until these checks pass, but do not send the retired settings alongside the new ones indefinitely. Migration warnings are temporary compatibility aids, not permanent configuration aliases.
