# Milestones

## v1.9 Model Selection in Node Setup — SHIPPED 2026-07-29

**Phases:** 1 | **Plans:** 1
**Timeline:** 2026-07-29 (1 day)
**Commits:** 1 feat
**Requirements:** 3/3 complete (100%)

### Key Accomplishments

1. Model selector dropdown on node detail page populated from NFS model catalog API
2. Setup blocked when no models downloaded, first model pre-selected when available

**Archive:** [v1.9-ROADMAP.md](milestones/v1.9-ROADMAP.md) | [v1.9-REQUIREMENTS.md](milestones/v1.9-REQUIREMENTS.md)

---

## v1.8 Nodes Power Control — SHIPPED 2026-07-29

**Phases:** 2 | **Plans:** 2
**Timeline:** 2026-07-29 (1 day)
**Commits:** 3 feat
**Requirements:** 5/5 complete (100%)

### Key Accomplishments

1. BMC power state badge on node detail page with amber Unknown/error styling
2. Power action buttons (Power On, Force Off, Graceful Restart, Force Restart) with context-aware visibility
3. In-flight feedback with amber badge and disabled buttons during power operations

**Archive:** [v1.8-ROADMAP.md](milestones/v1.8-ROADMAP.md) | [v1.8-REQUIREMENTS.md](milestones/v1.8-REQUIREMENTS.md)

---

## v1.6 LLMFit for Best Fit Models — SHIPPED 2026-07-26

**Phases:** 5 | **Plans:** 7
**Timeline:** 2026-07-24 to 2026-07-26 (3 days)
**Commits:** 9 feat
**Requirements:** 12/12 complete (100%)

### Key Accomplishments

1. llmfit Pydantic models, domain error hierarchy, and SSH-based LLMFitRunner with timeout protection
2. Non-fatal llmfit installation during provisioning via soft_step wrapper
3. Admin API endpoint GET /admin/nodes/{hostname}/recommendations with structured error responses
4. SetupRequest.model field with VLLM_MODEL injection through provisioner (shlex.quote for shell safety)
5. Dashboard recommendations card with button-triggered fetch for expensive remote operations

### Known Deferred Items

- Filtering/caching enhancements (FILT-01 through FILT-03, CACHE-01, FLEET-01) deferred to future milestone

**Archive:** [v1.6-ROADMAP.md](milestones/v1.6-ROADMAP.md) | [v1.6-REQUIREMENTS.md](milestones/v1.6-REQUIREMENTS.md)

---

## v1.7 HuggingFace Integration — SHIPPED 2026-07-29

**Phases:** 3 | **Plans:** 5 | **Tests:** 568 | **LOC:** 16,237 (Python + HTML/CSS/JS)
**Timeline:** 2026-07-28 to 2026-07-29 (2 days)
**Commits:** 8 feat, 1 fix
**Requirements:** 11/11 complete (100%)

### Key Accomplishments

1. HuggingFace settings with SecretStr token and NFS cache directory configuration
2. ModelCatalogService scanning HF cache via asyncio.to_thread with admin API endpoint
3. DownloadService with thread-safe status tracking, semaphore-gated concurrency, and background downloads
4. Admin API: GET /admin/models/catalog, POST /admin/models/download, GET /admin/models/downloads
5. Dashboard download column with catalog cross-reference, optimistic UI trigger, and lazy 4s polling

### Known Deferred Items

- Download enhancements (DLE-01 through DLE-05) deferred to future milestone
- In-memory download status (lost on restart)
- Verification gaps from v1.0 still open (Phases 3, 6)

**Archive:** [v1.7-ROADMAP.md](milestones/v1.7-ROADMAP.md) | [v1.7-REQUIREMENTS.md](milestones/v1.7-REQUIREMENTS.md)

---

## v1.5 Node Setup Enhancements (Shipped: 2026-07-22)

**Phases completed:** 4 phases, 6 plans, 5 tasks

**Key accomplishments:**

- 1. [Rule 1 - Bug] Fixed TestResolveBmcHost using deprecated asyncio.get_event_loop()
- 1. [Rule 3 - Blocking] Reordered main.py lifespan initialization

---

## v1.4 Chatbot Playground (Shipped: 2026-07-21)

**Phases completed:** 2 phases, 3 plans, 2 tasks

**Key accomplishments:**

- 1. [Rule 2 - Missing file] Created placeholder chat.js

---

## v1.3 QUADS Integration (Shipped: 2026-07-20)

**Phases completed:** 4 phases, 5 plans, 2 tasks

**Key accomplishments:**

- HTML template (dashboard.html):

---

## v1.2 Node Setup — SHIPPED 2026-07-08

**Phases:** 5 | **Plans:** 9 | **Tests:** 338 | **LOC:** 9,635 (Python + HTML/CSS/JS)
**Timeline:** 2026-07-01 to 2026-07-08 (7 days)
**Commits:** 16 feat commits

### Key Accomplishments

1. Hardened setup.sh and start-vllm.sh for idempotent, fail-fast automated execution
2. SSH provisioning via asyncssh — remote setup, container build, GPU auto-detection, health poll, etcd registration
3. Pre-flight validation (SSH, GPU, disk), state machine tracking (16-step ProvisioningStep), health checker coordination
4. Teardown lifecycle with graceful drain and force modes, etcd deregistration
5. Admin REST API — POST /admin/nodes/setup, GET /admin/provisioning/tasks, DELETE /admin/nodes/{id}
6. Dashboard operations UI — setup form, teardown buttons, provisioning tasks panel with step badges

### Known Deferred Items

- 2 verification gaps from v1.0 still open (Phases 3, 6 — require live vLLM/etcd)

**Archive:** [v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md) | [v1.2-REQUIREMENTS.md](milestones/v1.2-REQUIREMENTS.md)

---

## v1.1 Web UI — SHIPPED 2026-07-01

**Phases:** 3 | **Plans:** 5 | **Tests:** 265 | **LOC:** 7,618 (Python + HTML/CSS/JS)
**Timeline:** 2026-06-29 to 2026-07-01 (3 days)
**Commits:** 9 code commits (5 feat, 2 fix, 1 test)

### Key Accomplishments

1. Thread-safe request metrics tracking (per-node, per-model, total) with enriched admin API
2. Jinja2-rendered operations dashboard with node fleet table and Simple.css styling
3. Per-node request count display with JS polling auto-refresh at configurable interval
4. Badge-based visual distinction for node health and circuit breaker states

### Known Deferred Items

- 1 verification gap from v1.0 still open (Phases 3, 6 — require live vLLM/etcd)

**Archive:** [v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md) | [v1.1-REQUIREMENTS.md](milestones/v1.1-REQUIREMENTS.md)

---

## v1.0 MVP — SHIPPED 2026-06-25

**Phases:** 6 | **Plans:** 13 | **Tests:** 226 | **LOC:** 6,830 Python
**Timeline:** 2026-06-02 to 2026-06-25 (23 days)
**Commits:** 132 (20 feat, 18 test)

### Key Accomplishments

1. OpenAI-compatible API proxy with chat/text completions, model listing, and health endpoint
2. etcd-based service discovery with real-time watch updates and reconnection
3. SSE streaming for token-by-token responses with proper `[DONE]` termination
4. Least-connections load balancing with model-aware routing and drain coordination
5. Resilience layer: health checks, retry with failover, circuit breaker, graceful shutdown
6. Structured request logging and admin node status API for operational visibility

### Known Deferred Items

- 2 verification gaps requiring live vLLM/etcd testing (Phases 3, 6)

**Archive:** [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md) | [v1.0-REQUIREMENTS.md](milestones/v1.0-REQUIREMENTS.md)
