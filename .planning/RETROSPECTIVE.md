# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — Web UI

**Shipped:** 2026-07-01
**Phases:** 3 | **Plans:** 5

### What Was Built
- Thread-safe request metrics (per-node, per-model, total) with enriched admin API
- Jinja2 operations dashboard with node fleet table and Simple.css styling
- JS polling auto-refresh with configurable interval and per-node request counts

### What Worked
- Jinja2 + vanilla JS kept the stack simple — no build step, no Node.js toolchain
- TDD pattern from v1.0 carried forward cleanly (265 tests, all green)
- Small milestone scope (3 phases, 3 days) kept focus tight
- Existing admin API provided clean data layer for dashboard to consume

### What Was Inefficient
- Phase 08 visual verification required manual human check — automated tests covered structure but not visual rendering
- gsd-sdk `milestone.complete` extracted deviations as accomplishments — needed manual fix

### Patterns Established
- Jinja2 template + static CSS/JS pattern for server-rendered UI pages
- `DashboardSettings` sub-model pattern for feature-specific config
- Inline `<script>` for injecting server config into client JS (poll interval)

### Key Lessons
1. Visual verification gaps are unavoidable for UI work — budget for them explicitly in the phase plan
2. In-memory counters are fine for v1 ops dashboards — don't over-engineer persistence before there's a need

---

## Milestone: v1.7 — HuggingFace Integration

**Shipped:** 2026-07-29
**Phases:** 3 | **Plans:** 5

### What Was Built
- HuggingFace settings with SecretStr token and NFS cache dir configuration
- ModelCatalogService scanning HF cache with admin API endpoint
- DownloadService with thread-safe status tracking and semaphore-gated background downloads
- Dashboard download column with catalog cross-reference, optimistic UI, lazy polling

### What Worked
- llmfit model names being HF repo IDs eliminated mapping complexity — zero transformation needed
- Independent try/catch per fetch (catalog, downloads, recommendations) prevented cascade failures
- Optimistic UI pattern (immediate badge swap on download click) gave instant feedback without waiting for server
- 2-day execution for 3 phases — tight scope and clear requirements kept velocity high

### What Was Inefficient
- Phase 32 DASH requirements were left unchecked in REQUIREMENTS.md traceability despite code being complete — caught at milestone close
- XSS vectors found in code review after Phase 32 execution — security review should run before marking phase complete

### Patterns Established
- asyncio.to_thread wrapper for sync HF library calls (same pattern as etcd3gw)
- Module-level cache Set shared between initial load and poll updater for download state
- Lazy polling with single-timer guard — starts on user action, auto-stops when idle

### Key Lessons
1. When reusing a sync library in an async app, the thread pool pattern (asyncio.to_thread + ThreadPoolExecutor) is now proven across two subsystems (etcd, HF)
2. Dashboard features that poll should use lazy polling (start on trigger, stop when idle) rather than always-on intervals
3. Security review needs to run as part of phase execution, not as a post-hoc catch-up

---

## Milestone: v1.6 — LLMFit for Best Fit Models

**Shipped:** 2026-07-26
**Phases:** 5 | **Plans:** 7

### What Was Built
- llmfit Pydantic models, domain error hierarchy, SSH runner with timeout protection
- Non-fatal llmfit installation in provisioning via soft_step wrapper
- Admin API GET /admin/nodes/{hostname}/recommendations with structured errors
- Model selection in provisioning (SetupRequest.model, VLLM_MODEL injection)
- Dashboard recommendations card with button-triggered fetch

### What Worked
- Constructor injection (DIP) for LLMFitRunner made testing straightforward
- Domain error hierarchy translated SSH/parse failures into structured 502 responses
- soft_step pattern (non-fatal provisioning step) prevents llmfit install from blocking setup
- shlex.quote for shell safety when injecting model names into remote commands

### What Was Inefficient
- 5 phases for what was essentially 3 logical units (core, API, UI) — could have been tighter

### Patterns Established
- soft_step wrapper for non-fatal provisioning steps
- Button-triggered fetch for expensive remote operations (vs auto-load)
- Domain error hierarchy for structured API error responses

### Key Lessons
1. Non-fatal provisioning steps need explicit soft_step semantics — silent failures are worse than skipping
2. Button-triggered fetch is the right default for operations that SSH into remote hosts

---

## Milestone: v1.8 — Nodes Power Control

**Shipped:** 2026-07-29
**Phases:** 2 | **Plans:** 2

### What Was Built
- BMC power state badge on node detail page header
- Power action buttons (Power On, Force Off, Graceful Restart, Force Restart)
- Context-aware button visibility and in-flight feedback

### What Worked
- Frontend-only milestone — backend already existed from v1.5, zero backend code needed
- POWER_BADGE lookup pattern kept badge rendering clean
- refreshPowerState() designed for reuse between Phase 33 and 34

### What Was Inefficient
- Archive commit got dropped from main during subsequent milestone work — archival should be verified after

### Patterns Established
- POWER_BADGE lookup pattern: maps API string values to {cls, text} objects

### Key Lessons
1. When backend exists, frontend-only milestones can ship in a day
2. Verify milestone archive commits survive subsequent rebases/merges

---

## Milestone: v1.9 — Model Selection in Node Setup

**Shipped:** 2026-07-29
**Phases:** 1 | **Plans:** 1

### What Was Built
- Model selector dropdown on node detail page from NFS model catalog
- Setup blocked when no models downloaded

### What Worked
- Single-phase milestone executed in under a day
- Catalog API already existed from v1.7 — pure UI integration

### What Was Inefficient
- Nothing notable — cleanest execution of any milestone

### Key Lessons
1. When catalog/API infrastructure exists, UI-only features are fast wins

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 MVP | 6 | 13 | Established TDD, circuit breaker, structured logging patterns |
| v1.1 Web UI | 3 | 5 | Added frontend (Jinja2+JS), first UI verification gap |
| v1.6 LLMFit | 5 | 7 | SSH runner, domain errors, non-fatal provisioning steps |
| v1.7 HuggingFace | 3 | 5 | HF downloads integrated into dashboard, 2-day execution |
| v1.8 Power Control | 2 | 2 | Frontend-only milestone, backend reuse from v1.5 |
| v1.9 Model Selection | 1 | 1 | Single-phase, catalog API reuse from v1.7 |

### Cumulative Quality

| Milestone | Tests | LOC | New Dependencies |
|-----------|-------|-----|------------------|
| v1.0 | 226 | 6,830 | FastAPI, httpx, etcd3gw, structlog, pydantic-settings |
| v1.1 | 265 | 7,618 | jinja2 |
| v1.6 | ~500 | ~14,000 | (none) |
| v1.7 | 568 | 16,237 | huggingface-hub |
| v1.8 | ~600 | ~17,000 | (none) |
| v1.9 | 1,112 | 50,722 | (none) |

### Top Lessons (Verified Across Milestones)

1. Small, focused phases (2-3 plans each) execute faster than large ones
2. TDD catches integration issues early — wiring DI fixtures before new tests prevents cascading failures
