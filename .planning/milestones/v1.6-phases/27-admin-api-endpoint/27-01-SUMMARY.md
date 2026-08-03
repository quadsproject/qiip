---
phase: 27-admin-api-endpoint
plan: 01
subsystem: api
tags: [fastapi, pydantic, llmfit, ssh, dependency-injection]

requires:
  - phase: 26-llmfit-install
    provides: LLMFitRunner, LLMFitResult models, SSH-based llmfit execution
provides:
  - GET /admin/nodes/{hostname}/recommendations endpoint
  - LLMFitSettings configurable via env vars
  - RecommendationResponse Pydantic model
  - LLMFitRunner DI wiring via get_llmfit_runner
affects: [27-02 (tests for this endpoint)]

tech-stack:
  added: []
  patterns: [settings-driven runner config, structured 502 error responses with error_type field]

key-files:
  created: []
  modified:
    - inference_proxy/config/settings.py
    - inference_proxy/models/admin.py
    - inference_proxy/llmfit/runner.py
    - inference_proxy/config/dependencies.py
    - inference_proxy/main.py
    - inference_proxy/api/admin.py
    - .env.example

key-decisions:
  - "LLMFitSettings placed as BaseModel sub-model following SSHSettings pattern"
  - "Runner constructor keeps settings optional for backward compatibility"

patterns-established:
  - "Structured 502 errors: JSONResponse with {error_type, detail} for upstream failures"
  - "Raw error output logged via structlog, never exposed in API response body (D-01)"

requirements-completed: [API-01, API-02, API-03]

duration: 2min
completed: 2026-07-26
---

# Phase 27 Plan 01: Admin API Endpoint Summary

**GET /admin/nodes/{hostname}/recommendations endpoint with LLMFitSettings config, DI wiring, and structured 502 error handling**

## Performance

- **Duration:** 2 min
- **Started:** 2026-07-26T14:56:10Z
- **Completed:** 2026-07-26T14:58:38Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- LLMFitRunner refactored from hardcoded class vars to injectable LLMFitSettings
- GET /admin/nodes/{hostname}/recommendations returns ranked models with hardware info
- All 4 error types (timeout, parse_error, connection_error, ssh_error) return HTTP 502 with typed error_type field
- Raw llmfit output never exposed in API responses (D-01 security requirement)

## Task Commits

Each task was committed atomically:

1. **Task 1: LLMFitSettings, response model, runner refactor, DI wiring** - `4ac8131` (feat)
2. **Task 2: GET /admin/nodes/{hostname}/recommendations endpoint** - `6358135` (feat)

## Files Created/Modified
- `inference_proxy/config/settings.py` - Added LLMFitSettings sub-model with binary_path and timeout
- `inference_proxy/models/admin.py` - Added RecommendationResponse with hostname, system, models
- `inference_proxy/llmfit/runner.py` - Refactored to use injected LLMFitSettings instead of class vars
- `inference_proxy/config/dependencies.py` - Added get_llmfit_runner DI provider
- `inference_proxy/main.py` - LLMFitRunner initialization in lifespan
- `inference_proxy/api/admin.py` - GET recommendations endpoint with 4-way error handling
- `.env.example` - Documented INFERENCE_PROXY_LLMFIT__* env vars

## Decisions Made
- LLMFitSettings placed as BaseModel (not BaseSettings) following SSHSettings pattern -- nested env var resolution through root Settings class
- Runner constructor keeps `settings: LLMFitSettings | None = None` for backward compatibility so existing `LLMFitRunner(ssh_client=mock)` calls still work

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Endpoint is wired and importable, ready for test coverage in Plan 02
- All existing tests (54) continue passing

---
*Phase: 27-admin-api-endpoint*
*Completed: 2026-07-26*
