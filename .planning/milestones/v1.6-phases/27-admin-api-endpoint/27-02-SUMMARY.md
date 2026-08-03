---
phase: 27-admin-api-endpoint
plan: 02
subsystem: api
tags: [pytest, integration-tests, llmfit, error-handling]

requires:
  - phase: 27-admin-api-endpoint
    plan: 01
    provides: GET /admin/nodes/{hostname}/recommendations endpoint
provides:
  - TestRecommendations class (API-01, API-02 coverage)
  - TestRecommendationErrors class (API-03, D-01 coverage)
  - mock_llmfit_runner fixture in conftest.py
affects: []

tech-stack:
  added: []
  patterns: [DI override testing via app.dependency_overrides]

key-files:
  created: []
  modified:
    - tests/conftest.py
    - tests/api/test_admin.py

key-decisions:
  - "mock_llmfit_runner follows mock_provisioner fixture pattern for consistency"
  - "Invalid hostname test uses host@evil instead of path traversal (URL normalization prevents ../../ from reaching the handler)"

requirements-completed: [API-01, API-02, API-03]

duration: 4min
completed: 2026-07-26
---

# Phase 27 Plan 02: Recommendation Endpoint Tests Summary

**Integration tests for recommendations endpoint: 9 tests covering happy path, all 4 error types, hostname validation, and D-01 raw-output invariant**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-26T15:03:01Z
- **Completed:** 2026-07-26T15:07:36Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- mock_llmfit_runner fixture wired into conftest.py app fixture with DI override
- TestRecommendations: 4 tests proving 200 response with hostname, hardware info, model list, and hostname validation (API-01, API-02)
- TestRecommendationErrors: 5 tests proving all 4 error types return 502 with correct error_type field (API-03)
- D-01 invariant verified: raw_output string absent from API error response body
- Full suite (533 tests) passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Test fixtures and happy path tests** - `d86fa39` (test)
2. **Task 2: Error scenario tests** - `b6d9f1f` (test)

## Files Created/Modified
- `tests/conftest.py` - Added mock_llmfit_runner to app fixture and standalone fixture
- `tests/api/test_admin.py` - Added SAMPLE_RESULT, TestRecommendations (4 tests), TestRecommendationErrors (5 tests)

## Decisions Made
- mock_llmfit_runner follows the existing mock_provisioner fixture pattern (app.state + DI override) for consistency
- Invalid hostname test uses `host@evil` instead of path traversal `../../etc/passwd` because URL normalization resolves the path before it reaches the FastAPI route handler

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid hostname test data**
- **Found during:** Task 1
- **Issue:** Plan specified `../../etc/passwd` as invalid hostname, but URL normalization resolves this to `/etc/passwd/recommendations` which returns 404 (route not found) instead of 400 (validation error)
- **Fix:** Changed to `host@evil` which matches the route but fails `_validated_hostname` regex validation
- **Files modified:** tests/api/test_admin.py
- **Commit:** d86fa39

## Issues Encountered
None

## Self-Check: PASSED
