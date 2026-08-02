---
phase: 24-provisioning-error-diagnostics
verified: 2026-07-22T16:30:00Z
status: human_needed
score: 3/3 must-haves verified
re_verification: false
human_verification:
  - test: "Visual verification of expandable error sub-row in dashboard"
    expected: |
      1. Failed node shows red "failed" badge in State column
      2. Badge has pointer cursor on hover
      3. Clicking badge expands sub-row showing:
         - "failed at {step_name}" badge
         - Full error text in monospace
      4. Clicking again collapses the sub-row
      5. Sub-row has light red background (danger theme)
      6. Setup and Teardown action buttons appear for failed node
      7. Tab to badge and press Enter toggles sub-row (keyboard accessibility)
    why_human: "Visual appearance, interaction behavior, and accessibility require browser verification. Automated tests verify data structure and code logic but cannot confirm visual presentation or keyboard interaction UX."
---

# Phase 24: Provisioning Error Diagnostics Verification Report

**Phase Goal:** Operators can see why provisioning failed without checking logs
**Verified:** 2026-07-22T16:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Failed provisioning captures the specific step name where failure occurred | ✓ VERIFIED | Provisioner tracks `current_step` variable ("uploading_scripts", "starting_vllm", "health_poll", "registering") and passes it to `_update_state(failed_step=current_step)` in except block. Test `test_failed_state` verifies actual step name is captured, not exception class name. |
| 2 | Failed provisioning captures error details (stderr/exception message) | ✓ VERIFIED | `_update_state` called with `error=str(exc)` in provisioner except block. AdminNodeResponse has `error: str \| None` field. Full error text rendered via `textContent` with no truncation (pre-wrap CSS). |
| 3 | Dashboard displays failure details inline for failed nodes instead of just a status badge | ✓ VERIFIED | dashboard.js creates expandable error-subrow when `node.state === "failed"` with click/keyboard toggle. Sub-row displays step name badge and error text. CSS styled with `var(--danger-bg)`. Requires human verification for visual/interaction UX. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `inference_proxy/models/node.py` | NodeStatus.FAILED enum member | ✓ VERIFIED | Line 27: `FAILED = "failed"` — 6th enum member (after PROVISIONING, before UNKNOWN) |
| `inference_proxy/models/admin.py` | Error fields on AdminNodeResponse | ✓ VERIFIED | Lines 40-41: `failed_step: str \| None = None` and `error: str \| None = None` — matches TaskStatusResponse pattern |
| `inference_proxy/provisioning/provisioner.py` | Step tracking and node FAILED update | ✓ VERIFIED | Lines 266-301: `current_step` variable tracked before each phase, passed to `failed_step=current_step` on exception. Failed node written to etcd with `NodeStatus.FAILED` (lines 288-300). |
| `inference_proxy/services/unified_nodes.py` | Failed state actions and error merge | ✓ VERIFIED | Line 26: `"failed": ["setup", "teardown"]` in _STATE_ACTIONS. Lines 85-105: `_from_etcd` accepts `task_map`, merges `failed_step` and `error` from task lookup. |
| `inference_proxy/static/js/dashboard.js` | Expandable error sub-row for failed nodes | ✓ VERIFIED | Lines 247-296: Creates error-subrow when `node.state === "failed"`. Badge clickable with `role="button"`, `tabindex="0"`, `aria-expanded` toggle. Uses `textContent` (XSS-safe). |
| `inference_proxy/static/css/dashboard.css` | Sub-row and error detail styling | ✓ VERIFIED | Lines 466-476: `.error-subrow`, `.error-detail`, `.error-message` rules with danger theme, monospace font, pre-wrap. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `provisioner.py` | etcd provisioning state | `_update_state` with current_step | ✓ WIRED | Line 285: `failed_step=current_step, error=str(exc)` passed to `_update_state`. Lines 288-300: Failed node written to etcd with `NodeStatus.FAILED`. |
| `api/admin.py` | `unified_nodes.py` | task_map parameter | ✓ WIRED | Lines 77-86: `list_nodes` builds `task_map` from `provisioner.list_tasks_raw()` and passes to `service.get_unified_nodes(task_map=task_map)`. |
| `dashboard.js` | `/admin/nodes` API | fetch in refreshDashboard | ✓ WIRED | Lines 248-271: Reads `node.failed_step` and `node.error` from API response, renders in sub-row with textContent. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `dashboard.js` error sub-row | `node.failed_step`, `node.error` | AdminNodeResponse from GET /admin/nodes | Yes — from provisioning task state in etcd | ✓ FLOWING |
| `AdminNodeResponse` error fields | `failed_step`, `error` | TaskStatusResponse via task_map | Yes — from ProvisioningStep.FAILED state update | ✓ FLOWING |
| `unified_nodes._from_etcd` | `task.failed_step`, `task.error` | task_map lookup by node_id | Yes — from provisioner exception handler | ✓ FLOWING |

**Data flow verified:** Provisioner captures step name and error on exception → writes to etcd provisioning state → API builds task_map → service merges into AdminNodeResponse → dashboard renders in expandable sub-row.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite | `uv run pytest tests/ -q` | 502 passed, 10 warnings, exit 0 | ✓ PASS |
| NodeStatus enum has FAILED | `python -c "from inference_proxy.models.node import NodeStatus; assert NodeStatus.FAILED == 'failed'"` | No error | ✓ PASS |
| AdminNodeResponse accepts error fields | `python -c "from inference_proxy.models.admin import AdminNodeResponse; r = AdminNodeResponse(node_id='x', endpoint='x', model='x', status='x', active_connections=0, circuit_breaker_state='x', failed_step='step', error='err'); assert r.failed_step == 'step'"` | No error | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DIAG-01 | 24-01 | Failed provisioning step name and error details are captured and stored | ✓ SATISFIED | NodeStatus.FAILED exists, provisioner tracks `current_step` (not exception class name), failed node written to etcd with FAILED status, error stored in provisioning state and merged via task_map. |
| DIAG-02 | 24-02 | Dashboard displays failure details inline for failed nodes instead of just a state badge | ✓ SATISFIED | Dashboard creates expandable error-subrow on failed badge click, shows step name and full error text, styled with danger theme. Human verification required for UX. |

### Anti-Patterns Found

None. Clean implementation:

| Pattern | Scan Result | Status |
|---------|-------------|--------|
| Debt markers (TBD, FIXME, XXX) | 0 matches | ✓ CLEAN |
| innerHTML usage (XSS risk) | 0 matches in dashboard.js | ✓ SAFE |
| Hardcoded empty data in logic paths | 0 matches (only in tests) | ✓ CLEAN |
| Console.log-only implementations | 0 matches | ✓ CLEAN |

### Human Verification Required

#### 1. Visual verification of expandable error sub-row in dashboard

**Test:**
1. Start the dev server: `cd /home/developer/Sources/inference-proxy && uv run uvicorn inference_proxy.main:app --reload`
2. Open http://localhost:8000/dashboard in a browser
3. If no nodes are currently failed, trigger a failure: set up a node that will fail (e.g., a hostname with no SSH access)
4. Verify the failed node shows a red "failed" badge in the State column
5. Verify the badge has `cursor: pointer` (mouse cursor changes on hover)
6. Click the failed badge — a sub-row should expand below showing:
   - A "failed at {step_name}" badge (e.g., "failed at uploading_scripts")
   - The full error text in monospace below
7. Click the badge again — sub-row should collapse
8. Verify the sub-row has a light red background (matching danger theme)
9. Verify the "Setup" and "Teardown" action buttons appear for the failed node
10. Tab to the failed badge and press Enter — should toggle the sub-row (keyboard accessibility)

**Expected:**
- Failed node badge is visually distinct (red) and interactive (pointer cursor)
- Clicking badge toggles sub-row display smoothly
- Sub-row shows step name badge + error text with no truncation
- Sub-row background matches danger theme (light red)
- Failed nodes have both Setup and Teardown action buttons
- Keyboard navigation works (Tab to badge, Enter/Space toggles sub-row)
- Sub-rows toggle independently per node (if multiple failed nodes)

**Why human:**
Visual appearance (badge color, background color, font styling), interaction behavior (click/keyboard toggle, cursor change), and accessibility (keyboard focus, aria-expanded state) require browser verification. Automated tests verify data structure and code logic but cannot confirm visual presentation or user interaction UX.

---

_Verified: 2026-07-22T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
