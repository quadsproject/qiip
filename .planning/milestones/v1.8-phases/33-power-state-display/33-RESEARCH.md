# Phase 33: Power State Display - Research

**Researched:** 2026-07-29
**Domain:** Frontend — vanilla JS DOM manipulation, existing badge CSS
**Confidence:** HIGH

## Summary

This phase adds a single power state badge to the node detail page header. The backend API (`GET /admin/nodes/{hostname}/power`) already exists and returns `{ hostname, power_state }`. The work is entirely frontend: one HTML element in the Jinja2 template, one JS function in `node_detail.js`, zero new CSS, zero new packages.

The existing codebase already has every pattern needed — badge rendering, fetch-with-try/catch, DOM updates, CSS classes for green/red/amber states. The `refreshPowerState()` function follows the same shape as existing fetch calls but is intentionally kept standalone (not inside the polling loop) per decision D-04.

**Primary recommendation:** Add a `<p id="power-state">` element to the template header, write a ~20-line `refreshPowerState()` function in `node_detail.js`, call it on DOMContentLoaded.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Power state badge goes below `<p id="node-state">`, as a new line in the header. Keeps service state visually separate from BMC power state.
- **D-02:** Badge label reads "Power: On", "Power: Off", or "Power: Unknown" — prefixed to distinguish from service state.
- **D-03:** Reuse existing badge CSS classes — On = `badge-complete` (green), Off = `badge-failed` (red), Unknown = amber. **UI-SPEC correction:** Use `badge-unknown` (amber) not `badge-in-progress` (blue) for Unknown — matches user intent of amber color.
- **D-04:** Fetch power state once on page load (DOMContentLoaded). Do NOT add to the existing `refreshDetail()` polling loop. Expose a standalone `refreshPowerState()` function for Phase 34.
- **D-05:** When fetch fails, display "Power: Unknown" badge with amber styling. No error noise, no retry link.

### Claude's Discretion
None specified — all decisions locked.

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PWR-01 | Node detail page fetches and displays BMC power state (On/Off/Unknown) as a badge on page load | Badge element in template + `refreshPowerState()` called on DOMContentLoaded |
| PWR-02 | Power state badge auto-refreshes after any power action completes | `refreshPowerState()` exported as standalone function; Phase 34 action handlers call it after POST completes |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Power state display | Browser / Client | -- | Pure DOM update from existing API response |
| Power state fetch | Browser / Client | API / Backend | Client calls existing GET endpoint; backend already built |
| Badge styling | Browser / Client | -- | Existing CSS classes, no changes |

## Standard Stack

No new libraries. This phase uses only what already exists in the project:

| Asset | Location | Purpose |
|-------|----------|---------|
| Vanilla JS | `node_detail.js` | DOM manipulation, fetch API |
| CSS badge classes | `dashboard.css` | `badge-complete`, `badge-failed`, `badge-unknown` |
| Jinja2 template | `node_detail.html` | Static HTML structure |
| Backend API | `GET /admin/nodes/{hostname}/power` | Returns `PowerStateResponse` |

**Installation:** None required. Zero new dependencies.

## Package Legitimacy Audit

Not applicable — no packages installed in this phase.

## Architecture Patterns

### System Architecture Diagram

```
Page Load (DOMContentLoaded)
    |
    v
refreshPowerState()  <--- also callable from Phase 34 action handlers
    |
    v
GET /admin/nodes/{NODE_ID}/power
    |
    +---> 200: { power_state: "On"|"Off"|... }
    |         |
    |         v
    |     Map to badge class + label
    |     Update #power-state span
    |
    +---> Error (network, 4xx, 5xx)
              |
              v
          Set badge to "Power: Unknown" (amber)
```

### Recommended Project Structure

No new files. Changes touch exactly two existing files:

```
inference_proxy/
  templates/
    node_detail.html        # Add <p id="power-state"> to header
  static/
    js/
      node_detail.js        # Add refreshPowerState() function + DOMContentLoaded call
```

### Pattern 1: Fetch-and-Update Badge (existing codebase pattern)

**What:** Fetch an API endpoint, map response to badge class, update DOM element.
**When to use:** Whenever displaying server state as a colored badge.
**Example:**
```javascript
// Source: existing pattern in node_detail.js (stepBadgeClass, refreshDetail)
var POWER_BADGE = {
  On: { cls: "badge-complete", text: "Power: On" },
  Off: { cls: "badge-failed", text: "Power: Off" },
};
var POWER_UNKNOWN = { cls: "badge-unknown", text: "Power: Unknown" };

async function refreshPowerState() {
  var el = document.querySelector("#power-state span");
  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    var info = POWER_BADGE[data.power_state] || POWER_UNKNOWN;
  } catch (_) {
    var info = POWER_UNKNOWN;
  }
  el.className = "badge " + info.cls;
  el.textContent = info.text;
}
```

### Anti-Patterns to Avoid
- **Adding to poll loop:** D-04 explicitly forbids adding `refreshPowerState()` to `setInterval(refreshDetail, ...)`. BMC calls are expensive and the state rarely changes externally.
- **Separate loading state:** No spinner or "Loading..." text for the badge. Start as "Power: Unknown" (amber), resolve to actual state. Matches existing pattern where `node-state` shows "Loading..." then updates.

## Don't Hand-Roll

Not applicable. The entire implementation is ~20 lines of vanilla JS using existing patterns. No libraries solve this problem better than the code already in the codebase.

## Common Pitfalls

### Pitfall 1: Wrong badge class for amber
**What goes wrong:** Using `badge-in-progress` for Unknown renders blue, not amber.
**Why it happens:** CONTEXT.md D-03 says `badge-in-progress` but describes the color as amber. The CSS maps `badge-in-progress` to `var(--primary)` (blue) and `badge-unknown` to `var(--warning)` (amber).
**How to avoid:** Use `badge-unknown` class for Unknown/error states. UI-SPEC 33-UI-SPEC.md documents this correction.
**Warning signs:** Badge shows blue instead of amber for Unknown state.

### Pitfall 2: NODE_ID not URL-encoded
**What goes wrong:** Hostnames with special characters break the fetch URL.
**Why it happens:** Forgetting `encodeURIComponent()` around NODE_ID.
**How to avoid:** Use `encodeURIComponent(NODE_ID)` in the URL path, matching the existing pattern in `loadRecommendations()` (line 403 of node_detail.js).
**Warning signs:** 404 responses for hostnames with dots or hyphens.

### Pitfall 3: refreshPowerState not callable from Phase 34
**What goes wrong:** Phase 34 cannot call the function because it was declared inside an IIFE or block scope.
**Why it happens:** Over-encapsulating the function.
**How to avoid:** Declare `refreshPowerState` at module top level (same as `refreshDetail`, `handleAction`, etc.). The existing file uses no module system — all functions are global.
**Warning signs:** `refreshPowerState is not defined` error when Phase 34 calls it.

## Code Examples

### Template Addition (node_detail.html)
```html
<!-- After <p id="node-state">Loading...</p> -->
<p id="power-state" style="margin:0.25rem 0 0 0">
  <span class="badge badge-unknown">Power: Unknown</span>
</p>
```
Source: 33-UI-SPEC.md Component Inventory section.

### JS Function (node_detail.js)
```javascript
// ponytail: standalone power badge fetch — not in poll loop (D-04), callable by Phase 34
var POWER_BADGE = {
  On: { cls: "badge-complete", text: "Power: On" },
  Off: { cls: "badge-failed", text: "Power: Off" },
};
var POWER_UNKNOWN = { cls: "badge-unknown", text: "Power: Unknown" };

async function refreshPowerState() {
  var el = document.querySelector("#power-state span");
  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    var info = POWER_BADGE[data.power_state] || POWER_UNKNOWN;
  } catch (_) {
    var info = POWER_UNKNOWN;
  }
  el.className = "badge " + info.cls;
  el.textContent = info.text;
}
```
Source: derived from existing `refreshDetail()` and `stepBadgeClass()` patterns in node_detail.js.

### DOMContentLoaded Hook
```javascript
document.addEventListener("DOMContentLoaded", function () {
  refreshDetail();
  refreshPowerState();  // PWR-01: fetch power state on page load
  setInterval(refreshDetail, POLL_INTERVAL_MS);
});
```
Source: existing DOMContentLoaded handler at line 529 of node_detail.js.

## State of the Art

Not applicable — this is vanilla JS DOM manipulation. No framework evolution to track.

## Assumptions Log

No claims tagged `[ASSUMED]` in this research. All findings verified against the actual codebase files read during this session.

## Open Questions

None. The scope is fully constrained by locked decisions and the existing codebase patterns are clear.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio 1.4 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PWR-01 | Badge element exists in template, JS function fetches and updates it | manual-only | N/A — vanilla JS in browser, no server-side rendering to test | N/A |
| PWR-02 | `refreshPowerState()` is a callable function at module scope | manual-only | N/A — requires browser environment | N/A |

**Justification for manual-only:** This phase is pure client-side JS and HTML template changes. The project has no browser testing framework (no Playwright, no Cypress, no jsdom). The backend API that serves the data is already tested. Adding a browser testing framework for a ~20-line JS function would violate YAGNI. Manual verification: open node detail page, confirm badge appears with correct color.

### Sampling Rate
- **Per task commit:** Manual browser check — load node detail page, verify badge
- **Per wave merge:** Full backend suite `uv run pytest tests/` to ensure no regressions
- **Phase gate:** Visual confirmation of badge in all three states (On/Off/Unknown)

### Wave 0 Gaps
None — no test files needed for this frontend-only phase. Existing backend test suite remains green.

## Security Domain

No security concerns for this phase. It reads a single existing authenticated endpoint (admin API) and displays the result. No new inputs, no new endpoints, no user-controlled data rendered without encoding (badge text is hardcoded strings, not user input).

## Project Constraints (from CLAUDE.md)

| Directive | Compliance |
|-----------|------------|
| SOLID principles | N/A — ~20 lines of vanilla JS, no classes or abstractions |
| Update .env.example when env vars change | N/A — no env var changes |
| Tech stack: Python, FastAPI, httpx, etcd3 | N/A — frontend-only changes |
| Vanilla JS patterns (no frameworks) | Compliant — uses existing fetch + DOM pattern |

## Sources

### Primary (HIGH confidence)
- `inference_proxy/templates/node_detail.html` — current template structure (read this session)
- `inference_proxy/static/js/node_detail.js` — existing JS patterns (read this session)
- `inference_proxy/static/css/dashboard.css` — badge CSS classes (read this session)
- `inference_proxy/api/admin.py` lines 281-294 — GET power endpoint (read this session)
- `inference_proxy/models/admin.py` — `PowerStateResponse` model (read this session)
- `.planning/phases/33-power-state-display/33-UI-SPEC.md` — approved UI contract (read this session)

### Secondary (MEDIUM confidence)
None needed — all findings from direct codebase inspection.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new stack, reuses existing patterns verbatim
- Architecture: HIGH — single fetch + DOM update, pattern exists 5+ times in codebase
- Pitfalls: HIGH — verified CSS class mapping against actual dashboard.css

**Research date:** 2026-07-29
**Valid until:** indefinite — codebase patterns are stable, no external dependencies
