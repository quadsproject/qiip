# Phase 33: Power State Display - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Display the current BMC power state of a node on its detail page. A power state badge (On/Off/Unknown) appears in the header area, fetched from the existing backend power API on page load. A reusable refresh function is exposed for Phase 34 (power action controls) to call after actions complete.

</domain>

<decisions>
## Implementation Decisions

### Badge Placement
- **D-01:** Power state badge goes below the existing node state text (`<p id="node-state">`), as a new line in the header. Keeps service state (available/healthy/unhealthy) visually separate from BMC power state.
- **D-02:** Badge label reads "Power: On", "Power: Off", or "Power: Unknown" — prefixed to distinguish from service state.

### Badge Styling
- **D-03:** Reuse existing badge CSS classes — On = `badge-complete` (green), Off = `badge-failed` (red), Unknown = `badge-in-progress` (amber). No new CSS classes needed.

### Refresh Strategy
- **D-04:** Fetch power state once on page load (DOMContentLoaded). Do NOT add to the existing `refreshDetail()` polling loop — avoids repeated BMC calls for a value that rarely changes externally. Expose a standalone `refreshPowerState()` function that Phase 34 calls after power actions.

### Error Handling
- **D-05:** When BMC power state fetch fails (Redfish unreachable, no BMC configured, HTTP error), display "Power: Unknown" badge with `badge-in-progress` (amber). Treat all fetch failures as unknown state — no error noise, no retry link.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Backend Power API (already built in v1.5)
- `inference_proxy/api/admin.py` lines 281-311 — GET/POST `/admin/nodes/{hostname}/power` endpoints
- `inference_proxy/models/admin.py` — `PowerStateResponse` and `PowerActionRequest` models
- `inference_proxy/redfish/client.py` — Redfish BMC client implementation

### Node Detail Page (modify target)
- `inference_proxy/templates/node_detail.html` — Jinja2 template with header layout
- `inference_proxy/static/js/node_detail.js` — Page JS with `refreshDetail()`, action handling, badge patterns

### Styling
- `inference_proxy/static/css/dashboard.css` — Badge classes (`badge-complete`, `badge-failed`, `badge-in-progress`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Badge classes: `badge-complete` (green), `badge-failed` (red), `badge-in-progress` (amber) — reuse directly for power state mapping
- `showToast(message, type)` — available for error feedback if needed
- `NODE_ID` global — hostname already set from Jinja2 template variable

### Established Patterns
- Vanilla JS DOM manipulation (createElement, appendChild) — no frameworks
- `fetch()` with try/catch for API calls
- Badge rendering: `<span class="badge badge-{state}">{text}</span>`
- CSS custom properties (`var(--*)`) for all colors — dark/light theme compatible

### Integration Points
- Header div in `node_detail.html` — add new `<p>` element after `<p id="node-state">`
- `node_detail.js` DOMContentLoaded handler — add `refreshPowerState()` call
- `refreshPowerState()` function — standalone, callable from Phase 34's action handlers

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches matching existing page patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 33-Power State Display*
*Context gathered: 2026-07-29*
