# Phase 33: Power State Display - Pattern Map

**Mapped:** 2026-07-29
**Files analyzed:** 2 (both modifications, no new files)
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `inference_proxy/templates/node_detail.html` | template | request-response | itself (line 33 `<p id="node-state">`) | exact |
| `inference_proxy/static/js/node_detail.js` | client-script | request-response | itself (`refreshDetail()` + `loadRecommendations()`) | exact |

## Pattern Assignments

### `inference_proxy/templates/node_detail.html` (template, request-response)

**Analog:** Same file, header section.

**Header element pattern** (lines 32-33):
```html
<h1 id="node-title">{{ node_id }}</h1>
<p id="node-state">Loading...</p>
```
New `<p id="power-state">` goes immediately after line 33, before the closing `</div>` on line 34. Same structure: a `<p>` with an `id`, containing a `<span class="badge ...">` child.

**Insertion point** (line 33-34):
```html
<p id="node-state">Loading...</p>
<!-- INSERT HERE: <p id="power-state" ...> -->
</div>
```

---

### `inference_proxy/static/js/node_detail.js` (client-script, request-response)

**Analog:** Same file -- `loadRecommendations()` and `refreshDetail()`.

**Fetch + error handling pattern** (lines 393-419, `loadRecommendations`):
```javascript
async function loadRecommendations() {
  // ...
  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/recommendations");
    if (!resp.ok) {
      var err = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      // handle error...
      return;
    }
    var data = await resp.json();
    // use data...
  } catch (e) {
    // network error...
  }
}
```
`refreshPowerState()` follows this shape but simpler: no error toast, just fall through to "Unknown" badge on any failure (per D-05).

**Badge class mapping pattern** (lines 128-132, `stepBadgeClass`):
```javascript
function stepBadgeClass(step) {
  if (step === "complete" || step === "teardown_complete") return "badge-complete";
  if (step === "failed") return "badge-failed";
  return "badge-in-progress";
}
```
Power state uses a lookup object instead (simpler for a fixed mapping):
- `On` -> `badge-complete` (green)
- `Off` -> `badge-failed` (red)
- anything else / error -> `badge-unknown` (amber)

**DOM update pattern** (line 168):
```javascript
var sb = document.createElement("span"); sb.className = "badge badge-" + node.state; sb.textContent = node.state;
```
Power badge updates the *existing* span rather than creating one (the span is in the template already).

**DOMContentLoaded pattern** (lines 529-532):
```javascript
document.addEventListener("DOMContentLoaded", function () {
  refreshDetail();
  setInterval(refreshDetail, POLL_INTERVAL_MS);
});
```
Add `refreshPowerState();` call after `refreshDetail();` -- NOT inside the `setInterval` (D-04).

**URL encoding pattern** (line 403):
```javascript
var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/recommendations");
```
Power state fetch uses the same pattern: `"/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power"`.

**Function scope pattern**: All functions (`refreshDetail`, `loadRecommendations`, `handleAction`, etc.) are declared at module top level -- no IIFE, no module system. `refreshPowerState` must follow the same convention so Phase 34 can call it.

---

## Shared Patterns

### Badge CSS Classes
**Source:** `inference_proxy/static/css/dashboard.css` lines 331-358
**Apply to:** Power state badge

| CSS Class | Color Variable | Visual |
|-----------|---------------|--------|
| `badge-complete` | `var(--success)` / `var(--success-bg)` | Green |
| `badge-failed` | `var(--danger)` / `var(--danger-bg)` | Red |
| `badge-unknown` | `var(--warning)` / `var(--warning-bg)` | Amber |
| `badge-in-progress` | `var(--primary)` | Blue -- DO NOT use for Unknown |

**Critical:** CONTEXT.md D-03 says `badge-in-progress` but describes amber. The CSS maps `badge-in-progress` to blue (`var(--primary)`). Use `badge-unknown` for amber per RESEARCH.md correction.

### Error-as-Unknown Pattern
**Source:** Decision D-05
**Apply to:** `refreshPowerState()` catch block

All fetch failures (network error, non-200 status, missing BMC) silently resolve to `{ cls: "badge-unknown", text: "Power: Unknown" }`. No toast, no retry link, no console logging.

## No Analog Found

None -- both files modify existing code with patterns already present in those files.

## Metadata

**Analog search scope:** `inference_proxy/templates/`, `inference_proxy/static/js/`, `inference_proxy/static/css/`
**Files scanned:** 3
**Pattern extraction date:** 2026-07-29
