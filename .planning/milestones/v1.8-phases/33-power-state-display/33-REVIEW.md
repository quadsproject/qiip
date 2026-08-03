---
phase: 33-power-state-display
reviewed: 2026-07-29T10:52:34Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - inference_proxy/templates/node_detail.html
  - inference_proxy/static/js/node_detail.js
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: issues_found
---

# Phase 33: Code Review Report

**Reviewed:** 2026-07-29T10:52:34Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the node detail page template and its companion JavaScript. The page implements power state display, node info, provisioning tasks, live logs, and model recommendations. The power state badge renders once at load but is never refreshed -- a significant gap for the feature this phase delivers. A JS-context escaping issue on a user-controlled route parameter is the most security-relevant finding.

## Critical Issues

### CR-01: `node_id` injected into `<script>` block with HTML escaping instead of JS escaping

**File:** `inference_proxy/templates/node_detail.html:113`
**Issue:** `NODE_ID` is rendered via `"{{ node_id }}"`. Jinja2 auto-escaping applies HTML entity encoding (`&` -> `&amp;`, `"` -> `&quot;`), but the HTML parser does not decode entities inside `<script>` elements. This means: (a) any `node_id` containing `&`, `"`, `<`, `>`, or `'` will be garbled in the JS variable (the JS string gets literal `&amp;` instead of `&`), and (b) the route parameter is declared as `{node_id:path}` in `dashboard.py:37`, accepting arbitrary URL path content. While Jinja2 HTML auto-escaping happens to block XSS (by escaping `"` and `<`), it does so accidentally -- it is the wrong escaping strategy for JavaScript context. If auto-escaping were ever toggled off, or if a different template engine were used, this would be a direct XSS vector via crafted URL navigation to `/dashboard/nodes/PAYLOAD`.
**Fix:**
Use `tojson` filter, which produces a properly JS-escaped, quoted string:
```html
<script>
    const POLL_INTERVAL_MS = {{ poll_interval * 1000 }};
    const NODE_ID = {{ node_id | tojson }};
</script>
```
`tojson` escapes `\`, `"`, `<`, `>`, `&`, and `/` for safe embedding in `<script>` blocks and adds the surrounding quotes itself.

## Warnings

### WR-01: Power state is fetched once and never polled

**File:** `inference_proxy/static/js/node_detail.js:549-553`
**Issue:** `refreshPowerState()` is called once at `DOMContentLoaded` (line 551) but is not included in the `setInterval` on line 552. The `refreshDetail` function polls node info every `POLL_INTERVAL_MS`, but the power state badge goes stale immediately after page load. For a phase whose purpose is "power state display," a stale-on-load-only value is a significant omission.
**Fix:**
Add `refreshPowerState` to the poll interval, or call it from inside `refreshDetail`:
```javascript
document.addEventListener("DOMContentLoaded", function () {
  refreshDetail();
  refreshPowerState();
  setInterval(refreshDetail, POLL_INTERVAL_MS);
  setInterval(refreshPowerState, POLL_INTERVAL_MS);
});
```

### WR-02: Unguarded `.toFixed()` calls crash on null/undefined API fields

**File:** `inference_proxy/static/js/node_detail.js:450,485,489,490`
**Issue:** Several calls assume numeric values without null checks: `sys.gpu_vram_gb.toFixed(1)` (line 450), `m.score.toFixed(1)` (line 485), `m.estimated_tps.toFixed(1)` (line 489), `m.memory_required_gb.toFixed(1)` (line 490). If the recommendations API returns `null` or omits any of these fields, `toFixed` throws `TypeError: Cannot read properties of null`. The outer try/catch (line 510) would catch this but surfaces it as a generic "Network error" toast, hiding the real problem and preventing any recommendations from rendering.
**Fix:**
Guard with a fallback, e.g.:
```javascript
var vram = (sys.gpu_vram_gb != null ? sys.gpu_vram_gb : 0).toFixed(1) + " GB";
```
Or use a small helper:
```javascript
function fmt1(v) { return (v != null ? v : 0).toFixed(1); }
```

### WR-03: `ACTION_CONFIG` URL builders omit `encodeURIComponent` while other URL construction uses it

**File:** `inference_proxy/static/js/node_detail.js:26,44`
**Issue:** The teardown URL (`"/admin/nodes/" + id`, line 26) and force-teardown URL (`"/admin/nodes/" + id + "?force=true"`, line 44) inject `id` without `encodeURIComponent`. By contrast, the log stream URL (line 245), recommendations URL (line 403), and power state URL (line 538) all use `encodeURIComponent(NODE_ID)`. If `node_id` contains characters like `#`, `?`, or `/` (the route accepts `{node_id:path}`), the unencoded URLs will break or hit wrong endpoints.
**Fix:**
```javascript
teardown: {
  method: "DELETE", url: function (id) { return "/admin/nodes/" + encodeURIComponent(id); },
  ...
},
force_teardown: {
  method: "DELETE", url: function (id) { return "/admin/nodes/" + encodeURIComponent(id) + "?force=true"; },
  ...
},
```
Apply the same to `setup` and `retry` URL builders for consistency, even though they use a fixed path.

## Info

### IN-01: Silent empty catch in SSE message handler hides malformed log data

**File:** `inference_proxy/static/js/node_detail.js:274`
**Issue:** `catch (_) {}` in the SSE `message` event listener silently discards JSON parse errors. If the server sends malformed SSE data, there is zero indication in the UI or console. This makes debugging log stream issues unnecessarily difficult.
**Fix:**
Log to console at minimum:
```javascript
} catch (e) {
  console.warn("Failed to parse log entry:", e, ev.data);
}
```

---

_Reviewed: 2026-07-29T10:52:34Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
