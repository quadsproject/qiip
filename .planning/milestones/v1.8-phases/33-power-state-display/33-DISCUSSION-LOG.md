# Phase 33: Power State Display - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-29
**Phase:** 33-Power State Display
**Areas discussed:** Badge placement, Refresh strategy, Error/unknown display

---

## Badge Placement

| Option | Description | Selected |
|--------|-------------|----------|
| Next to node title | Inline with the h1 — e.g. 'mgmt-gpu01.example.com [On]'. Compact, immediately visible. | |
| Below state text | New line under the existing node state (p#node-state). Separates service state from BMC power state. | ✓ |
| Replace state text | Replace the current node-state paragraph with a richer status line showing both service state and power state. | |

**User's choice:** Below state text
**Notes:** None

### Badge Label

| Option | Description | Selected |
|--------|-------------|----------|
| Power: On / Off / Unknown | Prefixed with 'Power:' to distinguish from service state above. Clear at a glance. | ✓ |
| BMC: On / Off / Unknown | Uses the technical term. Operators know what BMC means. | |
| Just On / Off / Unknown | No prefix — minimal. Context is clear from placement below service state. | |

**User's choice:** Power: On / Off / Unknown
**Notes:** None

### Badge Color Mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing classes | On = badge-complete (green), Off = badge-failed (red), Unknown = badge-in-progress (amber). No new CSS needed. | ✓ |
| New power-specific classes | badge-power-on (green), badge-power-off (gray/neutral), badge-power-unknown (amber). Off as gray since it's not an error. | |

**User's choice:** Reuse existing classes
**Notes:** None

---

## Refresh Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Poll with page refresh | Add to existing refreshDetail() loop. Stays current if BMC state changes externally. Adds one API call per poll cycle. | |
| Fetch once on load | Single fetch on DOMContentLoaded. Phase 34 calls the refresh function after power actions. Simpler, avoids repeated BMC calls. | ✓ |
| You decide | Let Claude pick the approach that best fits existing patterns. | |

**User's choice:** Fetch once on load
**Notes:** None

---

## Error/Unknown Display

| Option | Description | Selected |
|--------|-------------|----------|
| Unknown badge (amber) | Show 'Power: Unknown' with badge-in-progress styling. Treats all failures as unknown state. Clean, no error noise. | ✓ |
| Hide badge entirely | Don't show power state if fetch fails. Absence = not available. Avoids confusing 'Unknown' on nodes without BMC. | |
| Error badge with retry | Show a distinct error state with a small retry link. More informative but adds UI complexity. | |

**User's choice:** Unknown badge (amber)
**Notes:** None

---

## Claude's Discretion

None — user made all decisions.

## Deferred Ideas

None — discussion stayed within phase scope.
