---
phase: 33
slug: power-state-display
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-29
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + pytest-asyncio |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After wave completion:** Run `uv run pytest tests/ -v`

---

## Validation Dimensions

| Dimension | Method | Threshold |
|-----------|--------|-----------|
| Badge renders correctly | Manual browser check | Power: On/Off/Unknown badge visible in node header |
| API integration | Existing endpoint tests | GET /admin/nodes/{hostname}/power returns 200 |
| No regressions | Full test suite | All existing tests pass |
| Dark/light theme | Manual browser check | Badge readable in both themes |

---

## Wave 0 Tasks

None required — test infrastructure already exists from prior phases.
