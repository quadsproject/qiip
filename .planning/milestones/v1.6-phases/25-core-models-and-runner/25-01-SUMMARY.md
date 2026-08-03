---
phase: 25-core-models-and-runner
plan: 01
subsystem: llmfit-models
tags: [pydantic, error-hierarchy, ssh, data-contracts]
dependency_graph:
  requires: []
  provides: [SystemInfo, ModelRecommendation, LLMFitResult, LLMFitError, LLMFitTimeoutError, LLMFitParseError, SSHClient.run]
  affects: [inference_proxy/provisioning/ssh_client.py]
tech_stack:
  added: []
  patterns: [frozen-pydantic-extra-ignore, domain-error-hierarchy, asyncio-wait-for-timeout]
key_files:
  created:
    - inference_proxy/models/llmfit.py
    - inference_proxy/llmfit/__init__.py
    - inference_proxy/llmfit/errors.py
  modified:
    - inference_proxy/provisioning/ssh_client.py
decisions:
  - "Followed D-02: asyncio.wait_for wraps conn.run() for timeout rather than conn.run(timeout=N)"
  - "Followed D-04: LLMFitParseError stores raw_output for debugging"
metrics:
  duration: 98s
  completed: 2026-07-25
---

# Phase 25 Plan 01: Core Models and Error Hierarchy Summary

Pydantic models for llmfit JSON output (SystemInfo, ModelRecommendation, LLMFitResult) with frozen+extra=ignore config, domain error hierarchy (LLMFitError base with timeout and parse subclasses), and SSHClient.run() for capture-all command execution with asyncio.wait_for timeout.

## Task Results

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create Pydantic models and error hierarchy | a903590 | inference_proxy/models/llmfit.py, inference_proxy/llmfit/__init__.py, inference_proxy/llmfit/errors.py |
| 2 | Add run() method to SSHClient | df7ef49 | inference_proxy/provisioning/ssh_client.py |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

All automated checks passed:
- Pydantic models parse llmfit JSON fixture with extra fields silently ignored
- Error hierarchy: LLMFitTimeoutError and LLMFitParseError subclass LLMFitError
- LLMFitParseError.raw_output stores raw stdout for debugging
- SSHClient.run() signature: (self, host, command, timeout=60.0) -> tuple[str, str, int]
- No new dependencies added

## Self-Check: PASSED
