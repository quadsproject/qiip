---
phase: 28-model-selection
plan: 01
subsystem: provisioning
tags: [model-selection, provisioning, admin-api, shell-safety]
dependency_graph:
  requires: []
  provides: [SetupRequest.model, VLLM_MODEL-injection]
  affects: [inference_proxy/models/admin.py, inference_proxy/provisioning/provisioner.py, inference_proxy/api/admin.py]
tech_stack:
  added: []
  patterns: [shlex.quote-for-shell-safety, optional-kwarg-threading]
key_files:
  created: []
  modified:
    - inference_proxy/models/admin.py
    - inference_proxy/provisioning/provisioner.py
    - inference_proxy/api/admin.py
    - tests/models/test_admin.py
    - tests/provisioning/test_provisioner.py
    - tests/api/test_admin.py
decisions:
  - No format validator on model string -- vLLM validates at startup
metrics:
  duration: 280s
  completed: 2026-07-26
---

# Phase 28 Plan 01: Model Selection Provisioning Summary

Optional model field threaded from admin API through provisioner to SSH command via VLLM_MODEL env var with shlex.quote() shell injection protection.

## What Was Done

### Task 1: Add model field and thread through provisioning (361ff57)
- Added `model: str | None = Field(default=None, max_length=256)` to SetupRequest
- Added `import shlex` and `model` kwarg to `provision()` and `_run_start_vllm()`
- Conditionally prepends `VLLM_MODEL={shlex.quote(model)}` to the SSH command
- Admin endpoint passes `model=body.model` to provisioner

### Task 2: Tests for model selection flow (0b84e35)
- TestSetupRequest: default None, explicit value, max_length rejection, frozen immutability
- TestModelExtraction: VLLM_MODEL prepend when model set, omission when None, shlex quoting of shell-unsafe chars
- TestSetupModelPassthrough: model param reaches provisioner.provision() via fire_background coroutine

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- Full test suite: 542 passed, 0 failures
- SetupRequest.model defaults to None, accepts strings, rejects >256 chars
- VLLM_MODEL prepended only when model is set
- shlex.quote() used on model string
- model=body.model passed from admin endpoint to provisioner

## Self-Check: PASSED
