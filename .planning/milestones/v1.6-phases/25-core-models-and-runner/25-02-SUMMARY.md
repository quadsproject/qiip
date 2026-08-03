---
phase: 25-core-models-and-runner
plan: 02
subsystem: llmfit-runner
tags: [llmfit, runner, ssh, tdd, pydantic]
dependency_graph:
  requires: [SystemInfo, ModelRecommendation, LLMFitResult, LLMFitTimeoutError, LLMFitParseError, SSHClient.run]
  provides: [LLMFitRunner, LLMFitRunner.recommend]
  affects: [inference_proxy/provisioning/ssh_client.py]
tech_stack:
  added: []
  patterns: [constructor-injection-dip, domain-error-translation]
key_files:
  created:
    - inference_proxy/llmfit/runner.py
    - tests/models/test_llmfit.py
    - tests/llmfit/__init__.py
    - tests/llmfit/test_runner.py
  modified:
    - tests/provisioning/test_ssh_client.py
    - inference_proxy/provisioning/ssh_client.py
decisions:
  - "D-05/D-06: hardcoded command and timeout as class constants, no LLMFitSettings"
  - "D-03: SSH errors bubble unchanged, not caught by runner"
metrics:
  duration: 283s
  completed: 2026-07-25
---

# Phase 25 Plan 02: LLMFitRunner and Test Suite Summary

LLMFitRunner.recommend(hostname) wires SSHClient.run() to Pydantic model parsing with typed error translation for timeout and parse failures, plus 21-test suite covering models, runner, and SSHClient.run().

## Task Results

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create LLMFitRunner (TDD RED) | cc81cd9 | tests/llmfit/__init__.py, tests/llmfit/test_runner.py |
| 1 | Create LLMFitRunner (TDD GREEN) | 31f176b | inference_proxy/llmfit/runner.py |
| 2 | Test suite for models, runner, SSHClient.run() | a35cd2e | tests/models/test_llmfit.py, tests/provisioning/test_ssh_client.py, inference_proxy/provisioning/ssh_client.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TimeoutError caught by except OSError in SSHClient.run()**
- **Found during:** Task 2
- **Issue:** In Python 3.11+, `asyncio.TimeoutError` is `TimeoutError` is `OSError`. The `except OSError` handler in `SSHClient.run()` caught timeout errors and wrapped them as `SSHConnectionError` instead of letting them bubble to the caller (violating D-02 contract).
- **Fix:** Added `except TimeoutError: raise` before `except OSError` in `SSHClient.run()`.
- **Files modified:** inference_proxy/provisioning/ssh_client.py
- **Commit:** a35cd2e

## TDD Gate Compliance

- RED gate: `cc81cd9` (test commit, runner module did not exist)
- GREEN gate: `31f176b` (feat commit, all 6 runner tests pass)
- No refactor needed (code already minimal)

## Verification

- `pytest tests/models/test_llmfit.py tests/llmfit/test_runner.py tests/provisioning/test_ssh_client.py -x -v`: 21 passed
- `pytest` (full suite): 524 passed, 0 failed
- LLMFitRunner.recommend() happy path: valid JSON -> typed LLMFitResult
- Timeout path: asyncio.TimeoutError -> LLMFitTimeoutError
- Parse errors: empty output, invalid JSON, wrong structure -> LLMFitParseError
- SSH error passthrough: SSHConnectionError bubbles unchanged (D-03)

## Self-Check: PASSED
