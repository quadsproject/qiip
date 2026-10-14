"""Unit tests for failure classification, retry policy, and setup identity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from inference_proxy.provisioning.recovery import (
    NEXT_ACTIONS,
    SETUP_STAGE_ORDER,
    FailureClass,
    RetryPolicy,
    classify_failure,
    is_retryable,
    next_action,
    setup_plan_id,
    setup_progress,
)


class _ResumeError(Exception):
    """Stand-in for the orchestrator's SetupResumeRequiredError."""

    resume_state = SimpleNamespace(value="maintenance_required")


class SSHConnectionError(Exception):
    """Transport error recognised by class name, not by an import."""


class TestFailureClassification:
    def test_resume_state_attribute_is_maintenance(self) -> None:
        assert classify_failure(_ResumeError("boom")) == FailureClass.MAINTENANCE

    @pytest.mark.parametrize(
        "message",
        [
            "[RESUME:reboot_required:install the driver]",
            "maintenance_required",
            "driver maintenance is waiting",
        ],
    )
    def test_maintenance_markers(self, message: str) -> None:
        assert classify_failure(RuntimeError(message)) == FailureClass.MAINTENANCE

    @pytest.mark.parametrize(
        "message",
        [
            "[REJECT:unsupported_hardware:no nvidia gpu pci device]",
            "unsupported architecture",
            "compute capability 9.0 not supported",
            "requires compute capability 8.0",
            "no CUDA-capable device detected",
        ],
    )
    def test_unsupported_hardware_markers(self, message: str) -> None:
        assert (
            classify_failure(RuntimeError(message)) == FailureClass.UNSUPPORTED_HARDWARE
        )

    def test_timeout_error_is_transient(self) -> None:
        assert (
            classify_failure(TimeoutError("operation timed out"))
            == FailureClass.TRANSIENT
        )

    def test_transport_exception_name_is_transient(self) -> None:
        assert classify_failure(SSHConnectionError("refused")) == FailureClass.TRANSIENT

    def test_transport_wrapped_cause_is_transient(self) -> None:
        try:
            try:
                raise TimeoutError("socket timeout")
            except TimeoutError as inner:
                raise RuntimeError("outer wrapper") from inner
        except RuntimeError as exc:
            assert classify_failure(exc) == FailureClass.TRANSIENT

    @pytest.mark.parametrize(
        "message",
        [
            "Could not resolve host: mirror.example.com",
            "error: failed to download metadata for repo 'baseos'",
            "curl: (28) Connection timed out after 30000 milliseconds",
            "wget: unable to resolve host address",
            "fatal: unable to access 'https://example/repo': Failed to connect",
            "TLS handshake timeout",
            "The requested URL returned error: 503",
        ],
    )
    def test_transient_download_and_network_markers(self, message: str) -> None:
        assert classify_failure(RuntimeError(message)) == FailureClass.TRANSIENT

    @pytest.mark.parametrize(
        "message",
        [
            "invalid configuration: missing required NFS export",
            "Unknown model 'org/missing'",
            "sha256 verification failed",
            "401 Unauthorized",
        ],
    )
    def test_configuration_markers(self, message: str) -> None:
        assert classify_failure(RuntimeError(message)) == FailureClass.CONFIGURATION

    def test_transient_wins_over_configuration(self) -> None:
        # A repository outage mentioning a config word must still retry.
        message = "failed to download metadata for repo; missing required mirror"
        assert classify_failure(RuntimeError(message)) == FailureClass.TRANSIENT

    @pytest.mark.parametrize(
        "message",
        [
            "401 Unauthorized",
            "403 Forbidden",
            "error: 404 Not Found for https://example/artifact",
            "SSL certificate problem: self-signed certificate",
            "host key verification failed",
            "curl: (60) SSL certificate problem: unable to get local issuer certificate",
        ],
    )
    def test_permanent_network_failures_are_configuration(self, message: str) -> None:
        assert classify_failure(RuntimeError(message)) == FailureClass.CONFIGURATION

    def test_ssh_auth_failure_is_configuration_not_transient(self) -> None:
        # SSHConnectionError is normally transient, but a public-key rejection
        # cannot be fixed by retrying.
        exc = SSHConnectionError("host1", "Permission denied (publickey).")
        assert classify_failure(exc) == FailureClass.CONFIGURATION

    @pytest.mark.parametrize("code", ["6", "7", "28", "35", "52", "56"])
    def test_transient_curl_codes_retry(self, code: str) -> None:
        assert (
            classify_failure(RuntimeError(f"curl: ({code}) boom"))
            == FailureClass.TRANSIENT
        )

    @pytest.mark.parametrize("code", ["1", "2", "3", "5", "60", "77", "90"])
    def test_permanent_curl_codes_do_not_retry(self, code: str) -> None:
        assert (
            classify_failure(RuntimeError(f"curl: ({code}) boom"))
            == FailureClass.CONFIGURATION
        )

    def test_setup_deadline_exceeded_is_terminal(self) -> None:
        # ``SetupDeadlineError`` (exercised end-to-end in test_resume_retry)
        # derives from ProvisioningError and carries no transient marker, so it
        # must never be auto-retried.
        class SetupDeadlineError(Exception):
            pass

        assert (
            classify_failure(SetupDeadlineError("setup exceeded its 60s deadline"))
            == FailureClass.TERMINAL
        )

    def test_unknown_failure_is_terminal(self) -> None:
        assert (
            classify_failure(RuntimeError("segmentation fault"))
            == FailureClass.TERMINAL
        )

    def test_output_tail_can_supply_the_symptom(self) -> None:
        error = RuntimeError("setup.sh exited 1")
        assert (
            classify_failure(error, output="Could not resolve host: mirror")
            == FailureClass.TRANSIENT
        )

    def test_is_retryable_only_for_transient(self) -> None:
        assert is_retryable(TimeoutError("x"))
        assert not is_retryable(RuntimeError("invalid configuration"))
        assert not is_retryable(RuntimeError("unsupported hardware"))
        assert not is_retryable(_ResumeError("reboot"))


class TestNextAction:
    @pytest.mark.parametrize("failure_class", list(FailureClass))
    def test_every_class_has_an_action(self, failure_class: FailureClass) -> None:
        assert NEXT_ACTIONS[failure_class]

    def test_returns_class_specific_action(self) -> None:
        assert next_action(TimeoutError("x")) == NEXT_ACTIONS[FailureClass.TRANSIENT]
        assert (
            next_action(RuntimeError("invalid configuration"))
            == NEXT_ACTIONS[FailureClass.CONFIGURATION]
        )
        assert (
            next_action(RuntimeError("no nvidia gpu pci device"))
            == NEXT_ACTIONS[FailureClass.UNSUPPORTED_HARDWARE]
        )

    def test_none_error_is_terminal_action(self) -> None:
        assert next_action(None) == NEXT_ACTIONS[FailureClass.TERMINAL]


class TestRetryPolicy:
    def test_max_retries_excludes_initial_attempt(self) -> None:
        assert RetryPolicy(max_attempts=3).max_retries == 2
        assert RetryPolicy(max_attempts=1).max_retries == 0

    def test_delay_is_exponential_without_jitter(self) -> None:
        policy = RetryPolicy(base_delay_seconds=5, max_delay_seconds=120, jitter=0)
        assert policy.delay(1) == 5
        assert policy.delay(2) == 10
        assert policy.delay(3) == 20

    def test_delay_is_capped(self) -> None:
        policy = RetryPolicy(base_delay_seconds=5, max_delay_seconds=12, jitter=0)
        assert policy.delay(1) == 5
        assert policy.delay(2) == 10
        assert policy.delay(3) == 12
        assert policy.delay(10) == 12

    def test_jitter_stays_within_bounds_and_never_exceeds_cap(self) -> None:
        policy = RetryPolicy(base_delay_seconds=5, max_delay_seconds=12, jitter=0.5)
        for retry in range(1, 8):
            low = policy.delay(retry, rand=lambda: 0.0)
            high = policy.delay(retry, rand=lambda: 1.0)
            assert 0.0 <= low <= high <= 12

    def test_retry_number_is_one_based(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy().delay(0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_attempts": 0},
            {"base_delay_seconds": 0},
            {"max_delay_seconds": 1, "base_delay_seconds": 5},
            {"jitter": 1.0},
            {"jitter": -0.1},
        ],
    )
    def test_invalid_settings_are_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)

    def test_policy_is_immutable(self) -> None:
        policy = RetryPolicy()
        with pytest.raises(FrozenInstanceError):
            policy.max_attempts = 9  # type: ignore[misc]


class TestSetupPlanId:
    def test_identity_is_stable_and_order_independent(self) -> None:
        first = setup_plan_id({"engine": "vllm", "model": "org/model"})
        second = setup_plan_id({"model": "org/model", "engine": "vllm"})
        assert first == second
        assert first.startswith("sha256:")

    def test_identity_changes_with_any_input(self) -> None:
        base = setup_plan_id({"engine": "vllm", "model": "org/model"})
        assert base != setup_plan_id({"engine": "vllm", "model": "org/other"})
        assert base != setup_plan_id({"engine": "llamacpp", "model": "org/model"})

    def test_non_json_values_are_stringified(self) -> None:
        assert setup_plan_id({"path": object()}).startswith("sha256:")


class TestSetupProgress:
    def test_counts_completed_and_skipped_only(self) -> None:
        completed = {SETUP_STAGE_ORDER[0]: "completed", SETUP_STAGE_ORDER[1]: "skipped"}
        progress = setup_progress(completed)
        assert progress["completed_count"] == 2
        assert progress["total_stages"] == len(SETUP_STAGE_ORDER)
        assert progress["completed_stages"] == list(SETUP_STAGE_ORDER[:2])

    def test_unknown_stage_never_inflates_progress(self) -> None:
        progress = setup_progress({"vendor_marker": "completed"})
        assert progress["completed_count"] == 0
        assert progress["percent"] == 0.0

    def test_failed_stages_are_reported(self) -> None:
        progress = setup_progress({SETUP_STAGE_ORDER[2]: "failed"})
        assert progress["failed_stages"] == [SETUP_STAGE_ORDER[2]]
        assert progress["completed_count"] == 0

    def test_full_completion_is_one_hundred_percent(self) -> None:
        progress = setup_progress({stage: "completed" for stage in SETUP_STAGE_ORDER})
        assert progress["percent"] == 100.0
