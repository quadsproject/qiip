"""Failure classification, bounded retries, and setup-plan identity.

A provisioning retry must repeat only work that was not verified complete and
must never loop automatically on invalid configuration or unsupported
hardware. This module keeps those decisions out of the SSH layer and the
orchestrator so they can be unit-tested and surfaced to operators.

The orchestrator still owns reconciliation and remote cleanup; this module only
*classifies* a failure, computes a bounded backoff, and defines the stable
identity used to decide whether a checkpoint is still valid.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from inference_proxy.provisioning.log_store import (
    SETUP_STAGE_ORDER,
    setup_progress,
)

# Keep this module free of SSH dependencies so it stays importable anywhere.
# Transport errors are recognised by class name and by ``TimeoutError``
# inheritance instead of importing ssh_client.
_TRANSPORT_ERROR_NAMES = frozenset(
    {"SSHConnectionError", "SSHCommandTimeoutError", "ConnectionError"}
)


class FailureClass(StrEnum):
    """Category of a provisioning failure used to decide automatic recovery."""

    TRANSIENT = "transient"
    CONFIGURATION = "configuration"
    UNSUPPORTED_HARDWARE = "unsupported_hardware"
    MAINTENANCE = "maintenance"
    TERMINAL = "terminal"


# Only TRANSIENT failures may schedule an automatic retry. Everything else
# stops and reports the operator action required before a manual retry.
NEXT_ACTIONS: dict[str, str] = {
    FailureClass.TRANSIENT: "Retry setup; verified stages resume automatically",
    FailureClass.CONFIGURATION: "Correct the setup configuration before retrying",
    FailureClass.UNSUPPORTED_HARDWARE: (
        "Select supported hardware or remove the host from managed setup"
    ),
    FailureClass.MAINTENANCE: "Complete the required maintenance, then retry setup",
    FailureClass.TERMINAL: (
        "Inspect the retained logs and diagnostics before retrying setup"
    ),
}

_MAINTENANCE_PATTERNS = (
    r"reboot_required",
    r"maintenance_required",
    r"driver maintenance is waiting",
    r"then retry setup",
)

_UNSUPPORTED_PATTERNS = (
    r"unsupported_hardware",
    r"unsupported (?:gpu|hardware|architecture|arch)",
    r"no nvidia gpu pci device",
    r"compute capability .*not supported",
    r"requires compute capability",
    r"no cuda-capable device",
)

# Network, mirror, and transport symptoms. Matched before configuration so a
# package repository outage does not look like a broken setup selection.
_TRANSIENT_PATTERNS = (
    r"could not resolve host",
    r"temporary failure in name resolution",
    r"name or service not known",
    r"network is unreachable",
    r"no route to host",
    r"connection (?:refused|reset|timed out|closed|aborted)",
    r"closed by remote host",
    r"ssh_exchange_identification",
    r"kex_exchange_identification",
    r"broken pipe",
    r"connection attempt failed",
    r"failed to connect",
    r"unable to access",
    r"operation timed out",
    r"\btimeout\b",
    r"\btimed out\b",
    r"failed to download",
    r"download failed",
    r"cannot download",
    r"error: failed to download metadata",
    r"wget: .*(?:unable|failed|error|timed out)",
    r"curl: \(\d+\)",
    r"resource temporarily unavailable",
    r"temporarily unavailable",
    r"mirror .* (?:failed|unavailable)",
    r"dnf.*(?:timeout|connection|mirror|repodata|metadata)",
    r"the requested url returned error: 5\d\d",
    r"tls handshake timeout",
    r"i/o error",
    r"input/output error",
    r"git (?:clone|fetch).*(?:failed|timed out|unable)",
    r"named pipe.*unavailable",
)

_CONFIGURATION_PATTERNS = (
    r"invalid configuration",
    r"configuration (?:error|invalid)",
    r"missing required",
    r"is required for node provisioning",
    r"unknown model",
    r"artifact .*not found",
    r"not found in (?:the )?artifact",
    r"401 unauthorized",
    r"403 forbidden",
    r"invalid (?:token|credential|sha256|configuration|selection)",
    r"sha-?256 verification failed",
    r"refusing to",
    r"not permitted",
    r"does not support",
    r"unsupported option",
)

_UNSUPPORTED_RE = re.compile("|".join(_UNSUPPORTED_PATTERNS), re.IGNORECASE)
_MAINTENANCE_RE = re.compile("|".join(_MAINTENANCE_PATTERNS), re.IGNORECASE)
_TRANSIENT_RE = re.compile("|".join(_TRANSIENT_PATTERNS), re.IGNORECASE)
_CONFIGURATION_RE = re.compile("|".join(_CONFIGURATION_PATTERNS), re.IGNORECASE)


def classify_failure(error: BaseException, *, output: str = "") -> FailureClass:
    """Classify *error* for retry decisions.

    ``output`` is optional captured remote output (for example the failing
    setup tail) that may carry a symptom the exception text omits.

    Explicit resume/maintenance exceptions win first, then the exception type,
    then the message. Configuration is checked after transient so a repository
    outage reported by a remote command is retried instead of treated as a
    selection error.
    """
    # ``SetupResumeRequiredError`` (defined in the orchestrator) exposes the
    # boundary as ``resume_state``; duck-typing avoids an import cycle.
    if getattr(error, "resume_state", None) is not None:
        return FailureClass.MAINTENANCE

    text = f"{error}\n{output}"
    if _MAINTENANCE_RE.search(text):
        return FailureClass.MAINTENANCE
    if _UNSUPPORTED_RE.search(text):
        return FailureClass.UNSUPPORTED_HARDWARE
    # Transport exceptions are always transient regardless of message.
    cause: BaseException | None = error
    while cause is not None:
        if (
            isinstance(cause, TimeoutError)
            or type(cause).__name__ in _TRANSPORT_ERROR_NAMES
        ):
            return FailureClass.TRANSIENT
        cause = cause.__cause__
    if _TRANSIENT_RE.search(text):
        return FailureClass.TRANSIENT
    if _CONFIGURATION_RE.search(text):
        return FailureClass.CONFIGURATION
    return FailureClass.TERMINAL


def is_retryable(error: BaseException, *, output: str = "") -> bool:
    return classify_failure(error, output=output) == FailureClass.TRANSIENT


def next_action(error: BaseException | None, *, output: str = "") -> str:
    """Return the operator action required before the next attempt."""
    if error is None:
        return NEXT_ACTIONS[FailureClass.TERMINAL]
    return NEXT_ACTIONS[classify_failure(error, output=output)]


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff for recognized transient failures."""

    max_attempts: int = 3
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 120.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be greater than zero")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")

    @property
    def max_retries(self) -> int:
        """Number of retries after the initial attempt."""
        return self.max_attempts - 1

    def delay(
        self, retry_number: int, *, rand: Callable[[], float] = random.random
    ) -> float:
        """Return the bounded delay before 1-based *retry_number*.

        The deterministic ceiling is ``base * 2 ** (retry_number - 1)`` capped
        at ``max_delay_seconds``. ``jitter`` removes the synchronized-retry
        herd without ever exceeding the cap.
        """
        if retry_number < 1:
            raise ValueError("retry_number is 1-based")
        ceiling = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2.0 ** (retry_number - 1)),
        )
        if self.jitter == 0:
            return ceiling
        spread = ceiling * self.jitter
        return max(
            0.0, min(self.max_delay_seconds, ceiling - spread + rand() * 2 * spread)
        )


def setup_plan_id(inputs: Mapping[str, Any]) -> str:
    """Return a stable identity for one compatible setup plan.

    Every input that can change what a stage installs, builds, or mounts must
    be included. The identity is order-independent and prefixed with the
    algorithm so a future schema can coexist with retained checkpoints.
    """
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "FailureClass",
    "NEXT_ACTIONS",
    "RetryPolicy",
    "SETUP_STAGE_ORDER",
    "classify_failure",
    "is_retryable",
    "next_action",
    "setup_plan_id",
    "setup_progress",
]
