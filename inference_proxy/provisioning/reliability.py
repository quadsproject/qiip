"""Fleet measurements from retained evidence; no node I/O or inferred successes."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import median
from typing import Any, Literal
from urllib.parse import quote

from inference_proxy.provisioning.log_store import AttemptLogStore

GroupBy = Literal[
    "signature",
    "stage",
    "engine",
    "runtime_version",
    "gpu_family",
    "os",
    "kernel",
    "bundle_version",
    "all",
]
DIMENSIONS = (
    "stage",
    "signature",
    "engine",
    "runtime_version",
    "gpu_family",
    "os",
    "kernel",
    "bundle_version",
)
TERMINAL = {"success", "failed", "cancelled", "unsupported"}


def _date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo else None
    except ValueError:
        return None


def _environment(attempt: dict[str, Any]) -> dict[str, str]:
    evidence = attempt.get("environment_evidence", {})
    release = dict(
        re.findall(r'^([A-Z_]+)=["\']?([^"\'\n]*)', evidence.get("os", ""), re.M)
    )
    os_version = " ".join(filter(None, (release.get("ID"), release.get("VERSION_ID"))))
    kernel = evidence.get("kernel", "").split()
    gpu_rows = list(csv.reader(io.StringIO(evidence.get("gpu", ""))))
    gpu_models = sorted(
        {row[2].strip() for row in gpu_rows[1:] if len(row) >= 3 and row[2].strip()}
    )
    profile = attempt.get("runtime_profile", "")
    family = re.sub(r"^(vllm|llamacpp)-", "", profile) if profile else ""
    runtime = evidence.get("runtime", "")
    pattern = (
        r"^vllm=([^\s]+)"
        if attempt.get("engine") == "vllm"
        else r"^(?:version:|build:)[ \t]*([^\r\n]+)"
    )
    match = re.search(pattern, runtime, re.M)
    return {
        "runtime_version": match.group(1)[:200] if match else "unknown",
        "gpu_family": family
        or (" / ".join(gpu_models)[:300] if gpu_models else "unknown"),
        "os": os_version[:200] or "unknown",
        # uname -a includes the hostname: group by kernel release, never host.
        "kernel": kernel[2][:200]
        if len(kernel) >= 3 and kernel[0] == "Linux"
        else "unknown",
    }


def _signature(attempt: dict[str, Any], original: str) -> tuple[str, bool]:
    """Versioned symptoms, not diagnoses. Unsupported requires an explicit marker."""
    unsupported = "[REJECT:unsupported_hardware:" in original or any(
        issue.startswith(("unsupported_hardware:", "Node: unsupported_hardware:"))
        for issue in attempt.get("issues", [])
    )
    if unsupported:
        return "v1:unsupported_hardware", True
    for signature, pattern in (
        ("cuda_out_of_memory", r"cuda out of memory|cuda error: out of memory"),
        ("disk_full", r"no space left on device"),
        (
            "kernel_headers",
            r"kernel.{0,40}(headers|devel).{0,80}(missing|not found|match)|no matching kernel",
        ),
        ("permission_denied", r"permission denied"),
        ("connection_refused", r"connection refused"),
        ("timeout", r"timed out|timeout|deadline exceeded"),
    ):
        if re.search(pattern, original, re.I):
            return "v1:" + signature, False
    if not original:
        return "v1:unknown", False
    normalized = original.lower().replace(
        str(attempt.get("hostname", "")).lower(), "<host>"
    )
    normalized = re.sub(
        r"\b[0-9a-f]{32,64}\b|\b(?:\d{1,3}\.){3}\d{1,3}\b", "<id>", normalized
    )
    normalized = re.sub(r"\b\d+(?:\.\d+)*\b", "<n>", normalized)
    normalized = " ".join(normalized.split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return "v1:unclassified:" + digest, False


def _attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    failure = attempt.get("failure") or {}
    original = failure.get("original_error") or attempt.get("failure_summary") or ""
    signature, unsupported = _signature(attempt, original)
    status, stage = attempt.get("status"), attempt.get("stage", "unknown")
    # A completed remote command or stopped log stream is not usable inference.
    if stage == "cancelled" or status == "cancelled":
        outcome = "cancelled"
    elif status == "complete" and stage == "complete":
        outcome = "success"
    elif status == "failed":
        outcome = "unsupported" if unsupported else "failed"
    elif status == "running":
        outcome = "running"
    else:
        outcome = "unknown"
    hostname, attempt_id = attempt["hostname"], attempt["attempt_id"]
    base = f"/admin/provisioning/{quote(hostname, safe='')}/attempts/{quote(attempt_id, safe='')}"
    environment = _environment(attempt)
    return {
        "attempt_id": attempt_id,
        "hostname": hostname,
        "engine": attempt.get("engine") or "unknown",
        "model": attempt.get("model"),
        "bundle_version": attempt.get("bundle_version") or "unknown",
        "started_at": attempt.get("started_at"),
        "finished_at": attempt.get("finished_at"),
        "ready_at": attempt.get("ready_at"),
        "outcome": outcome,
        "stage": failure.get("failed_stage") or stage,
        "signature": signature,
        "original_error": original,
        "error_type": failure.get("error_type", "unknown"),
        "series_id": attempt.get("series_id", attempt_id),
        "attempt_number": attempt.get("attempt_number"),
        "series_started_at": attempt.get("series_started_at"),
        "series_origin_known": attempt.get("series_origin_known", False),
        "incomplete": bool(
            attempt.get("incomplete") or outcome in {"unknown", "running"}
        ),
        "unknown_dimensions": [
            key for key, value in environment.items() if value == "unknown"
        ],
        "logs_url": base + "/logs",
        "bundle_url": base + "/bundle",
        "node_url": f"/dashboard/nodes/{quote(hostname, safe='')}#attempt-history-panel",
        **environment,
    }


def _rate(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
    return dict(
        numerator=numerator,
        denominator=denominator,
        percent=round(100 * numerator / denominator, 2) if denominator else None,
        definition=definition,
    )


def build_report(
    store: AttemptLogStore,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: GroupBy = "signature",
    hostnames: list[str] | None = None,
) -> dict[str, Any]:
    snapshot = store.reliability_snapshot()
    all_attempts = [
        _attempt(a) for a in snapshot["attempts"] if a.get("operation") == "provision"
    ]
    attempts = []
    invalid_dates = 0
    for attempt in all_attempts:
        if hostnames and attempt["hostname"] not in hostnames:
            continue
        started = _date(attempt["started_at"])
        if started is None:
            invalid_dates += 1
            if since or until:
                continue
        elif (since and started < since) or (until and started >= until):
            continue
        attempts.append(attempt)
    counts = Counter(a["outcome"] for a in attempts)
    terminal = sum(counts[name] for name in TERMINAL)
    first = [
        a
        for a in attempts
        if a["series_origin_known"]
        and a["attempt_number"] == 1
        and a["outcome"] in TERMINAL
    ]
    origins = {
        a["series_id"]: a
        for a in all_attempts
        if a["series_origin_known"] and a["attempt_number"] == 1
    }
    retries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        origin = origins.get(attempt["series_id"])
        if (
            origin
            and origin["outcome"] == "failed"
            and (attempt["attempt_number"] or 0) > 1
        ):
            retries[attempt["series_id"]].append(attempt)
    resolved_retries = [
        rows for rows in retries.values() if any(a["outcome"] in TERMINAL for a in rows)
    ]
    durations = []
    for attempt in attempts:
        if attempt["outcome"] != "success" or not attempt["series_origin_known"]:
            continue
        start, ready = _date(attempt["series_started_at"]), _date(attempt["ready_at"])
        if start and ready and ready >= start:
            durations.append((ready - start).total_seconds())
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    dimensions = DIMENSIONS if group_by == "all" else (group_by,)
    for attempt in attempts:
        if attempt["outcome"] in {"failed", "unsupported"}:
            grouped[tuple(attempt[key] for key in dimensions)].append(attempt)
    groups = [
        dict(
            dimensions=dict(zip(dimensions, key, strict=True)),
            count=len(rows),
            incomplete_count=sum(a["incomplete"] for a in rows),
            attempt_ids=[a["attempt_id"] for a in reversed(rows)],
        )
        for key, rows in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]
    warnings = [
        "Retained managed provisioning attempts only; API validation, power-on and reconcile failures before attempt creation are outside this denominator.",
        "Readiness means health checks passed and the node was registered; it does not measure time to a generated token.",
        "Environment versions are diagnostic snapshots at collection time, which may be later than the failure. Unknown values are never filled from current configuration.",
    ]
    if snapshot["evicted_attempts"]:
        warnings.append(
            "Retention evicted attempt manifests. This report is not a complete fleet history; export cohorts before they expire."
        )
    return {
        "schema_version": 1,
        "signature_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "window": dict(
            since=since.isoformat() if since else None,
            until=until.isoformat() if until else None,
            hostnames=hostnames or [],
        ),
        "group_by": group_by,
        "attempt_count": len(attempts),
        "outcomes": {
            name: counts[name] for name in (*sorted(TERMINAL), "running", "unknown")
        },
        "metrics": {
            "first_attempt_success": _rate(
                sum(a["outcome"] == "success" for a in first),
                len(first),
                "Successful first attempts / known first attempts with a terminal outcome, including cancellation and unsupported hardware.",
            ),
            "retry_recovery": _rate(
                sum(
                    any(a["outcome"] == "success" for a in rows)
                    for rows in resolved_retries
                ),
                len(resolved_retries),
                "Series recovered by a retry / initially failed series with at least one terminal retry starting in the window. The first failure must be retained; running-only retries and unattempted retries are excluded.",
            ),
            "cancellation": _rate(
                counts["cancelled"],
                terminal,
                "Explicitly cancelled attempts / all terminal attempts in the window.",
            ),
            "unsupported": _rate(
                counts["unsupported"],
                terminal,
                "Explicit unsupported-hardware rejections / all terminal attempts in the window.",
            ),
            "time_to_usable_inference": dict(
                samples=len(durations),
                eligible_successes=counts["success"],
                median_seconds=median(durations) if durations else None,
                definition="Seconds from series start to registered readiness, including retry delays; only successes with a known origin and recorded readiness timestamp.",
            ),
        },
        "evidence": dict(
            incomplete_attempts=sum(a["incomplete"] for a in attempts),
            unknown_environment_attempts=sum(
                bool(a["unknown_dimensions"]) for a in attempts
            ),
            unknown_origin_attempts=sum(not a["series_origin_known"] for a in attempts),
            invalid_start_times=invalid_dates,
            evicted_attempts=snapshot["evicted_attempts"],
            excluded_operations=len(snapshot["attempts"]) - len(all_attempts),
        ),
        "warnings": warnings,
        "groups": groups,
        "attempts": list(reversed(attempts)),
    }
