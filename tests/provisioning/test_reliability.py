"""Report measurements against durable fixtures and the shipped launch boundary."""

from __future__ import annotations

import asyncio
import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.provisioning.reliability import DIMENSIONS, build_report
from inference_proxy.provisioning.ssh_client import RemoteCommandError
from inference_proxy.provisioning.state import ProvisioningStep
from tests.provisioning.test_attempt_logs import LocalNodeSSH
from tests.provisioning.test_attempt_logs import harness as harness
from tests.provisioning.test_diagnostics import _prepare

START = datetime(2026, 9, 21, tzinfo=UTC)


@pytest.fixture
def fleet(tmp_path: Path) -> AttemptLogStore:
    store = AttemptLogStore(tmp_path / "fleet.sqlite3")
    records = json.loads(
        (Path(__file__).parent / "fixtures/reliability.json").read_text()
    )
    for record in records:
        attempt = store.create(
            record["host"],
            engine="vllm",
            model="org/model",
            bundle_version=record.get("bundle", "before"),
            operation=record.get("operation", "provision"),
            attempt_id=record["id"],
        )
        started = (START + timedelta(minutes=record["minute"])).isoformat()
        fields: dict[str, Any] = dict(
            started_at=started, status=record["status"], stage=record["stage"]
        )
        if store.get(attempt).get("attempt_number") == 1:
            fields["series_started_at"] = started
        if "ready_minute" in record:
            fields["ready_at"] = (
                START + timedelta(minutes=record["ready_minute"])
            ).isoformat()
        if "error" in record:
            fields["failure_summary"] = record["error"]
        if record.get("legacy"):
            fields.update(series_origin_known=False, attempt_number=None)
        store.update(attempt, **fields)
    return store


def test_fixture_denominators_retries_versions_and_readiness(
    fleet: AttemptLogStore,
) -> None:
    report = build_report(AttemptLogStore(fleet.path))
    assert report["attempt_count"] == 10
    assert report["outcomes"] == dict(
        success=3, failed=3, cancelled=1, unsupported=1, running=1, unknown=1
    )
    metrics = report["metrics"]
    assert metrics["first_attempt_success"]["numerator"] == 1
    assert metrics["first_attempt_success"]["denominator"] == 5
    assert metrics["retry_recovery"]["percent"] == 50
    assert metrics["cancellation"]["denominator"] == 8
    assert metrics["unsupported"]["numerator"] == 1
    assert metrics["time_to_usable_inference"]["samples"] == 2
    assert metrics["time_to_usable_inference"]["median_seconds"] == 510
    assert report["evidence"]["unknown_origin_attempts"] == 1
    assert report["evidence"]["excluded_operations"] == 1
    assert fleet.get("a2")["series_id"] == "a1"
    assert fleet.get("a2")["bundle_version"] == "after"
    assert fleet.get("a2")["attempt_number"] == 2


def test_windows_use_retry_start_and_half_open_utc_bounds(
    fleet: AttemptLogStore,
) -> None:
    report = build_report(
        fleet, since=START + timedelta(minutes=10), until=START + timedelta(minutes=12)
    )
    assert [a["attempt_id"] for a in report["attempts"]] == ["a2"]
    assert report["metrics"]["first_attempt_success"]["denominator"] == 0
    assert report["metrics"]["retry_recovery"]["percent"] == 100
    assert report["metrics"]["time_to_usable_inference"]["median_seconds"] == 900
    assert build_report(fleet, hostnames=["first-success"])["attempt_count"] == 1


def test_series_reset_for_new_target_completed_or_cancelled_operation(
    fleet: AttemptLogStore,
) -> None:
    for host, engine, model in (
        ("retry-host", "vllm", "org/model"),
        ("cancelled-host", "vllm", "org/model"),
        ("failed-retry", "vllm", "new/model"),
        ("failed-retry", "llama_cpp", "new/model"),
    ):
        attempt = fleet.create(host, engine=engine, model=model)
        assert fleet.get(attempt)["attempt_number"] == 1
        assert fleet.get(attempt)["series_id"] == attempt


def _source(
    store: AttemptLogStore, attempt: str, name: str, text: str, **fields: Any
) -> None:
    record = store.append(attempt, text, source="diagnostics." + name)
    assert record is not None
    sources = store.get(attempt).get("diagnostics", {}).get("sources", {})
    sources[name] = dict(
        status="collected", first_seq=record["seq"], last_seq=record["seq"], **fields
    )
    store.update(attempt, diagnostics=dict(sources=sources))


def test_environment_groups_keep_original_error_and_unknown_evidence(
    fleet: AttemptLogStore,
) -> None:
    _source(fleet, "a1", "runtime", "vllm=0.10.1\ntorch=2.7.0\n")
    _source(fleet, "a1", "os", 'ID="rhel"\nVERSION_ID="9.6"\n')
    _source(fleet, "a1", "kernel", "Linux retry-host 5.14.0-example #1 SMP x86_64\n")
    _source(
        fleet,
        "a1",
        "gpu",
        "index, uuid, name, memory.total\n0, GPU-1, NVIDIA A100, 40960 MiB\n",
    )
    row = next(a for a in build_report(fleet)["attempts"] if a["attempt_id"] == "a1")
    assert row["original_error"] == "CUDA out of memory"
    assert row["signature"] == "v1:cuda_out_of_memory"
    assert row["runtime_version"] == "0.10.1"
    assert row["os"] == "rhel 9.6"
    assert row["kernel"] == "5.14.0-example"
    assert row["gpu_family"] == "NVIDIA A100"
    assert row["unknown_dimensions"] == []
    fleet.append(
        "a1",
        "[PROFILE:select:vllm-ampere-a100 (model=A100 sm=8.0)]",
        source="setup.stdout",
    )
    assert fleet.get("a1")["runtime_profile"] == "vllm-ampere-a100"
    grouped = build_report(fleet, group_by="all")
    group = next(g for g in grouped["groups"] if "a1" in g["attempt_ids"])
    assert set(group["dimensions"]) == set(DIMENSIONS)
    assert group["dimensions"]["gpu_family"] == "ampere-a100"
    _source(fleet, "a1", "runtime", "vllm=wrong-version", deferred=True)
    row = next(a for a in build_report(fleet)["attempts"] if a["attempt_id"] == "a1")
    assert row["runtime_version"] == "unknown"
    assert row["incomplete"]


def test_eviction_legacy_and_empty_denominators(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "retention.sqlite3", max_attempts=1)
    old = store.create("host", engine="vllm")
    store.update(old, status="failed", stage="setup")
    retry = store.create("host", engine="vllm")
    store.update(retry, status="complete", stage="complete")
    report = build_report(store)
    assert report["evidence"]["evicted_attempts"] == 1
    assert report["metrics"]["retry_recovery"]["denominator"] == 0
    new = store.create("unseen-host", engine="vllm")
    assert store.get(new)["series_origin_known"] is True
    store.update(new, status="complete", stage="complete")
    report = build_report(store)
    assert report["metrics"]["first_attempt_success"]["percent"] == 100
    reopened = AttemptLogStore(store.path, max_attempts=1)
    returning = reopened.create("host", engine="vllm")
    assert reopened.get(returning)["series_origin_known"] is False
    empty = build_report(store, hostnames=["absent"])
    assert empty["metrics"]["first_attempt_success"]["percent"] is None
    assert empty["metrics"]["time_to_usable_inference"]["median_seconds"] is None


def test_expired_teardown_only_obscures_its_own_host(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "expired.sqlite3")
    old = store.create("old-host", operation="teardown")
    store.update(old, status="complete", stage="teardown_complete")
    with store._db() as db:
        db.execute("UPDATE attempts SET created=0 WHERE id=?", (old,))
    reopened = AttemptLogStore(store.path)
    new = reopened.create("new-host", engine="vllm")
    assert reopened.get(new)["series_origin_known"] is True
    returning = reopened.create("old-host", engine="vllm")
    assert reopened.get(returning)["series_origin_known"] is False
    reopened.update(returning, status="complete", stage="complete")
    fresh_series = reopened.create("old-host", engine="vllm")
    assert reopened.get(fresh_series)["series_origin_known"] is True


def test_invalid_start_is_retained_only_without_time_bounds(
    fleet: AttemptLogStore,
) -> None:
    fleet.update("b1", started_at="invalid")
    report = build_report(fleet, hostnames=["first-success"])
    assert report["attempt_count"] == 1
    assert report["evidence"]["invalid_start_times"] == 1
    assert report["metrics"]["first_attempt_success"]["percent"] == 100
    for since, until in ((START, None), (None, START + timedelta(days=1))):
        report = build_report(
            fleet, hostnames=["first-success"], since=since, until=until
        )
        assert report["attempt_count"] == 0
        assert report["evidence"]["invalid_start_times"] == 1
        assert report["metrics"]["first_attempt_success"]["percent"] is None


def test_upgrade_preserves_unattributed_historical_evictions(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "legacy.sqlite3")
    with store._db() as db:
        db.execute("DELETE FROM statistics WHERE name='unattributed_evicted_attempts'")
        db.execute("UPDATE statistics SET value=2 WHERE name='evicted_attempts'")
    reopened = AttemptLogStore(store.path)
    first = reopened.create("host", engine="vllm")
    assert not reopened.get(first)["series_origin_known"]
    reopened.update(first, status="complete", stage="complete")
    next_series = reopened.create("host", engine="vllm")
    assert reopened.get(next_series)["series_origin_known"]
    report = build_report(AttemptLogStore(store.path))
    assert report["evidence"]["unattributed_evicted_attempts"] == 2
    assert any("predate per-host tracking" in warning for warning in report["warnings"])


@pytest.mark.parametrize("count", [200, 201, 202])
def test_source_read_limits_preserve_latest_tail_and_exact_snapshot(
    tmp_path: Path, count: int
) -> None:
    store = AttemptLogStore(tmp_path / "source.sqlite3")
    attempt = store.create("host")
    _source(store, attempt, "kernel", "old snapshot")
    records = store.append_many(
        attempt,
        [dict(source="diagnostics.kernel", msg=f"line-{i}") for i in range(count)],
    )
    store.update(
        attempt,
        diagnostics={
            "sources": {
                "kernel": dict(
                    status="collected",
                    first_seq=records[0]["seq"],
                    last_seq=records[-1]["seq"],
                )
            }
        },
    )
    store.append(attempt, "different source")
    tail = store.tail(attempt, "diagnostics.kernel", max_bytes=65536)
    assert tail["truncated"]
    assert tail["output"].splitlines() == [
        f"line-{i}" for i in range(count - 200, count)
    ]
    evidence = store.reliability_snapshot()["attempts"][0]["environment_evidence"]
    if count <= 201:
        assert evidence["kernel"] == "".join(f"line-{i}" for i in range(count))
    else:
        assert "kernel" not in evidence


def test_success_requires_registration_and_readiness_time_is_not_invented(
    fleet: AttemptLogStore,
) -> None:
    fleet.update("e1", status="complete", stage="starting_engine")
    fleet.update("b1", ready_at="bad timestamp")
    fleet.update("a2", ready_at=START.isoformat())
    fleet.update("h1", started_at="bad timestamp")
    report = build_report(fleet)
    assert report["outcomes"]["unknown"] == 2
    assert (
        report["metrics"]["time_to_usable_inference"]["samples"] == 1
    )  # valid zero duration for a2
    assert report["evidence"]["invalid_start_times"] == 1
    assert build_report(fleet, since=START)["attempt_count"] == 9


@pytest.mark.parametrize(
    ("message", "signature"),
    [
        ("Permission denied", "permission_denied"),
        ("Connection refused", "connection_refused"),
        ("Health poll timed out", "timeout"),
        ("No matching kernel headers", "kernel_headers"),
        ("", "unknown"),
    ],
)
def test_symptoms_do_not_require_environment_evidence(
    fleet: AttemptLogStore, message: str, signature: str
) -> None:
    fleet.update("a1", failure_summary=message)
    row = build_report(fleet, hostnames=["retry-host"])["attempts"][-1]
    assert row["signature"] == "v1:" + signature
    assert row["original_error"] == message


def test_generic_signature_normalizes_host_ids_and_numbers(
    fleet: AttemptLogStore,
) -> None:
    fleet.update(
        "g1", failure_summary="failed-retry: unexpected value 123 from 10.0.0.1"
    )
    fleet.update(
        "g2", failure_summary="failed-retry: unexpected value 999 from 10.1.2.3"
    )
    report = build_report(fleet, hostnames=["failed-retry"])
    assert len(report["groups"]) == 1
    assert report["groups"][0]["dimensions"]["signature"].startswith("v1:unclassified:")


def test_cuda_probe_compile_failures_group_despite_different_log_tails(
    fleet: AttemptLogStore,
) -> None:
    errors = {
        "g1": (
            "cuda_proof: setup exited with status 1\n"
            "tzdata-java installed\nCUDA toolkit 13.0 installed\n"
            "FATAL: nvcc failed to compile the CUDA execution probe\n"
            "[STEP:cuda_proof:FAIL]"
        ),
        "g2": (
            "cuda_proof: setup exited with status 1\n"
            "CUDA toolkit 13.0 already installed, skipping\n"
            '/tmp/cuda-probe.ABC123/cuda_probe.cu(4): error: identifier "printf" '
            "is undefined\n"
            "FATAL: nvcc failed to compile the CUDA execution probe\n"
            "[STEP:cuda_proof:FAIL]"
        ),
    }
    for attempt_id, error in errors.items():
        fleet.update(attempt_id, failure_summary=error)
    report = build_report(fleet, hostnames=["failed-retry"])
    assert len(report["groups"]) == 1
    assert report["groups"][0]["dimensions"] == {"signature": "v1:cuda_probe_compile"}
    assert report["groups"][0]["count"] == 2
    assert {a["attempt_id"]: a["original_error"] for a in report["attempts"]} == errors


def test_unsupported_marker_in_remote_issues_and_legacy_retry(
    fleet: AttemptLogStore,
) -> None:
    fleet.update("g1", issues=["Node: unsupported_hardware: no tested profile"])
    assert (
        build_report(fleet, hostnames=["failed-retry"])["outcomes"]["unsupported"] == 1
    )
    fleet.update("h1", status="interrupted", stage="setup")
    retry = fleet.create("legacy-host", engine="vllm", model="org/model")
    assert not fleet.get(retry)["series_origin_known"]
    fleet.update(retry, status="complete", stage="complete")
    assert (
        build_report(fleet, hostnames=["legacy-host"])["metrics"]["retry_recovery"][
            "denominator"
        ]
        == 0
    )


def test_auto_selected_model_does_not_change_requested_target(tmp_path: Path) -> None:
    store = AttemptLogStore(tmp_path / "auto.sqlite3")
    first = store.create("host", engine="vllm")
    store.update(first, status="failed", model="selected/model")
    retry = store.create("host", engine="vllm")
    assert store.get(retry)["series_id"] == first
    assert store.get(retry)["attempt_number"] == 2


def test_incomplete_or_rotated_environment_is_not_a_measured_version(
    fleet: AttemptLogStore,
) -> None:
    _source(fleet, "a1", "runtime", "vllm=0.10.1", truncated=True)
    assert (
        "runtime"
        not in fleet.reliability_snapshot()["attempts"][0]["environment_evidence"]
    )
    _source(fleet, "a1", "runtime", "vllm=0.10.2")
    fleet.attempt_max_bytes = 1
    fleet.append("a1", "rotate all source output")
    assert (
        "runtime"
        not in fleet.reliability_snapshot()["attempts"][0]["environment_evidence"]
    )


def test_llamacpp_version_is_measured_from_runtime_source(
    fleet: AttemptLogStore,
) -> None:
    fleet.update("a1", engine="llama_cpp")
    _source(fleet, "a1", "runtime", "version: 12345 (abcdef)\nbuilt with gcc\n")
    row = build_report(fleet, hostnames=["retry-host"])["attempts"][-1]
    assert row["runtime_version"] == "12345 (abcdef)"


def test_report_api_auth_export_and_drilldown(
    client: TestClient,
    app: FastAPI,
    mock_provisioner: MagicMock,
    fleet: AttemptLogStore,
) -> None:
    mock_provisioner.log_buffer = ProvisioningLogBuffer(store=fleet)
    endpoint = "/admin/provisioning/reliability"
    assert TestClient(app).get(endpoint).status_code == 401
    response = client.get(endpoint + "?group_by=bundle_version&download=true")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    row = next(a for a in response.json()["attempts"] if a["attempt_id"] == "a1")
    assert client.get(row["logs_url"]).json()["attempt"]["attempt_id"] == "a1"
    bundle = client.get(row["bundle_url"])
    manifest = json.loads(gzip.decompress(bundle.content).splitlines()[0])["manifest"]
    assert manifest["failure_summary"] == row["original_error"]
    assert client.get(endpoint + "?hostname=first-success").json()["attempt_count"] == 1
    assert client.get("/dashboard/reliability").status_code == 200
    assert "reliability.js?v=" in client.get("/dashboard/reliability").text
    assert "Report filters" not in TestClient(app).get("/dashboard/reliability").text
    for query in (
        "since=2026-09-21T00:00:00",
        "since=2026-09-22T00:00:00Z&until=2026-09-21T00:00:00Z",
        "group_by=bad",
        "hostname=../bad",
    ):
        assert client.get(endpoint + "?" + query).status_code in {400, 422}
    mock_provisioner.log_buffer.store = None
    assert client.get(endpoint).status_code == 503


async def test_real_setup_failure_then_launch_success_is_one_recovered_series(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh, "CUDA out of memory")
    with pytest.raises(RemoteCommandError):
        await provisioner._provision("host1", model="org/model")
    setup = ssh.root / "auto-vllm/setup.sh"
    setup.write_text(
        setup.read_text().replace("echo 'CUDA out of memory' >&2; return 7;", ":;")
    )
    provisioner._poll_health = AsyncMock()  # type: ignore[method-assign]
    await provisioner._provision("host1", model="org/model")
    report = build_report(store)
    assert report["outcomes"]["success"] == 1
    assert report["outcomes"]["failed"] == 1
    assert report["metrics"]["first_attempt_success"]["percent"] == 0
    assert report["metrics"]["retry_recovery"]["percent"] == 100
    assert report["metrics"]["time_to_usable_inference"]["samples"] == 1
    assert ssh.launches == 3  # failed setup, retried setup, successful start


@pytest.mark.parametrize("entrypoint", ["_provision", "provision"])
async def test_registered_success_survives_shutdown_before_log_finalization(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
    entrypoint: str,
) -> None:
    provisioner, ssh, store = harness
    _prepare(provisioner, ssh)
    provisioner._poll_health = AsyncMock()  # type: ignore[method-assign]
    provisioner._finish_remote_logs = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )
    with pytest.raises(asyncio.CancelledError):
        await getattr(provisioner, entrypoint)("host1", model="org/model")
    reopened = AttemptLogStore(store.path)
    reopened.interrupt_running()
    report = build_report(reopened)
    assert report["outcomes"]["success"] == 1
    attempt = report["attempts"][0]
    assert attempt["ready_at"] == attempt["finished_at"]
    assert report["metrics"]["first_attempt_success"]["percent"] == 100
    assert report["metrics"]["time_to_usable_inference"]["samples"] == 1


async def test_late_cancellation_does_not_overwrite_recorded_success(
    harness: tuple[NodeProvisioner, LocalNodeSSH, AttemptLogStore],
) -> None:
    provisioner, _, store = harness
    provisioner.log_buffer.create("host1")
    await provisioner._update_state("host1", ProvisioningStep.COMPLETE)
    registered = asyncio.Event()

    async def finalize() -> None:
        try:
            registered.set()
            await asyncio.Event().wait()
        finally:
            provisioner._mark_log_complete("host1")

    task = asyncio.create_task(finalize())
    await registered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert build_report(store)["outcomes"]["success"] == 1
    attempt = store.get(provisioner.log_buffer.attempts["host1"])
    assert attempt["finished_at"] == attempt["ready_at"]
