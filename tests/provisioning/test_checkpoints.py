"""Checkpoint persistence and compatible-plan resume semantics (issue #133)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_proxy.provisioning.log_store import AttemptLogStore


@pytest.fixture
def store(tmp_path: Path) -> AttemptLogStore:
    return AttemptLogStore(tmp_path / "attempts.sqlite3")


def _create(
    store: AttemptLogStore,
    plan_id: str = "sha256:plan-a",
    *,
    hostname: str = "host1",
) -> str:
    return store.create(
        hostname,
        engine="vllm",
        model="org/model",
        bundle_version="sha256:bundle",
        plan_id=plan_id,
        setup_inputs={"engine": "vllm", "model": "org/model"},
    )


class TestCreateMetadata:
    def test_new_attempt_records_plan_and_defaults(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        manifest = store.get(attempt)
        assert manifest["plan_id"] == "sha256:plan-a"
        assert manifest["setup_inputs"] == {"engine": "vllm", "model": "org/model"}
        assert manifest["checkpoints"] == {}
        assert manifest["progress"] is None
        assert manifest["transient_retries"] == 0
        assert manifest["attempt_number"] == 1

    def test_retry_increments_attempt_number(self, store: AttemptLogStore) -> None:
        first = _create(store)
        store.update(first, status="failed", stage="setup")
        second = _create(store)
        assert store.get(second)["attempt_number"] == 2
        assert store.get(second)["series_id"] == store.get(first)["series_id"]


class TestRecordCheckpoint:
    def test_checkpoint_persists_status_evidence_and_progress(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        store.record_checkpoint(
            attempt, "nvidia_driver", "completed", evidence="[STEP:nvidia_driver:OK]"
        )
        manifest = store.get(attempt)
        checkpoint = manifest["checkpoints"]["nvidia_driver"]
        assert checkpoint["status"] == "completed"
        assert checkpoint["stage"] == "nvidia_driver"
        assert checkpoint["plan_id"] == "sha256:plan-a"
        assert checkpoint["attempt_id"] == attempt
        assert checkpoint["attempt_number"] == 1
        assert checkpoint["started_at"] is not None
        assert checkpoint["completed_at"] is not None
        assert checkpoint["evidence"] == "[STEP:nvidia_driver:OK]"
        assert manifest["progress"]["completed_count"] == 1
        assert manifest["current_stage"] == "nvidia_driver"

    def test_running_then_completed_preserves_started_at(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        store.record_checkpoint(attempt, "cuda_toolkit", "running")
        started = store.get(attempt)["checkpoints"]["cuda_toolkit"]["started_at"]
        store.record_checkpoint(attempt, "cuda_toolkit", "completed")
        checkpoint = store.get(attempt)["checkpoints"]["cuda_toolkit"]
        assert checkpoint["started_at"] == started
        assert checkpoint["completed_at"] is not None

    def test_rerun_clears_stale_terminal_evidence(self, store: AttemptLogStore) -> None:
        attempt = _create(store)
        store.record_checkpoint(attempt, "vllm_install", "completed")
        store.record_checkpoint(attempt, "vllm_install", "running")
        checkpoint = store.get(attempt)["checkpoints"]["vllm_install"]
        assert checkpoint["status"] == "running"
        assert checkpoint["completed_at"] is None

    def test_failed_checkpoint_is_reported_in_progress(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        store.record_checkpoint(attempt, "nfs_mount", "failed")
        progress = store.get(attempt)["progress"]
        assert progress["failed_stages"] == ["nfs_mount"]
        assert progress["completed_count"] == 0

    def test_running_clears_previous_failure_classification(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        store.update(attempt, failure_class="transient", next_action="retry")
        store.record_checkpoint(attempt, "vllm_install", "running")
        manifest = store.get(attempt)
        assert manifest["failure_class"] is None
        assert manifest["next_action"] is None

    def test_missing_attempt_raises(self, store: AttemptLogStore) -> None:
        with pytest.raises(KeyError):
            store.record_checkpoint("missing", "nvidia_driver", "completed")


class TestTransientRetries:
    def test_counts_each_increment(self, store: AttemptLogStore) -> None:
        attempt = _create(store)
        assert store.increment_transient_retry(attempt) == 1
        assert store.increment_transient_retry(attempt) == 2
        assert store.get(attempt)["transient_retries"] == 2
        assert store.get(attempt)["last_transient_retry_at"] is not None

    def test_unknown_attempt_raises(self, store: AttemptLogStore) -> None:
        with pytest.raises(KeyError):
            store.increment_transient_retry("missing")


class TestResumableCheckpoints:
    def test_completed_and_skipped_stages_are_resumable(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store)
        store.record_checkpoint(attempt, "nvidia_driver", "completed")
        store.record_checkpoint(attempt, "llmfit_install", "skipped")
        store.record_checkpoint(attempt, "cuda_toolkit", "running")
        resumable = store.resumable_checkpoints("host1", "sha256:plan-a")
        assert set(resumable) == {"nvidia_driver", "llmfit_install"}
        assert resumable["nvidia_driver"]["status"] == "completed"

    def test_changed_plan_invalidates_every_checkpoint(
        self, store: AttemptLogStore
    ) -> None:
        attempt = _create(store, "sha256:plan-a")
        store.record_checkpoint(attempt, "nvidia_driver", "completed")
        assert store.resumable_checkpoints("host1", "sha256:plan-b") == {}

    def test_newer_failure_blocks_older_completion(
        self, store: AttemptLogStore
    ) -> None:
        first = _create(store)
        store.record_checkpoint(first, "vllm_install", "completed")
        store.update(first, status="failed", stage="setup")
        second = _create(store)
        store.record_checkpoint(second, "vllm_install", "failed")
        # The newest attempt re-ran and failed the stage; the older completion
        # must not be reused.
        assert "vllm_install" not in store.resumable_checkpoints(
            "host1", "sha256:plan-a"
        )

    def test_newer_completion_supersedes_older_failure(
        self, store: AttemptLogStore
    ) -> None:
        first = _create(store)
        store.record_checkpoint(first, "vllm_install", "failed")
        store.update(first, status="failed", stage="setup")
        second = _create(store)
        store.record_checkpoint(second, "vllm_install", "completed")
        resumable = store.resumable_checkpoints("host1", "sha256:plan-a")
        assert resumable["vllm_install"]["status"] == "completed"

    def test_unrelated_host_is_isolated(self, store: AttemptLogStore) -> None:
        attempt = _create(store, hostname="host1")
        store.record_checkpoint(attempt, "nvidia_driver", "completed")
        assert store.resumable_checkpoints("host2", "sha256:plan-a") == {}

    def test_skipped_stage_without_verifier_still_resumes(
        self, store: AttemptLogStore
    ) -> None:
        # plan+stage agreement is enough for stages with no verifier hook.
        attempt = _create(store)
        store.record_checkpoint(attempt, "firewall", "skipped")
        assert "firewall" in store.resumable_checkpoints("host1", "sha256:plan-a")
