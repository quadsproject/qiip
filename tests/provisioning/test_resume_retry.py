"""Orchestrator tests for bounded retries, resume env, and checkpoints (#133)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
    SetupDeadlineError,
)
from inference_proxy.provisioning.ssh_client import SSHConnectionError

_HOST = "host1"
_PLAN = "sha256:plan-a"


def _make(
    store: AttemptLogStore,
    *,
    settings: ProvisioningSettings | None = None,
) -> tuple[NodeProvisioner, ProvisioningLogBuffer]:
    buffer = ProvisioningLogBuffer(store=store)
    buffer.create(
        _HOST,
        engine="vllm",
        model="org/model",
        bundle_version="sha256:bundle",
        plan_id=_PLAN,
        setup_inputs={"engine": "vllm"},
    )
    provisioner = NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=MagicMock(prefix="/nodes/"),
        settings=settings
        or ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            stage_retry_base_delay=0.01,
            stage_retry_max_delay=0.02,
            stage_retry_jitter=0,
        ),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=EndpointPolicy.from_values(
            allowed_hosts=[_HOST], allowed_networks=[], allowed_ports=[8000]
        ),
        log_buffer=buffer,
        nfs_export="nfs.example:/exports/huggingface",
    )
    return provisioner, buffer


@pytest.fixture
def store(tmp_path: Path) -> AttemptLogStore:
    return AttemptLogStore(tmp_path / "attempts.sqlite3")


class TestRunSetupWithRetries:
    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_then_succeeds(
        self, store: AttemptLogStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provisioner, buffer = _make(store)
        calls = 0

        async def flaky(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SSHConnectionError(_HOST, "connection reset by peer")

        monkeypatch.setattr(provisioner, "_run_setup", flaky)
        monkeypatch.setattr(
            provisioner, "_reconcile_host", AsyncMock(return_value=False)
        )

        await provisioner._run_setup_with_retries(
            _HOST,
            started_at=datetime.now(UTC),
            on_step=lambda _step: None,
            engine=InferenceEngine.VLLM,
            plan_id=_PLAN,
        )

        assert calls == 2
        manifest = store.get(buffer.attempts[_HOST])
        assert manifest["transient_retries"] == 1
        assert any("automatic retry 1" in issue for issue in manifest["issues"])
        # The retry recovered, so no failure classification is left behind.
        assert manifest["failure_class"] is None
        assert manifest["next_action"] is None

    @pytest.mark.asyncio
    async def test_configuration_failure_is_not_retried(
        self, store: AttemptLogStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provisioner, _buffer = _make(store)
        calls = 0

        async def invalid_config(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            raise ProvisioningError("invalid configuration: missing NFS export")

        monkeypatch.setattr(provisioner, "_run_setup", invalid_config)

        with pytest.raises(ProvisioningError):
            await provisioner._run_setup_with_retries(
                _HOST,
                started_at=datetime.now(UTC),
                on_step=lambda _step: None,
                engine=InferenceEngine.VLLM,
                plan_id=_PLAN,
            )

        assert calls == 1

    @pytest.mark.asyncio
    async def test_transient_retries_are_attempt_bounded(
        self, store: AttemptLogStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provisioner, buffer = _make(store)
        calls = 0

        async def always_transient(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            raise SSHConnectionError(_HOST, "operation timed out")

        monkeypatch.setattr(provisioner, "_run_setup", always_transient)
        monkeypatch.setattr(
            provisioner, "_reconcile_host", AsyncMock(return_value=False)
        )

        with pytest.raises(SSHConnectionError):
            await provisioner._run_setup_with_retries(
                _HOST,
                started_at=datetime.now(UTC),
                on_step=lambda _step: None,
                engine=InferenceEngine.VLLM,
                plan_id=_PLAN,
            )

        # max_attempts default is 3: initial attempt plus two bounded retries.
        assert calls == 3
        assert store.get(buffer.attempts[_HOST])["transient_retries"] == 2

    @pytest.mark.asyncio
    async def test_live_remote_phase_blocks_retry(
        self, store: AttemptLogStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provisioner, _buffer = _make(store)
        calls = 0

        async def transient(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            raise SSHConnectionError(_HOST, "connection reset by peer")

        monkeypatch.setattr(provisioner, "_run_setup", transient)
        reconcile = AsyncMock(return_value=True)
        monkeypatch.setattr(provisioner, "_reconcile_host", reconcile)

        with pytest.raises(SSHConnectionError):
            await provisioner._run_setup_with_retries(
                _HOST,
                started_at=datetime.now(UTC),
                on_step=lambda _step: None,
                engine=InferenceEngine.VLLM,
                plan_id=_PLAN,
            )

        assert calls == 1
        reconcile.assert_awaited_once_with(_HOST)

    @pytest.mark.asyncio
    async def test_stage_deadline_is_terminal_and_not_retried(
        self, store: AttemptLogStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        provisioner, _buffer = _make(
            store,
            settings=ProvisioningSettings(
                vllm_setup_timeout=0.05,
                stage_retry_max_attempts=3,
                stage_retry_base_delay=0.01,
                stage_retry_max_delay=0.02,
                stage_retry_jitter=0,
            ),
        )
        calls = 0

        async def never_finishes(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            await asyncio.sleep(5)

        monkeypatch.setattr(provisioner, "_run_setup", never_finishes)

        with pytest.raises(SetupDeadlineError, match="stage deadline"):
            await provisioner._run_setup_with_retries(
                _HOST,
                started_at=datetime.now(UTC),
                on_step=lambda _step: None,
                engine=InferenceEngine.VLLM,
                plan_id=_PLAN,
            )

        # Exhausting the run's own deadline must not trigger a retry loop.
        assert calls == 1


class TestResumeEnvironment:
    def test_emits_plan_and_verified_stages(self, store: AttemptLogStore) -> None:
        provisioner, buffer = _make(store)
        store.record_checkpoint(buffer.attempts[_HOST], "nvidia_driver", "completed")
        store.record_checkpoint(buffer.attempts[_HOST], "llmfit_install", "skipped")

        env = provisioner._setup_resume_env(_HOST, _PLAN)

        assert env["QIIP_PLAN_ID"] == _PLAN
        assert set(env["QIIP_RESUME_STAGES"].split(",")) == {
            "nvidia_driver",
            "llmfit_install",
        }

    def test_changed_plan_emits_no_resume_stages(self, store: AttemptLogStore) -> None:
        provisioner, buffer = _make(store)
        store.record_checkpoint(buffer.attempts[_HOST], "nvidia_driver", "completed")

        env = provisioner._setup_resume_env(_HOST, "sha256:different-plan")

        assert env["QIIP_PLAN_ID"] == "sha256:different-plan"
        assert env["QIIP_RESUME_STAGES"] == ""

    def test_no_plan_emits_no_resume_variables(self, store: AttemptLogStore) -> None:
        provisioner, _buffer = _make(store)
        assert provisioner._setup_resume_env(_HOST, None) == {}


class TestStepCheckpoints:
    def test_ok_marker_records_completion(self, store: AttemptLogStore) -> None:
        provisioner, buffer = _make(store)
        provisioner._record_step_checkpoint(
            _HOST, "nvidia_driver", "OK", "[STEP:nvidia_driver:OK]"
        )
        checkpoint = store.get(buffer.attempts[_HOST])["checkpoints"]["nvidia_driver"]
        assert checkpoint["status"] == "completed"
        assert checkpoint["evidence"] == "[STEP:nvidia_driver:OK]"

    def test_skip_marker_records_skipped(self, store: AttemptLogStore) -> None:
        provisioner, buffer = _make(store)
        provisioner._record_step_checkpoint(
            _HOST, "nvidia_driver", "SKIP", "[STEP:nvidia_driver:SKIP]"
        )
        assert (
            store.get(buffer.attempts[_HOST])["checkpoints"]["nvidia_driver"]["status"]
            == "skipped"
        )

    def test_unknown_status_is_ignored(self, store: AttemptLogStore) -> None:
        provisioner, buffer = _make(store)
        provisioner._record_step_checkpoint(_HOST, "nvidia_driver", "WARN", "warn")
        assert store.get(buffer.attempts[_HOST])["checkpoints"] == {}

    def test_no_active_attempt_is_a_noop(self, store: AttemptLogStore) -> None:
        provisioner, _buffer = _make(store)
        # No attempt opened for a different host must not raise.
        provisioner._record_step_checkpoint("host2", "nvidia_driver", "OK", "ok")


class TestPlanInputs:
    def test_plan_id_changes_when_an_input_changes(
        self, store: AttemptLogStore
    ) -> None:
        from inference_proxy.provisioning.recovery import setup_plan_id

        first, _buffer = _make(store)
        second, _buffer2 = _make(store)
        second._settings.nvidia_driver_version = "999.0"

        first_inputs = first._setup_plan_inputs(InferenceEngine.VLLM)
        second_inputs = second._setup_plan_inputs(InferenceEngine.VLLM)

        assert first_inputs != second_inputs
        assert setup_plan_id(first_inputs) != setup_plan_id(second_inputs)

    def test_plan_includes_bundle_identity_and_api_port(
        self, store: AttemptLogStore
    ) -> None:
        provisioner, _buffer = _make(store)
        inputs = provisioner._setup_plan_inputs(InferenceEngine.VLLM)
        # A re-uploaded script and a changed port both invalidate completed
        # stages even when no setting changed.
        assert inputs["bundle_version"] == provisioner._bundle_version(
            InferenceEngine.VLLM
        )
        assert inputs["api_port"] == provisioner._settings.vllm_port

    def test_llamacpp_plan_includes_catalog_inputs(
        self, store: AttemptLogStore
    ) -> None:
        provisioner, _buffer = _make(store)
        vllm_inputs = provisioner._setup_plan_inputs(InferenceEngine.VLLM)
        llamacpp_inputs = provisioner._setup_plan_inputs(InferenceEngine.LLAMA_CPP)
        assert "llamacpp_artifact_catalog_url" in llamacpp_inputs
        assert "llamacpp_artifact_catalog_url" not in vllm_inputs
