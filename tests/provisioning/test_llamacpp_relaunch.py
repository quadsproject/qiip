"""Transactional managed llama.cpp relaunch tests."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdRecord
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.huggingface.artifacts import GGUFArtifact, ResolvedGGUFArtifact
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppCacheType,
    LlamaCppFlashAttention,
    LlamaCppGPUState,
    LlamaCppRuntimeEffective,
    LlamaCppRuntimeRequest,
    LlamaCppRuntimeState,
    LlamaCppSizingMode,
    Node,
    NodeStatus,
)
from inference_proxy.provisioning.provisioner import (
    DrainOutcome,
    NodeProvisioner,
    ProvisioningError,
    RelaunchConflictError,
    RelaunchPreconditionError,
    RelaunchValidationError,
)

_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["host1"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _request(
    *,
    sizing: LlamaCppSizingMode = LlamaCppSizingMode.CUSTOM,
    fit_target_mib: int = 512,
    context_per_slot: int = 24576,
    slots: int = 2,
    allow_estimator_overrun: bool = False,
) -> LlamaCppRuntimeRequest:
    if sizing is LlamaCppSizingMode.AUTO:
        return LlamaCppRuntimeRequest(
            sizing=sizing,
            fit_target_mib=fit_target_mib,
        )
    return LlamaCppRuntimeRequest(
        sizing=sizing,
        fit_target_mib=fit_target_mib,
        context_per_slot=context_per_slot,
        slots=slots,
        cache_type=LlamaCppCacheType.Q8_0,
        allow_estimator_overrun=allow_estimator_overrun,
    )


def _runtime(
    request: LlamaCppRuntimeRequest | None = None,
) -> LlamaCppRuntimeState:
    requested = request or _request(sizing=LlamaCppSizingMode.AUTO)
    context_per_slot = requested.context_per_slot or 12544
    slots = requested.slots or 1
    aggregate = context_per_slot * slots
    cache_type = requested.cache_type or LlamaCppCacheType.Q8_0
    return LlamaCppRuntimeState(
        requested=requested,
        effective=LlamaCppRuntimeEffective(
            train_context=262144,
            context_per_slot=context_per_slot,
            slot_context_limit=min(262144, aggregate),
            slots=slots,
            aggregate_context=aggregate,
            cache_type_k=cache_type,
            cache_type_v=cache_type,
            flash_attn=LlamaCppFlashAttention.ON,
            kv_unified=True,
            gpu_layers=31,
            total_layers=31,
        ),
        gpus=(
            LlamaCppGPUState(
                index=0,
                total_mib=15360,
                used_mib=14117,
                free_mib=1243,
            ),
        ),
        observed_at=datetime(2026, 8, 5, 21, 55, 24, tzinfo=UTC),
    )


def _node(*, runtime: LlamaCppRuntimeState | None = None) -> Node:
    return Node(
        node_id="host1",
        endpoint="http://host1:8000",
        status=NodeStatus.HEALTHY,
        model="org/model-GGUF",
        engine=InferenceEngine.LLAMA_CPP,
        artifact_id="a" * 64,
        llamacpp_runtime=runtime or _runtime(),
        last_heartbeat=datetime(2026, 8, 5, 21, 54, tzinfo=UTC),
        managed=True,
    )


def _artifact() -> ResolvedGGUFArtifact:
    return ResolvedGGUFArtifact(
        artifact=GGUFArtifact(
            artifact_id="a" * 64,
            repo_id="org/model-GGUF",
            resolved_revision="b" * 40,
            files=("model-Q8_0.gguf",),
            entrypoint="model-Q8_0.gguf",
            model_alias="org/model-GGUF",
            file_sizes={"model-Q8_0.gguf": 1},
        ),
        node_relative_entrypoint=(
            "hub/models--org--model-GGUF/snapshots/" + "b" * 40 + "/model-Q8_0.gguf"
        ),
    )


def _node_from_value(value: bytes) -> Node:
    payload = json.loads(value)
    return Node(node_id="host1", **payload)


def _provisioner(
    node: Node | None = None,
) -> tuple[
    NodeProvisioner,
    NodeRegistry,
    MagicMock,
    dict[str, EtcdRecord],
    list[tuple[NodeStatus, int]],
]:
    initial = node or _node()
    key, value = node_to_etcd(initial, "/nodes/")
    state = {
        "record": EtcdRecord(
            key=key.encode(),
            value=value,
            mod_revision=10,
            lease_id=7001,
        )
    }
    writes: list[tuple[NodeStatus, int]] = []
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    etcd.node_lease_ttl = 600
    etcd.get_record.side_effect = lambda _key: state["record"]
    etcd.refresh_lease.return_value = 600
    etcd.grant_node_lease.return_value = 7002
    etcd.revoke_lease.return_value = True

    def replace_if_revision(
        candidate_key: str,
        candidate_value: bytes,
        *,
        expected_mod_revision: int,
        lease_id: int,
    ) -> int | None:
        current = state["record"]
        if current.mod_revision != expected_mod_revision:
            return None
        revision = current.mod_revision + 1
        state["record"] = EtcdRecord(
            key=candidate_key.encode(),
            value=candidate_value,
            mod_revision=revision,
            lease_id=lease_id,
        )
        writes.append((_node_from_value(candidate_value).status, lease_id))
        return revision

    etcd.replace_if_revision.side_effect = replace_if_revision
    registry = NodeRegistry()
    registry.add(initial)
    tracker = MagicMock()
    tracker.get.return_value = 0
    provisioner = NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=etcd,
        settings=ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
        ),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_POLICY,
        registry=registry,
        connection_tracker=tracker,
        nfs_export="nfs.example:/exports/huggingface",
    )
    provisioner.resolve_artifact_selection = AsyncMock(  # type: ignore[method-assign]
        return_value=_artifact()
    )
    provisioner._upload_scripts = AsyncMock()  # type: ignore[method-assign]
    provisioner._drain_wait = AsyncMock(  # type: ignore[method-assign]
        return_value=DrainOutcome.COMPLETE
    )
    provisioner._stop_llamacpp = AsyncMock()  # type: ignore[method-assign]
    provisioner._launch_llamacpp_runtime = AsyncMock()  # type: ignore[method-assign]
    provisioner._update_state = AsyncMock()  # type: ignore[method-assign]
    return provisioner, registry, etcd, state, writes


@pytest.mark.asyncio
async def test_relaunch_drains_then_commits_verified_runtime_with_same_lease() -> None:
    requested = _request()
    effective_runtime = _runtime(requested)
    provisioner, registry, etcd, state, writes = _provisioner()
    provisioner._launch_llamacpp_runtime.return_value = (  # type: ignore[attr-defined]
        "org/model-GGUF",
        effective_runtime,
    )

    async def observe_drained_state(_hostname: str) -> DrainOutcome:
        assert registry.get("host1").status is NodeStatus.RELAUNCHING  # type: ignore[union-attr]
        assert _node_from_value(state["record"].value).status is (
            NodeStatus.RELAUNCHING
        )
        return DrainOutcome.COMPLETE

    provisioner._drain_wait.side_effect = observe_drained_state  # type: ignore[attr-defined]

    await asyncio.wait_for(
        provisioner.relaunch_llamacpp("host1", requested),
        timeout=1,
    )

    assert writes == [
        (NodeStatus.RELAUNCHING, 7001),
        (NodeStatus.HEALTHY, 7001),
    ]
    current = registry.get("host1")
    assert current is not None
    assert current.status is NodeStatus.HEALTHY
    assert current.llamacpp_runtime == effective_runtime
    assert _node_from_value(state["record"].value) == current
    provisioner._upload_scripts.assert_awaited_once()  # type: ignore[attr-defined]
    provisioner._stop_llamacpp.assert_awaited_once_with("host1")  # type: ignore[attr-defined]
    etcd.refresh_lease.assert_called_once_with(7001)


@pytest.mark.asyncio
async def test_drain_timeout_restores_node_without_stopping_server() -> None:
    previous = _node()
    provisioner, registry, _etcd, state, writes = _provisioner(previous)
    provisioner._drain_wait.return_value = DrainOutcome.TIMED_OUT  # type: ignore[attr-defined]

    with pytest.raises(ProvisioningError, match="did not drain"):
        await asyncio.wait_for(
            provisioner.relaunch_llamacpp("host1", _request()),
            timeout=1,
        )

    assert writes == [
        (NodeStatus.RELAUNCHING, 7001),
        (NodeStatus.HEALTHY, 7001),
    ]
    restored = registry.get("host1")
    assert restored is not None
    assert restored.status is NodeStatus.HEALTHY
    assert restored.model == previous.model
    assert restored.llamacpp_runtime == previous.llamacpp_runtime
    assert _node_from_value(state["record"].value) == restored
    provisioner._stop_llamacpp.assert_not_awaited()  # type: ignore[attr-defined]
    provisioner._launch_llamacpp_runtime.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("previous_request", "restoration_message"),
    [
        pytest.param(
            _request(sizing=LlamaCppSizingMode.AUTO, fit_target_mib=768),
            "Automatic sizing restored",
            id="automatic",
        ),
        pytest.param(
            _request(
                fit_target_mib=768,
                context_per_slot=16384,
                slots=3,
                allow_estimator_overrun=True,
            ),
            "Previous custom sizing restored",
            id="custom",
        ),
    ],
)
@pytest.mark.asyncio
async def test_failed_request_rolls_back_previous_policy(
    previous_request: LlamaCppRuntimeRequest,
    restoration_message: str,
) -> None:
    previous = _node(runtime=_runtime(previous_request))
    provisioner, registry, _etcd, state, writes = _provisioner(previous)
    provisioner._launch_llamacpp_runtime.side_effect = [  # type: ignore[attr-defined]
        ProvisioningError("requested configuration does not fit"),
        ("org/model-GGUF", previous.llamacpp_runtime),
    ]

    with pytest.raises(ProvisioningError, match="does not fit"):
        await asyncio.wait_for(
            provisioner.relaunch_llamacpp("host1", _request()),
            timeout=1,
        )

    assert writes == [
        (NodeStatus.RELAUNCHING, 7001),
        (NodeStatus.HEALTHY, 7001),
    ]
    restored = registry.get("host1")
    assert restored is not None
    assert restored.status is NodeStatus.HEALTHY
    assert restored.model == previous.model
    assert restored.llamacpp_runtime == previous.llamacpp_runtime
    assert _node_from_value(state["record"].value) == restored
    assert provisioner._launch_llamacpp_runtime.await_args_list[1].args[2] == (  # type: ignore[attr-defined]
        previous_request
    )
    messages = [entry["msg"] for entry in provisioner.log_buffer.get_entries("host1")]
    assert restoration_message in messages


@pytest.mark.asyncio
async def test_cancellation_after_stop_rolls_back_before_propagating() -> None:
    previous = _node()
    provisioner, registry, _etcd, state, writes = _provisioner(previous)
    requested_launch_started = asyncio.Event()
    never_finishes: asyncio.Future[tuple[str, LlamaCppRuntimeState]] = asyncio.Future()

    async def launch_or_restore(
        _hostname: str,
        _artifact: ResolvedGGUFArtifact,
        request: LlamaCppRuntimeRequest,
    ) -> tuple[str, LlamaCppRuntimeState]:
        if request != previous.llamacpp_runtime.requested:  # type: ignore[union-attr]
            requested_launch_started.set()
            return await never_finishes
        return "org/model-GGUF", previous.llamacpp_runtime  # type: ignore[return-value]

    provisioner._launch_llamacpp_runtime.side_effect = launch_or_restore  # type: ignore[attr-defined]
    task = asyncio.create_task(provisioner.relaunch_llamacpp("host1", _request()))
    await asyncio.wait_for(requested_launch_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert writes == [
        (NodeStatus.RELAUNCHING, 7001),
        (NodeStatus.HEALTHY, 7001),
    ]
    restored = registry.get("host1")
    assert restored is not None
    assert restored.status is NodeStatus.HEALTHY
    assert _node_from_value(state["record"].value) == restored


@pytest.mark.asyncio
async def test_failed_request_and_rollback_enter_persistent_terminal_state() -> None:
    provisioner, registry, etcd, state, writes = _provisioner()
    provisioner._launch_llamacpp_runtime.side_effect = [  # type: ignore[attr-defined]
        ProvisioningError("requested configuration does not fit"),
        ProvisioningError("previous configuration no longer starts"),
    ]

    with pytest.raises(ProvisioningError, match="rollback failed"):
        await asyncio.wait_for(
            provisioner.relaunch_llamacpp("host1", _request()),
            timeout=1,
        )

    assert writes == [
        (NodeStatus.RELAUNCHING, 7001),
        (NodeStatus.RELAUNCH_FAILED, 0),
    ]
    current = registry.get("host1")
    assert current is not None
    assert current.status is NodeStatus.RELAUNCH_FAILED
    assert current.llamacpp_runtime is None
    assert state["record"].lease_id == 0
    assert _node_from_value(state["record"].value) == current
    etcd.revoke_lease.assert_called_once_with(7001)


@pytest.mark.asyncio
async def test_relaunch_starts_lease_keepalive_before_model_load() -> None:
    provisioner, _registry, _etcd, _state, _writes = _provisioner()
    keepalive_started = asyncio.Event()
    keepalive_cancelled = asyncio.Event()

    async def keepalive(_hostname: str, _lease_id: int) -> None:
        keepalive_started.set()
        try:
            await asyncio.Future()
        finally:
            keepalive_cancelled.set()

    async def launch(
        _hostname: str,
        _artifact: ResolvedGGUFArtifact,
        request: LlamaCppRuntimeRequest,
    ) -> tuple[str, LlamaCppRuntimeState]:
        # A real SSH/model load yields immediately; give the sibling keepalive
        # task the same scheduling opportunity before asserting it is live.
        await asyncio.sleep(0)
        assert keepalive_started.is_set()
        return "org/model-GGUF", _runtime(request)

    provisioner._keep_relaunch_lease = keepalive  # type: ignore[assignment]
    provisioner._launch_llamacpp_runtime.side_effect = launch  # type: ignore[attr-defined]

    await provisioner.relaunch_llamacpp("host1", _request())

    assert keepalive_cancelled.is_set()


@pytest.mark.asyncio
async def test_revision_conflict_before_stop_never_overwrites_or_stops() -> None:
    previous = _node()
    provisioner, registry, etcd, state, writes = _provisioner(previous)
    newer = previous.model_copy(update={"model": "changed-by-another-writer"})
    key, value = node_to_etcd(newer, "/nodes/")

    def conflict(
        _key: str,
        _value: bytes,
        **_kwargs: Any,
    ) -> None:
        state["record"] = EtcdRecord(
            key=key.encode(),
            value=value,
            mod_revision=11,
            lease_id=7001,
        )
        etcd.replace_if_revision.side_effect = lambda *_args, **_kw: None
        return None

    etcd.replace_if_revision.side_effect = conflict

    with pytest.raises(RelaunchConflictError, match="changed during relaunch"):
        await asyncio.wait_for(
            provisioner.relaunch_llamacpp("host1", _request()),
            timeout=1,
        )

    assert writes == []
    assert _node_from_value(state["record"].value) == newer
    assert registry.get("host1") == previous
    provisioner._stop_llamacpp.assert_not_awaited()  # type: ignore[attr-defined]


def test_relaunch_validation_uses_persisted_training_and_gpu_limits() -> None:
    provisioner, _registry, _etcd, _state, _writes = _provisioner()

    with pytest.raises(RelaunchValidationError, match="training context"):
        provisioner.validate_llamacpp_relaunch(
            _node(),
            _request(context_per_slot=262400),
        )
    with pytest.raises(RelaunchValidationError, match="total VRAM"):
        provisioner.validate_llamacpp_relaunch(
            _node(),
            _request(fit_target_mib=15360),
        )


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (None, "not registered"),
        (
            _node().model_copy(update={"status": NodeStatus.FAILED}),
            "requires a healthy node",
        ),
        (_node().model_copy(update={"managed": False}), "requires a managed"),
        (
            _node().model_copy(
                update={
                    "engine": InferenceEngine.VLLM,
                    "artifact_id": None,
                    "llamacpp_runtime": None,
                }
            ),
            "requires a managed",
        ),
        (
            _node().model_copy(update={"artifact_id": None, "llamacpp_runtime": None}),
            "artifact identity",
        ),
    ],
)
def test_relaunch_validation_rejects_ineligible_nodes(
    node: Node | None,
    message: str,
) -> None:
    provisioner, _registry, _etcd, _state, _writes = _provisioner()

    with pytest.raises(RelaunchPreconditionError, match=message):
        provisioner.validate_llamacpp_relaunch(node, _request())


def test_relaunch_validation_requires_connection_tracking() -> None:
    provisioner, _registry, _etcd, _state, _writes = _provisioner()
    provisioner._tracker = None

    assert provisioner.connection_tracking_available is False
    assert provisioner.connection_count("host1") == 0
    with pytest.raises(RelaunchPreconditionError, match="connection tracking"):
        provisioner.validate_llamacpp_relaunch(_node(), _request())


@pytest.mark.asyncio
async def test_relaunch_lease_is_granted_when_record_is_persistent() -> None:
    provisioner, _registry, etcd, state, _writes = _provisioner()
    record = state["record"]
    persistent = EtcdRecord(record.key, record.value, record.mod_revision, 0)

    assert await provisioner._prepare_relaunch_lease(persistent) == (7002, True)
    etcd.grant_node_lease.assert_called_once_with()


@pytest.mark.asyncio
async def test_relaunch_rejects_an_expired_managed_lease() -> None:
    provisioner, _registry, etcd, state, _writes = _provisioner()
    etcd.refresh_lease.return_value = -1

    with pytest.raises(RelaunchConflictError, match="lease has expired"):
        await provisioner._prepare_relaunch_lease(state["record"])


@pytest.mark.asyncio
async def test_keepalive_retries_errors_then_stops_after_lease_expiry() -> None:
    provisioner, _registry, etcd, _state, _writes = _provisioner()
    etcd.node_lease_ttl = 3
    etcd.refresh_lease.side_effect = [RuntimeError("etcd offline"), -1]

    with patch(
        "inference_proxy.provisioning.provisioner.asyncio.sleep",
        new=AsyncMock(side_effect=[None, None]),
    ) as sleep:
        await provisioner._keep_relaunch_lease("host1", 7001)

    assert sleep.await_args_list == [call(1.0), call(1.0)]
    assert etcd.refresh_lease.call_count == 2


@pytest.mark.asyncio
async def test_drain_wait_reports_complete_timeout_and_unavailable() -> None:
    provisioner, _registry, _etcd, _state, _writes = _provisioner()

    assert (
        await NodeProvisioner._drain_wait(provisioner, "host1") is DrainOutcome.COMPLETE
    )

    provisioner._tracker.get.return_value = 1  # type: ignore[union-attr]
    provisioner._settings = provisioner._settings.model_copy(
        update={"drain_timeout": 0}
    )
    assert (
        await NodeProvisioner._drain_wait(provisioner, "host1")
        is DrainOutcome.TIMED_OUT
    )

    provisioner._tracker = None
    assert (
        await NodeProvisioner._drain_wait(provisioner, "host1")
        is DrainOutcome.UNAVAILABLE
    )
