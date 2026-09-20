"""How a catalog profile travels through NodeProvisioner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.huggingface.artifacts import GGUFArtifact, ResolvedGGUFArtifact
from inference_proxy.models.endpoint import EndpointPolicy
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppRuntimeRequest,
    LlamaCppSizingMode,
    Node,
    NodeGPU,
    NodePlacement,
    NodeStatus,
)
from inference_proxy.placement.catalog import BUILTIN_PROFILES, GGUFRef
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningError,
    RelaunchPreconditionError,
)

QWEN38, _QWEN36, MUSE, GEMMA = BUILTIN_PROFILES
UUID = "GPU-33916087-1a73-3709-ee9c-2edf8788ee51"
PLACEMENT = NodePlacement(
    profile_id=QWEN38.profile_id, profile_version=QWEN38.version, claim_id="c" * 32
)


_L4_REQUEST = QWEN38.runtime_request(
    reserve_mib=256, draft_artifact_id=None, gpu_class="l4"
)
assert _L4_REQUEST.profile is not None
L4_PROFILE = _L4_REQUEST.profile


def _resolved(ref: GGUFRef, marker: str) -> ResolvedGGUFArtifact:
    return ResolvedGGUFArtifact(
        artifact=GGUFArtifact(
            artifact_id=marker * 64,
            repo_id=ref.repo_id,
            resolved_revision=ref.revision,
            files=(ref.filename,),
            entrypoint=ref.filename,
            model_alias=ref.repo_id,
            file_sizes={ref.filename: ref.size_bytes},
        ),
        node_relative_entrypoint=f"hub/x/snapshots/{ref.revision}/{ref.filename}",
    )


_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["host1"], allowed_networks=[], allowed_ports=[8000]
)


def _provisioner(etcd: MagicMock | None = None) -> NodeProvisioner:
    return NodeProvisioner(
        ssh_client=MagicMock(),
        etcd_client=etcd or MagicMock(),
        settings=ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0),
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_POLICY,
        nfs_export="nfs.example:/exports/huggingface",
    )


def test_mtp_profile_environment_is_complete_and_typed() -> None:
    request = QWEN38.runtime_request(
        reserve_mib=256, draft_artifact_id=None, gpu_class="l4"
    )

    env = _provisioner()._start_script_env(
        None,
        InferenceEngine.LLAMA_CPP,
        _resolved(QWEN38.target, "a"),
        llamacpp_request=request,
    )

    assert env["AUTOLLAMACPP_MANAGED_SIZING"] == "profile"
    assert env["AUTOLLAMACPP_FIT_TARGET_MIB"] == "256"
    assert env["AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT"] == "262144"
    assert env["AUTOLLAMACPP_MANAGED_PARALLEL"] == "1"
    assert env["AUTOLLAMACPP_MANAGED_CACHE_TYPE"] == "q4_0"
    profile_env = {
        k: v for k, v in env.items() if k.startswith("AUTOLLAMACPP_PROFILE_")
    }
    assert profile_env == {
        "AUTOLLAMACPP_PROFILE_ID": "qwen3.8-27b-24g",
        "AUTOLLAMACPP_PROFILE_VERSION": "1",
        "AUTOLLAMACPP_PROFILE_UBATCH": "256",
        "AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB": str(QWEN38.required_free_mib),
        "AUTOLLAMACPP_PROFILE_GPU_NAME": "NVIDIA L4",
        "AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB": "22900",
        "AUTOLLAMACPP_PROFILE_SPEC_TYPE": "draft-mtp",
        "AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX": "2",
        "AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE": "f16",
        "AUTOLLAMACPP_PROFILE_TEMPERATURE": "1.0",
        "AUTOLLAMACPP_PROFILE_TOP_P": "0.95",
        "AUTOLLAMACPP_PROFILE_TOP_K": "20",
        "AUTOLLAMACPP_PROFILE_MIN_P": "0.0",
        "AUTOLLAMACPP_PROFILE_PRESENCE_PENALTY": "0.0",
        "AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS": "0",
    }


def test_dflash_profile_environment_names_the_resolved_draft() -> None:
    draft = _resolved(MUSE.draft, "d") if MUSE.draft is not None else None
    assert draft is not None
    request = MUSE.runtime_request(
        reserve_mib=256, draft_artifact_id="d" * 64, gpu_class="l4"
    )

    env = _provisioner()._start_script_env(
        None,
        InferenceEngine.LLAMA_CPP,
        _resolved(MUSE.target, "c"),
        llamacpp_request=request,
        draft_artifact=draft,
    )

    assert env["AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH"] == draft.node_relative_entrypoint
    assert "AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE" not in env
    assert "AUTOLLAMACPP_PROFILE_MIN_P" not in env


def test_planner_requests_emit_no_profile_environment() -> None:
    request = LlamaCppRuntimeRequest(sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512)

    env = _provisioner()._start_script_env(
        None,
        InferenceEngine.LLAMA_CPP,
        _resolved(QWEN38.target, "a"),
        llamacpp_request=request,
    )

    assert not any(key.startswith("AUTOLLAMACPP_PROFILE_") for key in env)


@pytest.mark.parametrize("case", ["missing", "wrong", "unexpected", "planner"])
def test_a_draft_that_does_not_match_the_profile_is_refused(case: str) -> None:
    assert MUSE.draft is not None
    draft: ResolvedGGUFArtifact | None = _resolved(MUSE.draft, "d")
    request = MUSE.runtime_request(
        reserve_mib=256, draft_artifact_id="d" * 64, gpu_class="l4"
    )
    if case == "missing":
        draft = None
    elif case == "wrong":
        draft = _resolved(MUSE.draft, "e")
    elif case == "unexpected":
        request = QWEN38.runtime_request(
            reserve_mib=256, draft_artifact_id=None, gpu_class="l4"
        )
    else:
        request = LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512
        )

    with pytest.raises(ProvisioningError, match="draft artifact"):
        NodeProvisioner._profile_script_env(request, draft)


@pytest.mark.asyncio
async def test_gpu_inventory_is_read_from_the_node() -> None:
    provisioner = _provisioner()
    with patch.object(
        provisioner,
        "_ssh_run_command",
        new_callable=AsyncMock,
        return_value=f"0, {UUID}, NVIDIA L4, 23034\n",
    ):
        gpus = await provisioner._read_gpu_inventory("host1", profile=L4_PROFILE)

    assert gpus == (NodeGPU(index=0, uuid=UUID, name="NVIDIA L4", total_mib=23034),)


@pytest.mark.asyncio
async def test_a_profile_refuses_a_host_with_two_gpus() -> None:
    provisioner = _provisioner()
    rows = f"0, {UUID}, NVIDIA L4, 23034\n1, GPU-aaaaaaaa-bbbb, NVIDIA L4, 23034\n"
    with patch.object(
        provisioner, "_ssh_run_command", new_callable=AsyncMock, return_value=rows
    ):
        with pytest.raises(ProvisioningError, match="exactly one GPU per host"):
            await provisioner._read_gpu_inventory("host1", profile=L4_PROFILE)
        assert len(await provisioner._read_gpu_inventory("host1", profile=None)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"0, {UUID}, NVIDIA GeForce RTX 3090, 24576", "planned for a 'NVIDIA L4'"),
        (f"0, {UUID}, NVIDIA A30, 24576", "planned for a 'NVIDIA L4'"),
        (f"0, {UUID}, NVIDIA L4, 16384", "at least 22900 MiB"),
    ],
)
async def test_a_profile_refuses_a_gpu_other_than_the_one_it_was_planned_for(
    row: str, message: str
) -> None:
    """A stale QUADS string must not put an L4 profile on some other 24 GB card."""
    provisioner = _provisioner()
    with (
        patch.object(
            provisioner, "_ssh_run_command", new_callable=AsyncMock, return_value=row
        ),
        pytest.raises(ProvisioningError, match=message),
    ):
        await provisioner._read_gpu_inventory("host1", profile=L4_PROFILE)


def test_a_profile_cannot_be_requested_for_a_gpu_class_it_does_not_list() -> None:
    with pytest.raises(ValueError, match="does not run on GPU class 't4'"):
        QWEN38.runtime_request(reserve_mib=256, draft_artifact_id=None, gpu_class="t4")
    a30 = QWEN38.runtime_request(
        reserve_mib=256, draft_artifact_id=None, gpu_class="a30"
    )
    assert a30.profile is not None
    assert (a30.profile.gpu_name, a30.profile.gpu_min_total_mib) == (
        "NVIDIA A30",
        24000,
    )


@pytest.mark.asyncio
async def test_assigning_an_owner_takes_a_node_out_of_automation() -> None:
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    node = Node(
        node_id="host1",
        endpoint="http://host1:8000",
        status=NodeStatus.FAILED,
        managed=True,
        placement=PLACEMENT,
    )
    _, value = node_to_etcd(node, "/nodes/")
    etcd.get_record.return_value = MagicMock(
        key=b"/nodes/host1", value=value, mod_revision=5, lease_id=0
    )
    etcd.replace_if_revision.return_value = 6
    provisioner = _provisioner(etcd)

    updated = await provisioner.update_node_owner("host1", "new-owner@example.com")

    assert updated.owner == "new-owner@example.com"
    assert updated.placement is None
    # Clearing the owner of a node that is still a placement keeps the marker.
    assert (await provisioner.update_node_owner("host1", "")).placement == PLACEMENT


def _owner_etcd(value: bytes | None, *, revisions: list[int | None]) -> MagicMock:
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    etcd.get_record.return_value = (
        None
        if value is None
        else MagicMock(key=b"/nodes/host1", value=value, mod_revision=5, lease_id=0)
    )
    etcd.replace_if_revision.side_effect = revisions
    return etcd


def _placement_record() -> bytes:
    node = Node(
        node_id="host1",
        endpoint="http://host1:8000",
        status=NodeStatus.FAILED,
        managed=True,
        placement=PLACEMENT,
    )
    value = node_to_etcd(node, "/nodes/")[1]
    return value.encode() if isinstance(value, str) else value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("unknown", KeyError, "host1"),
        ("invalid", ProvisioningError, "invalid etcd registration"),
        ("contended", ProvisioningError, "kept changing during owner update"),
        ("vanished", ProvisioningError, "disappeared during owner update"),
    ],
)
async def test_a_failed_owner_update_releases_the_host(
    case: str, error: type[Exception], message: str
) -> None:
    if case == "unknown":
        etcd = _owner_etcd(None, revisions=[])
    elif case == "invalid":
        etcd = _owner_etcd(b"{not json", revisions=[])
    else:
        etcd = _owner_etcd(_placement_record(), revisions=[None, None, None])
    if case == "vanished":
        found = etcd.get_record.return_value
        etcd.get_record.side_effect = [found, None]
    provisioner = _provisioner(etcd)

    with pytest.raises(error, match=message):
        await provisioner.update_node_owner("host1", "new-owner@example.com")

    assert not provisioner.host_operation_in_progress("host1")
    assert provisioner._owner_updates == set()


@pytest.mark.asyncio
async def test_an_owner_update_survives_one_lost_revision_race() -> None:
    """A status write landed first: re-read, retry, and still free the host."""
    etcd = _owner_etcd(_placement_record(), revisions=[None, 7])
    provisioner = _provisioner(etcd)

    updated = await provisioner.update_node_owner("host1", "new-owner@example.com")

    assert (updated.owner, updated.placement) == ("new-owner@example.com", None)
    assert etcd.replace_if_revision.call_count == 2
    assert not provisioner.host_operation_in_progress("host1")


@pytest.mark.asyncio
async def test_an_owner_update_is_refused_while_the_host_is_reserved() -> None:
    etcd = _owner_etcd(_placement_record(), revisions=[6])
    provisioner = _provisioner(etcd)
    lease = await provisioner.try_reserve_host("host1")
    assert lease is not None

    with pytest.raises(ProvisioningError, match="lifecycle operation in progress"):
        await provisioner.update_node_owner("host1", "new-owner@example.com")
    etcd.get_record.assert_not_called()
    etcd.replace_if_revision.assert_not_called()
    assert provisioner.host_operation_in_progress("host1"), "the other holder's lease"

    lease.release()
    updated = await provisioner.update_node_owner("host1", "new-owner@example.com")
    assert updated.owner == "new-owner@example.com"
    assert not provisioner.host_operation_in_progress("host1")


@pytest.mark.asyncio
async def test_remote_lifecycle_processes_ignore_a_running_servers_log_sink() -> None:
    provisioner = _provisioner()
    ps = "\n".join(
        (
            "  101 sshd: root@notty",
            "  202 bash auto-llamacpp/setup.sh",
            "  203 bash auto-llamacpp/start-llamacpp.sh",
            "  204 python3 common/provision-logs.py worker",
            "  305 python3 /root/auto-llamacpp/../common/provision-logs.py engine",
            "  306 /opt/llama.cpp/v0.4.1-x/bin/llama-server --model /srv/m.gguf",
            "  407 ps -eo pid=,args=",
        )
    )
    with patch.object(
        provisioner, "_ssh_run_command", new_callable=AsyncMock, return_value=ps
    ):
        running = await provisioner.remote_lifecycle_processes("host1")

    assert [line.split()[0] for line in running] == ["202", "203", "204"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["", "garbage", "0, not-a-uuid, NVIDIA L4, 23034"])
async def test_unreadable_inventory_blocks_profiles_only(output: str) -> None:
    provisioner = _provisioner()
    with patch.object(
        provisioner, "_ssh_run_command", new_callable=AsyncMock, return_value=output
    ):
        assert await provisioner._read_gpu_inventory("host1", profile=None) == ()
        with pytest.raises(ProvisioningError, match="GPU inventory"):
            await provisioner._read_gpu_inventory("host1", profile=L4_PROFILE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"owner": "someone@example.com"},
        {"managed": False},
        {"llamacpp_request": None},
        {
            "llamacpp_request": LlamaCppRuntimeRequest(
                sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512
            )
        },
        {"placement": PLACEMENT.model_copy(update={"profile_version": 2})},
    ],
)
async def test_placement_only_provisions_a_managed_unowned_profile(
    kwargs: dict[str, object],
) -> None:
    provisioner = _provisioner()
    arguments: dict[str, object] = {
        "engine": InferenceEngine.LLAMA_CPP,
        "artifact_id": "a" * 64,
        "llamacpp_request": QWEN38.runtime_request(
            reserve_mib=256,
            draft_artifact_id=None,
            gpu_class="l4",
        ),
        "placement": PLACEMENT,
        **kwargs,
    }
    with (
        patch.object(
            provisioner,
            "resolve_artifact_selection",
            new_callable=AsyncMock,
            return_value=_resolved(QWEN38.target, "a"),
        ),
        patch.object(provisioner, "_provision", new_callable=AsyncMock) as body,
        pytest.raises(ProvisioningError, match="automatic placement"),
    ):
        await provisioner.provision("host1", **arguments)  # type: ignore[arg-type]

    body.assert_not_awaited()
    assert not provisioner.host_operation_in_progress("host1")


@pytest.mark.asyncio
async def test_the_draft_is_resolved_through_the_artifact_index() -> None:
    provisioner = _provisioner()
    assert MUSE.draft is not None
    target, draft = _resolved(MUSE.target, "c"), _resolved(MUSE.draft, "d")
    request = MUSE.runtime_request(
        reserve_mib=256, draft_artifact_id="d" * 64, gpu_class="l4"
    )
    with (
        patch.object(
            provisioner,
            "resolve_artifact_selection",
            new_callable=AsyncMock,
            side_effect=[target, draft],
        ) as resolve,
        patch.object(provisioner, "_provision", new_callable=AsyncMock) as body,
    ):
        await provisioner.provision(
            "host1",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="c" * 64,
            llamacpp_request=request,
        )

    assert [call.args for call in resolve.await_args_list] == [
        (InferenceEngine.LLAMA_CPP, "c" * 64),
        (InferenceEngine.LLAMA_CPP, "d" * 64),
    ]
    assert body.await_args is not None
    assert body.await_args.kwargs["draft_artifact"] == draft


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_every_node_record_of_a_placement_carries_its_claim(fails: bool) -> None:
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    puts: list[bytes] = []
    etcd.put.side_effect = lambda _key, value, **_kw: puts.append(value)

    def replace(_key: str, _old: bytes, new: bytes) -> bool:
        puts.append(new)
        return True

    etcd.replace.side_effect = replace
    etcd.grant_node_lease.return_value = 7
    provisioner = _provisioner(etcd)
    gpus = (NodeGPU(index=0, uuid=UUID, name="NVIDIA L4", total_mib=23034),)
    start = AsyncMock(return_value=QWEN38.target.repo_id)
    if fails:
        start.side_effect = ProvisioningError("llama-server exited")
    with (
        patch.object(provisioner, "_update_state", new_callable=AsyncMock),
        patch.object(provisioner, "_power_on_if_needed", new_callable=AsyncMock),
        patch.object(provisioner, "preflight", new_callable=AsyncMock),
        patch.object(provisioner, "_upload_scripts", new_callable=AsyncMock),
        patch.object(provisioner, "_run_setup", new_callable=AsyncMock),
        patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock),
        patch.object(
            provisioner,
            "_read_gpu_inventory",
            new_callable=AsyncMock,
            return_value=gpus,
        ) as inventory,
        patch.object(provisioner, "_run_start_vllm", start),
        patch.object(provisioner, "_poll_health", new_callable=AsyncMock),
        patch.object(
            provisioner,
            "_verify_llamacpp_runtime",
            new_callable=AsyncMock,
            return_value=None,
        ) as verify,
    ):
        run = provisioner._provision(
            "host1",
            engine=InferenceEngine.LLAMA_CPP,
            artifact=_resolved(QWEN38.target, "a"),
            llamacpp_request=QWEN38.runtime_request(
                reserve_mib=256,
                draft_artifact_id=None,
                gpu_class="l4",
            ),
            placement=PLACEMENT,
        )
        if fails:
            with pytest.raises(ProvisioningError):
                await run
        else:
            await run

    inventory.assert_awaited_once_with("host1", profile=L4_PROFILE)
    nodes: list[Node] = []
    for value in puts:
        node = node_from_etcd(
            b"/nodes/host1", value, "/nodes/", endpoint_policy=_POLICY
        )
        assert node is not None
        nodes.append(node)
    assert [n.status for n in nodes] == [
        NodeStatus.PROVISIONING,
        NodeStatus.FAILED if fails else NodeStatus.HEALTHY,
    ]
    assert all(n.placement == PLACEMENT and n.managed for n in nodes)
    if not fails:
        assert nodes[-1].gpus == gpus
        assert verify.await_args is not None
        assert verify.await_args.kwargs["gpus"] == gpus


def test_profile_nodes_refuse_a_custom_relaunch() -> None:
    provisioner = _provisioner()
    provisioner._tracker = MagicMock()
    runtime = MagicMock()
    runtime.requested.sizing = LlamaCppSizingMode.PROFILE
    node = MagicMock(
        status=NodeStatus.HEALTHY,
        managed=True,
        engine=InferenceEngine.LLAMA_CPP,
        artifact_id="a" * 64,
        llamacpp_runtime=runtime,
    )
    custom = LlamaCppRuntimeRequest(sizing=LlamaCppSizingMode.AUTO, fit_target_mib=512)

    with pytest.raises(RelaunchPreconditionError, match="automatic placement"):
        provisioner.validate_llamacpp_relaunch(node, custom)
