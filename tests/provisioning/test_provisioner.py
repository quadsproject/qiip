"""Unit tests for NodeProvisioner.

Tests mock SSHClient, EtcdClient, and httpx to verify the full
provisioning sequence: setup.sh -> start-vllm.sh -> health poll -> register.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import asyncssh
import httpx
import pytest

from inference_proxy.config.settings import (
    LLMFitSettings,
    ProvisioningSettings,
    RoutingSettings,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.artifacts import (
    GGUFArtifact,
    GGUFArtifactError,
    GGUFArtifactIndex,
    GGUFDownloadSpec,
    ResolvedGGUFArtifact,
)
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.models.endpoint import EndpointPolicy, EndpointValidationError
from inference_proxy.models.node import InferenceEngine, Node, NodeStatus
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    PreflightError,
    ProvisioningError,
    ProvisioningIdentity,
    _parse_llamacpp_runtime_fit,
)
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHConnectionError,
)
from inference_proxy.redfish.errors import RedfishError
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry

_TEST_ENDPOINT_POLICY = EndpointPolicy.from_values(
    allowed_hosts=["host1"],
    allowed_networks=[],
    allowed_ports=[8000],
)


def _artifact(alias: str = "org/model--with---separators") -> ResolvedGGUFArtifact:
    return ResolvedGGUFArtifact(
        artifact=GGUFArtifact(
            artifact_id="a" * 64,
            repo_id=alias,
            resolved_revision="b" * 40,
            files=("model -- Q4.gguf",),
            entrypoint="model -- Q4.gguf",
            model_alias=alias,
            file_sizes={"model -- Q4.gguf": 7},
        ),
        node_relative_entrypoint="hub/models--org--model/snapshots/"
        + "b" * 40
        + "/model -- Q4.gguf",
    )


def _llamacpp_fit_log(
    *,
    context_per_slot: int = 24576,
    slot_context_limit: int | None = None,
    slots: int = 4,
    runtime_slots: int | None = None,
    aggregate_context: int | None = None,
    runtime_aggregate_context: int | None = None,
    fit_target_mib: int = 1024,
    kv_unified: bool = True,
    gpu_layers: int = 37,
    total_layers: int = 37,
    context_sharing_warnings: bool = True,
) -> str:
    if slot_context_limit is None:
        slot_context_limit = context_per_slot
    if runtime_slots is None:
        runtime_slots = slots
    if aggregate_context is None:
        aggregate_context = context_per_slot * slots
    if runtime_aggregate_context is None:
        runtime_aggregate_context = aggregate_context
    lines = [
        "qiip_fit_plan: "
        f"context_per_slot={context_per_slot} slots={slots} "
        f"aggregate_context={aggregate_context} "
        f"fit_target_mib={fit_target_mib}",
        f"llama_model_load: offloaded {gpu_layers}/{total_layers} layers to GPU",
        f"llama_context: n_ctx = {runtime_aggregate_context}",
    ]
    if context_sharing_warnings and aggregate_context > slot_context_limit:
        lines.extend(
            (
                "llama_context: n_ctx_seq "
                f"({aggregate_context}) > n_ctx_train ({slot_context_limit}) "
                "-- possible training context overflow",
                "srv init: the slot context "
                f"({aggregate_context}) exceeds the training context of the model "
                f"({slot_context_limit}) - capping",
            )
        )
    lines.append(
        "srv init: initializing, "
        f"n_slots = {runtime_slots}, n_ctx_slot = {slot_context_limit}, "
        f"kv_unified = '{str(kv_unified).lower()}'"
    )
    return "\n".join(lines)


def _make_provisioner(
    *,
    ssh_client: MagicMock | None = None,
    etcd_client: MagicMock | None = None,
    settings: ProvisioningSettings | None = None,
    llmfit_settings: LLMFitSettings | None = None,
    registry: NodeRegistry | MagicMock | None = None,
    connection_tracker: MagicMock | None = None,
    circuit_breaker_registry: CircuitBreakerRegistry | MagicMock | None = None,
    redfish_client: MagicMock | None = None,
    endpoint_policy: EndpointPolicy = _TEST_ENDPOINT_POLICY,
    nfs_export: str | None = "nfs.example:/exports/huggingface",
    hf_token: str | None = None,
    artifact_index: GGUFArtifactIndex | None = None,
) -> NodeProvisioner:
    """Build a NodeProvisioner with mock dependencies."""
    return NodeProvisioner(
        ssh_client=ssh_client or MagicMock(),
        etcd_client=etcd_client or MagicMock(),
        settings=settings
        or ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0),
        llmfit_settings=llmfit_settings or LLMFitSettings(),
        endpoint_policy=endpoint_policy,
        registry=registry,
        connection_tracker=connection_tracker,
        circuit_breaker_registry=circuit_breaker_registry,
        redfish_client=redfish_client,
        hf_token=hf_token,
        nfs_export=nfs_export,
        artifact_index=artifact_index,
    )


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_owned_tasks() -> None:
    """C2: shutdown awaits cancellation and releases host task ownership."""
    provisioner = _make_provisioner()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def owned_task() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task: asyncio.Task[None] | None = None
    try:
        task = provisioner.fire_background(
            owned_task(),
            provisioning_hostname="host1",
            provisioning_identity=ProvisioningIdentity(InferenceEngine.VLLM),
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(provisioner.shutdown(), timeout=1)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert task is not None
    assert task.cancelled() is True
    assert cleaned.is_set()
    assert provisioner._background_tasks == set()
    assert provisioner._provisioning_tasks == {}


@pytest.mark.asyncio
async def test_llmfit_version_single_source() -> None:
    """E5: provisioning and repair install consume one LLMFit setting."""
    settings = LLMFitSettings(version="8.7.6", sha256="a" * 64)
    provisioner = _make_provisioner(llmfit_settings=settings)
    ssh = MagicMock()
    ssh.run = AsyncMock(return_value=("", "", 0))
    runner = LLMFitRunner(ssh_client=ssh, settings=settings)

    await runner._install("host1")

    assert provisioner._setup_script_env()["AUTOVLLM_LLMFIT_VERSION"] == "8.7.6"
    assert provisioner._setup_script_env()["AUTOVLLM_LLMFIT_SHA256"] == "a" * 64
    install_command = ssh.run.await_args.args[1]
    assert "/v8.7.6/llmfit-v8.7.6-" in install_command
    assert "a" * 64 in install_command
    assert "llmfit_version" not in ProvisioningSettings.model_fields


def test_script_env_prefix_exact() -> None:
    """E7/E9: each remote script receives only its ordered allowlist."""
    provisioner = _make_provisioner(
        settings=ProvisioningSettings(
            vllm_port=8123,
            nfs_mount_point="/srv/hf cache",
            nvidia_driver_version="999.1",
            nvidia_driver_sha256="b" * 64,
            llamacpp_fit_target_mib=1536,
        ),
        llmfit_settings=LLMFitSettings(version="8.7.6", sha256="c" * 64),
        nfs_export="nfs.example:/exports/hf cache",
        hf_token="hf secret",
    )

    assert provisioner._setup_script_env() == {
        "AUTOVLLM_NFS_EXPORT": "nfs.example:/exports/hf cache",
        "AUTOVLLM_NFS_MOUNT_POINT": "/srv/hf cache",
        "AUTOVLLM_NVIDIA_DRIVER_VERSION": "999.1",
        "AUTOVLLM_NVIDIA_DRIVER_SHA256": "b" * 64,
        "AUTOVLLM_API_PORT": "8123",
        "AUTOVLLM_LLMFIT_VERSION": "8.7.6",
        "AUTOVLLM_LLMFIT_SHA256": "c" * 64,
    }
    assert provisioner._start_script_env("org/model") == {
        "AUTOVLLM_NFS_MOUNT_POINT": "/srv/hf cache",
        "AUTOVLLM_API_PORT": "8123",
        "AUTOVLLM_MODEL": "org/model",
        "HF_TOKEN": "hf secret",
    }
    llama_setup = provisioner._setup_script_env(InferenceEngine.LLAMA_CPP)
    assert llama_setup == {
        "AUTOVLLM_NFS_EXPORT": "nfs.example:/exports/hf cache",
        "AUTOVLLM_NFS_MOUNT_POINT": "/srv/hf cache",
        "AUTOVLLM_NVIDIA_DRIVER_VERSION": "999.1",
        "AUTOVLLM_NVIDIA_DRIVER_SHA256": "b" * 64,
        "AUTOVLLM_API_PORT": "8123",
        "AUTOVLLM_LLMFIT_VERSION": "8.7.6",
        "AUTOVLLM_LLMFIT_SHA256": "c" * 64,
        "AUTOLLAMACPP_VERSION": "b10242",
        "AUTOLLAMACPP_SHA256": (
            "b5c2b0d09d2af9988e47570f7f96e8473b4e07fad2c99f6e2e0745e5b3935fe3"
        ),
        "AUTOLLAMACPP_SOURCE_URL": (
            "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10242.tar.gz"
        ),
    }
    artifact = _artifact()
    assert provisioner._start_script_env(None, InferenceEngine.LLAMA_CPP, artifact) == {
        "AUTOLLAMACPP_NFS_MOUNT_POINT": "/srv/hf cache",
        "AUTOLLAMACPP_PORT": "8123",
        "AUTOLLAMACPP_REQUIRE_CUDA": "1",
        "AUTOLLAMACPP_MANAGED": "1",
        "AUTOLLAMACPP_FIT_TARGET_MIB": "1536",
        "AUTOLLAMACPP_GGUF_PATH": artifact.node_relative_entrypoint,
        "AUTOLLAMACPP_MODEL_ALIAS": artifact.model_alias,
        "HF_TOKEN": "hf secret",
    }
    assert shlex.split(
        provisioner._script_command("setup.sh", env=provisioner._setup_script_env())
    ) == [
        "AUTOVLLM_NFS_EXPORT=nfs.example:/exports/hf cache",
        "AUTOVLLM_NFS_MOUNT_POINT=/srv/hf cache",
        "AUTOVLLM_NVIDIA_DRIVER_VERSION=999.1",
        f"AUTOVLLM_NVIDIA_DRIVER_SHA256={'b' * 64}",
        "AUTOVLLM_API_PORT=8123",
        "AUTOVLLM_LLMFIT_VERSION=8.7.6",
        f"AUTOVLLM_LLMFIT_SHA256={'c' * 64}",
        "bash",
        "auto-vllm/setup.sh",
    ]


@pytest.mark.asyncio
async def test_native_artifact_resolves_through_managed_launcher(
    tmp_path: Path,
) -> None:
    """Compose discovery, setup selection, environment, and node resolution."""
    repo_id = "org/model-with-separators-GGUF"
    revision = "c" * 40
    files = (
        "quant/model -- q4-00001-of-00002.gguf",
        "quant/model -- q4-00002-of-00002.gguf",
    )
    shared_root = tmp_path / "gateway shared root"
    cache_dir = shared_root / "hub"
    node_mount = tmp_path / "node mount"

    def populate_cache(root: Path) -> None:
        repo = root / "hub" / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo / "snapshots" / revision
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        for relative in files:
            content = relative.encode()
            blob = blobs / hashlib.sha256(content).hexdigest()
            blob.write_bytes(content)
            source = snapshot / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.symlink_to(Path(os.path.relpath(blob, source.parent)))
        refs = repo / "refs"
        refs.mkdir()
        (refs / "main").write_text(revision, encoding="utf-8")

    populate_cache(shared_root)
    populate_cache(node_mount)
    artifact_index = GGUFArtifactIndex(
        cache_dir,
        shared_root=shared_root,
    )
    published = artifact_index.artifact_from_download(
        repo_id=repo_id,
        resolved_revision=revision,
        snapshot_path=(
            cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        ),
        spec=GGUFDownloadSpec(files=files, entrypoint=files[0]),
    )
    provisioner = _make_provisioner(
        settings=ProvisioningSettings(nfs_mount_point=str(node_mount)),
        artifact_index=artifact_index,
    )

    selected = await provisioner.resolve_artifact_selection(
        InferenceEngine.LLAMA_CPP, published.artifact_id
    )
    assert selected is not None
    assert selected.artifact == published
    env = {
        **os.environ,
        **provisioner._start_script_env(None, InferenceEngine.LLAMA_CPP, selected),
    }
    start_script = (
        Path(__file__).resolve().parents[2] / "auto-llamacpp" / "start-llamacpp.sh"
    )
    command = "\n".join(
        (
            f"source {shlex.quote(str(start_script))}",
            "resolve_gguf_artifact",
            'printf "%s\\n%s\\n" "$GGUF_PATH" "$MODEL_ALIAS"',
        )
    )

    result = subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    expected_entrypoint = node_mount / selected.node_relative_entrypoint
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(expected_entrypoint), repo_id]
    assert expected_entrypoint.is_symlink()


@pytest.mark.asyncio
async def test_artifact_resolution_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_root = tmp_path / "shared root"
    cache_dir = shared_root / "hub"
    cache_dir.mkdir(parents=True)
    artifact_index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)
    started = threading.Event()
    release = threading.Event()

    def blocking_get(_artifact_id: str) -> ResolvedGGUFArtifact:
        started.set()
        if not release.wait(timeout=0.5):
            raise AssertionError("artifact lookup blocked the event loop")
        return _artifact()

    monkeypatch.setattr(artifact_index, "get", blocking_get)
    provisioner = _make_provisioner(artifact_index=artifact_index)
    resolution = asyncio.create_task(
        provisioner.resolve_artifact_selection(
            InferenceEngine.LLAMA_CPP,
            "a" * 64,
        )
    )
    try:
        deadline = asyncio.get_running_loop().time() + 0.2
        while not started.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        assert started.is_set()
        assert not resolution.done()
        await asyncio.sleep(0)
    finally:
        release.set()

    assert await asyncio.wait_for(resolution, timeout=0.2) == _artifact()


def test_env_prefix_quoting() -> None:
    """E10: hostile values round-trip through one command assembly path."""
    provisioner = _make_provisioner(
        settings=ProvisioningSettings(scripts_dir=Path("bundles/provision scripts")),
        hf_token="token '$(touch nope)'",
    )
    model = "org/model; printf 'unsafe'"

    command = provisioner._script_command(
        "start-vllm.sh", env=provisioner._start_script_env(model)
    )
    words = shlex.split(command)

    assert words == [
        "AUTOVLLM_NFS_MOUNT_POINT=/srv/hf-cache",
        "AUTOVLLM_API_PORT=8000",
        f"AUTOVLLM_MODEL={model}",
        "HF_TOKEN=token '$(touch nope)'",
        "bash",
        "provision scripts/start-vllm.sh",
    ]
    assert not command.endswith(" ")


def test_cache_paths_resolve_to_same_export() -> None:
    """E8: distinct mount paths share one declared backing export."""
    provisioner = _make_provisioner(
        settings=ProvisioningSettings(nfs_mount_point="/srv/hf-cache"),
        nfs_export="storage.example:/exports/huggingface",
    )

    assert provisioner._setup_script_env()["AUTOVLLM_NFS_EXPORT"] == (
        "storage.example:/exports/huggingface"
    )
    assert provisioner._setup_script_env()["AUTOVLLM_NFS_MOUNT_POINT"] == (
        "/srv/hf-cache"
    )
    assert "nfs_server" not in ProvisioningSettings.model_fields


@pytest.mark.asyncio
async def test_missing_nfs_export_fails_before_remote_work() -> None:
    """E3: proxy-only startup is valid, but setup fails before side effects."""
    ssh = MagicMock()
    etcd = MagicMock()
    provisioner = _make_provisioner(
        ssh_client=ssh,
        etcd_client=etcd,
        nfs_export=None,
    )

    with patch.object(provisioner, "_provision", new_callable=AsyncMock) as body:
        with pytest.raises(ProvisioningError, match="HUGGINGFACE__NFS_EXPORT"):
            await provisioner.provision("host1")
        body.assert_not_awaited()

    ssh.upload.assert_not_called()
    ssh.run_streaming.assert_not_called()
    etcd.put.assert_not_called()


@pytest.mark.asyncio
async def test_missing_engine_bundle_fails_before_remote_work(tmp_path: Path) -> None:
    ssh = MagicMock()
    provisioner = _make_provisioner(
        ssh_client=ssh,
        settings=ProvisioningSettings(scripts_dir=tmp_path / "auto-vllm"),
    )

    with patch.object(provisioner, "_provision", new_callable=AsyncMock) as body:
        with pytest.raises(ProvisioningError, match="bundle is incomplete"):
            await provisioner.provision("host1", engine=InferenceEngine.LLAMA_CPP)
        body.assert_not_awaited()

    ssh.upload.assert_not_called()
    ssh.run_streaming.assert_not_called()


@pytest.mark.asyncio
async def test_llamacpp_bundle_uses_configured_sibling_root(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle -- root with spaces"
    vllm_dir = bundle_root / "renamed vllm bundle"
    llama_dir = bundle_root / "auto-llamacpp"
    common_dir = bundle_root / "common"
    shutil.copytree(Path("auto-vllm"), vllm_dir)
    shutil.copytree(Path("auto-llamacpp"), llama_dir)
    shutil.copytree(Path("common"), common_dir)
    ssh = MagicMock()
    ssh.upload = AsyncMock()
    provisioner = _make_provisioner(
        ssh_client=ssh,
        settings=ProvisioningSettings(scripts_dir=vllm_dir),
    )

    await provisioner._upload_scripts("host1", InferenceEngine.LLAMA_CPP)

    assert [item.args for item in ssh.upload.await_args_list] == [
        ("host1", llama_dir),
        ("host1", common_dir),
    ]


@pytest.mark.asyncio
async def test_llamacpp_setup_uses_its_extended_total_timeout() -> None:
    seen: list[tuple[str, str, float | None]] = []

    async def mock_streaming(
        host: str,
        command: str,
        *,
        total_timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        seen.append((host, command, total_timeout))
        yield ("stdout", "[STEP:llamacpp_install:OK]")

    ssh = MagicMock()
    ssh.run_streaming = mock_streaming
    provisioner = _make_provisioner(
        ssh_client=ssh,
        settings=ProvisioningSettings(llamacpp_setup_timeout=4321),
    )

    await provisioner._run_setup(
        "host1",
        started_at=datetime.now(UTC),
        on_step=lambda _step: None,
        engine=InferenceEngine.LLAMA_CPP,
    )

    assert seen[0][0] == "host1"
    assert "auto-llamacpp/setup.sh" in seen[0][1]
    assert seen[0][2] == 4321


async def _async_iter(
    items: list[tuple[str, str]],
) -> AsyncIterator[tuple[str, str]]:
    """Helper: async generator yielding items."""
    for item in items:
        yield item


def _recording_etcd() -> tuple[MagicMock, dict[str, bytes], list[dict[str, object]]]:
    """Return an etcd double which implements put/replace/delete semantics."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    values: dict[str, bytes] = {}
    state_payloads: list[dict[str, object]] = []

    def put(
        key: str,
        value: bytes,
        *,
        lease_id: int | None = None,
    ) -> bool:
        del lease_id
        values[key] = value
        if key.startswith("/provisioning/"):
            state_payloads.append(json.loads(value))
        return True

    def replace(key: str, expected: bytes, value: bytes) -> bool:
        if values.get(key) != expected:
            return False
        values[key] = value
        return True

    def delete(key: str) -> bool:
        return values.pop(key, None) is not None

    etcd.put = MagicMock(side_effect=put)
    etcd.grant_node_lease = MagicMock(return_value=7001)
    etcd.replace = MagicMock(side_effect=replace)
    etcd.delete = MagicMock(side_effect=delete)
    return etcd, values, state_payloads


class TestProvisionSequence:
    """provision() orchestrates setup -> start -> poll -> register in order."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_calls_in_order(self, mock_httpx_cls: MagicMock) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        call_order: list[str] = []

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            if "setup.sh" in command:
                call_order.append("setup")
                for item in [
                    ("stdout", "[STEP:system_update:START]"),
                    ("stdout", "[STEP:system_update:OK]"),
                ]:
                    yield item
            elif "start-vllm.sh" in command:
                call_order.append("start_vllm")
                for item in [
                    ("stdout", "# Model:              Qwen/Qwen2.5-72B-Instruct")
                ]:
                    yield item

        ssh.run_streaming = mock_streaming
        ssh.upload = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.return_value = True
                    await provisioner.provision("host1")
                    call_order.append("register")

        assert "setup" in call_order
        assert "start_vllm" in call_order
        assert call_order.index("setup") < call_order.index("start_vllm")


class TestProvisionEndpointPolicy:
    """Provisioning cannot create a node discovery would reject."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("hostname", "expected_setting"),
        [
            ("gpu01", "routing.allowed_endpoint_hosts"),
            ("10.0.1.100", "routing.allowed_endpoint_networks"),
        ],
    )
    async def test_default_policy_rejects_lab_host_before_remote_work(
        self,
        hostname: str,
        expected_setting: str,
    ) -> None:
        ssh = MagicMock()
        ssh.upload = AsyncMock()
        etcd = MagicMock()
        provisioner = _make_provisioner(
            ssh_client=ssh,
            etcd_client=etcd,
            endpoint_policy=RoutingSettings().endpoint_policy(),
        )

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            new_callable=AsyncMock,
        ) as open_connection:
            with pytest.raises(EndpointValidationError) as caught:
                await provisioner.provision(hostname)

        message = str(caught.value)
        assert hostname in message
        assert expected_setting in message
        open_connection.assert_not_awaited()
        ssh.upload.assert_not_awaited()
        ssh.run_streaming.assert_not_called()
        etcd.put.assert_not_called()


@pytest.mark.asyncio
async def test_llamacpp_without_artifact_fails_before_remote_work() -> None:
    ssh = MagicMock()
    etcd = MagicMock()
    provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

    with patch.object(provisioner, "_provision", new_callable=AsyncMock) as body:
        with pytest.raises(ProvisioningError, match="requires artifact_id"):
            await provisioner.provision("host1", engine=InferenceEngine.LLAMA_CPP)
        body.assert_not_awaited()

    ssh.upload.assert_not_called()
    ssh.run_streaming.assert_not_called()
    etcd.put.assert_not_called()


@pytest.mark.asyncio
async def test_llamacpp_shared_root_is_required_before_remote_work(
    tmp_path: Path,
) -> None:
    ssh = MagicMock()
    etcd = MagicMock()
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()
    provisioner = _make_provisioner(
        ssh_client=ssh,
        etcd_client=etcd,
        artifact_index=GGUFArtifactIndex(cache_dir),
    )

    with patch.object(provisioner, "_provision", new_callable=AsyncMock) as body:
        with pytest.raises(ProvisioningError, match="HUGGINGFACE__SHARED_ROOT"):
            await provisioner.provision(
                "host1",
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id="a" * 64,
            )
        body.assert_not_awaited()

    ssh.upload.assert_not_called()
    ssh.run_streaming.assert_not_called()
    etcd.put.assert_not_called()


@pytest.mark.asyncio
async def test_artifact_selection_rejects_cross_engine_and_lookup_failures(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProvisioningError, match="only valid for llama_cpp"):
        await _make_provisioner().resolve_artifact_selection(
            InferenceEngine.VLLM, "a" * 64
        )

    with pytest.raises(ProvisioningError, match="discovery is not configured"):
        await _make_provisioner().resolve_artifact_selection(
            InferenceEngine.LLAMA_CPP, "a" * 64
        )

    artifact_index = MagicMock(spec=GGUFArtifactIndex)
    artifact_index.validate_shared_root.return_value = tmp_path
    artifact_index.get.return_value = None
    provisioner = _make_provisioner(artifact_index=artifact_index)
    with pytest.raises(ProvisioningError, match="was not found"):
        await provisioner.resolve_artifact_selection(
            InferenceEngine.LLAMA_CPP, "b" * 64
        )

    artifact_index.get.side_effect = GGUFArtifactError("broken cache entry")
    with pytest.raises(ProvisioningError, match="invalid: broken cache entry"):
        await provisioner.resolve_artifact_selection(
            InferenceEngine.LLAMA_CPP, "c" * 64
        )


class TestScriptUpload:
    """Scripts are uploaded to remote host before setup."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_upload_called_before_setup(self, mock_httpx_cls: MagicMock) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        call_order: list[str] = []

        async def mock_upload(
            host: str,
            local_path: Path,
            remote_path: str = ".",
        ) -> None:
            call_order.append("upload")

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            if "setup.sh" in command:
                call_order.append("setup")
                for item in [("stdout", "[STEP:system_update:START]")]:
                    yield item
            elif "start-vllm.sh" in command:
                for item in [
                    ("stdout", "# Model:              Qwen/Qwen2.5-72B-Instruct")
                ]:
                    yield item

        ssh.upload = mock_upload
        ssh.run_streaming = mock_streaming

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_tt:
                    mock_tt.return_value = True
                    await provisioner.provision("host1")

        assert call_order.index("upload") < call_order.index("setup")

    @pytest.mark.asyncio
    async def test_upload_failure_sets_failed_state(self) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        async def mock_upload(
            host: str,
            local_path: Path,
            remote_path: str = ".",
        ) -> None:
            raise SSHConnectionError("host1", "upload failed")

        ssh.upload = mock_upload
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_tt:
                mock_tt.return_value = True
                with pytest.raises(SSHConnectionError):
                    await provisioner.provision("host1")


class TestStepMarkerParsing:
    """D-05, D-06: Step markers parsed from setup.sh stdout."""

    @pytest.mark.asyncio
    async def test_parses_step_markers(self) -> None:
        ssh = MagicMock()

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            for item in [
                ("stdout", "[STEP:system_update:START]"),
                ("stdout", "some debug output"),
                ("stdout", "[STEP:system_update:OK]"),
                ("stdout", "[STEP:system_update:START]"),
                ("stdout", "[STEP:system_update:FAIL]"),
                ("stderr", "error details"),
            ]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        # _run_setup should not raise on FAIL markers -- that's a logging concern.
        # RemoteCommandError from SSHClient is what signals actual failure.
        await provisioner._run_setup(
            "host1",
            started_at=datetime.now(UTC),
            on_step=lambda _step: None,
        )


class TestModelExtraction:
    """Model name extracted from start-vllm.sh stdout."""

    @pytest.mark.asyncio
    async def test_extracts_model_name(self) -> None:
        ssh = MagicMock()

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            for item in [
                ("stdout", "Starting container..."),
                ("stdout", "# Model:              Qwen/Qwen2.5-72B-Instruct"),
                ("stdout", "Container started"),
            ]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        model = await provisioner._run_start_vllm("host1")
        assert model == "Qwen/Qwen2.5-72B-Instruct"

    @pytest.mark.asyncio
    async def test_raises_on_missing_model(self) -> None:
        ssh = MagicMock()

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            for item in [("stdout", "no model line here")]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        with pytest.raises(ProvisioningError, match="model name not found"):
            await provisioner._run_start_vllm("host1")

    @pytest.mark.asyncio
    async def test_llamacpp_reported_alias_must_match_selected_artifact(self) -> None:
        ssh = MagicMock()

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            yield ("stdout", "# Model:              wrong/model")

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        with pytest.raises(ProvisioningError, match="expected selected artifact"):
            await provisioner._run_start_vllm(
                "host1",
                engine=InferenceEngine.LLAMA_CPP,
                artifact=_artifact(),
            )

    @pytest.mark.asyncio
    async def test_includes_model_in_start_environment(self) -> None:
        """The model uses the same ordered start environment as other inputs."""
        ssh = MagicMock()
        captured_commands: list[str] = []

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            captured_commands.append(command)
            for item in [("stdout", "# Model:              org/model")]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        await provisioner._run_start_vllm("host1", model="org/model")
        assert len(captured_commands) == 1
        assert "AUTOVLLM_MODEL=" in captured_commands[0]
        assert "org/model" in captured_commands[0]
        assert captured_commands[0].endswith("bash auto-vllm/start-vllm.sh")

    @pytest.mark.asyncio
    async def test_omits_env_var_when_model_none(self) -> None:
        """When model is None, command is plain start-vllm.sh."""
        ssh = MagicMock()
        captured_commands: list[str] = []

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            captured_commands.append(command)
            for item in [("stdout", "# Model:              auto-detected")]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        await provisioner._run_start_vllm("host1")
        assert captured_commands[0].endswith("bash auto-vllm/start-vllm.sh")

    @pytest.mark.asyncio
    async def test_quotes_model_with_special_chars(self) -> None:
        """Shell-unsafe characters in model name are quoted via shlex.quote()."""
        ssh = MagicMock()
        captured_commands: list[str] = []

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            captured_commands.append(command)
            for item in [("stdout", "# Model:              safe")]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        await provisioner._run_start_vllm("host1", model="model; rm -rf /")
        cmd = captured_commands[0]
        # shlex.quote wraps in single quotes so the shell treats it as one token
        assert "AUTOVLLM_MODEL='model; rm -rf /'" in cmd
        assert "bash auto-vllm/start-vllm.sh" in cmd


class TestHealthPoll:
    """D-10, D-09: Health polling via httpx."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_success_on_200(self, mock_httpx_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        provisioner = _make_provisioner()
        await provisioner._poll_health("host1")

        mock_client.get.assert_called_with("http://host1:8000/health")

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_timeout_raises(self, mock_httpx_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        settings = ProvisioningSettings(health_poll_timeout=0, health_poll_interval=0)
        provisioner = _make_provisioner(settings=settings)

        with pytest.raises(ProvisioningError, match="timed out"):
            await provisioner._poll_health("host1")


class TestNodeRegistration:
    """D-11, D-12: Node registered in etcd with correct fields."""

    @pytest.mark.asyncio
    async def test_registers_with_correct_fields(self) -> None:
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        provisioner = _make_provisioner(etcd_client=etcd)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_to_thread:
            mock_to_thread.side_effect = [7001, True]
            with patch(
                "inference_proxy.provisioning.provisioner.node_to_etcd"
            ) as mock_serialize:
                mock_serialize.return_value = ("/nodes/host1", b'{"model":"test"}')
                await provisioner._register_node(
                    "host1",
                    "test-model",
                    engine=InferenceEngine.LLAMA_CPP,
                    artifact_id="a" * 64,
                )

                # Verify Node was constructed correctly
                call_args = mock_serialize.call_args
                node = call_args[0][0]
                assert node.node_id == "host1"
                assert node.status == NodeStatus.HEALTHY
                assert node.model == "test-model"
                assert node.endpoint == "http://host1:8000"
                assert node.last_heartbeat is not None
                assert node.engine is InferenceEngine.LLAMA_CPP
                assert node.artifact_id == "a" * 64

                # Each managed registration gets a fresh lease before its PUT.
                assert mock_to_thread.call_args_list == [
                    call(etcd.grant_node_lease),
                    call(
                        etcd.put,
                        "/nodes/host1",
                        b'{"model":"test"}',
                        lease_id=7001,
                    ),
                ]

    @pytest.mark.asyncio
    async def test_unmanaged_registration_remains_unleased(self) -> None:
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        provisioner = _make_provisioner(etcd_client=etcd)

        await provisioner._register_node("host1", "test-model", managed=False)

        etcd.grant_node_lease.assert_not_called()
        etcd.put.assert_called_once()
        assert "lease_id" not in etcd.put.call_args.kwargs

    @pytest.mark.asyncio
    async def test_managed_healthy_registration_uses_unique_lease(self) -> None:
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.grant_node_lease.side_effect = [7001, 7002]
        provisioner = _make_provisioner(
            etcd_client=etcd,
            endpoint_policy=EndpointPolicy.from_values(
                allowed_hosts=["host1", "host2"],
                allowed_networks=[],
                allowed_ports=[8000],
            ),
        )

        await provisioner._register_node("host1", "test-model")
        await provisioner._register_node("host2", "test-model")

        assert etcd.put.call_args_list[0].kwargs["lease_id"] == 7001
        assert etcd.put.call_args_list[1].kwargs["lease_id"] == 7002

    @pytest.mark.asyncio
    async def test_lease_grant_failure_falls_back_to_unleased_registration(
        self,
    ) -> None:
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.grant_node_lease.side_effect = RuntimeError("etcd lease unavailable")
        provisioner = _make_provisioner(etcd_client=etcd)

        await provisioner._register_node("host1", "test-model")

        etcd.put.assert_called_once()
        assert "lease_id" not in etcd.put.call_args.kwargs


class TestSetupFailure:
    """D-08: setup failures retain their original typed exception."""

    @pytest.mark.asyncio
    async def test_remote_command_error_wraps(self) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            yield ("stdout", "[STEP:system_update:START]")
            raise RemoteCommandError("host1", "bash setup.sh", 1)

        ssh.run_streaming = mock_streaming
        ssh.upload = AsyncMock()
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread:
                mock_to_thread.return_value = True
                with pytest.raises(RemoteCommandError):
                    await provisioner.provision("host1")

    @pytest.mark.asyncio
    async def test_ssh_connection_error_wraps(self) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            raise SSHConnectionError("host1", "connection refused")
            # Make it an async generator
            yield  # pragma: no cover

        ssh.run_streaming = mock_streaming
        ssh.upload = AsyncMock()
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread:
                mock_to_thread.return_value = True
                with pytest.raises(SSHConnectionError):
                    await provisioner.provision("host1")


class TestProvisioningFailureAccuracy:
    """P3/P7/P8/E14: failures are terminal, accurate, and diagnosable."""

    @pytest.mark.asyncio
    async def test_initial_registration_failure_aborts_before_ssh(self) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        registration_error = RuntimeError("etcd registration unavailable")
        normal_put: Callable[[str, bytes], bool] = etcd.put.side_effect

        def fail_node_registration(key: str, value: bytes) -> bool:
            if key == "/nodes/host1":
                raise registration_error
            return normal_put(key, value)

        etcd.put.side_effect = fail_node_registration
        ssh = MagicMock()
        ssh.upload = AsyncMock(
            side_effect=RuntimeError("remote work started after registration failure")
        )
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as caught:
                await provisioner.provision("host1")

        assert caught.value is registration_error
        ssh.upload.assert_not_awaited()
        assert state_payloads[-1]["current_step"] == "failed"
        assert state_payloads[-1]["failed_step"] == "registering_node"

    @pytest.mark.asyncio
    async def test_unexpected_provision_error_records_terminal_state(self) -> None:
        etcd, values, state_payloads = _recording_etcd()
        upload_error = RuntimeError("unexpected upload failure")
        ssh = MagicMock()
        ssh.upload = AsyncMock(side_effect=upload_error)
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as caught:
                await provisioner.provision("host1")

        assert caught.value is upload_error
        assert state_payloads[-1]["current_step"] == "failed"
        assert state_payloads[-1]["failed_step"] == "uploading_scripts"
        assert json.loads(values["/nodes/host1"])["status"] == "failed"

    @pytest.mark.asyncio
    async def test_llamacpp_artifact_identity_survives_provision_failure(
        self,
    ) -> None:
        """PROVISIONING and FAILED records retain the selected generation."""
        etcd, values, _state_payloads = _recording_etcd()
        ssh = MagicMock()
        ssh.upload = AsyncMock(side_effect=RuntimeError("upload failed"))
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)
        artifact = _artifact()

        async def run_inline(
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return function(*args, **kwargs)

        with (
            patch.object(provisioner, "preflight", new_callable=AsyncMock),
            patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                side_effect=run_inline,
            ),
            pytest.raises(RuntimeError, match="upload failed"),
        ):
            await provisioner._provision(
                "host1",
                engine=InferenceEngine.LLAMA_CPP,
                artifact=artifact,
            )

        initial_payload = next(
            json.loads(call.args[1])
            for call in etcd.put.call_args_list
            if call.args[0] == "/nodes/host1"
        )
        failed_payload = json.loads(values["/nodes/host1"])
        assert initial_payload["status"] == "provisioning"
        assert initial_payload["engine"] == "llama_cpp"
        assert initial_payload["artifact_id"] == artifact.artifact_id
        assert failed_payload["status"] == "failed"
        assert failed_payload["engine"] == "llama_cpp"
        assert failed_payload["artifact_id"] == artifact.artifact_id

    @pytest.mark.asyncio
    async def test_setup_failure_records_emitted_step(self) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        command_error = RemoteCommandError("host1", "setup", 1)
        ssh = MagicMock()
        ssh.upload = AsyncMock()

        async def setup_failure(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            assert host == "host1"
            assert "setup.sh" in command
            yield ("stdout", "[STEP:nvidia_driver:START]")
            raise command_error

        ssh.run_streaming = setup_failure
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with pytest.raises((RemoteCommandError, ProvisioningError)):
                await provisioner.provision("host1")

        assert state_payloads[-1]["failed_step"] == "nvidia_driver"

    @pytest.mark.asyncio
    async def test_setup_markers_preserve_original_started_at(self) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        ssh = MagicMock()
        ssh.upload = AsyncMock()

        async def setup_failure(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            yield ("stdout", "[STEP:system_update:START]")
            raise RuntimeError("setup interrupted")

        ssh.run_streaming = setup_failure
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await provisioner.provision("host1")

        starts = {payload["started_at"] for payload in state_payloads}
        assert len(starts) == 1
        assert any(
            payload["current_step"] == "system_update" for payload in state_payloads
        )

    @pytest.mark.asyncio
    async def test_warn_marker_parsed(self) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        ssh = MagicMock()
        ssh.upload = AsyncMock()

        async def setup_warning(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            yield ("stdout", "[STEP:llmfit_install:WARN]")
            raise RuntimeError("later setup failure")

        ssh.run_streaming = setup_warning
        provisioner = _make_provisioner(ssh_client=ssh, etcd_client=etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await provisioner.provision("host1")

        assert any(
            payload["current_step"] == "llmfit_install" for payload in state_payloads
        )
        assert state_payloads[-1]["failed_step"] == "llmfit_install"
        assert any(
            entry["level"] == "warning" and entry["msg"] == "[STEP:llmfit_install:WARN]"
            for entry in provisioner.log_buffer.get_entries("host1")
        )

    @pytest.mark.asyncio
    async def test_unexpected_teardown_error_marks_node_failed_and_not_routable(
        self,
    ) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        delete_error = RuntimeError("etcd delete failed")
        etcd.delete.side_effect = delete_error
        registry = NodeRegistry()
        registry.add(
            Node(
                node_id="host1",
                endpoint="host1:8000",
                status=NodeStatus.HEALTHY,
                model="model-a",
                managed=True,
            )
        )
        ssh = MagicMock()
        ssh.upload = AsyncMock()

        async def stopped(
            _host: str,
            _command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            if False:  # pragma: no cover - defines an empty async generator
                yield ("stdout", "")

        ssh.run_streaming = stopped
        provisioner = _make_provisioner(
            ssh_client=ssh,
            etcd_client=etcd,
            registry=registry,
        )

        with pytest.raises(RuntimeError) as caught:
            await provisioner.teardown("host1", force=True)

        assert caught.value is delete_error
        failed_node = registry.get("host1")
        assert failed_node is not None
        assert failed_node.status == NodeStatus.FAILED
        assert state_payloads[-1]["current_step"] == "failed"
        assert state_payloads[-1]["failed_step"] == "teardown"


class TestPreflight:
    """D-01 through D-04: Pre-flight validation with collected errors."""

    @pytest.mark.asyncio
    async def test_tcp_unreachable(self) -> None:
        """TCP probe failure raises PreflightError immediately."""
        provisioner = _make_provisioner()

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            side_effect=OSError("Connection refused"),
        ):
            with pytest.raises(
                PreflightError, match="SSH port 22 unreachable"
            ) as exc_info:
                await provisioner.preflight("host1")
            assert exc_info.value.hostname == "host1"
            assert len(exc_info.value.failures) == 1

    @pytest.mark.asyncio
    async def test_insufficient_disk(self) -> None:
        """Insufficient disk space raises PreflightError."""
        settings = ProvisioningSettings(
            health_poll_timeout=2, health_poll_interval=0, min_disk_gb=20
        )
        provisioner = _make_provisioner(settings=settings)
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            return_value=(MagicMock(), mock_writer),
        ):
            with patch.object(
                provisioner, "_ssh_run_command", new_callable=AsyncMock
            ) as mock_cmd:
                mock_cmd.return_value = "5242880"
                with pytest.raises(
                    PreflightError, match="Insufficient disk"
                ) as exc_info:
                    await provisioner.preflight("host1")
                assert "5.0" in str(exc_info.value) or "5" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_collects_all_failures(self) -> None:
        """D-03: All failures collected before raising single PreflightError."""
        settings = ProvisioningSettings(
            health_poll_timeout=2, health_poll_interval=0, min_disk_gb=20
        )
        provisioner = _make_provisioner(settings=settings)
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            return_value=(MagicMock(), mock_writer),
        ):
            with patch.object(
                provisioner, "_ssh_run_command", new_callable=AsyncMock
            ) as mock_cmd:
                # Disk check fails + SSH diagnostic error on a second call
                mock_cmd.side_effect = [
                    "5242880",  # disk: 5GB, below min_disk_gb=20
                ]
                with pytest.raises(PreflightError) as exc_info:
                    await provisioner.preflight("host1")
                assert len(exc_info.value.failures) == 1

    @pytest.mark.asyncio
    async def test_standalone_preflight(self) -> None:
        """D-04: preflight() works independently when all checks pass."""
        provisioner = _make_provisioner()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            return_value=(MagicMock(), mock_writer),
        ):
            with patch.object(
                provisioner, "_ssh_run_command", new_callable=AsyncMock
            ) as mock_cmd:
                mock_cmd.return_value = "52428800"  # 50GB disk in KB
                await provisioner.preflight("host1")  # Should not raise

    @pytest.mark.asyncio
    async def test_ssh_diagnostic_failure(self) -> None:
        """SSH diagnostic errors are collected as failures."""
        provisioner = _make_provisioner()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            return_value=(MagicMock(), mock_writer),
        ):
            with patch.object(
                provisioner, "_ssh_run_command", new_callable=AsyncMock
            ) as mock_cmd:
                mock_cmd.side_effect = SSHConnectionError("host1", "connection reset")
                with pytest.raises(
                    PreflightError, match="SSH diagnostic failed"
                ) as exc_info:
                    await provisioner.preflight("host1")
                assert len(exc_info.value.failures) >= 1


class TestLlamaCppRuntimeFit:
    def test_parses_final_runtime_records(self) -> None:
        log_text = "\n".join(
            (
                _llamacpp_fit_log(context_per_slot=4096, gpu_layers=1),
                _llamacpp_fit_log(
                    context_per_slot=24577,
                    slots=4,
                    gpu_layers=37,
                    total_layers=37,
                ),
            )
        )

        fit = _parse_llamacpp_runtime_fit(log_text)

        assert fit.context_per_slot == 24577
        assert fit.slot_context_limit == 24577
        assert fit.slots == 4
        assert fit.aggregate_context == 98308
        assert fit.fit_target_mib == 1024
        assert fit.kv_unified is True
        assert fit.gpu_layers == 37
        assert fit.total_layers == 37

    def test_accepts_expected_full_context_unified_kv_warnings(self) -> None:
        fit = _parse_llamacpp_runtime_fit(
            _llamacpp_fit_log(
                context_per_slot=128000,
                slot_context_limit=128000,
                slots=8,
                aggregate_context=1024000,
            )
        )

        assert fit.context_per_slot == 128000
        assert fit.slot_context_limit == 128000
        assert fit.slots == 8
        assert fit.aggregate_context == 1024000

    @pytest.mark.parametrize(
        ("log_text", "message"),
        [
            (
                "llama_model_load: offloaded 37/37 layers to GPU\n"
                "llama_context: n_ctx = 98304\n"
                "srv init: initializing, n_slots = 4, n_ctx_slot = 24576, "
                "kv_unified = 'true'",
                "no QIIP VRAM plan",
            ),
            (
                "qiip_fit_plan: context_per_slot=24576 slots=4 "
                "aggregate_context=98304 fit_target_mib=1024\n"
                "llama_model_load: offloaded 37/37 layers to GPU\n"
                "srv init: initializing, n_slots = 4, n_ctx_slot = 24576, "
                "kv_unified = 'true'",
                "no aggregate context record",
            ),
            (
                "qiip_fit_plan: context_per_slot=24576 slots=4 "
                "aggregate_context=98304 fit_target_mib=1024\n"
                "llama_context: n_ctx = 98304\n"
                "llama_model_load: offloaded 37/37 layers to GPU",
                "no effective context record",
            ),
            (
                "qiip_fit_plan: context_per_slot=24576 slots=4 "
                "aggregate_context=98304 fit_target_mib=1024\n"
                "llama_context: n_ctx = 98304\n"
                "srv init: initializing, n_slots = 4, n_ctx_slot = 24576, "
                "kv_unified = 'true'",
                "no GPU offload record",
            ),
            (_llamacpp_fit_log(kv_unified=False), "requires unified KV cache"),
            (_llamacpp_fit_log(context_per_slot=0), "invalid effective context"),
            (
                _llamacpp_fit_log(slot_context_limit=100000),
                "invalid effective context",
            ),
            (
                _llamacpp_fit_log(runtime_slots=3),
                "runtime slot count differs",
            ),
            (
                _llamacpp_fit_log(aggregate_context=98303),
                "invalid aggregate context",
            ),
            (
                _llamacpp_fit_log(runtime_aggregate_context=98000),
                "runtime aggregate context differs",
            ),
            (
                _llamacpp_fit_log(gpu_layers=36, total_layers=37),
                "did not fully offload",
            ),
            (
                _llamacpp_fit_log(gpu_layers=0, total_layers=0),
                "did not fully offload",
            ),
            (
                _llamacpp_fit_log(context_sharing_warnings=False),
                "missing expected unified-KV context-sharing warnings",
            ),
            (
                _llamacpp_fit_log().replace(
                    "n_ctx_train (24576)", "n_ctx_train (24575)", 1
                ),
                "context-sharing warnings differ from runtime",
            ),
            (
                _llamacpp_fit_log()
                + "\ncommon_fit_params: failed to fit params to free device memory",
                "runtime fitting failure",
            ),
        ],
    )
    def test_rejects_missing_or_invalid_runtime_evidence(
        self, log_text: str, message: str
    ) -> None:
        with pytest.raises(ProvisioningError, match=message):
            _parse_llamacpp_runtime_fit(log_text)

    @pytest.mark.asyncio
    async def test_reports_parsed_fit_and_post_load_memory(self) -> None:
        provisioner = _make_provisioner(
            settings=ProvisioningSettings(llamacpp_fit_target_mib=1536)
        )
        with (
            patch.object(
                provisioner,
                "_ssh_run_command",
                new_callable=AsyncMock,
                side_effect=[
                    _llamacpp_fit_log(
                        context_per_slot=24577,
                        slot_context_limit=98308,
                        fit_target_mib=1536,
                    ),
                    "20854, 2180\n20860, 2174",
                ],
            ) as run_command,
            patch.object(provisioner, "_log") as log,
        ):
            await provisioner._verify_llamacpp_runtime("host1")

        assert run_command.await_args_list == [
            call("host1", "cat -- /var/log/llamacpp-serve.log"),
            call(
                "host1",
                "nvidia-smi --query-gpu=memory.used,memory.free "
                "--format=csv,noheader,nounits",
            ),
        ]
        log.assert_called_once_with(
            "host1",
            "info",
            "llama.cpp fitted: context_per_slot=24577 slot_context_limit=98308 "
            "slots=4 aggregate_context=98308 kv_unified=true "
            "gpu_layers=37/37 fit_target_mib=1536 "
            "gpu_used_mib=20854,20860 gpu_free_mib=2180,2174",
        )

    @pytest.mark.asyncio
    async def test_below_target_post_load_memory_stops_llamacpp(self) -> None:
        provisioner = _make_provisioner(
            settings=ProvisioningSettings(llamacpp_fit_target_mib=1536)
        )
        run_command = AsyncMock(
            side_effect=[
                _llamacpp_fit_log(fit_target_mib=1536),
                "21854, 1180",
                "",
            ]
        )
        with patch.object(provisioner, "_ssh_run_command", run_command):
            with pytest.raises(ProvisioningError, match="below the configured"):
                await provisioner._verify_llamacpp_runtime("host1")

        assert run_command.await_args_list[-1] == call(
            "host1", "bash auto-llamacpp/stop-llamacpp.sh"
        )

    @pytest.mark.asyncio
    async def test_failed_runtime_contract_stops_llamacpp(self) -> None:
        provisioner = _make_provisioner()
        run_command = AsyncMock(
            side_effect=[
                _llamacpp_fit_log(gpu_layers=12, total_layers=37),
                "",
            ]
        )
        with patch.object(provisioner, "_ssh_run_command", run_command):
            with pytest.raises(ProvisioningError, match="did not fully offload"):
                await provisioner._verify_llamacpp_runtime("host1")

        assert run_command.await_args_list == [
            call("host1", "cat -- /var/log/llamacpp-serve.log"),
            call("host1", "bash auto-llamacpp/stop-llamacpp.sh"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_text", ["not-a-memory-record", "\n"])
    async def test_invalid_memory_telemetry_stops_llamacpp(
        self, memory_text: str
    ) -> None:
        provisioner = _make_provisioner()
        run_command = AsyncMock(
            side_effect=[
                _llamacpp_fit_log(),
                memory_text,
                "",
            ]
        )
        with patch.object(provisioner, "_ssh_run_command", run_command):
            with pytest.raises(ProvisioningError, match="memory telemetry"):
                await provisioner._verify_llamacpp_runtime("host1")

        assert run_command.await_args_list[-1] == call(
            "host1", "bash auto-llamacpp/stop-llamacpp.sh"
        )

    @pytest.mark.asyncio
    async def test_cleanup_failure_preserves_runtime_contract_error(self) -> None:
        provisioner = _make_provisioner()
        cleanup_error = SSHConnectionError("host1", "connection lost")
        run_command = AsyncMock(
            side_effect=[
                _llamacpp_fit_log(gpu_layers=12, total_layers=37),
                cleanup_error,
            ]
        )
        with (
            patch.object(provisioner, "_ssh_run_command", run_command),
            patch.object(provisioner, "_log") as log,
            pytest.raises(ProvisioningError, match="did not fully offload"),
        ):
            await provisioner._verify_llamacpp_runtime("host1")

        log.assert_called_once_with(
            "host1",
            "warning",
            "llama.cpp runtime verification failed and the stop script also "
            f"failed: {cleanup_error}",
        )


class TestVerifyGpu:
    """GPU verification after setup.sh installs the NVIDIA driver."""

    @pytest.mark.asyncio
    async def test_no_gpu_after_setup(self) -> None:
        """No GPUs detected raises ProvisioningError."""
        provisioner = _make_provisioner()
        with patch.object(
            provisioner, "_ssh_run_command", new_callable=AsyncMock, return_value=""
        ):
            with pytest.raises(ProvisioningError, match="No GPUs detected"):
                await provisioner._verify_gpu("host1")

    @pytest.mark.asyncio
    async def test_gpu_detected(self) -> None:
        """GPUs detected passes without error."""
        provisioner = _make_provisioner()
        with patch.object(
            provisioner,
            "_ssh_run_command",
            new_callable=AsyncMock,
            return_value="Tesla V100\nTesla V100",
        ):
            await provisioner._verify_gpu("host1")  # Should not raise

    @pytest.mark.asyncio
    async def test_llamacpp_provision_requires_gpu_before_engine_start(self) -> None:
        """Managed llama.cpp never degrades into its standalone CPU branch."""
        provisioner = _make_provisioner()
        with (
            patch.object(provisioner, "_update_state", new_callable=AsyncMock),
            patch.object(provisioner, "_power_on_if_needed", new_callable=AsyncMock),
            patch.object(provisioner, "preflight", new_callable=AsyncMock),
            patch.object(provisioner, "_upload_scripts", new_callable=AsyncMock),
            patch.object(provisioner, "_run_setup", new_callable=AsyncMock),
            patch.object(
                provisioner,
                "_verify_gpu",
                new_callable=AsyncMock,
                side_effect=ProvisioningError("No GPUs detected"),
            ) as verify_gpu,
            patch.object(
                provisioner, "_run_start_vllm", new_callable=AsyncMock
            ) as start_engine,
            patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            with pytest.raises(ProvisioningError, match="No GPUs detected"):
                await provisioner._provision(
                    "host1",
                    model="org/model",
                    engine=InferenceEngine.LLAMA_CPP,
                )

        verify_gpu.assert_awaited_once_with("host1")
        start_engine.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llamacpp_runtime_fit_is_verified_after_health_before_register(
        self,
    ) -> None:
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        provisioner = _make_provisioner(etcd_client=etcd)
        events: list[str] = []

        with (
            patch.object(provisioner, "_update_state", new_callable=AsyncMock),
            patch.object(provisioner, "_power_on_if_needed", new_callable=AsyncMock),
            patch.object(provisioner, "preflight", new_callable=AsyncMock),
            patch.object(provisioner, "_upload_scripts", new_callable=AsyncMock),
            patch.object(provisioner, "_run_setup", new_callable=AsyncMock),
            patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock),
            patch.object(
                provisioner,
                "_run_start_vllm",
                new_callable=AsyncMock,
                return_value=_artifact().model_alias,
            ),
            patch.object(
                provisioner, "_poll_health", new_callable=AsyncMock
            ) as poll_health,
            patch.object(
                provisioner, "_verify_llamacpp_runtime", new_callable=AsyncMock
            ) as verify_runtime,
            patch.object(
                provisioner, "_register_node", new_callable=AsyncMock
            ) as register,
            patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            poll_health.side_effect = lambda *_args, **_kwargs: events.append("health")
            verify_runtime.side_effect = lambda *_args, **_kwargs: events.append(
                "runtime"
            )
            register.side_effect = lambda *_args, **_kwargs: events.append("register")

            await provisioner._provision(
                "host1",
                engine=InferenceEngine.LLAMA_CPP,
                artifact=_artifact(),
            )

        assert events == ["health", "runtime", "register"]


def _make_full_provisioner(etcd: MagicMock) -> tuple[NodeProvisioner, MagicMock]:
    """Build a provisioner with mocks suitable for full provision() tests."""
    ssh = MagicMock()

    async def mock_streaming(
        host: str,
        command: str,
    ) -> AsyncIterator[tuple[str, str]]:
        if "setup.sh" in command:
            for item in [
                ("stdout", "[STEP:system_update:START]"),
                ("stdout", "[STEP:system_update:OK]"),
            ]:
                yield item
        elif "start-vllm.sh" in command:
            for item in [("stdout", "# Model:              Qwen/Qwen2.5-72B-Instruct")]:
                yield item

    ssh.run_streaming = mock_streaming
    ssh.upload = AsyncMock()
    etcd.prefix = "/nodes/"
    etcd.put = MagicMock(return_value=True)

    settings = ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0)
    provisioner = NodeProvisioner(
        ssh_client=ssh,
        etcd_client=etcd,
        settings=settings,
        llmfit_settings=LLMFitSettings(),
        endpoint_policy=_TEST_ENDPOINT_POLICY,
        nfs_export="nfs.example:/exports/huggingface",
    )
    return provisioner, ssh


class TestStateTracking:
    """D-05 through D-11: State machine tracking and PROVISIONING registration."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_full_success_transitions(self, mock_httpx_cls: MagicMock) -> None:
        """State writes go PENDING -> PREFLIGHT -> steps -> ... -> COMPLETE."""
        etcd = MagicMock()
        provisioner, _ = _make_full_provisioner(etcd)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.return_value = True
                    await provisioner.provision("host1")

        # Collect all etcd put calls -- both via to_thread and direct mock
        put_calls = mock_to_thread.call_args_list
        state_keys = []
        node_keys = []
        for c in put_calls:
            args = c[0]  # positional args: (etcd.put, key, value)
            if len(args) >= 3:
                key = args[1]
                if "/provisioning/" in key:
                    value = json.loads(args[2])
                    state_keys.append(value["current_step"])
                elif "/nodes/" in key:
                    node_keys.append(key)

        assert "pending" in state_keys
        assert "complete" in state_keys
        assert len(node_keys) >= 1  # PROVISIONING registration

    @pytest.mark.asyncio
    async def test_failed_state(self) -> None:
        """On failure, last state write has current_step=failed with details."""
        etcd = MagicMock()
        ssh = MagicMock()

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            if "setup.sh" in command:
                raise RemoteCommandError("host1", "bash setup.sh", 1)
                yield  # pragma: no cover

        ssh.run_streaming = mock_streaming
        ssh.upload = AsyncMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        settings = ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0)
        provisioner = NodeProvisioner(
            ssh_client=ssh,
            etcd_client=etcd,
            settings=settings,
            llmfit_settings=LLMFitSettings(),
            endpoint_policy=_TEST_ENDPOINT_POLICY,
            nfs_export="nfs.example:/exports/huggingface",
        )

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread:
                mock_to_thread.return_value = True
                with pytest.raises(RemoteCommandError):
                    await provisioner.provision("host1")

        # Find the last /provisioning/ state write
        state_writes = []
        for c in mock_to_thread.call_args_list:
            args = c[0]
            if len(args) >= 3 and "/provisioning/" in str(args[1]):
                state_writes.append(json.loads(args[2]))

        assert len(state_writes) > 0
        last_state = state_writes[-1]
        assert last_state["current_step"] == "failed"
        # D-03: failed_step must be the actual step name, not the exception class name
        assert last_state["failed_step"] == "uploading_scripts"
        assert last_state["failed_step"] != "RemoteCommandError"
        assert last_state["error"] is not None

    @pytest.mark.asyncio
    async def test_etcd_prefix(self) -> None:
        """State writes use /provisioning/{hostname}, not /nodes/ prefix (D-05)."""
        etcd = MagicMock()
        provisioner, _ = _make_full_provisioner(etcd)

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.return_value = True
                    with patch(
                        "inference_proxy.provisioning.provisioner.httpx.AsyncClient"
                    ) as mock_httpx:
                        mock_resp = MagicMock()
                        mock_resp.status_code = 200
                        mock_cl = AsyncMock()
                        mock_cl.get = AsyncMock(return_value=mock_resp)
                        mock_cl.__aenter__ = AsyncMock(return_value=mock_cl)
                        mock_cl.__aexit__ = AsyncMock(return_value=False)
                        mock_httpx.return_value = mock_cl
                        await provisioner.provision("host1")

        # All state writes should use /provisioning/ prefix
        for c in mock_to_thread.call_args_list:
            args = c[0]
            if len(args) >= 3:
                key = str(args[1])
                if "provisioning" in key.lower() and "nodes" not in key:
                    assert key.startswith("/provisioning/")

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_state_write_failure_continues(
        self, mock_httpx_cls: MagicMock
    ) -> None:
        """State write exceptions are swallowed -- provisioning continues (Pitfall 3)."""
        etcd = MagicMock()
        provisioner, _ = _make_full_provisioner(etcd)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        call_count = 0

        async def flaky_to_thread(
            fn: Callable[..., Any],
            *args: Any,
            **_kwargs: Any,
        ) -> int:
            nonlocal call_count
            call_count += 1
            if fn is etcd.grant_node_lease:
                return 7001
            key = args[0] if args else ""
            # Fail all /provisioning/ writes, allow /nodes/ writes
            if isinstance(key, str) and "/provisioning/" in key:
                raise ConnectionError("etcd down")
            return True

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    side_effect=flaky_to_thread,
                ):
                    # Should complete despite state write failures
                    await provisioner.provision("host1")

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.provisioner.httpx.AsyncClient")
    async def test_registers_provisioning_before_setup(
        self, mock_httpx_cls: MagicMock
    ) -> None:
        """D-09: First /nodes/ write creates node with status=provisioning."""
        etcd = MagicMock()
        provisioner, _ = _make_full_provisioner(etcd)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        with patch.object(provisioner, "preflight", new_callable=AsyncMock):
            with patch.object(provisioner, "_verify_gpu", new_callable=AsyncMock):
                with patch(
                    "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_to_thread:
                    mock_to_thread.return_value = True
                    with patch(
                        "inference_proxy.provisioning.provisioner.node_to_etcd"
                    ) as mock_ser:
                        mock_ser.return_value = (
                            "/nodes/host1",
                            b'{"status":"provisioning"}',
                        )
                        await provisioner.provision("host1")

                        # Find the first call to node_to_etcd
                        first_call = mock_ser.call_args_list[0]
                        node = first_call[0][0]
                        assert node.status == NodeStatus.PROVISIONING
                        node_puts = [
                            call
                            for call in mock_to_thread.call_args_list
                            if call.args and call.args[0] is etcd.put
                        ]
                        assert node_puts
                        assert "lease_id" not in node_puts[0].kwargs

    @pytest.mark.asyncio
    async def test_preflight_called_before_setup(self) -> None:
        """Preflight failure prevents _run_setup from running."""
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)
        provisioner, ssh = _make_full_provisioner(etcd)

        setup_called = False
        original_run_setup = provisioner._run_setup

        async def tracking_setup(
            hostname: str,
            *,
            started_at: datetime,
            on_step: Callable[[str], None],
        ) -> None:
            nonlocal setup_called
            setup_called = True
            await original_run_setup(
                hostname,
                started_at=started_at,
                on_step=on_step,
            )

        with (
            patch.object(provisioner, "_run_setup", side_effect=tracking_setup),
            patch.object(provisioner, "preflight", new_callable=AsyncMock) as mock_pf,
        ):
            mock_pf.side_effect = PreflightError("host1", ["no gpus"])
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_to_thread:
                mock_to_thread.return_value = True
                with pytest.raises(PreflightError):
                    await provisioner.provision("host1")

        assert not setup_called


def _make_teardown_provisioner(
    *,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    tracker_get_returns: int | list[int] = 0,
    force: bool = False,
    scripts_dir: Path = Path("auto-vllm"),
) -> tuple[NodeProvisioner, MagicMock, MagicMock, MagicMock, list[str]]:
    """Build a provisioner wired for teardown testing.

    Returns (provisioner, ssh_mock, etcd_mock, registry_mock, state_steps).
    state_steps is populated during the test via side_effect on to_thread.
    """
    from inference_proxy.models.node import Node, NodeStatus

    ssh = MagicMock()
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    etcd.put = MagicMock(return_value=True)
    etcd.delete = MagicMock(return_value=True)

    registry = MagicMock()
    node = Node(
        node_id="host1",
        endpoint="host1:8000",
        status=NodeStatus.HEALTHY,
        model=model,
        last_heartbeat=datetime.now(UTC),
    )
    registry.get.return_value = node
    registry.drain.return_value = True

    tracker = MagicMock()
    if isinstance(tracker_get_returns, list):
        tracker.get.side_effect = tracker_get_returns
    else:
        tracker.get.return_value = tracker_get_returns

    async def mock_streaming(
        host: str,
        command: str,
    ) -> AsyncIterator[tuple[str, str]]:
        for item in [("stdout", "ok")]:
            yield item

    ssh.run_streaming = mock_streaming
    ssh.upload = AsyncMock()

    provisioner = _make_provisioner(
        ssh_client=ssh,
        etcd_client=etcd,
        registry=registry,
        connection_tracker=tracker,
        settings=ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            drain_timeout=2,
            scripts_dir=scripts_dir,
        ),
    )
    return provisioner, ssh, etcd, registry, tracker


class TestTeardownIdentity:
    @pytest.mark.asyncio
    async def test_unregistered_host_without_recovery_identity_fails_closed(
        self,
    ) -> None:
        ssh = MagicMock()
        ssh.upload = AsyncMock()
        provisioner = _make_provisioner(
            ssh_client=ssh,
            registry=NodeRegistry(),
        )

        with pytest.raises(ProvisioningError, match="Cannot determine"):
            await provisioner.teardown("host1", force=True)

        ssh.upload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_force_recovery_supplies_missing_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provisioner = _make_provisioner(registry=NodeRegistry())
        teardown = AsyncMock()
        monkeypatch.setattr(provisioner, "_teardown", teardown)

        await provisioner.teardown(
            "host1",
            force=True,
            recovery_engine=InferenceEngine.LLAMA_CPP,
        )

        teardown.assert_awaited_once_with(
            "host1",
            force=True,
            engine=InferenceEngine.LLAMA_CPP,
        )

    @pytest.mark.parametrize(
        ("registered", "provisioning_identity", "force", "message"),
        [
            (
                False,
                ProvisioningIdentity(InferenceEngine.VLLM),
                True,
                "active provisioning identity",
            ),
            (True, None, True, "registered node identity"),
            (False, None, False, "requires force=true"),
        ],
    )
    def test_recovery_identity_cannot_override_authoritative_state(
        self,
        registered: bool,
        provisioning_identity: ProvisioningIdentity | None,
        force: bool,
        message: str,
    ) -> None:
        registry = NodeRegistry()
        if registered:
            registry.add(
                Node(
                    node_id="host1",
                    endpoint="host1:8000",
                    engine=InferenceEngine.VLLM,
                )
            )
        provisioner = _make_provisioner(registry=registry)

        with pytest.raises(ProvisioningError, match=message):
            provisioner._resolve_teardown_engine(
                "host1",
                force=force,
                provisioning_identity=provisioning_identity,
                recovery_engine=InferenceEngine.LLAMA_CPP,
            )


class TestTeardownGraceful:
    """D-01, D-08, D-11, D-12: Graceful teardown drains, stops, deregisters."""

    @pytest.mark.asyncio
    async def test_graceful_teardown_sequence(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            # Capture state writes to track step progression
            if fn == etcd.put and len(args) >= 2:
                key = args[0]
                if "/provisioning/" in str(key):
                    data = json.loads(args[1])
                    state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1")

        # Verify drain was called
        registry.drain.assert_called_once_with("host1")
        # Verify state progression: DRAINING -> STOPPING_CONTAINER -> DEREGISTERING -> TEARDOWN_COMPLETE
        assert "draining" in state_steps
        assert "stopping_vllm" in state_steps
        assert "deregistering" in state_steps
        assert "teardown_complete" in state_steps

    @pytest.mark.asyncio
    async def test_graceful_teardown_ssh_command(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        commands: list[str] = []
        operation_order: list[str] = []

        async def mock_upload(host: str, local_path: Path) -> None:
            assert host == "host1"
            assert local_path in {
                provisioner._settings.scripts_dir,
                provisioner._settings.scripts_dir.parent / "common",
            }
            operation_order.append("upload")

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            commands.append(command)
            operation_order.append("stop")
            for item in [("stdout", "ok")]:
                yield item

        ssh.upload = AsyncMock(side_effect=mock_upload)
        ssh.run_streaming = mock_streaming

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_tt:
            mock_tt.return_value = True
            await provisioner.teardown("host1")

        assert commands == ["bash auto-vllm/stop-vllm.sh"]
        assert operation_order == ["upload", "upload", "stop"]

    @pytest.mark.asyncio
    async def test_scripts_dir_respected_in_commands(self, tmp_path: Path) -> None:
        """E12: upload, setup, start, and stop share the configured bundle."""
        bundle_root = tmp_path / "bundles -- root"
        scripts_dir = bundle_root / "provision scripts -- vllm"
        shutil.copytree(Path("auto-vllm"), scripts_dir)
        common_dir = bundle_root / "common"
        shutil.copytree(Path("common"), common_dir)
        provisioner, ssh, _etcd, _registry, _tracker = _make_teardown_provisioner(
            scripts_dir=scripts_dir
        )
        commands: list[str] = []

        async def capture_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            assert host == "host1"
            commands.append(command)
            if "start-vllm.sh" in command:
                yield ("stdout", "# Model:              org/model")
            else:
                yield ("stdout", "ok")

        ssh.run_streaming = capture_streaming

        await provisioner._upload_scripts("host1")
        await provisioner._run_setup(
            "host1",
            started_at=datetime.now(UTC),
            on_step=lambda _step: None,
        )
        await provisioner._run_start_vllm("host1", model="org/model")

        assert [item.args for item in ssh.upload.await_args_list[:2]] == [
            ("host1", scripts_dir),
            ("host1", common_dir),
        ]
        assert [shlex.split(command)[-1] for command in commands[:2]] == [
            "provision scripts -- vllm/setup.sh",
            "provision scripts -- vllm/start-vllm.sh",
        ]
        assert all("auto-vllm/" not in command for command in commands)

        await asyncio.wait_for(provisioner.teardown("host1", force=True), timeout=1)

        assert shlex.split(commands[2]) == [
            "bash",
            "provision scripts -- vllm/stop-vllm.sh",
            "--force",
        ]
        assert all("auto-vllm/" not in command for command in commands)

    @pytest.mark.asyncio
    async def test_etcd_node_key_deleted(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        deleted_keys: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.delete:
                deleted_keys.append(args[0])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1")

        # D-11: should delete /nodes/host1
        assert "/nodes/host1" in deleted_keys

    @pytest.mark.asyncio
    @pytest.mark.parametrize("force", [False, True])
    async def test_upload_failure_still_attempts_stop_and_deregisters(
        self,
        force: bool,
    ) -> None:
        """Script refresh failure cannot prevent the teardown attempt."""
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        commands: list[str] = []
        ssh.upload = AsyncMock(side_effect=asyncssh.SFTPFailure("disk full"))

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            assert host == "host1"
            commands.append(command)
            yield ("stdout", "stopped")

        ssh.run_streaming = mock_streaming

        await provisioner.teardown("host1", force=force)

        expected_command = "bash auto-vllm/stop-vllm.sh"
        if force:
            expected_command += " --force"
        assert commands == [expected_command]
        etcd.delete.assert_called_once_with("/nodes/host1")
        registry.remove.assert_called_once_with("host1")
        assert any(
            "Could not refresh teardown scripts" in entry["msg"]
            for entry in provisioner.log_buffer.get_entries("host1")
        )


class TestTeardownForce:
    """Force teardown skips drain, uses kill -9."""

    @pytest.mark.asyncio
    async def test_force_skips_drain(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1", force=True)

        # Force mode should NOT have DRAINING step
        assert "draining" not in state_steps
        # But should still have the rest
        assert "stopping_vllm" in state_steps
        assert "deregistering" in state_steps
        assert "teardown_complete" in state_steps
        # registry.drain should NOT be called
        registry.drain.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_uses_verified_stop_helper(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        commands: list[str] = []

        async def mock_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            commands.append(command)
            for item in [("stdout", "ok")]:
                yield item

        ssh.run_streaming = mock_streaming

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_tt:
            mock_tt.return_value = True
            await provisioner.teardown("host1", force=True)

        assert commands == ["bash auto-vllm/stop-vllm.sh --force"]


class TestDrainTimeout:
    """D-09: Drain timeout expiry proceeds to container stop."""

    @pytest.mark.asyncio
    async def test_timeout_proceeds_to_stop(self) -> None:
        """Connections never reach 0 but teardown still completes after timeout."""
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner(
            tracker_get_returns=5,  # always 5 connections
        )
        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1")

        # Should still complete despite never draining
        assert "stopping_vllm" in state_steps
        assert "teardown_complete" in state_steps


class TestTeardownStateProgression:
    """D-05: State tracked step-by-step in etcd."""

    @pytest.mark.asyncio
    async def test_graceful_state_order(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1")

        expected_order = [
            "draining",
            "stopping_vllm",
            "deregistering",
            "teardown_complete",
        ]
        assert state_steps == expected_order

    @pytest.mark.asyncio
    async def test_force_state_order(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await provisioner.teardown("host1", force=True)

        expected_order = ["stopping_vllm", "deregistering", "teardown_complete"]
        assert state_steps == expected_order


class TestTeardownSSHFailure:
    """Teardown with SSH failure updates state to FAILED."""

    @pytest.mark.asyncio
    async def test_ssh_failure_sets_failed_state(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def failing_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            raise SSHConnectionError("host1", "connection refused")
            yield  # pragma: no cover

        ssh.run_streaming = failing_streaming

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            with pytest.raises(SSHConnectionError):
                await provisioner.teardown("host1", force=True)

        assert "failed" in state_steps

    @pytest.mark.asyncio
    async def test_stop_failure_preserves_registration(self) -> None:
        """A failed verified stop cannot deregister a possibly-live backend."""
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()

        async def failing_streaming(
            host: str,
            command: str,
        ) -> AsyncIterator[tuple[str, str]]:
            assert command == "bash auto-vllm/stop-vllm.sh --force"
            raise RemoteCommandError(host, command, 1, "process survived")
            yield  # pragma: no cover

        ssh.run_streaming = failing_streaming

        with pytest.raises(RemoteCommandError, match="exited with status 1"):
            await provisioner.teardown("host1", force=True)

        etcd.delete.assert_not_called()
        registry.remove.assert_not_called()


class TestPowerOnIfNeeded:
    """D-01, D-04, D-06, D-07: Power-on logic before SSH provisioning."""

    @pytest.mark.asyncio
    async def test_skips_when_redfish_none(self) -> None:
        """D-01: No POWERING_ON state write or power_action when redfish_client is None."""
        etcd = MagicMock()
        etcd.put = MagicMock()
        provisioner = _make_provisioner(etcd_client=etcd)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_tt:
            mock_tt.return_value = True
            await provisioner._power_on_if_needed("host1")

        # No etcd writes should happen (no POWERING_ON state)
        mock_tt.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_power_action_on(self) -> None:
        """D-04: Calls power_action("On") and writes POWERING_ON state."""
        etcd = MagicMock()
        etcd.put = MagicMock()
        redfish = MagicMock()
        redfish.power_action = AsyncMock(return_value="On")

        provisioner = _make_provisioner(etcd_client=etcd, redfish_client=redfish)

        state_steps: list[str] = []

        async def capture_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                state_steps.append(data["current_step"])
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            with patch.object(provisioner, "_wait_for_ssh", new_callable=AsyncMock):
                await provisioner._power_on_if_needed("host1")

        redfish.power_action.assert_awaited_once_with("host1", "On")
        assert "powering_on" in state_steps

    @pytest.mark.asyncio
    async def test_catches_redfish_error(self) -> None:
        """D-06: RedfishError caught, logged, continues to _wait_for_ssh."""
        redfish = MagicMock()
        redfish.power_action = AsyncMock(side_effect=RedfishError("BMC unreachable"))

        provisioner = _make_provisioner(redfish_client=redfish)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            new_callable=AsyncMock,
        ):
            with patch.object(
                provisioner, "_wait_for_ssh", new_callable=AsyncMock
            ) as mock_wait:
                await provisioner._power_on_if_needed("host1")

        # _wait_for_ssh should still be called despite RedfishError
        mock_wait.assert_awaited_once_with("host1")

    @pytest.mark.asyncio
    async def test_powering_on_state_written_before_action(self) -> None:
        """D-07: POWERING_ON state is written before power_action is called."""
        etcd = MagicMock()
        etcd.put = MagicMock()
        redfish = MagicMock()

        call_order: list[str] = []

        async def tracking_power_action(hostname: str, action: str) -> str:
            call_order.append("power_action")
            return "On"

        redfish.power_action = tracking_power_action

        provisioner = _make_provisioner(etcd_client=etcd, redfish_client=redfish)

        async def tracking_to_thread(
            fn: Callable[..., Any],
            *args: Any,
        ) -> bool:
            if fn == etcd.put and len(args) >= 2 and "/provisioning/" in str(args[0]):
                data = json.loads(args[1])
                if data["current_step"] == "powering_on":
                    call_order.append("state_write")
            return True

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.to_thread",
            side_effect=tracking_to_thread,
        ):
            with patch.object(provisioner, "_wait_for_ssh", new_callable=AsyncMock):
                await provisioner._power_on_if_needed("host1")

        assert call_order.index("state_write") < call_order.index("power_action")


class TestWaitForSsh:
    """D-03, D-05: SSH wait loop with TCP probe retries."""

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self) -> None:
        """When open_connection succeeds immediately, returns without sleeping."""
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        settings = ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            boot_wait_timeout=10,
            boot_wait_interval=0,
        )
        provisioner = _make_provisioner(settings=settings)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            return_value=(MagicMock(), mock_writer),
        ) as mock_conn:
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep:
                await provisioner._wait_for_ssh("host1")

        mock_conn.assert_called_once()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_until_success(self) -> None:
        """Fails twice then succeeds on third attempt."""
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        attempt = 0

        async def flaky_open(host: str, port: int) -> tuple[MagicMock, MagicMock]:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise OSError("Connection refused")
            return (MagicMock(), mock_writer)

        settings = ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            boot_wait_timeout=60,
            boot_wait_interval=0,
        )
        provisioner = _make_provisioner(settings=settings)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            side_effect=flaky_open,
        ):
            with patch(
                "inference_proxy.provisioning.provisioner.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                await provisioner._wait_for_ssh("host1")

        assert attempt == 3

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self) -> None:
        """When open_connection never succeeds, returns after timeout without raising."""
        settings = ProvisioningSettings(
            health_poll_timeout=2,
            health_poll_interval=0,
            boot_wait_timeout=0,
            boot_wait_interval=0,
        )
        provisioner = _make_provisioner(settings=settings)

        with patch(
            "inference_proxy.provisioning.provisioner.asyncio.open_connection",
            side_effect=OSError("refused"),
        ):
            # Should not raise -- just returns after timeout
            await provisioner._wait_for_ssh("host1")
