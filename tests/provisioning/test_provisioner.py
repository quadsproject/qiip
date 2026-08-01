"""Unit tests for NodeProvisioner.

Tests mock SSHClient, EtcdClient, and httpx to verify the full
provisioning sequence: setup.sh -> start-vllm.sh -> health poll -> register.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import httpx
import pytest

from inference_proxy.config.settings import (
    LLMFitSettings,
    ProvisioningSettings,
    RoutingSettings,
)
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.models.endpoint import EndpointPolicy, EndpointValidationError
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    PreflightError,
    ProvisioningError,
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
) -> NodeProvisioner:
    """Build a NodeProvisioner with mock dependencies."""
    constructor_args: dict[str, object] = {
        "ssh_client": ssh_client or MagicMock(),
        "etcd_client": etcd_client or MagicMock(),
        "settings": settings
        or ProvisioningSettings(health_poll_timeout=2, health_poll_interval=0),
        "endpoint_policy": endpoint_policy,
        "registry": registry,
        "connection_tracker": connection_tracker,
        "circuit_breaker_registry": circuit_breaker_registry,
        "redfish_client": redfish_client,
        "hf_token": hf_token,
    }
    # Baseline comparisons run these tests against the pre-PR constructor so
    # failures reach behavioral assertions rather than stopping at TypeError.
    if "llmfit_settings" in inspect.signature(NodeProvisioner).parameters:
        constructor_args["llmfit_settings"] = llmfit_settings or LLMFitSettings()
    if "nfs_export" in inspect.signature(NodeProvisioner).parameters:
        constructor_args["nfs_export"] = nfs_export
    return NodeProvisioner(**constructor_args)


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


async def _async_iter(items: list[tuple[str, str]]):
    """Helper: async generator yielding items."""
    for item in items:
        yield item


def _recording_etcd() -> tuple[MagicMock, dict[str, bytes], list[dict[str, object]]]:
    """Return an etcd double which implements put/replace/delete semantics."""
    etcd = MagicMock()
    etcd.prefix = "/nodes/"
    values: dict[str, bytes] = {}
    state_payloads: list[dict[str, object]] = []

    def put(key: str, value: bytes) -> bool:
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

        async def mock_streaming(host: str, command: str):
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

        async def mock_upload(host, local_path, remote_path="."):
            call_order.append("upload")

        async def mock_streaming(host: str, command: str):
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

        async def mock_upload(host, local_path, remote_path="."):
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

        async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
            for item in [("stdout", "no model line here")]:
                yield item

        ssh.run_streaming = mock_streaming
        provisioner = _make_provisioner(ssh_client=ssh)

        with pytest.raises(ProvisioningError, match="model name not found"):
            await provisioner._run_start_vllm("host1")

    @pytest.mark.asyncio
    async def test_includes_model_in_start_environment(self) -> None:
        """The model uses the same ordered start environment as other inputs."""
        ssh = MagicMock()
        captured_commands: list[str] = []

        async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
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
            mock_to_thread.return_value = True
            with patch(
                "inference_proxy.provisioning.provisioner.node_to_etcd"
            ) as mock_serialize:
                mock_serialize.return_value = ("/nodes/host1", b'{"model":"test"}')
                await provisioner._register_node("host1", "test-model")

                # Verify Node was constructed correctly
                call_args = mock_serialize.call_args
                node = call_args[0][0]
                assert node.node_id == "host1"
                assert node.status == NodeStatus.HEALTHY
                assert node.model == "test-model"
                assert node.endpoint == "http://host1:8000"
                assert node.last_heartbeat is not None

                # Verify etcd.put called via asyncio.to_thread
                mock_to_thread.assert_called_once_with(
                    etcd.put, "/nodes/host1", b'{"model":"test"}'
                )


class TestSetupFailure:
    """D-08: setup failures retain their original typed exception."""

    @pytest.mark.asyncio
    async def test_remote_command_error_wraps(self) -> None:
        ssh = MagicMock()
        etcd = MagicMock()
        etcd.prefix = "/nodes/"
        etcd.put = MagicMock(return_value=True)

        async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
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
        normal_put = etcd.put.side_effect

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
    async def test_setup_failure_records_emitted_step(self) -> None:
        etcd, _values, state_payloads = _recording_etcd()
        command_error = RemoteCommandError("host1", "setup", 1)
        ssh = MagicMock()
        ssh.upload = AsyncMock()

        async def setup_failure(host: str, command: str):
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

        async def setup_failure(host: str, command: str):
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

        async def setup_warning(host: str, command: str):
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

        async def stopped(_host: str, _command: str):
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


def _make_full_provisioner(etcd: MagicMock) -> tuple[NodeProvisioner, MagicMock]:
    """Build a provisioner with mocks suitable for full provision() tests."""
    ssh = MagicMock()

    async def mock_streaming(host: str, command: str):
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

        async def mock_streaming(host: str, command: str):
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

        async def flaky_to_thread(fn, *args):
            nonlocal call_count
            call_count += 1
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

        provisioner._run_setup = tracking_setup

        with patch.object(provisioner, "preflight", new_callable=AsyncMock) as mock_pf:
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

    async def mock_streaming(host: str, command: str):
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


class TestTeardownGraceful:
    """D-01, D-08, D-11, D-12: Graceful teardown drains, stops, deregisters."""

    @pytest.mark.asyncio
    async def test_graceful_teardown_sequence(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        state_steps: list[str] = []

        async def capture_to_thread(fn, *args):
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

        async def mock_upload(host: str, local_path) -> None:
            assert host == "host1"
            assert local_path == provisioner._settings.scripts_dir
            operation_order.append("upload")

        async def mock_streaming(host: str, command: str):
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
        assert operation_order == ["upload", "stop"]

    @pytest.mark.asyncio
    async def test_scripts_dir_respected_in_commands(self) -> None:
        """E12: upload, setup, start, and stop share the configured bundle."""
        scripts_dir = Path("bundles/provision scripts")
        provisioner, ssh, _etcd, _registry, _tracker = _make_teardown_provisioner(
            scripts_dir=scripts_dir
        )
        commands: list[str] = []

        async def capture_streaming(host: str, command: str):
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

        assert ssh.upload.await_args_list[0].args == ("host1", scripts_dir)
        assert [shlex.split(command)[-1] for command in commands[:2]] == [
            "provision scripts/setup.sh",
            "provision scripts/start-vllm.sh",
        ]
        assert all("auto-vllm/" not in command for command in commands)

        await asyncio.wait_for(provisioner.teardown("host1", force=True), timeout=1)

        assert shlex.split(commands[2]) == [
            "bash",
            "provision scripts/stop-vllm.sh",
            "--force",
        ]
        assert all("auto-vllm/" not in command for command in commands)

    @pytest.mark.asyncio
    async def test_etcd_node_key_deleted(self) -> None:
        provisioner, ssh, etcd, registry, tracker = _make_teardown_provisioner()
        deleted_keys: list[str] = []

        async def capture_to_thread(fn, *args):
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

        async def mock_streaming(host: str, command: str):
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

        async def capture_to_thread(fn, *args):
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

        async def mock_streaming(host: str, command: str):
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

        async def capture_to_thread(fn, *args):
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

        async def capture_to_thread(fn, *args):
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

        async def capture_to_thread(fn, *args):
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

        async def failing_streaming(host: str, command: str):
            raise SSHConnectionError("host1", "connection refused")
            yield  # pragma: no cover

        ssh.run_streaming = failing_streaming

        async def capture_to_thread(fn, *args):
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

        async def failing_streaming(host: str, command: str):
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

        async def capture_to_thread(fn, *args):
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

        async def tracking_power_action(hostname, action):
            call_order.append("power_action")
            return "On"

        redfish.power_action = tracking_power_action

        provisioner = _make_provisioner(etcd_client=etcd, redfish_client=redfish)

        async def tracking_to_thread(fn, *args):
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

        async def flaky_open(host, port):
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
