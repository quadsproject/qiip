"""Unit tests for LLMFitRunner.recommend()."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from inference_proxy.llmfit.errors import (
    LLMFitParseError,
    LLMFitTimeoutError,
)
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)

FIXTURE_JSON = json.dumps(
    {
        "system": {
            "total_ram_gb": 64.0,
            "available_ram_gb": 58.24,
            "cpu_cores": 16,
            "cpu_name": "AMD EPYC 7742",
            "has_gpu": True,
            "gpu_vram_gb": 80.0,
            "unified_memory": False,
            "backend": "CUDA",
        },
        "models": [
            {
                "name": "llama-3.3-70b",
                "provider": "Meta",
                "parameter_count": "70B",
                "params_b": 70.0,
                "context_length": 131072,
                "use_case": "general",
                "category": "General",
                "release_date": "2024-12-06",
                "fit_level": "perfect",
                "run_mode": "gpu",
                "score": 95.2,
                "estimated_tps": 42.5,
                "runtime": "vLLM",
                "best_quant": "4bit",
                "memory_required_gb": 43.68,
                "utilization_pct": 68.2,
            },
            {
                "name": "qwen-2.5-72b-instruct",
                "provider": "Alibaba",
                "parameter_count": "72B",
                "params_b": 72.0,
                "context_length": 131072,
                "use_case": "general",
                "category": "General",
                "release_date": "2025-01-15",
                "fit_level": "good",
                "run_mode": "gpu",
                "score": 88.7,
                "estimated_tps": 38.1,
                "runtime": "vLLM",
                "best_quant": "4bit",
                "memory_required_gb": 45.2,
                "utilization_pct": 72.5,
            },
        ],
    }
)


@pytest.fixture
def mock_ssh_client() -> MagicMock:
    client = MagicMock(spec=SSHClient)
    client.run = AsyncMock()
    return client


@pytest.fixture
def runner(mock_ssh_client: MagicMock) -> LLMFitRunner:
    return LLMFitRunner(ssh_client=mock_ssh_client)


class TestRecommend:
    @pytest.mark.asyncio
    async def test_parses_valid_json(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.return_value = (FIXTURE_JSON, "", 0)
        result = await runner.recommend("gpu-host-01")

        assert result.system.has_gpu is True
        assert result.system.gpu_vram_gb == 80.0
        assert result.system.backend == "CUDA"
        assert len(result.models) == 2
        assert result.models[0].name == "llama-3.3-70b"
        assert result.models[0].score == 95.2
        assert result.models[1].fit_level == "good"
        mock_ssh_client.run.assert_called_once_with(
            "gpu-host-01",
            "/usr/local/bin/llmfit recommend --json --runtime vllm -n 30",
            timeout=60.0,
        )


class TestRecommendTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises_typed_error(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.side_effect = TimeoutError()
        with pytest.raises(LLMFitTimeoutError) as exc_info:
            await runner.recommend("gpu-host-01")
        assert exc_info.value.host == "gpu-host-01"
        assert exc_info.value.timeout == 60.0

    @pytest.mark.asyncio
    async def test_install_timeout_raises_typed_error(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.side_effect = [
            RemoteCommandError("gpu-host-01", "llmfit recommend", 127),
            TimeoutError(),
        ]

        with pytest.raises(LLMFitTimeoutError) as caught:
            await runner.recommend("gpu-host-01")

        assert caught.value.host == "gpu-host-01"
        assert caught.value.timeout == 60.0
        assert mock_ssh_client.run.await_count == 2


class TestFirstRecommendationInstall:
    """T23: first use installs the requested binary and retries exactly once."""

    @pytest.mark.asyncio
    async def test_first_recommendation_installs_then_retries_exactly_once(
        self, mock_ssh_client: MagicMock
    ) -> None:
        from inference_proxy.config.settings import LLMFitSettings

        settings = LLMFitSettings(
            version="2.3.4",
            sha256="a" * 64,
            binary_path="/opt/llmfit",
            timeout=17.0,
            install_url="https://downloads.example/llmfit-{version}.tar.gz",
        )
        runner = LLMFitRunner(mock_ssh_client, settings)
        recommend_command = "/opt/llmfit recommend --json --runtime vllm -n 30"
        mock_ssh_client.run.side_effect = [
            RemoteCommandError("gpu-host-01", recommend_command, 127),
            ("", "", 0),
            (FIXTURE_JSON, "", 0),
        ]

        result = await runner.recommend("gpu-host-01")

        assert len(result.models) == 2
        assert mock_ssh_client.run.await_args_list[0] == call(
            "gpu-host-01", recommend_command, timeout=17.0
        )
        install_call = mock_ssh_client.run.await_args_list[1]
        assert install_call.args[0] == "gpu-host-01"
        assert install_call.kwargs == {"timeout": 17.0}
        shell_command = shlex.split(install_call.args[1])
        assert shell_command[:2] == ["bash", "-c"]
        assert "https://downloads.example/llmfit-2.3.4.tar.gz" in shell_command[2]
        assert "a" * 64 in shell_command[2]
        assert "sha256sum -c -" in shell_command[2]
        assert mock_ssh_client.run.await_args_list[2] == call(
            "gpu-host-01", recommend_command, timeout=17.0
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_status", [1, 2, 126])
    async def test_non_command_not_found_failure_does_not_install(
        self,
        mock_ssh_client: MagicMock,
        exit_status: int,
    ) -> None:
        error = RemoteCommandError(
            "gpu-host-01", "llmfit recommend", exit_status, "recommend failed"
        )
        mock_ssh_client.run.side_effect = error
        runner = LLMFitRunner(mock_ssh_client)

        with pytest.raises(RemoteCommandError) as caught:
            await runner.recommend("gpu-host-01")

        assert caught.value is error
        mock_ssh_client.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_install_checksum_mismatch_never_reaches_root_install(
    mock_ssh_client: MagicMock,
    tmp_path: Path,
) -> None:
    from inference_proxy.config.settings import LLMFitSettings

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    operation_log = tmp_path / "operations.log"
    wget = fake_bin / "wget"
    wget.write_text(
        "#!/bin/bash\n"
        'echo wget >> "$LLMFIT_TEST_LOG"\n'
        'while [[ "$#" -gt 0 ]]; do\n'
        '  if [[ "$1" == "-O" ]]; then printf bad > "$2"; exit 0; fi\n'
        "  shift\n"
        "done\n"
        "exit 2\n"
    )
    wget.chmod(0o755)
    for command in ("tar", "sudo"):
        executable = fake_bin / command
        executable.write_text(
            f'#!/bin/bash\necho {command} >> "$LLMFIT_TEST_LOG"\nexit 99\n'
        )
        executable.chmod(0o755)

    settings = LLMFitSettings(
        binary_path=str(tmp_path / "llmfit"),
        sha256="a" * 64,
        install_url="https://downloads.example/llmfit-{version}.tar.gz",
    )
    runner = LLMFitRunner(mock_ssh_client, settings)
    await runner._install("gpu-host-01")
    command = mock_ssh_client.run.await_args.args[1]

    result = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "LLMFIT_TEST_LOG": str(operation_log),
        },
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert operation_log.read_text().splitlines() == ["wget"]


class TestRecommendEmptyOutput:
    @pytest.mark.asyncio
    async def test_empty_stdout_raises_parse_error(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.return_value = ("", "", 0)
        with pytest.raises(LLMFitParseError) as exc_info:
            await runner.recommend("gpu-host-01")
        assert "empty output" in str(exc_info.value)


class TestRecommendInvalidJSON:
    @pytest.mark.asyncio
    async def test_invalid_json_raises_parse_error(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.return_value = ("not json", "", 0)
        with pytest.raises(LLMFitParseError) as exc_info:
            await runner.recommend("gpu-host-01")
        assert exc_info.value.raw_output == "not json"


class TestRecommendValidationError:
    @pytest.mark.asyncio
    async def test_missing_required_field_raises_parse_error(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.return_value = ('{"wrong": "structure"}', "", 0)
        with pytest.raises(LLMFitParseError):
            await runner.recommend("gpu-host-01")


class TestRecommendProviderFilter:
    @pytest.mark.asyncio
    async def test_filters_by_allowed_providers(
        self, mock_ssh_client: MagicMock
    ) -> None:
        from inference_proxy.config.settings import LLMFitSettings

        settings = LLMFitSettings(allowed_providers=["Meta"])
        runner = LLMFitRunner(ssh_client=mock_ssh_client, settings=settings)
        mock_ssh_client.run.return_value = (FIXTURE_JSON, "", 0)
        result = await runner.recommend("gpu-host-01")

        assert len(result.models) == 1
        assert result.models[0].provider == "Meta"

    @pytest.mark.asyncio
    async def test_filter_is_case_insensitive(self, mock_ssh_client: MagicMock) -> None:
        from inference_proxy.config.settings import LLMFitSettings

        settings = LLMFitSettings(allowed_providers=["alibaba"])
        runner = LLMFitRunner(ssh_client=mock_ssh_client, settings=settings)
        mock_ssh_client.run.return_value = (FIXTURE_JSON, "", 0)
        result = await runner.recommend("gpu-host-01")

        assert len(result.models) == 1
        assert result.models[0].provider == "Alibaba"

    @pytest.mark.asyncio
    async def test_empty_allowed_providers_returns_all(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.return_value = (FIXTURE_JSON, "", 0)
        result = await runner.recommend("gpu-host-01")

        assert len(result.models) == 2

    @pytest.mark.asyncio
    async def test_caps_filtered_results_to_ten(
        self, mock_ssh_client: MagicMock
    ) -> None:
        from inference_proxy.config.settings import LLMFitSettings

        payload = json.loads(FIXTURE_JSON)
        template = payload["models"][0]
        payload["models"] = [
            {**template, "name": f"meta-model-{index}"} for index in range(12)
        ]
        mock_ssh_client.run.return_value = (json.dumps(payload), "", 0)
        runner = LLMFitRunner(
            ssh_client=mock_ssh_client,
            settings=LLMFitSettings(allowed_providers=["Meta"]),
        )

        result = await runner.recommend("gpu-host-01")

        assert len(result.models) == 10
        assert [model.name for model in result.models] == [
            f"meta-model-{index}" for index in range(10)
        ]


class TestRecommendSSHErrorBubbles:
    @pytest.mark.asyncio
    async def test_ssh_connection_error_not_caught(
        self, runner: LLMFitRunner, mock_ssh_client: MagicMock
    ) -> None:
        mock_ssh_client.run.side_effect = SSHConnectionError("host1", "refused")
        with pytest.raises(SSHConnectionError):
            await runner.recommend("host1")
