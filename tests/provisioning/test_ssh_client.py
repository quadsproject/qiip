"""Unit tests for the SSHClient wrapper.

All tests mock asyncssh to avoid requiring real SSH connections.
Tests verify connection parameters, stdout/stderr streaming,
error handling, and DIP compliance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_proxy.config.settings import SSHSettings
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)


def _make_settings(**overrides: object) -> SSHSettings:
    defaults: dict[str, object] = {
        "key_path": Path("/tmp/test_key"),
        "username": "testuser",
        "connect_timeout": 5,
    }
    defaults.update(overrides)
    return SSHSettings(**defaults)


def _setup_mock_asyncssh(
    mock_asyncssh: MagicMock,
    stdout_lines: list[str] | None = None,
    stderr_text: str = "",
    exit_status: int = 0,
) -> None:
    """Wire up mock_asyncssh with real exception classes and a mock process."""
    # Must set real exception classes so `except asyncssh.X` works
    mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
    mock_asyncssh.DisconnectError = type(
        "DisconnectError", (Exception,), {"reason": ""}
    )

    mock_process = MagicMock()
    mock_process.stdout = _AsyncLineIter(stdout_lines or [])
    mock_process.stderr = MagicMock()
    mock_process.stderr.read = AsyncMock(return_value=stderr_text)
    mock_process.exit_status = exit_status

    mock_conn = MagicMock()
    mock_conn.create_process = MagicMock(return_value=_AsyncCM(mock_process))

    mock_asyncssh.connect = MagicMock(return_value=_AsyncCM(mock_conn))


class TestSSHClientConnectParams:
    """D-03, D-04: asyncssh.connect called with correct parameters."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_connect_params(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh(mock_asyncssh)
        client = SSHClient(_make_settings())

        _ = [line async for line in client.run_streaming("host1", "echo hi")]

        mock_asyncssh.connect.assert_called_once_with(
            "host1",
            username="testuser",
            client_keys=["/tmp/test_key"],
            known_hosts=None,
            connect_timeout=5,
        )


class TestSSHClientStdoutStreaming:
    """D-05: run_streaming yields (stdout, line) tuples."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_yields_stdout_lines(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh(mock_asyncssh, stdout_lines=["line1\n", "line2\n"])
        client = SSHClient(_make_settings())

        lines = [line async for line in client.run_streaming("host1", "ls")]

        assert lines == [("stdout", "line1"), ("stdout", "line2")]


class TestSSHClientStderrStreaming:
    """D-07: stderr lines yielded separately after stdout."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_yields_stderr_lines(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh(
            mock_asyncssh, stdout_lines=["out\n"], stderr_text="warn1\nwarn2\n"
        )
        client = SSHClient(_make_settings())

        lines = [line async for line in client.run_streaming("host1", "cmd")]

        assert ("stdout", "out") in lines
        assert ("stderr", "warn1") in lines
        assert ("stderr", "warn2") in lines


class TestSSHClientNonZeroExit:
    """RemoteCommandError raised on non-zero exit status."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_raises_remote_command_error(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh(mock_asyncssh, exit_status=1)
        client = SSHClient(_make_settings())

        with pytest.raises(RemoteCommandError) as exc_info:
            async for _ in client.run_streaming("host1", "fail"):
                pass

        assert exc_info.value.host == "host1"
        assert exc_info.value.command == "fail"
        assert exc_info.value.exit_status == 1

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_error_includes_stderr(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh(
            mock_asyncssh, exit_status=1, stderr_text="error: package not found\n"
        )
        client = SSHClient(_make_settings())

        with pytest.raises(RemoteCommandError) as exc_info:
            async for _ in client.run_streaming("host1", "fail"):
                pass

        assert exc_info.value.stderr == "error: package not found\n"
        assert "package not found" in str(exc_info.value)


class TestSSHClientConnectionError:
    """SSHConnectionError wraps asyncssh auth/disconnect/OS errors."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_wraps_permission_denied(self, mock_asyncssh: MagicMock) -> None:
        mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
        mock_asyncssh.DisconnectError = type(
            "DisconnectError", (Exception,), {"reason": "test"}
        )
        mock_asyncssh.connect = MagicMock(
            return_value=_AsyncCMRaises(mock_asyncssh.PermissionDenied("denied"))
        )
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as exc_info:
            async for _ in client.run_streaming("host1", "cmd"):
                pass

        assert exc_info.value.host == "host1"
        assert "authentication failed" in exc_info.value.reason

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_wraps_os_error(self, mock_asyncssh: MagicMock) -> None:
        mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
        mock_asyncssh.DisconnectError = type(
            "DisconnectError", (Exception,), {"reason": "test"}
        )
        mock_asyncssh.connect = MagicMock(
            return_value=_AsyncCMRaises(OSError("Connection refused"))
        )
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as exc_info:
            async for _ in client.run_streaming("host1", "cmd"):
                pass

        assert exc_info.value.host == "host1"
        assert "Connection refused" in exc_info.value.reason


# -- Helpers --


class _AsyncLineIter:
    """Async iterable that yields lines from a list."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self):  # noqa: ANN204
        return self._iter()

    async def _iter(self):  # noqa: ANN201
        for line in self._lines:
            yield line


class _AsyncCM:
    """Minimal async context manager wrapping a return value."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self):  # noqa: ANN204
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


class _AsyncCMRaises:
    """Async context manager that raises on __aenter__."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self):  # noqa: ANN204
        raise self._exc

    async def __aexit__(self, *args: object) -> None:
        pass


def _setup_mock_asyncssh_run(
    mock_asyncssh: MagicMock,
    stdout: str = "",
    stderr: str = "",
    exit_status: int = 0,
) -> None:
    """Wire mock_asyncssh for SSHClient.run() (conn.run, not create_process)."""
    mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
    mock_asyncssh.DisconnectError = type(
        "DisconnectError", (Exception,), {"reason": ""}
    )

    mock_result = MagicMock()
    mock_result.stdout = stdout
    mock_result.stderr = stderr
    mock_result.exit_status = exit_status

    mock_conn = MagicMock()
    mock_conn.run = AsyncMock(return_value=mock_result)

    mock_asyncssh.connect = MagicMock(return_value=_AsyncCM(mock_conn))


class TestSSHClientRun:
    """SSHClient.run() returns (stdout, stderr, exit_status) tuple."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_returns_tuple(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh_run(
            mock_asyncssh, stdout="hello", stderr="", exit_status=0
        )
        client = SSHClient(_make_settings())

        result = await client.run("host1", "echo hello")
        assert result == ("hello", "", 0)


class TestSSHClientRunNonZeroExit:
    """SSHClient.run() raises RemoteCommandError on non-zero exit."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_raises_remote_command_error(self, mock_asyncssh: MagicMock) -> None:
        _setup_mock_asyncssh_run(mock_asyncssh, stderr="error", exit_status=1)
        client = SSHClient(_make_settings())

        with pytest.raises(RemoteCommandError) as exc_info:
            await client.run("host1", "fail")
        assert exc_info.value.exit_status == 1


class TestSSHClientRunConnectionError:
    """SSHClient.run() wraps asyncssh auth errors as SSHConnectionError."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_wraps_permission_denied(self, mock_asyncssh: MagicMock) -> None:
        mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
        mock_asyncssh.DisconnectError = type(
            "DisconnectError", (Exception,), {"reason": "test"}
        )
        mock_asyncssh.connect = MagicMock(
            return_value=_AsyncCMRaises(mock_asyncssh.PermissionDenied("denied"))
        )
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as exc_info:
            await client.run("host1", "cmd")
        assert "authentication failed" in exc_info.value.reason


class TestSSHClientRunTimeoutBubbles:
    """asyncio.TimeoutError from run() bubbles to caller."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_timeout_propagates(self, mock_asyncssh: MagicMock) -> None:
        mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
        mock_asyncssh.DisconnectError = type(
            "DisconnectError", (Exception,), {"reason": ""}
        )

        mock_conn = MagicMock()
        mock_conn.run = AsyncMock(side_effect=TimeoutError())
        mock_asyncssh.connect = MagicMock(return_value=_AsyncCM(mock_conn))
        client = SSHClient(_make_settings())

        with pytest.raises(asyncio.TimeoutError):
            await client.run("host1", "slow-cmd")
