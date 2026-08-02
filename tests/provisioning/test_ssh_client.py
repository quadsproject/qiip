"""Unit tests for the SSHClient wrapper.

All tests mock asyncssh to avoid requiring real SSH connections.
Tests verify connection parameters, stdout/stderr streaming,
error handling, and DIP compliance.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

import inference_proxy.provisioning.ssh_client as ssh_client_module
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
        "streaming_command_timeout": 1.0,
        "streaming_inactivity_timeout": 0.5,
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
    mock_asyncssh.Error = asyncssh.Error
    mock_asyncssh.PermissionDenied = type("PermissionDenied", (Exception,), {})
    mock_asyncssh.DisconnectError = type(
        "DisconnectError", (Exception,), {"reason": ""}
    )

    mock_process = MagicMock()
    mock_process.stdout = _AsyncLineIter(stdout_lines or [])
    mock_process.stderr = _AsyncLineIter(stderr_text.splitlines(keepends=True))
    mock_process.exit_status = exit_status

    mock_conn = MagicMock()
    mock_conn.create_process = MagicMock(return_value=_AsyncCM(mock_process))

    mock_asyncssh.connect = MagicMock(return_value=_AsyncCM(mock_conn))


def _setup_streaming_process(
    mock_asyncssh: MagicMock,
    *,
    stdout: AsyncIterator[str],
    stderr: AsyncIterator[str],
) -> None:
    """Install a process with caller-controlled asynchronous streams."""
    mock_asyncssh.Error = asyncssh.Error
    mock_asyncssh.PermissionDenied = asyncssh.PermissionDenied
    mock_asyncssh.DisconnectError = asyncssh.DisconnectError
    process = MagicMock(
        stdout=stdout,
        stderr=_AsyncStreamAdapter(stderr),
        exit_status=0,
    )
    connection = MagicMock()
    connection.create_process.return_value = _AsyncCM(process)
    mock_asyncssh.connect.return_value = _AsyncCM(connection)


async def _collect_stream(client: SSHClient, command: str) -> list[tuple[str, str]]:
    return [item async for item in client.run_streaming("host1", command)]


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


class TestSSHClientConcurrentStreaming:
    """P1: both pipes are drained and bounded without relying on test timeouts."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_drains_stderr_concurrently(
        self, mock_asyncssh: MagicMock
    ) -> None:
        stderr_drained = asyncio.Event()

        async def stdout() -> AsyncIterator[str]:
            await stderr_drained.wait()
            yield "stdout-ready\n"

        async def stderr() -> AsyncIterator[str]:
            for index in range(4096):
                yield f"stderr-{index}\n"
            stderr_drained.set()

        _setup_streaming_process(
            mock_asyncssh,
            stdout=stdout(),
            stderr=stderr(),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=1.0,
                streaming_inactivity_timeout=0.5,
            )
        )

        async def collect() -> list[tuple[str, str]]:
            return [item async for item in client.run_streaming("host1", "setup")]

        try:
            lines = await asyncio.wait_for(collect(), timeout=0.75)
        except TimeoutError:
            pytest.fail("run_streaming blocked on stdout before draining stderr")

        assert ("stdout", "stdout-ready") in lines
        assert sum(stream == "stderr" for stream, _line in lines) == 4096

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_applies_shared_bounded_backpressure(
        self,
        mock_asyncssh: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ssh_client_module, "_STREAM_QUEUE_MAXSIZE", 2)
        produced = {"stdout": 0, "stderr": 0}
        producers_advanced = asyncio.Event()

        async def source(stream: str) -> AsyncIterator[str]:
            for index in range(100):
                produced[stream] += 1
                if sum(produced.values()) >= 4:
                    producers_advanced.set()
                yield f"{stream}-{index}\n"

        _setup_streaming_process(
            mock_asyncssh,
            stdout=source("stdout"),
            stderr=source("stderr"),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=1.0,
                streaming_inactivity_timeout=0.5,
            )
        )
        stream = client.run_streaming("host1", "bounded")

        first = await anext(stream)
        await asyncio.wait_for(producers_advanced.wait(), timeout=0.1)

        # With an unbounded queue, each synchronous producer drains all 100
        # lines before this task resumes. A size-two queue instead blocks both
        # pumps after only the buffered and in-flight lines are produced.
        assert sum(produced.values()) < 10

        remaining = [item async for item in stream]
        all_lines = [first, *remaining]
        assert sum(kind == "stdout" for kind, _line in all_lines) == 100
        assert sum(kind == "stderr" for kind, _line in all_lines) == 100

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_inactivity_timeout_is_independently_bounded(
        self, mock_asyncssh: MagicMock
    ) -> None:
        async def never() -> AsyncIterator[str]:
            await asyncio.Event().wait()
            yield "unreachable"

        _setup_streaming_process(
            mock_asyncssh,
            stdout=never(),
            stderr=never(),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=1.0,
                streaming_inactivity_timeout=0.02,
            )
        )

        async def assert_production_timeout() -> None:
            with pytest.raises(TimeoutError) as caught:
                async for _ in client.run_streaming("host1", "stalled"):
                    pass
            assert getattr(caught.value, "deadline", None) == "inactivity"

        try:
            await asyncio.wait_for(assert_production_timeout(), timeout=0.25)
        except TimeoutError:
            pytest.fail("test safety deadline fired before the inactivity deadline")

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_inactivity_deadline_resets_on_output(
        self, mock_asyncssh: MagicMock
    ) -> None:
        async def stdout() -> AsyncIterator[str]:
            for index in range(5):
                await asyncio.sleep(0.015)
                yield f"progress-{index}\n"

        async def stderr() -> AsyncIterator[str]:
            if False:  # pragma: no cover - defines an empty async generator
                yield ""

        _setup_streaming_process(
            mock_asyncssh,
            stdout=stdout(),
            stderr=stderr(),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=0.5,
                streaming_inactivity_timeout=0.03,
            )
        )

        lines = await asyncio.wait_for(
            _collect_stream(client, "progress"),
            timeout=0.4,
        )

        assert [line for _stream, line in lines] == [
            f"progress-{index}" for index in range(5)
        ]

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_total_timeout_is_independently_bounded(
        self, mock_asyncssh: MagicMock
    ) -> None:
        async def chatter() -> AsyncIterator[str]:
            index = 0
            while True:
                await asyncio.sleep(0.005)
                yield f"progress-{index}\n"
                index += 1

        async def stderr() -> AsyncIterator[str]:
            if False:  # pragma: no cover - defines an empty async generator
                yield ""

        _setup_streaming_process(
            mock_asyncssh,
            stdout=chatter(),
            stderr=stderr(),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=0.04,
                streaming_inactivity_timeout=0.02,
            )
        )

        async def assert_production_timeout() -> None:
            with pytest.raises(TimeoutError) as caught:
                async for _ in client.run_streaming("host1", "chatty"):
                    pass
            assert getattr(caught.value, "deadline", None) == "total"

        try:
            await asyncio.wait_for(assert_production_timeout(), timeout=0.25)
        except TimeoutError:
            pytest.fail("test safety deadline fired before the total deadline")

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_total_timeout_does_not_cancel_output_consumer(
        self, mock_asyncssh: MagicMock
    ) -> None:
        output_started = asyncio.Event()

        async def stdout() -> AsyncIterator[str]:
            output_started.set()
            yield "started\n"
            await asyncio.Event().wait()

        async def stderr() -> AsyncIterator[str]:
            if False:  # pragma: no cover - defines an empty async generator
                yield ""

        _setup_streaming_process(
            mock_asyncssh,
            stdout=stdout(),
            stderr=stderr(),
        )
        client = SSHClient(
            _make_settings(
                streaming_command_timeout=0.03,
                streaming_inactivity_timeout=0.2,
            )
        )

        async def consume_slowly() -> None:
            stream = client.run_streaming("host1", "slow-consumer")
            assert await anext(stream) == ("stdout", "started")
            await asyncio.wait_for(output_started.wait(), timeout=0.05)

            # A deadline owned by the async generator would cancel this caller
            # while it processes a yielded line. The SSH supervisor must only
            # stop the remote command and report the timeout on the next read.
            await asyncio.sleep(0.06)

            with pytest.raises(TimeoutError) as caught:
                await anext(stream)
            assert getattr(caught.value, "deadline", None) == "total"

        try:
            await asyncio.wait_for(consume_slowly(), timeout=0.25)
        except TimeoutError:
            pytest.fail("the SSH deadline cancelled or hung the output consumer")


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
        mock_asyncssh.Error = asyncssh.Error
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
        mock_asyncssh.Error = asyncssh.Error
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


class TestAsyncSSHErrorMapping:
    """P2: every public operation normalizes the asyncssh error hierarchy."""

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_streaming_wraps_channel_open_error(
        self, mock_asyncssh: MagicMock
    ) -> None:
        error = asyncssh.ChannelOpenError(
            asyncssh.OPEN_CONNECT_FAILED,
            "maximum sessions exceeded",
        )
        _install_real_asyncssh_errors(mock_asyncssh)
        mock_asyncssh.connect.return_value = _AsyncCMRaises(error)
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as caught:
            async for _ in client.run_streaming("host1", "cmd"):
                pass

        assert caught.value.host == "host1"
        assert caught.value.__cause__ is error

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_run_wraps_key_import_error(self, mock_asyncssh: MagicMock) -> None:
        error = asyncssh.KeyImportError("malformed key")
        _install_real_asyncssh_errors(mock_asyncssh)
        mock_asyncssh.connect.return_value = _AsyncCMRaises(error)
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as caught:
            await client.run("host1", "cmd")

        assert caught.value.host == "host1"
        assert caught.value.__cause__ is error

    @pytest.mark.asyncio
    @patch("inference_proxy.provisioning.ssh_client.asyncssh")
    async def test_upload_wraps_sftp_error(self, mock_asyncssh: MagicMock) -> None:
        error = asyncssh.SFTPError(asyncssh.FX_FAILURE, "disk full")
        _install_real_asyncssh_errors(mock_asyncssh)
        mock_asyncssh.connect.return_value = _AsyncCM(MagicMock())
        mock_asyncssh.scp = AsyncMock(side_effect=error)
        client = SSHClient(_make_settings())

        with pytest.raises(SSHConnectionError) as caught:
            await client.upload("host1", Path("/tmp/scripts"))

        assert caught.value.host == "host1"
        assert caught.value.__cause__ is error


# -- Helpers --


class _AsyncLineIter:
    """Async iterable that yields lines from a list."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def read(self) -> str:
        """Compatibility with the pre-PR sequential stderr reader."""
        return "".join(self._lines)


class _AsyncStreamAdapter:
    """Expose one stream through both iteration and legacy bulk reads."""

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._source = source.__aiter__()

    def __aiter__(self) -> _AsyncStreamAdapter:
        return self

    async def __anext__(self) -> str:
        return await self._source.__anext__()

    async def read(self) -> str:
        return "".join([line async for line in self])


def _install_real_asyncssh_errors(mock_asyncssh: MagicMock) -> None:
    mock_asyncssh.Error = asyncssh.Error
    mock_asyncssh.PermissionDenied = asyncssh.PermissionDenied
    mock_asyncssh.DisconnectError = asyncssh.DisconnectError


class _AsyncCM:
    """Minimal async context manager wrapping a return value."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


class _AsyncCMRaises:
    """Async context manager that raises on __aenter__."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __aenter__(self) -> object:
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
    mock_asyncssh.Error = asyncssh.Error
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
        mock_asyncssh.Error = asyncssh.Error
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
        mock_asyncssh.Error = asyncssh.Error
        mock_asyncssh.DisconnectError = type(
            "DisconnectError", (Exception,), {"reason": ""}
        )

        mock_conn = MagicMock()
        mock_conn.run = AsyncMock(side_effect=TimeoutError())
        mock_asyncssh.connect = MagicMock(return_value=_AsyncCM(mock_conn))
        client = SSHClient(_make_settings())

        with pytest.raises(asyncio.TimeoutError):
            await client.run("host1", "slow-cmd")
