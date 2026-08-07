"""Thin wrapper around asyncssh for remote command execution.

This module is the **sole consumer** of ``asyncssh`` in the codebase,
following the Dependency Inversion Principle (DIP): all other modules
depend on this wrapper rather than importing ``asyncssh`` directly.

Per D-14: Mirrors the EtcdClient wrapper pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import asyncssh
import structlog

from inference_proxy.config.settings import SSHSettings

logger = structlog.get_logger()

# KeyImportError is a ValueError in asyncssh 2.24 rather than part of the
# asyncssh.Error hierarchy. Keep it beside the hierarchy so malformed key
# material is normalized consistently with connection and channel failures.
_ASYNCSSH_CONNECTION_ERRORS = (asyncssh.Error, asyncssh.KeyImportError)
_STREAM_QUEUE_MAXSIZE = 1024
# Remote tools can write arbitrary model metadata bytes to their logs. Preserve
# the UTF-8 text contract without allowing one invalid byte to abort an SSH job.
_REMOTE_TEXT_ENCODING = "utf-8"
_REMOTE_TEXT_ERRORS = "replace"


class SSHConnectionError(Exception):
    """Raised when SSH connection to a host fails."""

    def __init__(self, host: str, reason: str) -> None:
        self.host = host
        self.reason = reason
        super().__init__(f"SSH connection to {host} failed: {reason}")


class SSHCommandTimeoutError(TimeoutError):
    """Raised when a streaming SSH command exceeds a bounded deadline."""

    def __init__(
        self,
        host: str,
        command: str,
        timeout: float,
        *,
        deadline: str,
    ) -> None:
        self.host = host
        self.command = command
        self.timeout = timeout
        self.deadline = deadline
        super().__init__(
            f"Streaming command on {host} exceeded the {deadline} timeout of {timeout}s"
        )


class RemoteCommandError(Exception):
    """Raised when a remote command exits with non-zero status."""

    def __init__(
        self, host: str, command: str, exit_status: int, stderr: str = ""
    ) -> None:
        self.host = host
        self.command = command
        self.exit_status = exit_status
        self.stderr = stderr
        tail = _stderr_tail(stderr) if stderr else ""
        msg = f"Command '{command}' on {host} exited with status {exit_status}"
        if tail:
            msg += f"\n--- stderr (last 50 lines) ---\n{tail}"
        super().__init__(msg)


def _stderr_tail(stderr: str, max_lines: int = 50) -> str:
    lines = stderr.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[-max_lines:])
    return stderr


@dataclass(frozen=True)
class _StreamLine:
    stream: str
    value: str


@dataclass(frozen=True)
class _StreamError:
    error: Exception


@dataclass(frozen=True)
class _StreamDone:
    pass


_StreamEvent = _StreamLine | _StreamError | _StreamDone


class SSHClient:
    """Sole consumer of asyncssh in the codebase (DIP).

    Wraps ``asyncssh`` to provide streaming remote command execution
    with typed error handling.  Constructor extracts values from
    ``SSHSettings`` rather than storing the settings object.
    """

    def __init__(self, settings: SSHSettings) -> None:
        self._username = settings.username
        self._key_path = settings.key_path
        self._connect_timeout = settings.connect_timeout
        self._streaming_command_timeout = settings.streaming_command_timeout
        self._streaming_inactivity_timeout = settings.streaming_inactivity_timeout

    async def run_streaming(
        self,
        host: str,
        command: str,
        *,
        total_timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Run *command* on *host*, yielding ``(stream, line)`` tuples.

        *stream* is ``"stdout"`` or ``"stderr"``. Both streams are drained
        concurrently so either remote pipe can exceed asyncssh's receive
        window without deadlocking the other. The total command and
        no-output intervals are bounded independently. ``total_timeout`` may
        extend one known-long operation without weakening the client default.

        Raises:
            SSHConnectionError: On auth failure, disconnect, or OS error.
            SSHCommandTimeoutError: On total or inactivity deadline expiry.
            RemoteCommandError: When the remote process exits non-zero.
        """
        # Bound buffering without returning to the old one-stream-at-a-time
        # deadlock: both pumps share the same queue and receive backpressure
        # symmetrically when a consumer falls behind.
        effective_total_timeout = (
            self._streaming_command_timeout if total_timeout is None else total_timeout
        )
        if effective_total_timeout <= 0:
            raise ValueError("total_timeout must be greater than zero")

        queue: asyncio.Queue[_StreamEvent] = asyncio.Queue(
            maxsize=_STREAM_QUEUE_MAXSIZE
        )

        async def run_remote() -> None:
            stderr_chunks: list[str] = []

            async def pump(stream: str, reader: AsyncIterator[str]) -> None:
                async for line in reader:
                    if stream == "stderr":
                        stderr_chunks.append(line)
                    await queue.put(_StreamLine(stream, line))

            async with (
                asyncssh.connect(
                    host,
                    username=self._username,
                    client_keys=[str(self._key_path)],
                    known_hosts=None,  # D-03: lab servers reimaged frequently
                    connect_timeout=self._connect_timeout,
                ) as conn,
                conn.create_process(
                    command,
                    encoding=_REMOTE_TEXT_ENCODING,
                    errors=_REMOTE_TEXT_ERRORS,
                ) as process,
            ):
                pumps = [
                    asyncio.create_task(
                        pump("stdout", cast(AsyncIterator[str], process.stdout))
                    ),
                    asyncio.create_task(
                        pump("stderr", cast(AsyncIterator[str], process.stderr))
                    ),
                ]
                try:
                    done, pending = await asyncio.wait(
                        pumps,
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    failure: BaseException | None = None
                    for task in done:
                        if not task.cancelled():
                            failure = task.exception()
                            if failure is not None:
                                break
                    if failure is not None:
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        raise failure
                    await asyncio.gather(*pending)
                finally:
                    for task in pumps:
                        task.cancel()
                    await asyncio.gather(*pumps, return_exceptions=True)

                stderr_output = "".join(stderr_chunks)
                if process.exit_status is not None and process.exit_status != 0:
                    raise RemoteCommandError(
                        host,
                        command,
                        process.exit_status,
                        stderr=stderr_output,
                    )

        async def supervise() -> None:
            command_timeout = asyncio.timeout(effective_total_timeout)
            try:
                async with command_timeout:
                    await run_remote()
            except asyncio.CancelledError:
                raise
            except TimeoutError as timeout_error:
                error: Exception = timeout_error
                if command_timeout.expired():
                    error = SSHCommandTimeoutError(
                        host,
                        command,
                        effective_total_timeout,
                        deadline="total",
                    )
                await queue.put(_StreamError(error))
            except Exception as exc:
                await queue.put(_StreamError(exc))
            # Deliberately skipped on external cancellation: the consumer is
            # already unwinding and may have left a full queue behind.
            await queue.put(_StreamDone())

        try:
            supervisor = asyncio.create_task(supervise())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(),
                            timeout=self._streaming_inactivity_timeout,
                        )
                    except TimeoutError as exc:
                        raise SSHCommandTimeoutError(
                            host,
                            command,
                            self._streaming_inactivity_timeout,
                            deadline="inactivity",
                        ) from exc

                    if isinstance(event, _StreamDone):
                        break
                    if isinstance(event, _StreamError):
                        raise event.error
                    yield (event.stream, event.value.rstrip("\n"))
            finally:
                supervisor.cancel()
                await asyncio.gather(supervisor, return_exceptions=True)
        except asyncssh.PermissionDenied as exc:
            raise SSHConnectionError(host, f"authentication failed: {exc}") from exc
        except asyncssh.DisconnectError as exc:
            raise SSHConnectionError(host, f"disconnected: {exc.reason}") from exc
        except TimeoutError:
            # TimeoutError is an OSError subclass. Preserve both SSH command
            # deadlines and any caller-visible asyncio timeout unchanged.
            raise
        except _ASYNCSSH_CONNECTION_ERRORS as exc:
            raise SSHConnectionError(host, str(exc)) from exc
        except OSError as exc:
            raise SSHConnectionError(host, str(exc)) from exc

    async def run(
        self,
        host: str,
        command: str,
        timeout: float = 60.0,
    ) -> tuple[str, str, int]:
        """Run *command* on *host*, return ``(stdout, stderr, exit_status)``.

        Timeout via ``asyncio.wait_for`` (D-02).  Raises
        ``SSHConnectionError`` on auth/disconnect/OS errors.  Raises
        ``RemoteCommandError`` on non-zero exit.
        ``asyncio.TimeoutError`` bubbles to caller.
        """
        log = logger.bind(host=host, command=command)
        log.debug("ssh_run_start")
        try:
            async with asyncssh.connect(
                host,
                username=self._username,
                client_keys=[str(self._key_path)],
                known_hosts=None,
                connect_timeout=self._connect_timeout,
            ) as conn:
                result = await asyncio.wait_for(
                    conn.run(
                        command,
                        encoding=_REMOTE_TEXT_ENCODING,
                        errors=_REMOTE_TEXT_ERRORS,
                    ),
                    timeout=timeout,
                )
                exit_status = (
                    result.exit_status if result.exit_status is not None else 0
                )
                stdout = cast(str, result.stdout or "")
                stderr = cast(str, result.stderr or "")

                if exit_status != 0:
                    raise RemoteCommandError(
                        host,
                        command,
                        exit_status,
                        stderr=stderr,
                    )

                log.debug("ssh_run_complete", exit_status=exit_status)
                return (stdout, stderr, exit_status)
        except asyncssh.PermissionDenied as exc:
            raise SSHConnectionError(host, f"authentication failed: {exc}") from exc
        except asyncssh.DisconnectError as exc:
            raise SSHConnectionError(host, f"disconnected: {exc.reason}") from exc
        except TimeoutError:
            raise  # asyncio.TimeoutError is TimeoutError is OSError in 3.11+
        except _ASYNCSSH_CONNECTION_ERRORS as exc:
            raise SSHConnectionError(host, str(exc)) from exc
        except OSError as exc:
            raise SSHConnectionError(host, str(exc)) from exc

    async def upload(
        self,
        host: str,
        local_path: Path,
        remote_path: str = ".",
    ) -> None:
        """Copy *local_path* to *host* via SCP.

        Directories are copied recursively.
        """
        try:
            async with asyncssh.connect(
                host,
                username=self._username,
                client_keys=[str(self._key_path)],
                known_hosts=None,
                connect_timeout=self._connect_timeout,
            ) as conn:
                await asyncssh.scp(
                    str(local_path),
                    (conn, remote_path),
                    recurse=True,
                )
        except asyncssh.PermissionDenied as exc:
            raise SSHConnectionError(host, f"authentication failed: {exc}") from exc
        except asyncssh.DisconnectError as exc:
            raise SSHConnectionError(host, f"disconnected: {exc.reason}") from exc
        except _ASYNCSSH_CONNECTION_ERRORS as exc:
            raise SSHConnectionError(host, str(exc)) from exc
        except OSError as exc:
            raise SSHConnectionError(host, str(exc)) from exc
