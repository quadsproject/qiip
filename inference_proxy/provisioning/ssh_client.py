"""Thin wrapper around asyncssh for remote command execution.

This module is the **sole consumer** of ``asyncssh`` in the codebase,
following the Dependency Inversion Principle (DIP): all other modules
depend on this wrapper rather than importing ``asyncssh`` directly.

Per D-14: Mirrors the EtcdClient wrapper pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import asyncssh
import structlog

from inference_proxy.config.settings import SSHSettings

logger = structlog.get_logger()


class SSHConnectionError(Exception):
    """Raised when SSH connection to a host fails."""

    def __init__(self, host: str, reason: str) -> None:
        self.host = host
        self.reason = reason
        super().__init__(f"SSH connection to {host} failed: {reason}")


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

    async def run_streaming(
        self, host: str, command: str
    ) -> AsyncIterator[tuple[str, str]]:
        """Run *command* on *host*, yielding ``(stream, line)`` tuples.

        *stream* is ``"stdout"`` or ``"stderr"``.  Stdout lines are
        yielded in real-time as they arrive (D-05).  After stdout is
        exhausted, any stderr output is read in bulk and yielded
        line-by-line (D-07).

        Raises:
            SSHConnectionError: On auth failure, disconnect, or OS error.
            RemoteCommandError: When the remote process exits non-zero.
        """
        try:
            async with (
                asyncssh.connect(
                    host,
                    username=self._username,
                    client_keys=[str(self._key_path)],
                    known_hosts=None,  # D-03: lab servers reimaged frequently
                    connect_timeout=self._connect_timeout,
                ) as conn,
                conn.create_process(command) as process,
            ):
                async for line in process.stdout:
                    yield ("stdout", line.rstrip("\n"))

                # D-07: read stderr after stdout exhausted
                stderr_output = await process.stderr.read()
                if stderr_output:
                    for err_line in stderr_output.splitlines():
                        if err_line:
                            yield ("stderr", err_line)

                if process.exit_status is not None and process.exit_status != 0:
                    raise RemoteCommandError(
                        host,
                        command,
                        process.exit_status,
                        stderr=stderr_output or "",
                    )
        except asyncssh.PermissionDenied as exc:
            raise SSHConnectionError(host, f"authentication failed: {exc}") from exc
        except asyncssh.DisconnectError as exc:
            raise SSHConnectionError(host, f"disconnected: {exc.reason}") from exc
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
                    conn.run(command),
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
        except OSError as exc:
            raise SSHConnectionError(host, str(exc)) from exc
