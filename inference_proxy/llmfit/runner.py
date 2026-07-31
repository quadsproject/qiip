"""LLMFit remote execution via SSH.

Runs ``llmfit recommend --json --runtime vllm -n 30`` on a remote host
and returns a typed ``LLMFitResult``.  SSH transport is injected via
constructor (DIP).
"""

from __future__ import annotations

import json

import structlog
from pydantic import ValidationError

from inference_proxy.config.settings import LLMFitSettings
from inference_proxy.llmfit.errors import LLMFitParseError, LLMFitTimeoutError
from inference_proxy.models.llmfit import LLMFitResult
from inference_proxy.provisioning.ssh_client import RemoteCommandError, SSHClient

logger = structlog.get_logger()


class LLMFitRunner:
    """Run llmfit on a remote host and parse the result.

    Constructor-injected ``SSHClient`` keeps this testable without SSH.
    Settings are optional for backward compatibility.
    """

    def __init__(
        self, ssh_client: SSHClient, settings: LLMFitSettings | None = None
    ) -> None:
        self._ssh = ssh_client
        self._settings = settings or LLMFitSettings()

    async def _install(self, hostname: str) -> None:
        """Install llmfit binary on remote host via wget+tar."""
        url = self._settings.install_url.format(version=self._settings.version)
        cmd = (
            f"wget -q '{url}' -O /tmp/llmfit.tar.gz"
            " && tar -xzf /tmp/llmfit.tar.gz -C /tmp/"
            ' && sudo install -m 755 "$(find /tmp/ -name llmfit -type f -print -quit)" {binary}'
            " && rm -rf /tmp/llmfit.tar.gz /tmp/llmfit-*"
        ).format(binary=self._settings.binary_path)
        await self._ssh.run(hostname, cmd, timeout=self._settings.timeout)

    async def _run_recommend(self, hostname: str) -> tuple[str, str, int]:
        """Run the llmfit recommend command, raising TimeoutError on timeout."""
        command = f"{self._settings.binary_path} recommend --json --runtime vllm -n 30"
        return await self._ssh.run(hostname, command, timeout=self._settings.timeout)

    async def recommend(self, hostname: str) -> LLMFitResult:
        """Run llmfit on *hostname* and return parsed recommendations.

        If llmfit is not installed, installs it first then retries.

        Raises:
            LLMFitTimeoutError: When execution exceeds the configured timeout.
            LLMFitParseError: When stdout is empty, not valid JSON,
                or fails Pydantic validation.  Stores raw stdout.
            SSHConnectionError: Bubbles unchanged from SSHClient (D-03).
            RemoteCommandError: Bubbles unchanged from SSHClient (D-03).
        """
        log = logger.bind(host=hostname)
        log.debug("llmfit_recommend_start")

        try:
            stdout, _stderr, _exit = await self._run_recommend(hostname)
        except RemoteCommandError:
            log.info("llmfit_not_found_installing", host=hostname)
            try:
                await self._install(hostname)
            except TimeoutError as exc:
                raise LLMFitTimeoutError(hostname, self._settings.timeout) from exc
            try:
                stdout, _stderr, _exit = await self._run_recommend(hostname)
            except TimeoutError as exc:
                raise LLMFitTimeoutError(hostname, self._settings.timeout) from exc
        except TimeoutError as exc:
            raise LLMFitTimeoutError(hostname, self._settings.timeout) from exc

        if not stdout.strip():
            raise LLMFitParseError("empty output", raw_output=stdout)

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LLMFitParseError(str(exc), raw_output=stdout) from exc

        try:
            result = LLMFitResult.model_validate(data)
        except ValidationError as exc:
            raise LLMFitParseError(str(exc), raw_output=stdout) from exc

        if self._settings.allowed_providers:
            allowed = {p.lower() for p in self._settings.allowed_providers}
            result = LLMFitResult(
                system=result.system,
                models=[m for m in result.models if m.provider.lower() in allowed][:10],
            )

        log.debug("llmfit_recommend_complete", model_count=len(result.models))
        return result
