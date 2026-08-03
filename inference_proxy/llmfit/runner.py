"""LLMFit remote execution via SSH.

Runs ``llmfit recommend --json --runtime vllm -n 30`` on a remote host
and returns a typed ``LLMFitResult``.  SSH transport is injected via
constructor (DIP).
"""

from __future__ import annotations

import json
import shlex

import structlog
from pydantic import ValidationError

from inference_proxy.config.settings import LLMFitSettings
from inference_proxy.llmfit.errors import LLMFitParseError, LLMFitTimeoutError
from inference_proxy.models.llmfit import LLMFitResult
from inference_proxy.models.node import InferenceEngine
from inference_proxy.provisioning.ssh_client import RemoteCommandError, SSHClient

_LLMFIT_RUNTIMES: dict[str, str] = {
    InferenceEngine.VLLM: "vllm",
    InferenceEngine.LLAMA_CPP: "llamacpp",
}

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
        """Install one verified llmfit release artifact on the remote host."""
        url = self._settings.install_url.format(version=self._settings.version)
        script = (
            "set -euo pipefail; "
            "work_dir=$(mktemp -d /tmp/llmfit-install.XXXXXX); "
            "trap 'rm -rf \"$work_dir\"' EXIT; "
            f'wget -q -- {shlex.quote(url)} -O "$work_dir/llmfit.tar.gz"; '
            f"printf '%s  %s\\n' {shlex.quote(self._settings.sha256)} "
            '"$work_dir/llmfit.tar.gz" | sha256sum -c - >/dev/null; '
            'tar -xzf "$work_dir/llmfit.tar.gz" -C "$work_dir"; '
            'mapfile -t binaries < <(find "$work_dir" -name llmfit -type f -print); '
            'if [ "${#binaries[@]}" -ne 1 ]; then '
            "echo 'verified llmfit archive must contain exactly one binary' >&2; "
            "exit 1; fi; "
            f'sudo install -m 755 "${{binaries[0]}}" '
            f"{shlex.quote(self._settings.binary_path)}"
        )
        cmd = shlex.join(("bash", "-c", script))
        await self._ssh.run(hostname, cmd, timeout=self._settings.timeout)

    async def _run_recommend(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> tuple[str, str, int]:
        """Run the llmfit recommend command, raising TimeoutError on timeout."""
        runtime = _LLMFIT_RUNTIMES.get(engine, "vllm")
        command = shlex.join(
            (
                self._settings.binary_path,
                "recommend",
                "--json",
                "--runtime",
                runtime,
                "-n",
                "30",
            )
        )
        return await self._ssh.run(hostname, command, timeout=self._settings.timeout)

    async def recommend(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> LLMFitResult:
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
            stdout, _stderr, _exit = await self._run_recommend(hostname, engine)
        except RemoteCommandError as exc:
            if exc.exit_status != 127:
                raise
            log.info("llmfit_not_found_installing", host=hostname)
            try:
                await self._install(hostname)
            except TimeoutError as exc:
                raise LLMFitTimeoutError(hostname, self._settings.timeout) from exc
            try:
                stdout, _stderr, _exit = await self._run_recommend(hostname, engine)
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
