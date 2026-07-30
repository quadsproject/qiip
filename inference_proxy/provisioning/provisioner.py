"""Node provisioning orchestrator.

Runs the full provisioning sequence on a remote host: setup.sh,
start-vllm.sh, health poll, etcd registration.

Per D-15: Concrete class, no protocol/interface.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Coroutine
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog

from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)
from inference_proxy.provisioning.state import ProvisioningState, ProvisioningStep
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError
from inference_proxy.routing.connection_tracker import ConnectionTracker

if TYPE_CHECKING:
    from etcd3gw.types import KeyValue

logger = structlog.get_logger()

STEP_PATTERN = re.compile(r"\[STEP:(\w+):(START|OK|FAIL)\]")
MODEL_PATTERN = re.compile(r"#\s*Model:\s+(.+)")


class ProvisioningError(Exception):
    """Raised when any stage of provisioning fails."""


class PreflightError(Exception):
    """Raised when pre-flight validation fails (D-01 through D-04).

    Collects all failures before raising so operators see every problem
    at once (D-03).
    """

    def __init__(self, hostname: str, failures: list[str]) -> None:
        self.hostname = hostname
        self.failures = failures
        super().__init__(f"Pre-flight failed on {hostname}: {'; '.join(failures)}")


class NodeProvisioner:
    """Orchestrates full provisioning of a vLLM node on a remote host.

    Accepts SSHClient, EtcdClient, and ProvisioningSettings via
    constructor injection (DIP).
    """

    def __init__(
        self,
        ssh_client: SSHClient,
        etcd_client: EtcdClient,
        settings: ProvisioningSettings,
        registry: NodeRegistry | None = None,
        connection_tracker: ConnectionTracker | None = None,
        redfish_client: RedfishClient | None = None,
        log_buffer: ProvisioningLogBuffer | None = None,
    ) -> None:
        self._ssh_client = ssh_client
        self._etcd_client = etcd_client
        self._settings = settings
        self._registry = registry
        self._tracker = connection_tracker
        self._redfish_client = redfish_client
        self._log_buffer = log_buffer or ProvisioningLogBuffer()
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def log_buffer(self) -> ProvisioningLogBuffer:
        return self._log_buffer

    def _script_env_prefix(self) -> str:
        """Build env var prefix for remote script invocation."""
        s = self._settings
        return (
            f"NFS_SERVER={shlex.quote(s.nfs_server)} "
            f"NFS_MOUNT_POINT={shlex.quote(s.nfs_mount_point)} "
            f"NVIDIA_DRIVER_VERSION={shlex.quote(s.nvidia_driver_version)} "
            f"VLLM_PORT={s.vllm_port} "
            f"LLMFIT_VERSION={shlex.quote(s.llmfit_version)} "
        )

    def _log(
        self,
        hostname: str,
        level: str,
        msg: str,
        *,
        stream: str | None = None,
    ) -> None:
        self._log_buffer.append(hostname, level, msg, stream=stream)

    async def list_tasks_raw(self) -> list[tuple[bytes, KeyValue]]:
        """Return raw provisioning task entries from etcd."""
        return await asyncio.to_thread(self._etcd_client.get_prefix, "/provisioning/")

    async def _update_state(
        self,
        hostname: str,
        step: ProvisioningStep,
        *,
        failed_step: str | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """Write provisioning state to etcd (D-05). Best-effort (Pitfall 3)."""
        now = datetime.now(UTC)
        state = ProvisioningState(
            hostname=hostname,
            current_step=step,
            started_at=started_at or now,
            updated_at=now,
            failed_step=failed_step,
            error=error,
        )
        key = f"/provisioning/{hostname}"
        value = json.dumps(state.model_dump(mode="json")).encode("utf-8")
        try:
            await asyncio.to_thread(self._etcd_client.put, key, value)
        except Exception:
            logger.warning("state_write_failed", hostname=hostname, step=step)

    async def _ssh_run_command(self, hostname: str, command: str) -> str:
        """Run a command via SSH and return collected stdout as a string."""
        lines: list[str] = []
        async for stream, line in self._ssh_client.run_streaming(hostname, command):
            if stream == "stdout":
                lines.append(line)
        return "\n".join(lines)

    async def _power_on_if_needed(self, hostname: str) -> None:
        """Power on the host via Redfish if configured (D-01, D-06, D-07).

        Best-effort: RedfishError is caught and logged so provisioning
        can continue even if the BMC is unreachable (server may already be on).
        """
        if self._redfish_client is None:
            logger.info("redfish_not_configured", msg="skipping power check")
            return

        await self._update_state(hostname, ProvisioningStep.POWERING_ON)
        try:
            state = await self._redfish_client.power_action(hostname, "On")
            logger.info("power_on_result", hostname=hostname, state=state)
        except RedfishError as exc:
            logger.warning("power_on_failed", hostname=hostname, error=str(exc))

        await self._wait_for_ssh(hostname)

    async def _wait_for_ssh(self, hostname: str) -> None:
        """Wait for SSH port 22 to become reachable (D-03, D-05).

        Deadline-based retry loop mirroring _poll_health(). On timeout,
        logs a warning and returns -- preflight will fail naturally if
        SSH is truly unreachable.
        """
        deadline = asyncio.get_running_loop().time() + self._settings.boot_wait_timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(hostname, 22), timeout=10
                )
                writer.close()
                await writer.wait_closed()
                logger.info("ssh_ready", hostname=hostname)
                return
            except (OSError, TimeoutError):
                pass
            await asyncio.sleep(self._settings.boot_wait_interval)
        logger.warning(
            "ssh_wait_timeout",
            hostname=hostname,
            timeout=self._settings.boot_wait_timeout,
        )

    async def preflight(self, hostname: str) -> None:
        """Pre-flight validation: TCP probe + disk check (D-01, D-04).

        Stage 1: TCP probe to port 22.  If unreachable, raises immediately
        (cannot proceed to SSH diagnostics).

        Stage 2: Disk check via SSH.  Failures collected before raising
        a single PreflightError (D-03).

        Note: GPU check runs after setup.sh installs the NVIDIA driver,
        not here — nvidia-smi doesn't exist on a fresh node.
        """
        failures: list[str] = []

        # Stage 1: TCP probe (D-01)
        try:
            # ponytail: hardcoded 10s timeout matches SSHSettings default
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, 22), timeout=10
            )
            writer.close()
            await writer.wait_closed()
        except (OSError, TimeoutError) as exc:
            failures.append(f"SSH port 22 unreachable: {exc}")
            raise PreflightError(hostname, failures) from exc

        # Disk check
        try:
            disk_output = await self._ssh_run_command(
                hostname, "df --output=avail / | tail -1"
            )
            kb = int(disk_output.strip())
            gb = kb / 1024 / 1024
            if gb < self._settings.min_disk_gb:
                failures.append(
                    f"Insufficient disk: {gb:.1f}GB available, {self._settings.min_disk_gb}GB required"
                )
        except (SSHConnectionError, RemoteCommandError) as exc:
            failures.append(f"SSH diagnostic failed: {exc}")
        except (ValueError, IndexError) as exc:
            failures.append(
                f"SSH diagnostic failed: could not parse disk output: {exc}"
            )

        if failures:
            raise PreflightError(hostname, failures)

    async def provision(
        self, hostname: str, *, managed: bool = True, model: str | None = None
    ) -> None:
        """Run full provisioning sequence on *hostname*.

        Sequence: preflight -> register PROVISIONING -> setup.sh ->
        start-vllm.sh -> health poll -> register HEALTHY.
        Tracks state in etcd at each step (D-05 through D-11).
        """
        provision_started_at = datetime.now(UTC)
        logger.info("provisioning_start", hostname=hostname)
        self._log_buffer.create(hostname)
        self._log(hostname, "info", "Provisioning started")

        await self._update_state(
            hostname, ProvisioningStep.PENDING, started_at=provision_started_at
        )
        await self._power_on_if_needed(hostname)
        await self._update_state(
            hostname, ProvisioningStep.PREFLIGHT, started_at=provision_started_at
        )
        self._log(hostname, "info", "Running pre-flight checks")

        # D-04: preflight before any setup work
        try:
            await self.preflight(hostname)
        except PreflightError as exc:
            self._log(hostname, "error", f"Pre-flight failed: {exc}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="preflight",
                error=str(exc),
                started_at=provision_started_at,
            )
            self._log_buffer.mark_complete(hostname)
            raise

        # D-09: Register node as PROVISIONING before setup
        node = Node(
            node_id=hostname,
            endpoint=f"{hostname}:{self._settings.vllm_port}",
            status=NodeStatus.PROVISIONING,
            model="",
            last_heartbeat=datetime.now(UTC),
            managed=managed,
        )
        key, value = node_to_etcd(node, self._etcd_client.prefix)
        try:
            await asyncio.to_thread(self._etcd_client.put, key, value)
        except Exception:
            logger.warning("provisioning_registration_failed", hostname=hostname)

        current_step = "uploading_scripts"
        try:
            await self._update_state(
                hostname,
                ProvisioningStep.UPLOADING_SCRIPTS,
                started_at=provision_started_at,
            )
            self._log(hostname, "info", "Uploading provisioning scripts")
            await self._upload_scripts(hostname)
            self._log(hostname, "info", "Running setup.sh")
            await self._run_setup(hostname)
            current_step = "gpu_verify"
            await self._verify_gpu(hostname)
            current_step = "starting_vllm"
            await self._update_state(
                hostname,
                ProvisioningStep.STARTING_VLLM,
                started_at=provision_started_at,
            )
            self._log(hostname, "info", "Running start-vllm.sh")
            model_name = await self._run_start_vllm(hostname, model=model)
            current_step = "health_poll"
            await self._update_state(
                hostname, ProvisioningStep.HEALTH_POLL, started_at=provision_started_at
            )
            self._log(hostname, "info", "Waiting for vLLM health endpoint")
            await self._poll_health(hostname)
            current_step = "registering"
            await self._update_state(
                hostname, ProvisioningStep.REGISTERING, started_at=provision_started_at
            )
            self._log(hostname, "info", f"Registering node (model={model_name})")
            await self._register_node(hostname, model_name, managed=managed)
            await self._update_state(
                hostname, ProvisioningStep.COMPLETE, started_at=provision_started_at
            )
            self._log(hostname, "info", "Provisioning complete")
        except (RemoteCommandError, SSHConnectionError, ProvisioningError) as exc:
            self._log(hostname, "error", f"Failed at step '{current_step}': {exc}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step=current_step,
                error=str(exc),
                started_at=provision_started_at,
            )
            # Update node entry to FAILED so it doesn't stay stuck as PROVISIONING
            failed_node = Node(
                node_id=hostname,
                endpoint=f"{hostname}:{self._settings.vllm_port}",
                status=NodeStatus.FAILED,
                model="",
                last_heartbeat=datetime.now(UTC),
                managed=managed,
            )
            f_key, f_value = node_to_etcd(failed_node, self._etcd_client.prefix)
            try:
                await asyncio.to_thread(self._etcd_client.put, f_key, f_value)
            except Exception:
                logger.warning("failed_node_update_failed", hostname=hostname)
            raise ProvisioningError(str(exc)) from exc
        finally:
            self._log_buffer.mark_complete(hostname)

        logger.info("provisioning_complete", hostname=hostname)

    async def _upload_scripts(self, hostname: str) -> None:
        """Copy provisioning scripts to the remote host via SCP."""
        await self._ssh_client.upload(hostname, self._settings.scripts_dir)

    async def _run_setup(self, hostname: str) -> None:
        """Run setup.sh and parse step markers from stdout (D-05, D-06)."""
        async for stream, line in self._ssh_client.run_streaming(
            hostname, f"{self._script_env_prefix()}bash auto-vllm/setup.sh"
        ):
            if stream == "stdout":
                match = STEP_PATTERN.search(line)
                if match:
                    step_name, status = match.group(1), match.group(2)
                    if status == "START":
                        with suppress(ValueError):
                            await self._update_state(
                                hostname, ProvisioningStep(step_name)
                            )
                    if status == "FAIL":
                        logger.error("step_failed", step=step_name, hostname=hostname)
                        self._log(hostname, "error", f"[STEP:{step_name}:FAIL]")
                    else:
                        logger.info(
                            "step_marker",
                            step=step_name,
                            status=status,
                            hostname=hostname,
                        )
                        self._log(hostname, "info", f"[STEP:{step_name}:{status}]")
                else:
                    logger.debug("setup_stdout", line=line, hostname=hostname)
                    self._log(hostname, "debug", line, stream="stdout")
            else:
                logger.warning("setup_stderr", line=line, hostname=hostname)
                self._log(hostname, "warning", line, stream="stderr")

    async def _verify_gpu(self, hostname: str) -> None:
        """Verify GPUs are visible after setup.sh installs the NVIDIA driver."""
        gpu_output = await self._ssh_run_command(
            hostname, "nvidia-smi --query-gpu=name --format=csv,noheader"
        )
        gpu_lines = [ln for ln in gpu_output.strip().splitlines() if ln.strip()]
        if not gpu_lines:
            raise ProvisioningError(f"No GPUs detected on {hostname} after setup")
        self._log(hostname, "info", f"Detected {len(gpu_lines)} GPU(s)")

    async def _run_start_vllm(self, hostname: str, *, model: str | None = None) -> str:
        """Run start-vllm.sh and extract model name from stdout."""
        command = f"{self._script_env_prefix()}bash auto-vllm/start-vllm.sh"
        if model:
            command = f"VLLM_MODEL={shlex.quote(model)} {command}"
        model_name: str | None = None
        async for stream, line in self._ssh_client.run_streaming(hostname, command):
            logger.debug(
                "start_vllm_output", stream=stream, line=line, hostname=hostname
            )
            self._log(hostname, "debug", line, stream=stream)
            if stream == "stdout":
                match = MODEL_PATTERN.search(line)
                if match:
                    model_name = match.group(1).strip()
                    self._log(hostname, "info", f"Detected model: {model_name}")

        if model_name is None:
            raise ProvisioningError(
                f"model name not found in start-vllm.sh output on {hostname}"
            )
        return model_name

    async def _tail_vllm_log(self, hostname: str) -> None:
        """Tail vLLM log and feed lines into the provisioning log buffer."""
        try:
            async for _stream, line in self._ssh_client.run_streaming(
                hostname, "tail -n +1 -f /var/log/vllm-serve.log"
            ):
                self._log(hostname, "info", line, stream="vllm")
        except (SSHConnectionError, RemoteCommandError, asyncio.CancelledError):
            pass

    async def _poll_health(self, hostname: str) -> None:
        """Poll /health endpoint until 200 OK or timeout (D-10, D-09).

        Tails /var/log/vllm-serve.log concurrently so vLLM startup output
        appears in the live log pane while waiting.
        """
        tail_task = asyncio.create_task(self._tail_vllm_log(hostname))

        url = f"http://{hostname}:{self._settings.vllm_port}/health"
        deadline = (
            asyncio.get_running_loop().time() + self._settings.health_poll_timeout
        )

        try:
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            logger.info("health_poll_success", hostname=hostname)
                            return
                        logger.debug(
                            "health_poll_non_200",
                            status=response.status_code,
                            hostname=hostname,
                        )
                    except httpx.HTTPError as exc:
                        logger.debug(
                            "health_poll_retry",
                            hostname=hostname,
                            error=str(exc),
                        )

                    if asyncio.get_running_loop().time() >= deadline:
                        raise ProvisioningError(
                            f"health poll timed out after "
                            f"{self._settings.health_poll_timeout}s for {hostname}"
                        )
                    await asyncio.sleep(self._settings.health_poll_interval)
        finally:
            tail_task.cancel()
            with suppress(asyncio.CancelledError):
                await tail_task

    async def _register_node(
        self, hostname: str, model: str, *, managed: bool = True
    ) -> None:
        """Register node in etcd with correct fields (D-11, D-12)."""
        node = Node(
            node_id=hostname,
            endpoint=f"{hostname}:{self._settings.vllm_port}",
            status=NodeStatus.HEALTHY,
            model=model,
            last_heartbeat=datetime.now(UTC),
            managed=managed,
        )
        key, value = node_to_etcd(node, self._etcd_client.prefix)
        # ponytail: etcd3gw is sync, asyncio.to_thread wraps it (Pitfall 5)
        await asyncio.to_thread(self._etcd_client.put, key, value)
        logger.info("node_registered", hostname=hostname, model=model, key=key)

    def fire_background(
        self, coro: Coroutine[object, object, None]
    ) -> asyncio.Task[None]:
        """Schedule a coroutine as a background task, preventing GC."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _drain_wait(self, hostname: str) -> None:
        """Wait for active connections to reach zero or timeout (D-08, D-09)."""
        if self._tracker is None:
            logger.warning("drain_skip_no_tracker", hostname=hostname)
            return
        deadline = asyncio.get_running_loop().time() + self._settings.drain_timeout
        while True:
            if self._tracker.get(hostname) == 0:
                return
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("drain_timeout_expired", hostname=hostname)
                return
            await asyncio.sleep(1)

    async def teardown(self, hostname: str, *, force: bool = False) -> None:
        """Teardown a provisioned node.

        Graceful: drain -> kill vllm -> deregister.
        Force: kill -9 -> deregister.
        """
        teardown_started_at = datetime.now(UTC)
        logger.info("teardown_start", hostname=hostname, force=force)
        self._log_buffer.create(hostname)
        self._log(hostname, "info", f"Teardown started (force={force})")

        try:
            if not force:
                await self._update_state(
                    hostname, ProvisioningStep.DRAINING, started_at=teardown_started_at
                )
                self._log(hostname, "info", "Draining active connections")
                if self._registry is not None:
                    self._registry.drain(hostname)
                await self._drain_wait(hostname)
                self._log(hostname, "info", "Drain complete")

            await self._update_state(
                hostname, ProvisioningStep.STOPPING_VLLM, started_at=teardown_started_at
            )
            self._log(hostname, "info", "Stopping vLLM process")
            if force:
                await self._ssh_run_command(
                    hostname,
                    "kill -9 $(cat /var/run/vllm.pid) 2>/dev/null; rm -f /var/run/vllm.pid",
                )
            else:
                await self._ssh_run_command(
                    hostname,
                    "kill $(cat /var/run/vllm.pid) 2>/dev/null; rm -f /var/run/vllm.pid",
                )

            await self._update_state(
                hostname, ProvisioningStep.DEREGISTERING, started_at=teardown_started_at
            )
            self._log(hostname, "info", "Deregistering node from etcd")
            await asyncio.to_thread(
                self._etcd_client.delete, f"{self._etcd_client.prefix}{hostname}"
            )
            if self._registry is not None:
                self._registry.remove(hostname)

            await self._update_state(
                hostname,
                ProvisioningStep.TEARDOWN_COMPLETE,
                started_at=teardown_started_at,
            )
            self._log(hostname, "info", "Teardown complete")
        except (RemoteCommandError, SSHConnectionError) as exc:
            self._log(hostname, "error", f"Teardown failed: {exc}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="teardown",
                error=str(exc),
                started_at=teardown_started_at,
            )
            raise ProvisioningError(str(exc)) from exc
        finally:
            self._log_buffer.mark_complete(hostname)

        logger.info("teardown_complete", hostname=hostname)
