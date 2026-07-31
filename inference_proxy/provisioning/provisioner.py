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
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import ip_address
from typing import TYPE_CHECKING

import httpx
import structlog

from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.models.endpoint import EndpointPolicy, EndpointValidationError
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.host_lifecycle import (
    HostLifecycleCoordinator,
    HostLifecycleLease,
)
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)
from inference_proxy.provisioning.state import ProvisioningState, ProvisioningStep
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
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


@dataclass
class _ProvisioningTask:
    task: asyncio.Task[None]
    started: bool = False


class NodeProvisioner:
    """Orchestrates full provisioning of a vLLM node on a remote host.

    Accepts SSHClient, EtcdClient, ProvisioningSettings, and EndpointPolicy
    via constructor injection (DIP).
    """

    def __init__(
        self,
        ssh_client: SSHClient,
        etcd_client: EtcdClient,
        settings: ProvisioningSettings,
        endpoint_policy: EndpointPolicy,
        registry: NodeRegistry | None = None,
        connection_tracker: ConnectionTracker | None = None,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
        redfish_client: RedfishClient | None = None,
        log_buffer: ProvisioningLogBuffer | None = None,
        lifecycle_coordinator: HostLifecycleCoordinator | None = None,
    ) -> None:
        self._ssh_client = ssh_client
        self._etcd_client = etcd_client
        self._settings = settings
        self._endpoint_policy = endpoint_policy
        self._registry = registry
        self._tracker = connection_tracker
        self._cb_registry = circuit_breaker_registry
        self._redfish_client = redfish_client
        self._log_buffer = log_buffer or ProvisioningLogBuffer()
        self._lifecycle = lifecycle_coordinator or HostLifecycleCoordinator()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._provisioning_tasks: dict[str, _ProvisioningTask] = {}

    @property
    def log_buffer(self) -> ProvisioningLogBuffer:
        return self._log_buffer

    def validate_endpoint(self, hostname: str) -> str:
        """Return the canonical provisioned endpoint or fail with a config hint."""
        candidate = f"{hostname}:{self._settings.vllm_port}"
        try:
            return self._endpoint_policy.normalize(candidate)
        except EndpointValidationError as exc:
            try:
                ip_address(hostname.strip("[]"))
            except ValueError:
                allowlist_hint = (
                    f"add {hostname!r} to routing.allowed_endpoint_hosts "
                    "or add an approved wildcard suffix"
                )
            else:
                allowlist_hint = (
                    f"add a CIDR containing {hostname!r} to "
                    "routing.allowed_endpoint_networks"
                )
            raise EndpointValidationError(
                f"Provisioning host {hostname!r} would create disallowed backend "
                f"endpoint {candidate!r}: {exc}; {allowlist_hint}"
            ) from exc

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

    async def try_reserve_host(self, hostname: str) -> HostLifecycleLease | None:
        """Reserve *hostname* without waiting for another lifecycle operation."""
        return await self._lifecycle.try_acquire(hostname)

    def host_operation_in_progress(self, hostname: str) -> bool:
        """Return whether provisioning or teardown owns or awaits *hostname*."""
        return self._lifecycle.is_busy(hostname)

    def connection_count(self, hostname: str) -> int:
        """Return active requests for *hostname*, or zero without a tracker."""
        if self._tracker is None:
            return 0
        return self._tracker.get(hostname)

    async def cleanup_stale_node(self, hostname: str) -> None:
        """Delete stale discovery and local routing state before a retry.

        The etcd delete completes before local state is removed and before the
        caller starts provisioning. A replacement registration therefore has
        a newer mod revision, allowing the revision-aware watcher to reject a
        delayed cleanup DELETE.
        """
        await asyncio.to_thread(
            self._etcd_client.delete,
            f"{self._etcd_client.prefix}{hostname}",
        )
        if self._registry is not None:
            self._registry.remove(hostname)
        if self._tracker is not None:
            self._tracker.remove(hostname)
        if self._cb_registry is not None:
            self._cb_registry.remove(hostname)

    async def provision(
        self,
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        lifecycle_lease: HostLifecycleLease | None = None,
    ) -> None:
        """Provision *hostname* under the shared host lifecycle coordinator."""
        # Validate before acquiring the lifecycle lease or touching the host.
        # The API also calls this synchronously so configuration errors become
        # immediate 400 responses instead of failed background operations.
        self.validate_endpoint(hostname)
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            await self._provision(hostname, managed=managed, model=model)
        except asyncio.CancelledError:
            self._log(hostname, "error", "Provisioning cancelled by teardown")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="cancelled",
                error="Provisioning cancelled by teardown",
            )
            self._log_buffer.mark_complete(hostname)
            raise
        finally:
            lease.release()

    async def _provision(
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
            endpoint=self.validate_endpoint(hostname),
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
                endpoint=self.validate_endpoint(hostname),
                status=NodeStatus.FAILED,
                model="",
                last_heartbeat=datetime.now(UTC),
                managed=managed,
            )
            f_key, f_value = node_to_etcd(failed_node, self._etcd_client.prefix)
            try:
                replaced = await asyncio.to_thread(
                    self._etcd_client.replace,
                    f_key,
                    value,
                    f_value,
                )
                if not replaced:
                    logger.info(
                        "failed_node_update_skipped",
                        hostname=hostname,
                        reason="node changed or removed",
                    )
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
            endpoint=self.validate_endpoint(hostname),
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
        self,
        coro: Coroutine[object, object, None],
        *,
        provisioning_hostname: str | None = None,
    ) -> asyncio.Task[None]:
        """Schedule a coroutine and optionally track host provisioning ownership."""
        if provisioning_hostname is not None:
            current = self._provisioning_tasks.get(provisioning_hostname)
            if current is not None and not current.task.done():
                raise RuntimeError(
                    f"Provisioning task already active for '{provisioning_hostname}'"
                )

        record: _ProvisioningTask | None = None
        scheduled_coro = coro
        if provisioning_hostname is not None:

            async def _tracked_provision() -> None:
                if record is None:
                    raise RuntimeError("provisioning task record was not initialized")
                record.started = True
                await coro

            scheduled_coro = _tracked_provision()

        try:
            task = asyncio.create_task(scheduled_coro)
        except Exception:
            if scheduled_coro is not coro:
                scheduled_coro.close()
            raise
        self._background_tasks.add(task)

        if provisioning_hostname is not None:
            record = _ProvisioningTask(task)
            self._provisioning_tasks[provisioning_hostname] = record

        def _task_done(done_task: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done_task)
            if record is not None and not record.started:
                coro.close()
            if (
                provisioning_hostname is not None
                and self._provisioning_tasks.get(provisioning_hostname) is record
            ):
                self._provisioning_tasks.pop(provisioning_hostname, None)

        task.add_done_callback(_task_done)
        return task

    async def cancel_active_provision(self, hostname: str) -> asyncio.Task[None] | None:
        """Cancel and await the active provisioning task for *hostname*."""
        record = self._provisioning_tasks.get(hostname)
        if record is None or record.task.done():
            return None

        task = record.task
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        if not task.cancelled():
            raise RuntimeError(
                f"Provisioning task for '{hostname}' swallowed cancellation"
            )

        if not record.started:
            # A task cancelled before its coroutine first runs cannot record
            # its own terminal state. Normal cancellation is recorded inside
            # provision() while that task still owns the host lease.
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="cancelled",
                error="Provisioning cancelled by teardown",
            )
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

    async def teardown(
        self,
        hostname: str,
        *,
        force: bool = False,
        lifecycle_lease: HostLifecycleLease | None = None,
    ) -> None:
        """Teardown *hostname* under the shared host lifecycle coordinator."""
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            await self._teardown(hostname, force=force)
        finally:
            lease.release()

    async def _teardown(self, hostname: str, *, force: bool = False) -> None:
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
            # Existing nodes may predate stop-vllm.sh, and a cancelled setup
            # may not have reached its upload step. Refresh the bundle before
            # invoking the verified stop path.
            try:
                await self._upload_scripts(hostname)
            except Exception as exc:
                # The refresh is upgrade compatibility, not a teardown
                # precondition. The existing remote copy may still work; if
                # it does not, the verified stop command below reports the
                # operation as FAILED without deregistering the node.
                logger.warning(
                    "teardown_script_refresh_failed",
                    hostname=hostname,
                    error=str(exc),
                    exc_info=True,
                )
                self._log(
                    hostname,
                    "warning",
                    "Could not refresh teardown scripts; attempting the "
                    "existing remote stop script",
                )
            stop_command = "bash auto-vllm/stop-vllm.sh"
            if force:
                stop_command += " --force"
            await self._ssh_run_command(hostname, stop_command)

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
