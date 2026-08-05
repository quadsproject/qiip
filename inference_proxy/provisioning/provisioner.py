"""Node provisioning orchestrator.

Runs the full provisioning sequence on a remote host: setup.sh,
engine start script, health poll, etcd registration.

Per D-15: Concrete class, no protocol/interface.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import httpx
import structlog

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.discovery.serializer import node_to_etcd
from inference_proxy.huggingface.artifacts import (
    GGUFArtifactError,
    GGUFArtifactIndex,
    ResolvedGGUFArtifact,
)
from inference_proxy.models.endpoint import EndpointPolicy, EndpointValidationError
from inference_proxy.models.node import InferenceEngine, Node, NodeStatus
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

STEP_PATTERN = re.compile(r"\[STEP:(\w+):(START|OK|FAIL|WARN)\]")
MODEL_PATTERN = re.compile(r"#\s*Model:\s+(.+)")
LLAMACPP_CONTEXT_PATTERN = re.compile(
    r"initializing, n_slots = (?P<slots>\d+), "
    r"n_ctx_slot = (?P<context>\d+), "
    r"kv_unified = '(?P<unified>true|false)'"
)
LLAMACPP_AGGREGATE_CONTEXT_PATTERN = re.compile(
    r"llama_context:\s+n_ctx\s*=\s*(?P<context>\d+)"
)
LLAMACPP_PLAN_PATTERN = re.compile(
    r"qiip_fit_plan: context_per_slot=(?P<context>\d+) "
    r"slots=(?P<slots>\d+) "
    r"aggregate_context=(?P<aggregate>\d+) "
    r"fit_target_mib=(?P<target>\d+) "
    r"cache_type_k=(?P<cache_type_k>f16|q8_0) "
    r"cache_type_v=(?P<cache_type_v>f16|q8_0) "
    r"flash_attn=(?P<flash_attn>auto|on)"
)
LLAMACPP_KV_CACHE_PATTERN = re.compile(
    r"llama_kv_cache[^:]*:\s+size\s*=.*?"
    r"K \((?P<cache_type_k>[a-z0-9_]+)\):.*?"
    r"V \((?P<cache_type_v>[a-z0-9_]+)\):"
)
LLAMACPP_OFFLOAD_PATTERN = re.compile(
    r"offloaded (?P<loaded>\d+)/(?P<total>\d+) layers to GPU"
)
LLAMACPP_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"n_ctx_seq \((?P<context>\d+)\) > n_ctx_train \((?P<train>\d+)\) "
    r"-- possible training context overflow"
)
LLAMACPP_SLOT_CAP_PATTERN = re.compile(
    r"the slot context \((?P<context>\d+)\) exceeds the training context "
    r"of the model \((?P<train>\d+)\) - capping"
)
LLAMACPP_FIT_FAILURE_PATTERN = re.compile(r"failed to fit params to free device memory")
_ENGINE_BUNDLE_FILES = {
    InferenceEngine.VLLM: {
        ".uv-version",
        "pyproject.toml",
        "setup.sh",
        "start-vllm.sh",
        "stop-vllm.sh",
        "uv.lock",
        "uv-x86_64-unknown-linux-gnu.tar.gz.sha256",
        "vllm-process.sh",
    },
    InferenceEngine.LLAMA_CPP: {
        "llamacpp-process.sh",
        "setup.sh",
        "start-llamacpp.sh",
        "stop-llamacpp.sh",
    },
}


class ProvisioningError(Exception):
    """Raised when any stage of provisioning fails."""


class ProvisioningCapacityError(RuntimeError):
    """Raised when the configured concurrent-provision limit is full."""

    def __init__(self, *, active: int, limit: int) -> None:
        self.active = active
        self.limit = limit
        super().__init__(
            f"Provisioning capacity reached ({active}/{limit} active tasks)"
        )


class PreflightError(Exception):
    """Raised when pre-flight validation fails (D-01 through D-04).

    Collects all failures before raising so operators see every problem
    at once (D-03).
    """

    def __init__(self, hostname: str, failures: list[str]) -> None:
        self.hostname = hostname
        self.failures = failures
        super().__init__(f"Pre-flight failed on {hostname}: {'; '.join(failures)}")


@dataclass(frozen=True)
class ProvisioningIdentity:
    """Engine and immutable artifact selected for one provisioning operation."""

    engine: InferenceEngine
    artifact_id: str | None = None


@dataclass(frozen=True)
class LlamaCppRuntimeFit:
    """Effective llama.cpp sizing parsed from the completed startup log."""

    context_per_slot: int
    slot_context_limit: int
    slots: int
    aggregate_context: int
    fit_target_mib: int
    cache_type_k: str
    cache_type_v: str
    flash_attn: str
    kv_unified: bool
    gpu_layers: int
    total_layers: int


def _parse_llamacpp_runtime_fit(log_text: str) -> LlamaCppRuntimeFit:
    """Validate and return the final managed llama.cpp fit evidence."""
    if LLAMACPP_FIT_FAILURE_PATTERN.search(log_text):
        raise ProvisioningError("llama.cpp reported a runtime fitting failure")
    plans = list(LLAMACPP_PLAN_PATTERN.finditer(log_text))
    if not plans:
        raise ProvisioningError("llama.cpp startup log has no QIIP VRAM plan")
    contexts = list(LLAMACPP_CONTEXT_PATTERN.finditer(log_text))
    if not contexts:
        raise ProvisioningError("llama.cpp startup log has no effective context record")
    aggregate_contexts = list(LLAMACPP_AGGREGATE_CONTEXT_PATTERN.finditer(log_text))
    if not aggregate_contexts:
        raise ProvisioningError("llama.cpp startup log has no aggregate context record")
    offloads = list(LLAMACPP_OFFLOAD_PATTERN.finditer(log_text))
    if not offloads:
        raise ProvisioningError("llama.cpp startup log has no GPU offload record")
    cache_records = list(LLAMACPP_KV_CACHE_PATTERN.finditer(log_text))
    if not cache_records:
        raise ProvisioningError("llama.cpp startup log has no KV cache type record")

    plan = plans[-1].groupdict()
    context = contexts[-1].groupdict()
    aggregate_context = int(aggregate_contexts[-1].group("context"))
    offload = offloads[-1].groupdict()
    cache_record = cache_records[-1].groupdict()
    fit = LlamaCppRuntimeFit(
        context_per_slot=int(plan["context"]),
        slot_context_limit=int(context["context"]),
        slots=int(plan["slots"]),
        aggregate_context=int(plan["aggregate"]),
        fit_target_mib=int(plan["target"]),
        cache_type_k=plan["cache_type_k"],
        cache_type_v=plan["cache_type_v"],
        flash_attn=plan["flash_attn"],
        kv_unified=context["unified"] == "true",
        gpu_layers=int(offload["loaded"]),
        total_layers=int(offload["total"]),
    )
    if (
        fit.context_per_slot < 1
        or fit.slot_context_limit < fit.context_per_slot
        or fit.slot_context_limit > fit.aggregate_context
        or fit.slots < 1
        or fit.fit_target_mib < 1
    ):
        raise ProvisioningError("llama.cpp reported an invalid effective context")
    if int(context["slots"]) != fit.slots:
        raise ProvisioningError(
            "llama.cpp runtime slot count differs from its VRAM plan"
        )
    requested_context = fit.context_per_slot * fit.slots
    if not requested_context <= fit.aggregate_context < requested_context + 256:
        raise ProvisioningError("llama.cpp VRAM plan has invalid aggregate context")
    if aggregate_context != fit.aggregate_context:
        raise ProvisioningError(
            "llama.cpp runtime aggregate context differs from its VRAM plan"
        )
    if fit.cache_type_k != fit.cache_type_v:
        raise ProvisioningError("llama.cpp managed startup requires matching K/V types")
    expected_flash_attn = "auto" if fit.cache_type_k == "f16" else "on"
    if fit.flash_attn != expected_flash_attn:
        raise ProvisioningError(
            "llama.cpp managed cache type has an invalid Flash Attention policy"
        )
    if cache_record != {
        "cache_type_k": fit.cache_type_k,
        "cache_type_v": fit.cache_type_v,
    }:
        raise ProvisioningError(
            "llama.cpp runtime KV cache types differ from its VRAM plan"
        )
    overflows = list(LLAMACPP_CONTEXT_OVERFLOW_PATTERN.finditer(log_text))
    caps = list(LLAMACPP_SLOT_CAP_PATTERN.finditer(log_text))
    if fit.aggregate_context > fit.slot_context_limit:
        if not overflows or not caps:
            raise ProvisioningError(
                "llama.cpp startup log is missing expected unified-KV "
                "context-sharing warnings"
            )
        overflow = overflows[-1].groupdict()
        cap = caps[-1].groupdict()
        expected = {
            "context": str(fit.aggregate_context),
            "train": str(fit.slot_context_limit),
        }
        if overflow != expected or cap != expected:
            raise ProvisioningError(
                "llama.cpp unified-KV context-sharing warnings differ from runtime"
            )
    elif overflows or caps:
        raise ProvisioningError(
            "llama.cpp reported unexpected unified-KV context-sharing warnings"
        )
    if not fit.kv_unified:
        raise ProvisioningError("llama.cpp managed startup requires unified KV cache")
    if fit.total_layers < 1 or fit.gpu_layers != fit.total_layers:
        raise ProvisioningError(
            "llama.cpp did not fully offload the model to GPU "
            f"({fit.gpu_layers}/{fit.total_layers} layers)"
        )
    return fit


@dataclass
class _ProvisioningTask:
    task: asyncio.Task[None]
    identity: ProvisioningIdentity
    started: bool = False


class NodeProvisioner:
    """Orchestrates full provisioning of an inference node on a remote host.

    Accepts SSHClient, EtcdClient, ProvisioningSettings, and EndpointPolicy
    via constructor injection (DIP).
    """

    def __init__(
        self,
        ssh_client: SSHClient,
        etcd_client: EtcdClient,
        settings: ProvisioningSettings,
        llmfit_settings: LLMFitSettings,
        endpoint_policy: EndpointPolicy,
        registry: NodeRegistry | None = None,
        connection_tracker: ConnectionTracker | None = None,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
        redfish_client: RedfishClient | None = None,
        log_buffer: ProvisioningLogBuffer | None = None,
        lifecycle_coordinator: HostLifecycleCoordinator | None = None,
        hf_token: str | None = None,
        nfs_export: str | None = None,
        artifact_index: GGUFArtifactIndex | None = None,
    ) -> None:
        self._ssh_client = ssh_client
        self._etcd_client = etcd_client
        self._settings = settings
        self._llmfit_version = llmfit_settings.version
        self._llmfit_sha256 = llmfit_settings.sha256
        self._endpoint_policy = endpoint_policy
        self._registry = registry
        self._tracker = connection_tracker
        self._cb_registry = circuit_breaker_registry
        self._redfish_client = redfish_client
        self._log_buffer = log_buffer or ProvisioningLogBuffer(
            max_entries_per_host=settings.log_max_entries_per_host,
            max_bytes_per_host=settings.log_max_bytes_per_host,
            max_entry_bytes=settings.log_max_entry_bytes,
            max_completed_hosts=settings.log_max_completed_hosts,
        )
        self._lifecycle = lifecycle_coordinator or HostLifecycleCoordinator()
        self._hf_token = hf_token
        self._nfs_export = nfs_export
        self._artifact_index = artifact_index
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

    def validate_setup_configuration(
        self, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> None:
        """Require settings used only by the node-provisioning workflow."""
        self._required_nfs_export()
        self._required_script_bundles(engine)

    async def resolve_artifact_selection(
        self,
        engine: InferenceEngine,
        artifact_id: str | None,
    ) -> ResolvedGGUFArtifact | None:
        """Resolve the exact llama.cpp artifact or reject the engine contract."""
        if engine == InferenceEngine.VLLM:
            if artifact_id is not None:
                raise ProvisioningError(
                    "artifact_id is only valid for llama_cpp provisioning"
                )
            return None
        if artifact_id is None:
            raise ProvisioningError("llama_cpp provisioning requires artifact_id")
        artifact_index = self._required_llamacpp_shared_root()
        try:
            artifact = await asyncio.to_thread(artifact_index.get, artifact_id)
        except GGUFArtifactError as exc:
            raise ProvisioningError(
                f"GGUF artifact {artifact_id!r} is invalid: {exc}"
            ) from exc
        if artifact is None:
            raise ProvisioningError(f"GGUF artifact {artifact_id!r} was not found")
        return artifact

    def _required_llamacpp_shared_root(self) -> GGUFArtifactIndex:
        """Require a valid shared-root mapping for llama.cpp provisioning."""
        if self._artifact_index is None:
            raise ProvisioningError("GGUF artifact discovery is not configured")
        try:
            self._artifact_index.validate_shared_root()
        except GGUFArtifactError as exc:
            raise ProvisioningError(
                "llama_cpp provisioning requires "
                "INFERENCE_PROXY_HUGGINGFACE__SHARED_ROOT containing the "
                f"configured cache_dir: {exc}"
            ) from exc
        return self._artifact_index

    def _required_nfs_export(self) -> str:
        """Return the canonical NFS export or reject node provisioning."""
        if self._nfs_export is None:
            raise ProvisioningError(
                "Node provisioning requires "
                "INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT; proxy-only "
                "deployments may leave it unset"
            )
        return self._nfs_export

    def _engine_scripts_dir(self, engine: InferenceEngine) -> Path:
        """Resolve an engine bundle beside the configured vLLM bundle."""
        if engine == InferenceEngine.VLLM:
            return self._settings.scripts_dir
        return self._settings.scripts_dir.parent / "auto-llamacpp"

    def _common_scripts_dir(self) -> Path:
        return self._settings.scripts_dir.parent / "common"

    def _required_script_bundles(self, engine: InferenceEngine) -> tuple[Path, Path]:
        """Return complete engine/common bundles or fail before remote work."""
        engine_dir = self._engine_scripts_dir(engine)
        common_dir = self._common_scripts_dir()
        required = {
            *(engine_dir / name for name in _ENGINE_BUNDLE_FILES[engine]),
            common_dir / "setup-base.sh",
        }
        missing = sorted(str(path) for path in required if not path.is_file())
        if missing:
            raise ProvisioningError(
                "Provisioning script bundle is incomplete; missing: "
                + ", ".join(missing)
            )
        return engine_dir, common_dir

    def _setup_script_env(
        self, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> dict[str, str]:
        """Return the exact environment accepted by setup.sh."""
        env = {
            "AUTOVLLM_NFS_EXPORT": self._required_nfs_export(),
            "AUTOVLLM_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
            "AUTOVLLM_NVIDIA_DRIVER_VERSION": self._settings.nvidia_driver_version,
            "AUTOVLLM_NVIDIA_DRIVER_SHA256": self._settings.nvidia_driver_sha256,
            "AUTOVLLM_API_PORT": str(self._settings.vllm_port),
            "AUTOVLLM_LLMFIT_VERSION": self._llmfit_version,
            "AUTOVLLM_LLMFIT_SHA256": self._llmfit_sha256,
        }
        if engine == InferenceEngine.LLAMA_CPP:
            env["AUTOLLAMACPP_VERSION"] = self._settings.llamacpp_version
            env["AUTOLLAMACPP_SHA256"] = self._settings.llamacpp_sha256
            env["AUTOLLAMACPP_SOURCE_URL"] = (
                self._settings.llamacpp_source_download_url()
            )
        return env

    def _start_script_env(
        self,
        model: str | None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact: ResolvedGGUFArtifact | None = None,
    ) -> dict[str, str]:
        """Return the exact environment accepted by the engine start script."""
        if engine == InferenceEngine.LLAMA_CPP:
            env = {
                "AUTOLLAMACPP_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
                "AUTOLLAMACPP_PORT": str(self._settings.vllm_port),
                "AUTOLLAMACPP_REQUIRE_CUDA": "1",
                "AUTOLLAMACPP_MANAGED": "1",
                "AUTOLLAMACPP_FIT_TARGET_MIB": str(
                    self._settings.llamacpp_fit_target_mib
                ),
            }
            if artifact is None:
                raise ProvisioningError("llama_cpp start requires a GGUF artifact")
            env["AUTOLLAMACPP_GGUF_PATH"] = artifact.node_relative_entrypoint
            env["AUTOLLAMACPP_MODEL_ALIAS"] = artifact.model_alias
        else:
            env = {
                "AUTOVLLM_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
                "AUTOVLLM_API_PORT": str(self._settings.vllm_port),
            }
            if model is not None:
                env["AUTOVLLM_MODEL"] = model
        if self._hf_token:
            env["HF_TOKEN"] = self._hf_token
        return env

    def _script_command(
        self,
        script_name: str,
        *,
        env: dict[str, str] | None = None,
        args: tuple[str, ...] = (),
        scripts_dir: str | None = None,
    ) -> str:
        """Build one uniformly quoted remote script command."""
        dir_name = scripts_dir or self._settings.scripts_dir.name
        script_path = str(PurePosixPath(dir_name, script_name))
        command = shlex.join(("bash", script_path, *args))
        if not env:
            return command
        assignments = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in env.items()
        )
        return f"{assignments} {command}"

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
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact_id: str | None = None,
        lifecycle_lease: HostLifecycleLease | None = None,
    ) -> None:
        """Provision *hostname* under the shared host lifecycle coordinator."""
        # Validate before acquiring the lifecycle lease or touching the host.
        # The API also calls this synchronously so configuration errors become
        # immediate 400 responses instead of failed background operations.
        self.validate_setup_configuration(engine)
        self.validate_endpoint(hostname)
        artifact = await self.resolve_artifact_selection(engine, artifact_id)
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            await self._provision(
                hostname,
                managed=managed,
                model=model,
                engine=engine,
                artifact=artifact,
            )
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
        self,
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact: ResolvedGGUFArtifact | None = None,
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

        current_step = "registering_node"
        value: bytes | None = None
        artifact_id = artifact.artifact_id if artifact is not None else None
        try:
            # D-09: Register node as PROVISIONING before setup. Failure is
            # terminal: remote mutation must not start without an ownership
            # record in discovery.
            node = Node(
                node_id=hostname,
                endpoint=self.validate_endpoint(hostname),
                status=NodeStatus.PROVISIONING,
                model="",
                engine=engine,
                artifact_id=artifact_id,
                last_heartbeat=datetime.now(UTC),
                managed=managed,
            )
            key, value = node_to_etcd(node, self._etcd_client.prefix)
            await asyncio.to_thread(self._etcd_client.put, key, value)
            current_step = "uploading_scripts"
            await self._update_state(
                hostname,
                ProvisioningStep.UPLOADING_SCRIPTS,
                started_at=provision_started_at,
            )
            self._log(hostname, "info", "Uploading provisioning scripts")
            await self._upload_scripts(hostname, engine=engine)
            self._log(hostname, "info", "Running setup.sh")

            def set_current_step(step: str) -> None:
                nonlocal current_step
                current_step = step

            await self._run_setup(
                hostname,
                started_at=provision_started_at,
                on_step=set_current_step,
                engine=engine,
            )
            current_step = "gpu_verify"
            await self._verify_gpu(hostname)
            current_step = "starting_engine"
            if engine == InferenceEngine.LLAMA_CPP:
                starting_step = ProvisioningStep.STARTING_LLAMACPP
            else:
                starting_step = ProvisioningStep.STARTING_VLLM
            await self._update_state(
                hostname,
                starting_step,
                started_at=provision_started_at,
            )
            self._log(hostname, "info", f"Starting {engine} inference engine")
            model_name = await self._run_start_vllm(
                hostname, model=model, engine=engine, artifact=artifact
            )
            current_step = "health_poll"
            await self._update_state(
                hostname, ProvisioningStep.HEALTH_POLL, started_at=provision_started_at
            )
            self._log(hostname, "info", "Waiting for health endpoint")
            await self._poll_health(hostname, engine=engine)
            if engine == InferenceEngine.LLAMA_CPP:
                current_step = "llamacpp_runtime_verify"
                await self._verify_llamacpp_runtime(hostname)
            current_step = "registering"
            await self._update_state(
                hostname, ProvisioningStep.REGISTERING, started_at=provision_started_at
            )
            self._log(hostname, "info", f"Registering node (model={model_name})")
            await self._register_node(
                hostname,
                model_name,
                managed=managed,
                engine=engine,
                artifact_id=artifact_id,
            )
            await self._update_state(
                hostname, ProvisioningStep.COMPLETE, started_at=provision_started_at
            )
            self._log(hostname, "info", "Provisioning complete")
        except Exception as exc:
            self._log(hostname, "error", f"Failed at step '{current_step}': {exc}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step=current_step,
                error=str(exc),
                started_at=provision_started_at,
            )
            # Update node entry to FAILED so it doesn't stay stuck as PROVISIONING
            try:
                failed_node = Node(
                    node_id=hostname,
                    endpoint=self.validate_endpoint(hostname),
                    status=NodeStatus.FAILED,
                    model="",
                    engine=engine,
                    artifact_id=artifact_id,
                    last_heartbeat=datetime.now(UTC),
                    managed=managed,
                )
                f_key, f_value = node_to_etcd(failed_node, self._etcd_client.prefix)
                if value is not None:
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
            raise
        finally:
            self._log_buffer.mark_complete(hostname)

        logger.info("provisioning_complete", hostname=hostname)

    async def _upload_scripts(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> None:
        """Copy provisioning scripts to the remote host via SCP."""
        scripts_dir, common_dir = self._required_script_bundles(engine)
        await self._ssh_client.upload(hostname, scripts_dir)
        await self._ssh_client.upload(hostname, common_dir)

    async def _run_setup(
        self,
        hostname: str,
        *,
        started_at: datetime,
        on_step: Callable[[str], None],
        engine: InferenceEngine = InferenceEngine.VLLM,
    ) -> None:
        """Run setup.sh and parse step markers from stdout (D-05, D-06)."""
        command = self._script_command(
            "setup.sh",
            env=self._setup_script_env(engine),
            scripts_dir=self._engine_scripts_dir(engine).name,
        )
        if engine == InferenceEngine.LLAMA_CPP:
            output = self._ssh_client.run_streaming(
                hostname,
                command,
                total_timeout=self._settings.llamacpp_setup_timeout,
            )
        else:
            output = self._ssh_client.run_streaming(hostname, command)
        async for stream, line in output:
            if stream == "stdout":
                match = STEP_PATTERN.search(line)
                if match:
                    step_name, status = match.group(1), match.group(2)
                    on_step(step_name)
                    if status in {"START", "WARN"}:
                        with suppress(ValueError):
                            await self._update_state(
                                hostname,
                                ProvisioningStep(step_name),
                                started_at=started_at,
                            )
                    if status == "FAIL":
                        logger.error("step_failed", step=step_name, hostname=hostname)
                        self._log(hostname, "error", f"[STEP:{step_name}:FAIL]")
                    elif status == "WARN":
                        logger.warning(
                            "step_warning", step=step_name, hostname=hostname
                        )
                        self._log(hostname, "warning", f"[STEP:{step_name}:WARN]")
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

    async def _stop_failed_llamacpp_start(self, hostname: str) -> None:
        """Best-effort cleanup after post-health llama.cpp verification fails."""
        command = self._script_command(
            "stop-llamacpp.sh",
            scripts_dir=self._engine_scripts_dir(InferenceEngine.LLAMA_CPP).name,
        )
        try:
            await self._ssh_run_command(hostname, command)
        except Exception as exc:
            logger.warning(
                "llamacpp_verification_cleanup_failed",
                hostname=hostname,
                error=str(exc),
                exc_info=True,
            )
            self._log(
                hostname,
                "warning",
                "llama.cpp runtime verification failed and the stop script "
                f"also failed: {exc}",
            )

    async def _verify_llamacpp_runtime(self, hostname: str) -> None:
        """Fail closed unless the healthy server proves the managed fit contract."""
        try:
            log_text = await self._ssh_run_command(
                hostname, "cat -- /var/log/llamacpp-serve.log"
            )
            fit = _parse_llamacpp_runtime_fit(log_text)
            memory_text = await self._ssh_run_command(
                hostname,
                "nvidia-smi --query-gpu=memory.used,memory.free "
                "--format=csv,noheader,nounits",
            )
            memory_rows: list[tuple[int, int]] = []
            try:
                for line in memory_text.splitlines():
                    if not line.strip():
                        continue
                    used_text, free_text = line.split(",", maxsplit=1)
                    memory_rows.append((int(used_text.strip()), int(free_text.strip())))
            except ValueError as exc:
                raise ProvisioningError(
                    "could not parse post-load NVIDIA memory telemetry"
                ) from exc
            if not memory_rows:
                raise ProvisioningError(
                    "post-load NVIDIA memory telemetry returned no GPUs"
                )
            if fit.fit_target_mib != self._settings.llamacpp_fit_target_mib:
                raise ProvisioningError(
                    "llama.cpp runtime fit target differs from gateway configuration"
                )
            if any(row[1] < fit.fit_target_mib for row in memory_rows):
                raise ProvisioningError(
                    "llama.cpp post-load free VRAM is below the configured fit target"
                )

            used = ",".join(str(row[0]) for row in memory_rows)
            free = ",".join(str(row[1]) for row in memory_rows)
            self._log(
                hostname,
                "info",
                "llama.cpp fitted: "
                f"context_per_slot={fit.context_per_slot} "
                f"slot_context_limit={fit.slot_context_limit} "
                f"slots={fit.slots} "
                f"aggregate_context={fit.aggregate_context} "
                f"kv_unified={str(fit.kv_unified).lower()} "
                f"cache_type_k={fit.cache_type_k} "
                f"cache_type_v={fit.cache_type_v} "
                f"flash_attn={fit.flash_attn} "
                f"gpu_layers={fit.gpu_layers}/{fit.total_layers} "
                f"fit_target_mib={self._settings.llamacpp_fit_target_mib} "
                f"gpu_used_mib={used} gpu_free_mib={free}",
            )
        except Exception:
            await self._stop_failed_llamacpp_start(hostname)
            raise

    async def _run_start_vllm(
        self,
        hostname: str,
        *,
        model: str | None = None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact: ResolvedGGUFArtifact | None = None,
    ) -> str:
        """Run the engine start script and extract model name from stdout."""
        if engine == InferenceEngine.LLAMA_CPP:
            script = "start-llamacpp.sh"
        else:
            script = "start-vllm.sh"
        command = self._script_command(
            script,
            env=self._start_script_env(model, engine, artifact),
            scripts_dir=self._engine_scripts_dir(engine).name,
        )
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
                f"model name not found in {script} output on {hostname}"
            )
        if artifact is not None and model_name != artifact.model_alias:
            raise ProvisioningError(
                f"{script} reported model {model_name!r}, expected selected "
                f"artifact alias {artifact.model_alias!r}"
            )
        return artifact.model_alias if artifact is not None else model_name

    async def _tail_vllm_log(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> None:
        """Tail engine log and feed lines into the provisioning log buffer."""
        if engine == InferenceEngine.LLAMA_CPP:
            log_path = "/var/log/llamacpp-serve.log"
        else:
            log_path = "/var/log/vllm-serve.log"
        try:
            async for _stream, line in self._ssh_client.run_streaming(
                hostname, f"tail -n +1 -f {log_path}"
            ):
                self._log(hostname, "info", line, stream=engine)
        except (SSHConnectionError, RemoteCommandError, asyncio.CancelledError):
            pass

    async def _poll_health(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> None:
        """Poll /health endpoint until 200 OK or timeout (D-10, D-09).

        Tails the engine log concurrently so startup output appears in
        the live log pane while waiting.
        """
        tail_task = asyncio.create_task(self._tail_vllm_log(hostname, engine))

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
        self,
        hostname: str,
        model: str,
        *,
        managed: bool = True,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact_id: str | None = None,
    ) -> None:
        """Register node in etcd with correct fields (D-11, D-12)."""
        node = Node(
            node_id=hostname,
            endpoint=self.validate_endpoint(hostname),
            status=NodeStatus.HEALTHY,
            model=model,
            engine=engine,
            artifact_id=artifact_id,
            last_heartbeat=datetime.now(UTC),
            managed=managed,
        )
        key, value = node_to_etcd(node, self._etcd_client.prefix)
        # ponytail: etcd3gw is sync, asyncio.to_thread wraps it (Pitfall 5)
        lease_id: int | None = None
        if managed:
            try:
                lease_id = await asyncio.to_thread(self._etcd_client.grant_node_lease)
            except Exception:
                # Lease protection is convergent rather than a precondition
                # for completing an otherwise successful hour-long provision.
                # The health checker adopts this managed key after its next
                # successful probe.
                logger.warning(
                    "node_lease_grant_failed_during_registration",
                    hostname=hostname,
                    exc_info=True,
                )
                self._log(
                    hostname,
                    "warning",
                    "Node registered without a lease; health checking will retry",
                )

        try:
            if lease_id is None:
                await asyncio.to_thread(self._etcd_client.put, key, value)
            else:
                await asyncio.to_thread(
                    self._etcd_client.put,
                    key,
                    value,
                    lease_id=lease_id,
                )
        except Exception:
            if lease_id is not None:
                with suppress(Exception):
                    await asyncio.to_thread(
                        self._etcd_client.revoke_lease,
                        lease_id,
                    )
            raise
        logger.info("node_registered", hostname=hostname, model=model, key=key)

    def fire_background(
        self,
        coro: Coroutine[object, object, None],
        *,
        provisioning_hostname: str | None = None,
        provisioning_identity: ProvisioningIdentity | None = None,
        task_name: str | None = None,
    ) -> asyncio.Task[None]:
        """Schedule and observe an owned background operation.

        Provisioning capacity is reserved only after ``create_task`` succeeds,
        so scheduling failures cannot leak a slot. Teardown and other cleanup
        work omit ``provisioning_hostname`` and remain admissible when the
        provisioning limit is full.
        """
        if (provisioning_hostname is None) != (provisioning_identity is None):
            raise ValueError(
                "provisioning_hostname and provisioning_identity must be "
                "supplied together"
            )

        if provisioning_hostname is not None:
            current = self._provisioning_tasks.get(provisioning_hostname)
            if current is not None and not current.task.done():
                raise RuntimeError(
                    f"Provisioning task already active for '{provisioning_hostname}'"
                )
            active = sum(
                not record.task.done() for record in self._provisioning_tasks.values()
            )
            if active >= self._settings.max_concurrent_provisions:
                raise ProvisioningCapacityError(
                    active=active,
                    limit=self._settings.max_concurrent_provisions,
                )

            if task_name is None:
                task_name = f"provision:{provisioning_hostname}"

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
            if task_name is None:
                task = asyncio.create_task(scheduled_coro)
            else:
                task = asyncio.create_task(scheduled_coro, name=task_name)
        except Exception:
            if scheduled_coro is not coro:
                scheduled_coro.close()
            raise
        self._background_tasks.add(task)

        if provisioning_hostname is not None:
            if provisioning_identity is None:  # narrowed by the paired check above
                raise RuntimeError("provisioning identity was not initialized")
            record = _ProvisioningTask(task, provisioning_identity)
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

            # Cancellation is the expected PR 20 shutdown path. Calling
            # exception() on a cancelled task raises CancelledError and would
            # turn clean shutdown into callback noise.
            if done_task.cancelled():
                return

            error = done_task.exception()
            if error is not None:
                logger.error(
                    "background_task_failed",
                    task_name=done_task.get_name(),
                    hostname=provisioning_hostname,
                    error=str(error),
                    exception_type=type(error).__name__,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(_task_done)
        return task

    async def shutdown(self) -> None:
        """Cancel and await every lifecycle task still owned by this process.

        This method owns shutdown lifecycle only. Per-task exception observation
        remains the responsibility of ``fire_background``'s done callback so a
        future generic observer cannot double-log failures during shutdown.
        """
        tasks = tuple(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_active_provision(
        self, hostname: str
    ) -> ProvisioningIdentity | None:
        """Cancel *hostname* provisioning and return its serving identity."""
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
        return record.identity

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
        provisioning_identity: ProvisioningIdentity | None = None,
        recovery_engine: InferenceEngine | None = None,
        lifecycle_lease: HostLifecycleLease | None = None,
    ) -> None:
        """Teardown *hostname* under the shared host lifecycle coordinator."""
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            engine = self._resolve_teardown_engine(
                hostname,
                force=force,
                provisioning_identity=provisioning_identity,
                recovery_engine=recovery_engine,
            )
            await self._teardown(hostname, force=force, engine=engine)
        finally:
            lease.release()

    def _resolve_teardown_engine(
        self,
        hostname: str,
        *,
        force: bool,
        provisioning_identity: ProvisioningIdentity | None,
        recovery_engine: InferenceEngine | None,
    ) -> InferenceEngine:
        """Resolve one authoritative engine without silently guessing vLLM."""
        node = None
        if self._registry is not None:
            node = self._registry.get(hostname)
        if provisioning_identity is not None:
            if recovery_engine is not None:
                raise ProvisioningError(
                    "recovery_engine cannot override an active provisioning identity"
                )
            if node is not None and node.engine != provisioning_identity.engine:
                logger.warning(
                    "teardown_identity_prefers_cancelled_provision",
                    hostname=hostname,
                    cancelled_engine=provisioning_identity.engine,
                    registry_engine=node.engine,
                )
            return provisioning_identity.engine
        if node is not None:
            if recovery_engine is not None:
                raise ProvisioningError(
                    "recovery_engine cannot override a registered node identity"
                )
            return node.engine
        if recovery_engine is not None:
            if not force:
                raise ProvisioningError(
                    "recovery_engine requires force=true when no node identity exists"
                )
            return recovery_engine
        raise ProvisioningError(
            f"Cannot determine the inference engine for unregistered host {hostname!r}"
        )

    async def _teardown(
        self,
        hostname: str,
        *,
        force: bool = False,
        engine: InferenceEngine,
    ) -> None:
        """Teardown a provisioned node.

        Graceful: drain -> stop engine -> deregister.
        Force: kill -9 -> deregister.
        """
        teardown_started_at = datetime.now(UTC)
        logger.info("teardown_start", hostname=hostname, force=force, engine=engine)
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

            if engine == InferenceEngine.LLAMA_CPP:
                stopping_step = ProvisioningStep.STOPPING_LLAMACPP
            else:
                stopping_step = ProvisioningStep.STOPPING_VLLM
            await self._update_state(
                hostname, stopping_step, started_at=teardown_started_at
            )
            self._log(hostname, "info", f"Stopping {engine} process")
            # Existing nodes may predate stop-vllm.sh, and a cancelled setup
            # may not have reached its upload step. Refresh the bundle before
            # invoking the verified stop path.
            try:
                await self._upload_scripts(hostname, engine=engine)
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
            if engine == InferenceEngine.LLAMA_CPP:
                stop_script = "stop-llamacpp.sh"
            else:
                stop_script = "stop-vllm.sh"
            stop_command = self._script_command(
                stop_script,
                args=("--force",) if force else (),
                scripts_dir=self._engine_scripts_dir(engine).name,
            )
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
        except Exception as exc:
            self._log(hostname, "error", f"Teardown failed: {exc}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="teardown",
                error=str(exc),
                started_at=teardown_started_at,
            )
            if self._registry is not None:
                self._registry.update_status(
                    hostname,
                    NodeStatus.FAILED,
                    allowed_from=set(NodeStatus),
                )
            raise
        finally:
            self._log_buffer.mark_complete(hostname)

        logger.info("teardown_complete", hostname=hostname)
