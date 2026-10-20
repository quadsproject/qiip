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
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from inference_proxy.config.settings import LLMFitSettings, ProvisioningSettings
from inference_proxy.discovery.etcd_client import EtcdClient, EtcdRecord
from inference_proxy.discovery.registry import NodeRegistry, node_with_status
from inference_proxy.discovery.serializer import node_from_etcd, node_to_etcd
from inference_proxy.huggingface.artifacts import (
    GGUFArtifactError,
    GGUFArtifactIndex,
    ResolvedGGUFArtifact,
)
from inference_proxy.models.endpoint import (
    EndpointPolicy,
    EndpointValidationError,
    build_backend_url,
)
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppCacheType,
    LlamaCppFlashAttention,
    LlamaCppGPUState,
    LlamaCppProfileRuntime,
    LlamaCppRuntimeEffective,
    LlamaCppRuntimeRequest,
    LlamaCppRuntimeState,
    LlamaCppSizingMode,
    LlamaCppSpeculativeEffective,
    LlamaCppSpeculativeType,
    Node,
    NodeGPU,
    NodePlacement,
    NodeStatus,
    VllmAdjustment,
    VllmGPUState,
    VllmMemoryEstimate,
    VllmModelSelection,
    VllmParams,
    VllmRuntimeEffective,
    VllmRuntimeRequest,
    VllmRuntimeState,
    VllmServingVerification,
)
from inference_proxy.provisioning.bundles import (
    bundle_files,
    bundle_manifest,
    write_bundle,
)
from inference_proxy.provisioning.host_lifecycle import (
    HostLifecycleCoordinator,
    HostLifecycleLease,
)
from inference_proxy.provisioning.llamacpp_profile import (
    ProfileEvidence,
    ProfileEvidenceError,
    latest_launch,
    parse_profile_evidence,
    split_profile_sections,
)
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.recovery import (
    FailureClass,
    RetryPolicy,
    classify_failure,
    next_action,
    setup_plan_id,
)
from inference_proxy.provisioning.remote_logs import RemoteLogCollector
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)
from inference_proxy.provisioning.state import (
    ProvisioningResumeState,
    ProvisioningState,
    ProvisioningStep,
)
from inference_proxy.provisioning.vllm_fit import (
    VllmDeviceMemory,
    VllmFitPlan,
    VllmRequested,
    is_memory_failure,
    plan_vllm_launch,
    vllm_profile_bucket,
)
from inference_proxy.provisioning.vllm_serving import (
    VllmServingError,
    fetch_server_identity,
    parse_vllm_processes,
    probe_inference,
    revision_from_root,
    select_model,
    select_process,
)
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker

if TYPE_CHECKING:
    from etcd3gw.types import KeyValue

logger = structlog.get_logger()

_SELF_SETUP_PROBE_TIMEOUT = 5.0
# /health responses that mean the endpoint is not implemented, so a server is
# still adoptable via the OpenAI-compatible /v1/models contract. An explicit
# unhealthy response (e.g. 503) is authoritative and refuses adoption.
_OPTIONAL_HEALTH_STATUSES = frozenset({404, 405, 501})

STEP_PATTERN = re.compile(r"\[STEP:(\w+):(START|OK|SKIP|FAIL|WARN)\]")
RESUME_PATTERN = re.compile(
    r"\[RESUME:(maintenance_required|reboot_required):([^\]\r\n]+)\]"
)
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
    r"qiip_fit_plan: sizing=(?P<sizing>auto|custom|profile) "
    r"train_context=(?P<train_context>\d+) "
    r"context_per_slot=(?P<context>\d+) "
    r"slots=(?P<slots>\d+) "
    r"aggregate_context=(?P<aggregate>\d+) "
    r"fit_target_mib=(?P<target>\d+) "
    r"cache_type_k=(?P<cache_type_k>f16|q8_0|q4_0) "
    r"cache_type_v=(?P<cache_type_v>f16|q8_0|q4_0) "
    r"flash_attn=(?P<flash_attn>auto|on) "
    r"estimator_overrun_used=(?P<estimator_overrun_used>true|false)"
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
# qiip's own node-side provisioning commands, as they appear in ``ps`` args:
# setup, engine start, and the detached command worker. The long-lived
# ``provision-logs.py engine`` sink belongs to a running server, not to
# provisioning, and must not match.
_REMOTE_LIFECYCLE_PATTERN = re.compile(
    r"(?:auto-(?:llamacpp|vllm)/(?:setup|start-[a-z]+)\.sh"
    r"|common/provision-logs\.py worker\b)"
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
        "preflight.sh",
        "wait-fabric.sh",
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


class SetupResumeRequiredError(ProvisioningError):
    """Setup stopped at a maintenance boundary; retry after operator action."""

    def __init__(self, state: ProvisioningResumeState, reason: str) -> None:
        super().__init__(reason)
        self.resume_state = state


class SetupDeadlineError(ProvisioningError):
    """One setup attempt exceeded its stage-specific total deadline.

    Deliberately not a ``TimeoutError``: a download socket timeout is
    transient, but the run's own deadline expiring is terminal. Retrying it
    automatically would only repeat the exhausted wait.
    """


class SelfSetupError(Exception):
    """Raised when an already-running OpenAI-compatible server cannot be adopted."""


def served_model_id(payload: object) -> str | None:
    """Return the primary OpenAI-compatible model id from a ``/v1/models`` body.

    Self-setup adoption tracks exactly one model: the first entry the server
    reports. Additional aliases and LoRA entries are intentionally not
    tracked (single-model contract). Re-adoption re-probes the live server
    and reconciles the stored model, so an external restart with another
    primary model is refreshed on each re-adoption.
    """
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    if not isinstance(model_id, str):
        return None
    model_id = model_id.strip()
    return model_id or None


class ProvisioningCapacityError(RuntimeError):
    """Raised when the configured concurrent-provision limit is full."""

    def __init__(self, *, active: int, limit: int) -> None:
        self.active = active
        self.limit = limit
        super().__init__(
            f"Provisioning capacity reached ({active}/{limit} active tasks)"
        )


class RelaunchPreconditionError(RuntimeError):
    """Raised when a node cannot safely accept a managed relaunch."""


class RelaunchValidationError(ValueError):
    """Raised when a relaunch policy conflicts with observed node limits."""


class RelaunchConflictError(RuntimeError):
    """Raised when etcd ownership changes during a relaunch transaction."""


class PreflightError(Exception):
    """Raised when pre-flight validation fails (D-01 through D-04).

    Collects all failures before raising so operators see every problem
    at once (D-03).
    """

    def __init__(self, hostname: str, failures: list[str]) -> None:
        self.hostname = hostname
        self.failures = failures
        super().__init__(f"Pre-flight failed on {hostname}: {'; '.join(failures)}")


def _plain_decimal(value: float) -> str:
    """Format a bounded sampling value the way the start script validates it."""
    text = f"{value:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


@dataclass(frozen=True)
class ProvisioningIdentity:
    """Engine and immutable artifact selected for one provisioning operation."""

    engine: InferenceEngine
    artifact_id: str | None = None


@dataclass(frozen=True)
class LlamaCppRuntimeFit:
    """Effective llama.cpp sizing parsed from the completed startup log."""

    sizing: str
    train_context: int
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
    estimator_overrun_used: bool
    profile: ProfileEvidence | None = None


def _parse_llamacpp_runtime_fit(log_text: str) -> LlamaCppRuntimeFit:
    """Validate and return the final managed llama.cpp fit evidence."""
    if LLAMACPP_FIT_FAILURE_PATTERN.search(log_text):
        raise ProvisioningError("llama.cpp reported a runtime fitting failure")
    plans = list(LLAMACPP_PLAN_PATTERN.finditer(log_text))
    if not plans:
        raise ProvisioningError("llama.cpp startup log has no QIIP VRAM plan")
    profile: ProfileEvidence | None = None
    if plans[-1].group("sizing") == LlamaCppSizingMode.PROFILE.value:
        # A profile launch also loads a speculative draft, which writes the
        # same record kinds as the target. Read the target's records from the
        # target's section only; the draft is verified separately.
        try:
            profile = parse_profile_evidence(log_text)
            launch_text = latest_launch(log_text)
            target_text = split_profile_sections(launch_text).target
        except ProfileEvidenceError as exc:
            raise ProvisioningError(str(exc)) from exc
        slot_text = launch_text
        log_text = target_text
    else:
        slot_text = log_text
    contexts = list(LLAMACPP_CONTEXT_PATTERN.finditer(slot_text))
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
        raise ProvisioningError(
            "llama.cpp startup log has no KV cache type record "
            "(recurrent-state models are unsupported by managed provisioning)"
        )

    plan = plans[-1].groupdict()
    context = contexts[-1].groupdict()
    aggregate_context = int(aggregate_contexts[-1].group("context"))
    offload = offloads[-1].groupdict()
    cache_record = cache_records[-1].groupdict()
    fit = LlamaCppRuntimeFit(
        sizing=plan["sizing"],
        train_context=int(plan["train_context"]),
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
        estimator_overrun_used=plan["estimator_overrun_used"] == "true",
        profile=profile,
    )
    if profile is None and LlamaCppCacheType.Q4_0.value in (
        fit.cache_type_k,
        fit.cache_type_v,
    ):
        raise ProvisioningError(
            "llama.cpp planner sizing does not support a q4_0 KV cache"
        )
    if profile is not None and any(
        record != (fit.cache_type_k, fit.cache_type_v)
        for record in profile.target_cache_types
    ):
        raise ProvisioningError(
            "llama.cpp runtime KV cache types differ from its VRAM plan"
        )
    if (
        fit.train_context < 1
        or fit.context_per_slot < 1
        or fit.context_per_slot > fit.train_context
        or fit.slot_context_limit != min(fit.train_context, fit.aggregate_context)
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
    expected_flash_attn = (
        "auto" if fit.cache_type_k == "f16" and profile is None else "on"
    )
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


class ProvisioningOperationChangedError(RuntimeError):
    """The inspected provisioning operation no longer owns the host."""


@dataclass
class ProvisioningTask:
    task: asyncio.Task[None]
    identity: ProvisioningIdentity
    operation: BackgroundOperation
    started: bool = False
    attempt_id: str | None = None


class BackgroundOperation(StrEnum):
    """Capacity-counted background operation type."""

    PROVISION = "provision"
    RELAUNCH = "relaunch"


class DrainOutcome(StrEnum):
    """Result of waiting for a node's active requests to drain."""

    COMPLETE = "complete"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"


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
        hf_cache_dir: str | None = None,
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
        self._remote_logs = (
            RemoteLogCollector(
                ssh_client, self._log_buffer.store, self._log_buffer, settings
            )
            if self._log_buffer.store is not None
            else None
        )
        self._lifecycle = lifecycle_coordinator or HostLifecycleCoordinator()
        self._hf_token = hf_token
        self._nfs_export = nfs_export
        self._artifact_index = artifact_index
        self._hf_cache_dir = hf_cache_dir
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._owner_updates: set[asyncio.Task[Node]] = set()
        self._provisioning_tasks: dict[str, ProvisioningTask] = {}
        self._explicit_cancel_tasks: set[asyncio.Task[None]] = set()
        self._remote_bundles: dict[tuple[str, InferenceEngine], str] = {}

    @property
    def log_buffer(self) -> ProvisioningLogBuffer:
        return self._log_buffer

    def _bundle_version(self, engine: InferenceEngine) -> str:
        """Return the content digest of the scripts uploaded for *engine*.

        The bundle identity is part of the setup plan: a re-uploaded script
        that changes an install or build must invalidate completed stages even
        when every configured setting is unchanged.
        """
        identity, _manifest = bundle_manifest(
            bundle_files(self._engine_scripts_dir(engine), self._common_scripts_dir())
        )
        return "sha256:" + identity

    def _begin_log(
        self,
        hostname: str,
        engine: InferenceEngine,
        *,
        model: str | None = None,
        operation: str = "provision",
        plan_id: str | None = None,
        setup_inputs: dict[str, Any] | None = None,
    ) -> None:
        bundle_version = self._bundle_version(engine)
        self._log_buffer.create(
            hostname,
            engine=engine.value,
            model=model,
            operation=operation,
            bundle_version=bundle_version,
            plan_id=plan_id or bundle_version,
            setup_inputs=setup_inputs,
        )
        if self._log_buffer.store is not None and self._remote_logs is not None:
            recorder = self._remote_logs.recorders.get(hostname)
            if recorder:
                self._log_buffer.store.update(
                    self._log_buffer.attempts[hostname], recorder_path=recorder
                )

    async def collect_logs(self, hostname: str, attempt_id: str) -> dict[str, object]:
        store = self._log_buffer.store
        if (
            store is None
            or self._remote_logs is None
            or store.get(attempt_id)["hostname"] != hostname
        ):
            raise KeyError(attempt_id)
        # The durable manifest describes any unavailable source.
        with suppress(SSHConnectionError, TimeoutError):
            await self._collect_remote_logs(attempt_id)
        await self._remote_logs.diagnose(attempt_id)
        return store.get(attempt_id)

    async def _collect_remote_logs(
        self, attempt_id: str, *, finish: bool = False, cancel: bool = False
    ) -> None:
        assert self._remote_logs is not None
        try:
            async with asyncio.timeout(self._settings.diagnostics_timeout):
                await self._remote_logs.collect(
                    attempt_id, finish=finish, cancel=cancel
                )
        except TimeoutError:
            # The outer deadline cancels collect(), bypassing its error handler.
            self._remote_logs.store.issue(
                attempt_id, "Remote log retrieval deadline exceeded", source="remote"
            )
            raise

    def _record_failure_classification(
        self, hostname: str, error: BaseException
    ) -> None:
        """Persist the retry class and operator action on the attempt.

        The original error stays in the retained failure/diagnostics records;
        this only adds the recovery decision so the UI can show what to do next.
        """
        store = self._log_buffer.store
        attempt_id = self._log_buffer.attempts.get(hostname)
        if store is None or attempt_id is None:
            return
        with suppress(KeyError):
            store.update(
                attempt_id,
                failure_class=str(classify_failure(error)),
                next_action=next_action(error),
            )

    async def _capture_failure(
        self, hostname: str, stage: str, error: BaseException
    ) -> None:
        try:
            attempt_id = self._log_buffer.attempts.get(hostname)
            if self._remote_logs is not None and attempt_id is not None:
                self._remote_logs.record_failure(attempt_id, stage, error)
                await self._remote_logs.diagnose(attempt_id)
        except Exception:
            logger.warning(
                "failure_diagnostics_unavailable", hostname=hostname, exc_info=True
            )

    async def _finish_remote_logs(self, hostname: str, *, cancel: bool = False) -> None:
        try:
            attempt_id = self._log_buffer.attempts.get(hostname)
            if self._remote_logs is None or attempt_id is None:
                return
            manifest = await asyncio.to_thread(self._remote_logs.store.get, attempt_id)
            if not manifest.get("phases"):
                return
            task = asyncio.current_task()
            # A gateway shutdown stops retrieval, not the detached node worker.
            interrupted = bool(task and task.cancelling()) and not cancel
            await self._collect_remote_logs(
                attempt_id, finish=not interrupted, cancel=cancel
            )
        except Exception:
            logger.warning(
                "remote_logs_finish_failed", hostname=hostname, exc_info=True
            )

    def _mark_log_complete(self, hostname: str) -> None:
        try:
            task = asyncio.current_task()
            attempt_id = self._log_buffer.attempts.get(hostname)
            store = self._log_buffer.store
            if (
                task
                and task.cancelling()
                and task not in self._explicit_cancel_tasks
                and store
                and attempt_id
            ):
                stored = store.get(attempt_id)
                status = stored["status"]
                # A genuinely recorded failure (structured failure object set
                # by record_failure before diagnostics) must survive shutdown,
                # with the interrupted collection recorded as a separate issue.
                if status == "failed" and stored.get("failure"):
                    store.issue(
                        attempt_id,
                        "Gateway collection interrupted; remote command left running",
                    )
                elif status != "complete":
                    store.update(
                        attempt_id,
                        status="interrupted",
                        failure_summary="Gateway collection interrupted; remote command left running",
                    )
        except Exception:
            logger.warning(
                "log_interrupt_status_failed", hostname=hostname, exc_info=True
            )
        try:
            self._log_buffer.mark_complete(hostname)
        except Exception:
            logger.warning("log_completion_failed", hostname=hostname, exc_info=True)

    def _engine_log_path(self, hostname: str, engine: InferenceEngine) -> str:
        if self._remote_logs is not None:
            return f"{self._settings.log_remote_root}/{self._log_buffer.attempts[hostname]}.engine.log"
        return (
            "/var/log/llamacpp-serve.log"
            if engine == InferenceEngine.LLAMA_CPP
            else "/var/log/vllm-serve.log"
        )

    def validate_endpoint(self, hostname: str, port: int | None = None) -> str:
        """Return the canonical provisioned endpoint or fail with a config hint."""
        candidate = f"{hostname}:{port or self._settings.vllm_port}"
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
            common_dir / "profiles.sh",
            common_dir / "generations.py",
        }
        if engine == InferenceEngine.LLAMA_CPP:
            required.add(common_dir / "llamacpp_artifacts.py")
        if self._remote_logs is not None:
            required.add(common_dir / "provision-logs.py")
        missing = sorted(str(path) for path in required if not path.is_file())
        if missing:
            raise ProvisioningError(
                "Provisioning script bundle is incomplete; missing: "
                + ", ".join(missing)
            )
        return engine_dir, common_dir

    def _setup_plan_inputs(
        self,
        engine: InferenceEngine = InferenceEngine.VLLM,
        *,
        artifact: ResolvedGGUFArtifact | None = None,
        llamacpp_request: LlamaCppRuntimeRequest | None = None,
        vllm_params: VllmParams | None = None,
    ) -> dict[str, Any]:
        """Return the inputs whose change invalidates a completed stage.

        The identity must cover everything a setup stage installs, builds, or
        mounts. The caller passes the resulting mapping to ``_begin_log`` so a
        retry with different inputs cannot resume stale checkpoints.
        """
        inputs: dict[str, Any] = {
            "engine": engine.value,
            "bundle_version": self._bundle_version(engine),
            # AUTOVLLM_API_PORT / AUTOLLAMACPP_PORT feed configure_firewall,
            # the systemd unit, and the generation config, so a port change
            # must invalidate those completed stages too.
            "api_port": self._settings.vllm_port,
            "nfs_export": self._required_nfs_export(),
            "nfs_mount_point": self._settings.nfs_mount_point,
            "nvidia_driver_version": self._settings.nvidia_driver_version,
            "nvidia_driver_sha256": self._settings.nvidia_driver_sha256,
            "llmfit_version": self._llmfit_version,
            "llmfit_sha256": self._llmfit_sha256,
            "min_disk_gb": self._settings.min_disk_gb,
            "generation_root": str(self._settings.generation_root / engine.value),
        }
        if engine == InferenceEngine.LLAMA_CPP:
            inputs.update(
                llamacpp_version=self._settings.llamacpp_version,
                llamacpp_sha256=self._settings.llamacpp_sha256,
                llamacpp_source_url=self._settings.llamacpp_source_download_url(),
                llamacpp_artifact_catalog_url=(
                    self._settings.llamacpp_artifact_catalog_url
                ),
                llamacpp_artifact_catalog_sha256=(
                    self._settings.llamacpp_artifact_catalog_sha256
                ),
                llamacpp_allow_source_build=self._settings.llamacpp_allow_source_build,
                llamacpp_build_jobs=self._settings.llamacpp_build_jobs,
                artifact_id=artifact.artifact_id if artifact is not None else None,
                runtime_request=(
                    llamacpp_request.model_dump(mode="json")
                    if llamacpp_request is not None
                    else None
                ),
            )
        else:
            inputs.update(
                model=artifact.model_alias if artifact is not None else None,
                vllm_params=(
                    vllm_params.model_dump(mode="json")
                    if vllm_params is not None
                    else None
                ),
            )
        return inputs

    def _setup_script_env(
        self, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> dict[str, str]:
        """Return the exact environment accepted by setup.sh."""
        env = {
            "QIIP_GENERATION_ROOT": str(self._settings.generation_root / engine.value),
            "AUTOVLLM_NFS_EXPORT": self._required_nfs_export(),
            "AUTOVLLM_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
            "AUTOVLLM_NVIDIA_DRIVER_VERSION": self._settings.nvidia_driver_version,
            "AUTOVLLM_NVIDIA_DRIVER_SHA256": self._settings.nvidia_driver_sha256,
            "AUTOVLLM_API_PORT": str(self._settings.vllm_port),
            "AUTOVLLM_MIN_FREE_GB": str(self._settings.min_disk_gb),
            "AUTOVLLM_LLMFIT_VERSION": self._llmfit_version,
            "AUTOVLLM_LLMFIT_SHA256": self._llmfit_sha256,
        }
        if engine == InferenceEngine.LLAMA_CPP:
            env["AUTOLLAMACPP_VERSION"] = self._settings.llamacpp_version
            env["AUTOLLAMACPP_SHA256"] = self._settings.llamacpp_sha256
            env["AUTOLLAMACPP_SOURCE_URL"] = (
                self._settings.llamacpp_source_download_url()
            )
            env["AUTOLLAMACPP_ARTIFACT_CATALOG_URL"] = (
                self._settings.llamacpp_artifact_catalog_url
            )
            env["AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256"] = (
                self._settings.llamacpp_artifact_catalog_sha256
            )
            env["AUTOLLAMACPP_ALLOW_SOURCE_BUILD"] = (
                "1" if self._settings.llamacpp_allow_source_build else "0"
            )
            if self._settings.llamacpp_build_jobs:
                env["AUTOLLAMACPP_BUILD_JOBS"] = str(self._settings.llamacpp_build_jobs)
        return env

    def _start_script_env(
        self,
        model: str | None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact: ResolvedGGUFArtifact | None = None,
        llamacpp_request: LlamaCppRuntimeRequest | None = None,
        vllm_params: VllmParams | None = None,
        draft_artifact: ResolvedGGUFArtifact | None = None,
        vllm_plan: VllmFitPlan | None = None,
    ) -> dict[str, str]:
        """Return the exact environment accepted by the engine start script."""
        if engine == InferenceEngine.LLAMA_CPP:
            request = llamacpp_request or self._default_llamacpp_runtime_request()
            env = {
                "AUTOLLAMACPP_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
                "AUTOLLAMACPP_PORT": str(self._settings.vllm_port),
                "AUTOLLAMACPP_REQUIRE_CUDA": "1",
                "AUTOLLAMACPP_MANAGED": "1",
                "AUTOLLAMACPP_FIT_TARGET_MIB": str(request.fit_target_mib),
                "AUTOLLAMACPP_MANAGED_SIZING": request.sizing.value,
                "AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT": (
                    str(request.context_per_slot)
                    if request.context_per_slot is not None
                    else ""
                ),
                "AUTOLLAMACPP_MANAGED_PARALLEL": (
                    str(request.slots) if request.slots is not None else ""
                ),
                "AUTOLLAMACPP_MANAGED_CACHE_TYPE": (
                    request.cache_type.value if request.cache_type is not None else ""
                ),
                "AUTOLLAMACPP_MANAGED_ALLOW_ESTIMATOR_OVERRUN": (
                    "1" if request.allow_estimator_overrun else "0"
                ),
            }
            if artifact is None:
                raise ProvisioningError("llama_cpp start requires a GGUF artifact")
            env["AUTOLLAMACPP_GGUF_PATH"] = artifact.node_relative_entrypoint
            env["AUTOLLAMACPP_MODEL_ALIAS"] = artifact.model_alias
            if self._nfs_export is not None:
                env["AUTOVLLM_NFS_EXPORT"] = self._nfs_export
            env.update(self._profile_script_env(request, draft_artifact))
        else:
            if llamacpp_request is not None:
                raise ProvisioningError(
                    "llama.cpp sizing policy is only valid for llama_cpp starts"
                )
            env = {
                "AUTOVLLM_NFS_MOUNT_POINT": self._settings.nfs_mount_point,
                "AUTOVLLM_API_PORT": str(self._settings.vllm_port),
                "AUTOVLLM_MIN_FREE_GB": str(self._settings.min_disk_gb),
            }
            if self._nfs_export is not None:
                env["AUTOVLLM_NFS_EXPORT"] = self._nfs_export
            if model is not None:
                env["AUTOVLLM_MODEL"] = model
            if vllm_params is not None:
                if vllm_params.tensor_parallel_size is not None:
                    env["AUTOVLLM_TENSOR_PARALLEL"] = str(
                        vllm_params.tensor_parallel_size
                    )
                if vllm_params.max_model_len is not None:
                    env["AUTOVLLM_MAX_MODEL_LEN"] = str(vllm_params.max_model_len)
                if vllm_params.gpu_memory_utilization is not None:
                    env["AUTOVLLM_GPU_MEM_UTIL"] = str(
                        vllm_params.gpu_memory_utilization
                    )
                if vllm_params.max_num_batched_tokens is not None:
                    env["AUTOVLLM_MAX_BATCHED_TOKENS"] = str(
                        vllm_params.max_num_batched_tokens
                    )
                if vllm_params.tool_call_parser is not None:
                    env["AUTOVLLM_TOOL_CALL_PARSER"] = vllm_params.tool_call_parser
                if vllm_params.reasoning_parser is not None:
                    env["AUTOVLLM_REASONING_PARSER"] = vllm_params.reasoning_parser
                if vllm_params.dtype is not None:
                    env["AUTOVLLM_DTYPE"] = vllm_params.dtype
                if vllm_params.gpu_devices is not None:
                    env["AUTOVLLM_GPU_DEVICES"] = ",".join(
                        str(device) for device in vllm_params.gpu_devices
                    )
            if vllm_plan is not None:
                if vllm_plan.effective.attention_backend is not None:
                    env["AUTOVLLM_ATTENTION_BACKEND"] = (
                        vllm_plan.effective.attention_backend
                    )
                env["AUTOVLLM_FIT_BUDGET_MIB"] = str(
                    vllm_plan.estimate.usable_budget_mib
                )
                env["AUTOVLLM_FIT_RESERVE_MIB"] = str(vllm_plan.estimate.reserve_mib)
                env["AUTOVLLM_FIT_OVERHEAD_MIB"] = str(
                    vllm_plan.estimate.runtime_overhead_mib
                )
        if self._hf_token:
            env["HF_TOKEN"] = self._hf_token
        return env

    @staticmethod
    def _profile_script_env(
        request: LlamaCppRuntimeRequest,
        draft_artifact: ResolvedGGUFArtifact | None,
    ) -> dict[str, str]:
        """Return the typed profile inputs, or nothing for planner sizing."""
        profile = request.profile
        if profile is None:
            if draft_artifact is not None:
                raise ProvisioningError(
                    "a draft artifact is only valid with a llama.cpp profile"
                )
            return {}
        if (profile.draft_artifact_id is None) != (draft_artifact is None) or (
            draft_artifact is not None
            and draft_artifact.artifact_id != profile.draft_artifact_id
        ):
            raise ProvisioningError(
                "the resolved draft artifact differs from the llama.cpp profile"
            )
        sampling = profile.sampling
        env = {
            "AUTOLLAMACPP_PROFILE_ID": profile.profile_id,
            "AUTOLLAMACPP_PROFILE_VERSION": str(profile.profile_version),
            "AUTOLLAMACPP_PROFILE_UBATCH": str(profile.ubatch),
            "AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB": str(profile.required_free_mib),
            "AUTOLLAMACPP_PROFILE_GPU_NAME": profile.gpu_name,
            "AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB": str(profile.gpu_min_total_mib),
            "AUTOLLAMACPP_PROFILE_SPEC_TYPE": profile.speculative_type.value,
            "AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX": str(
                profile.speculative_draft_n_max
            ),
            "AUTOLLAMACPP_PROFILE_TEMPERATURE": _plain_decimal(sampling.temperature),
            "AUTOLLAMACPP_PROFILE_TOP_P": _plain_decimal(sampling.top_p),
            "AUTOLLAMACPP_PROFILE_TOP_K": str(sampling.top_k),
            "AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS": (
                "1" if profile.disable_cuda_graphs else "0"
            ),
        }
        if profile.draft_cache_type is not None:
            env["AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE"] = (
                profile.draft_cache_type.value
            )
        if draft_artifact is not None:
            env["AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH"] = (
                draft_artifact.node_relative_entrypoint
            )
        if sampling.min_p is not None:
            env["AUTOLLAMACPP_PROFILE_MIN_P"] = _plain_decimal(sampling.min_p)
        if sampling.presence_penalty is not None:
            env["AUTOLLAMACPP_PROFILE_PRESENCE_PENALTY"] = _plain_decimal(
                sampling.presence_penalty
            )
        return env

    def _default_llamacpp_runtime_request(self) -> LlamaCppRuntimeRequest:
        """Return the configured automatic policy for a fresh managed start."""
        return LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.AUTO,
            fit_target_mib=self._settings.llamacpp_fit_target_mib,
        )

    def _script_command(
        self,
        script_name: str,
        *,
        env: dict[str, str] | None = None,
        args: tuple[str, ...] = (),
        scripts_dir: str | None = None,
    ) -> str:
        """Build quoted shell input for SSH or the recorder, never shell argv."""
        dir_name = scripts_dir or self._settings.scripts_dir.name
        script_path = str(PurePosixPath(dir_name, script_name))
        command = shlex.join(("bash", script_path, *args))
        if not env:
            return command
        assignments = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in env.items()
        )
        return f"{assignments} {command}"

    def _remote_script_command(
        self,
        hostname: str,
        engine: InferenceEngine,
        script: str,
        *,
        env: dict[str, str] | None = None,
        args: tuple[str, ...] = (),
    ) -> str:
        bundle = self._remote_bundles.get((hostname, engine))
        directory = self._engine_scripts_dir(engine).name
        if bundle is None:
            return self._script_command(
                script, scripts_dir=directory, env=env, args=args
            )
        if script == "setup.sh":
            return self._script_command(
                script, scripts_dir=f"{bundle}/{directory}", env=env, args=args
            )
        command = shlex.join(
            (
                "python3",
                f"{bundle}/common/generations.py",
                "exec",
                str(self._settings.generation_root / engine.value),
                f"{bundle}/{directory}",
                script,
                *args,
            )
        )
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in (env or {}).items()
        )
        return f"{assignments} {command}" if assignments else command

    def _record_generation(self, hostname: str, line: str) -> None:
        if not line.startswith("[GENERATION:") or not line.endswith("]"):
            return
        try:
            selected = json.loads(line[len("[GENERATION:") : -1])
            if not isinstance(selected, dict) or not isinstance(
                selected.get("runtime_path"), str
            ):
                raise ValueError("expected generation metadata")
        except ValueError:
            logger.warning("invalid_generation_marker", hostname=hostname)
            return
        if self._log_buffer.store is not None:
            self._log_buffer.store.update(
                self._log_buffer.attempts[hostname],
                selected_generation=selected,
            )

    def _log(
        self,
        hostname: str,
        level: str,
        msg: str,
        *,
        stream: str | None = None,
    ) -> None:
        if self._remote_logs is not None and stream is not None:
            return  # Remote raw output was already committed by its sequence.
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
        resume_state: ProvisioningResumeState | None = None,
        started_at: datetime | None = None,
        history: bool = True,
    ) -> None:
        """Write provisioning state to etcd (D-05). Best-effort (Pitfall 3).

        ``history`` records the state transition on the current attempt.
        Callers that run before ``_begin_log`` (power-on, a cancel before the
        attempt starts) pass ``history=False``: the buffer still tracks the
        previous finished attempt, so a history write would corrupt it.
        """
        now = datetime.now(UTC)
        store = self._log_buffer.store
        attempt_id = self._log_buffer.attempts.get(hostname)
        if not isinstance(attempt_id, str):
            attempt_id = None
        if history and store is not None and attempt_id is not None:
            fields: dict[str, object] = {"stage": failed_step or step.value}
            if step == ProvisioningStep.COMPLETE:
                fields.update(
                    status="complete",
                    ready_at=now.isoformat(),
                    finished_at=now.isoformat(),
                )
            if error:
                failure = store.get(attempt_id).get("failure")
                fields.update(
                    status="failed",
                    failure_summary=(
                        f"{failure['failed_stage']}: {failure['original_error']}"
                        if failure
                        else f"{failed_step or step.value}: {error}"
                    ),
                )
            store.update(attempt_id, **fields)
        attempt_number: int | None = None
        retry_count = 0
        progress: dict[str, object] | None = None
        failure_class: str | None = None
        required_action: str | None = None
        if store is not None and attempt_id is not None:
            with suppress(KeyError):
                manifest = store.get(attempt_id)
                attempt_number = manifest.get("attempt_number")
                retry_count = int(manifest.get("transient_retries") or 0)
                progress = manifest.get("progress")
                failure_class = manifest.get("failure_class")
                required_action = manifest.get("next_action")
        state = ProvisioningState(
            hostname=hostname,
            attempt_id=attempt_id,
            current_step=step,
            started_at=started_at or now,
            updated_at=now,
            failed_step=failed_step,
            error=error,
            resume_state=resume_state,
            attempt_number=attempt_number,
            retry_count=retry_count,
            progress=progress,
            failure_class=failure_class,
            next_action=required_action,
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

    async def _reconcile_host(self, hostname: str, *, block: bool = True) -> bool:
        """Recover evidence and detect live remote work on the node.

        Runs under the host lifecycle lease before any new remote mutation.
        The node is the authority on its own process groups: a read-only
        ``active`` probe reports whether any attempt on this host still has a
        live or surviving phase, so a newer local attempt cannot shadow an
        older survivor and a second gateway cannot fence-bypass. Fresh or
        legacy nodes get the compatible recorder placed first so the probe is
        authoritative. Returns True when a live remote operation blocks a new
        mutation (block=True) or a live/survivor operation was found so a
        teardown can warn (block=False). Never re-runs remote setup or engine
        launch.
        """
        store = self._log_buffer.store
        if store is None or self._remote_logs is None:
            return False
        # A fresh node has no recorder yet and a legacy node lacks the
        # ``active`` action; placing the compatible recorder first makes the
        # probe authoritative. Uploading never starts a process, so fencing
        # before setup begins is preserved. Best-effort: an unreachable node
        # still blocks below through the probe itself.
        with suppress(Exception):
            await self._ensure_remote_recorder(hostname)
        probe = await self._remote_logs.host_active(hostname)
        if probe.get("unreachable"):
            history = await asyncio.to_thread(store.history, hostname, limit=1)
            if history["attempts"]:
                store.issue(
                    history["attempts"][0]["attempt_id"],
                    f"Host unreachable during reconcile: {probe.get('error', '')}",
                )
            return block
        if not probe.get("active"):
            # A command can finish while the gateway is down; recover its
            # terminal phase, exit status, and unseen records even when
            # nothing is live anymore.
            await self._recover_pending_evidence(hostname)
            return False

        status = probe.get("status")
        reason = (
            "Remote operation is still running on the node; "
            "wait for it to finish or tear the host down before retrying"
            if status in {"running", "launching"}
            else (
                "Remote process survived cancellation; it must be torn down "
                "before retrying this host"
            )
        )
        holder = probe.get("holder")
        match = None
        if isinstance(holder, str):
            history = await asyncio.to_thread(store.history, hostname)
            match = next(
                (a for a in history["attempts"] if a["attempt_id"] == holder), None
            )
        if match is not None:
            # Mirror remaining evidence for the holder before writing, and
            # re-check: a phase that completed during the probe is not a block.
            # A failed collection must retain the block; stale cached phases
            # are no proof that the probed operation finished.
            collected = True
            try:
                await self._remote_logs.collect(match["attempt_id"])
            except Exception:
                collected = False
            phases = store.get(match["attempt_id"]).get("remote_phases") or {}
            phase_id = probe.get("phase")
            probed = phases.get(phase_id) if isinstance(phase_id, str) else None
            if probed is None and phases:
                probed = next(reversed(phases.values()))
            if collected and probed is not None and probed.get("status") == "complete":
                return False
        else:
            history = await asyncio.to_thread(store.history, hostname, limit=1)
            match = history["attempts"][0] if history["attempts"] else None
        if match is None:
            # A live phase exists even if no local attempt row can be matched.
            # It still blocks a mutation and must warn a teardown.
            return True
        survivor = status == "survivor"
        existing = match.get("failure_summary") or ""
        summary = (
            existing
            if reason in existing
            else (f"{existing} (reconcile: {reason})" if existing else reason)
        )
        store.update(
            match["attempt_id"],
            status="interrupted",
            survivor=survivor,
            failure_summary=summary,
        )
        store.issue(match["attempt_id"], reason + "; conflicting retries blocked")
        # A live/survivor operation was found: block a mutation (block=True)
        # and warn a teardown that a remote process may still be running.
        return True

    async def _recover_pending_evidence(self, hostname: str) -> None:
        """Mirror remote evidence for unfinished attempts when nothing is live.

        A detached command can finish while the gateway is down; the active
        probe then reports no live phase, but the terminal phase, exit status,
        and unseen records still belong in the gateway store before the next
        mutation is allowed.
        """
        store = self._log_buffer.store
        if store is None or self._remote_logs is None:
            return
        history = await asyncio.to_thread(store.history, hostname, limit=5)
        for attempt in history["attempts"]:
            if attempt.get("status") not in {"running", "interrupted", "failed"}:
                continue
            with suppress(Exception):
                await self.collect_logs(hostname, attempt["attempt_id"])

    async def reconcile_pending_operations(self) -> None:
        """Best-effort startup recovery for hosts with unfinished remote work.

        Runs in the background so a slow or unreachable host cannot block
        startup; the per-mutation ``_reconcile_host`` hook is the safety gate.
        """
        store = self._log_buffer.store
        if store is None or self._remote_logs is None:
            return
        hostnames = await asyncio.to_thread(store.pending_hosts)
        for hostname in hostnames:
            # Hold the host lifecycle lease so a concurrent placement retry or
            # user mutation cannot race the probe's write to a new attempt.
            lease = await self.try_reserve_host(hostname)
            if lease is None:
                logger.info("startup_reconcile_skipped", hostname=hostname)
                continue
            try:
                with suppress(Exception):
                    await self._reconcile_host(hostname)
            finally:
                lease.release()
            logger.info("startup_reconcile_finished", hostname=hostname)

    async def power_state(self, hostname: str) -> str | None:
        """Current BMC power state, or ``None`` when Redfish is not configured.

        Placement reads this when its SSH probe fails: a host the BMC reports
        ``Off`` runs no earlier setup, so the failed probe does not block and
        provisioning powers it on. ``None`` keeps today's probe-first behavior.
        """
        if self._redfish_client is None:
            return None
        return await self._redfish_client.get_power_state(hostname)

    async def _power_on_if_needed(self, hostname: str) -> None:
        """Power on the host via Redfish if configured (D-01, D-06, D-07).

        Best-effort: RedfishError is caught and logged so provisioning
        can continue even if the BMC is unreachable (server may already be on).
        """
        if self._redfish_client is None:
            logger.info("redfish_not_configured", msg="skipping power check")
            return

        await self._update_state(hostname, ProvisioningStep.POWERING_ON, history=False)
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
        except (SSHConnectionError, RemoteCommandError, TimeoutError) as exc:
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

    @property
    def connection_tracking_available(self) -> bool:
        """Return whether lifecycle operations can observe active requests."""
        return self._tracker is not None

    def validate_llamacpp_relaunch(
        self,
        node: Node | None,
        request: LlamaCppRuntimeRequest,
    ) -> Node:
        """Return a relaunchable node or raise a typed boundary error."""
        if node is None:
            raise RelaunchPreconditionError("Node is not registered")
        if node.status is not NodeStatus.HEALTHY:
            raise RelaunchPreconditionError(
                f"Node is {node.status.value}; relaunch requires a healthy node"
            )
        if not node.managed or node.engine is not InferenceEngine.LLAMA_CPP:
            raise RelaunchPreconditionError(
                "Relaunch requires a managed llama_cpp node"
            )
        if node.artifact_id is None or node.llamacpp_runtime is None:
            raise RelaunchPreconditionError(
                "Relaunch requires an artifact identity and verified runtime state"
            )
        if self._tracker is None:
            raise RelaunchPreconditionError(
                "Relaunch requires active-connection tracking"
            )

        runtime = node.llamacpp_runtime
        if (
            request.sizing is LlamaCppSizingMode.PROFILE
            or node.placement is not None
            or (
                runtime.requested.sizing is LlamaCppSizingMode.PROFILE
                and not node.owner
            )
        ):
            # An admin must take ownership before overriding automation.
            # The old runtime remains intact until a verified replacement is
            # committed, so a failed override can restore the exact profile.
            raise RelaunchPreconditionError(
                "Catalog-profile nodes are relaunched by automatic placement, "
                "not with a custom sizing policy"
            )
        if request.fit_target_mib >= min(gpu.total_mib for gpu in runtime.gpus):
            raise RelaunchValidationError(
                "fit_target_mib must be smaller than every GPU's total VRAM"
            )
        if (
            request.sizing is LlamaCppSizingMode.CUSTOM
            and request.context_per_slot is not None
            and request.context_per_slot > runtime.effective.train_context
        ):
            raise RelaunchValidationError(
                "context_per_slot exceeds the model training context"
            )
        return node

    async def stored_node(self, hostname: str) -> tuple[bool, Node | None]:
        """Read *hostname*'s node key from etcd itself, not the registry.

        Returns whether the key exists and the parsed node, which is ``None``
        when the key is absent or its value cannot be parsed.
        """
        key = f"{self._etcd_client.prefix}{hostname}"
        record = await asyncio.to_thread(self._etcd_client.get_record, key)
        if record is None:
            return False, None
        return True, node_from_etcd(
            record.key,
            record.value,
            self._etcd_client.prefix,
            endpoint_policy=self._endpoint_policy,
        )

    async def _read_node_record(self, hostname: str) -> tuple[EtcdRecord, Node]:
        key = f"{self._etcd_client.prefix}{hostname}"
        record = await asyncio.to_thread(self._etcd_client.get_record, key)
        if record is None:
            raise RelaunchConflictError(
                f"Node {hostname!r} disappeared from etcd during relaunch"
            )
        node = node_from_etcd(
            record.key,
            record.value,
            self._etcd_client.prefix,
            endpoint_policy=self._endpoint_policy,
        )
        if node is None:
            raise RelaunchConflictError(
                f"Node {hostname!r} has an invalid etcd registration"
            )
        return record, node

    async def _replace_node_revision(
        self,
        record: EtcdRecord,
        expected_node: Node,
        replacement: Node,
        *,
        lease_id: int,
    ) -> EtcdRecord:
        """Replace one node revision, retrying lease-only interleavings."""
        key, value = node_to_etcd(replacement, self._etcd_client.prefix)
        current_record = record
        for _attempt in range(3):
            replacement_revision = await asyncio.to_thread(
                self._etcd_client.replace_if_revision,
                key,
                value,
                expected_mod_revision=current_record.mod_revision,
                lease_id=lease_id,
            )
            if replacement_revision is not None:
                return EtcdRecord(
                    key=key.encode("utf-8"),
                    value=value,
                    mod_revision=replacement_revision,
                    lease_id=lease_id,
                )

            current_record, current_node = await self._read_node_record(
                replacement.node_id
            )
            if current_node != expected_node:
                raise RelaunchConflictError(
                    f"Node {replacement.node_id!r} changed during relaunch"
                )

        raise RelaunchConflictError(
            f"Node {replacement.node_id!r} revision changed repeatedly during relaunch"
        )

    async def _prepare_relaunch_lease(self, record: EtcdRecord) -> tuple[int, bool]:
        """Return a fresh managed lease and whether this call granted it."""
        if record.lease_id:
            remaining = await asyncio.to_thread(
                self._etcd_client.refresh_lease,
                record.lease_id,
            )
            if remaining == -1:
                raise RelaunchConflictError("The managed node lease has expired")
            return record.lease_id, False
        lease_id = await asyncio.to_thread(self._etcd_client.grant_node_lease)
        return lease_id, True

    async def _keep_relaunch_lease(
        self,
        hostname: str,
        lease_id: int,
    ) -> None:
        """Keep a RELAUNCHING registration alive until terminal persistence."""
        interval = max(1.0, self._etcd_client.node_lease_ttl / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                remaining = await asyncio.to_thread(
                    self._etcd_client.refresh_lease,
                    lease_id,
                )
            except Exception:
                logger.warning(
                    "relaunch_lease_refresh_failed",
                    hostname=hostname,
                    lease_id=lease_id,
                    exc_info=True,
                )
                continue
            if remaining == -1:
                logger.error(
                    "relaunch_lease_expired",
                    hostname=hostname,
                    lease_id=lease_id,
                )
                return

    async def relaunch_llamacpp(
        self,
        hostname: str,
        request: LlamaCppRuntimeRequest,
        *,
        lifecycle_lease: HostLifecycleLease | None = None,
    ) -> None:
        """Transactionally relaunch one healthy managed llama.cpp node."""
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            if await self._reconcile_host(hostname):
                raise RelaunchPreconditionError(
                    "A prior provisioning operation is still active on the node; "
                    "wait for it to finish or tear the host down before retrying"
                )
            await self._relaunch_llamacpp(hostname, request)
        finally:
            lease.release()

    async def _launch_llamacpp_runtime(
        self,
        hostname: str,
        artifact: ResolvedGGUFArtifact,
        request: LlamaCppRuntimeRequest,
    ) -> tuple[str, LlamaCppRuntimeState]:
        """Launch and verify one exact llama.cpp runtime policy."""
        draft_artifact = await self._resolve_draft_artifact(request)
        gpus = await self._read_gpu_inventory(hostname, profile=request.profile)
        model_name = await self._run_start_vllm(
            hostname,
            engine=InferenceEngine.LLAMA_CPP,
            artifact=artifact,
            llamacpp_request=request,
            draft_artifact=draft_artifact,
        )
        await self._poll_health(hostname, engine=InferenceEngine.LLAMA_CPP)
        runtime = await self._verify_llamacpp_runtime(
            hostname,
            expected_request=request,
            draft_artifact=draft_artifact,
            gpus=gpus,
        )
        return model_name, runtime

    async def _stop_llamacpp(self, hostname: str) -> None:
        """Stop llama.cpp through the verified engine-specific script."""
        command = self._remote_script_command(
            hostname,
            InferenceEngine.LLAMA_CPP,
            "stop-llamacpp.sh",
        )
        await self._ssh_run_command(hostname, command)

    async def _relaunch_llamacpp(
        self,
        hostname: str,
        request: LlamaCppRuntimeRequest,
    ) -> None:
        started_at = datetime.now(UTC)
        self._begin_log(hostname, InferenceEngine.LLAMA_CPP, operation="relaunch")
        self._log(hostname, "info", "llama.cpp relaunch started")
        await self._update_state(
            hostname,
            ProvisioningStep.RELAUNCH_VALIDATING,
            started_at=started_at,
        )

        relaunch_record: EtcdRecord | None = None
        previous_node: Node | None = None
        artifact: ResolvedGGUFArtifact | None = None
        lease_id: int | None = None
        granted_lease = False
        keepalive: asyncio.Task[None] | None = None
        stop_attempted = False
        relaunch_active = False
        current_step = ProvisioningStep.RELAUNCH_VALIDATING.value

        try:
            local_node = self._registry.get(hostname) if self._registry else None
            self.validate_llamacpp_relaunch(local_node, request)
            record, persisted_node = await self._read_node_record(hostname)
            previous_node = self.validate_llamacpp_relaunch(persisted_node, request)
            if local_node is None or (
                local_node.engine,
                local_node.artifact_id,
                local_node.llamacpp_runtime,
            ) != (
                previous_node.engine,
                previous_node.artifact_id,
                previous_node.llamacpp_runtime,
            ):
                raise RelaunchConflictError(
                    "The registry and persisted node identity differ"
                )
            artifact = await self.resolve_artifact_selection(
                InferenceEngine.LLAMA_CPP,
                previous_node.artifact_id,
            )
            if artifact is None:
                raise RelaunchPreconditionError(
                    "The persisted llama.cpp artifact is unavailable"
                )

            # Refresh the gateway-owned scripts before taking the node out of
            # rotation so upload failure cannot create avoidable downtime.
            self._log(hostname, "info", "Refreshing llama.cpp lifecycle scripts")
            await self._upload_scripts(hostname, engine=InferenceEngine.LLAMA_CPP)

            lease_id, granted_lease = await self._prepare_relaunch_lease(record)
            relaunching = node_with_status(
                previous_node,
                NodeStatus.RELAUNCHING,
            )
            if self._registry is None:
                raise RelaunchPreconditionError(
                    "Relaunch requires the managed node registry"
                )
            # DRAINING stops selection immediately. Replacing it with the
            # protected lifecycle state under the same registry lock prevents
            # zero-connection drain cleanup from removing the node.
            with self._registry.locked():
                if self._registry.get(hostname) != local_node:
                    raise RelaunchConflictError(
                        "The node changed before relaunch could begin"
                    )
                self._registry.drain(hostname)
                self._registry.add(relaunching)
            try:
                relaunch_record = await self._replace_node_revision(
                    record,
                    previous_node,
                    relaunching,
                    lease_id=lease_id,
                )
                relaunch_active = True
            except Exception:
                self._registry.add(local_node)
                if granted_lease:
                    with suppress(Exception):
                        await asyncio.to_thread(
                            self._etcd_client.revoke_lease,
                            lease_id,
                        )
                raise

            keepalive = asyncio.create_task(
                self._keep_relaunch_lease(hostname, lease_id),
                name=f"relaunch-lease:{hostname}",
            )
            current_step = ProvisioningStep.DRAINING.value
            await self._update_state(
                hostname,
                ProvisioningStep.DRAINING,
                started_at=started_at,
            )
            self._log(hostname, "info", "Draining active connections")
            drain_outcome = await self._drain_wait(hostname)
            if drain_outcome is not DrainOutcome.COMPLETE:
                restored = await self._replace_node_revision(
                    relaunch_record,
                    relaunching,
                    previous_node,
                    lease_id=lease_id,
                )
                relaunch_record = restored
                relaunch_active = False
                self._registry.add(previous_node)
                if drain_outcome is DrainOutcome.TIMED_OUT:
                    raise ProvisioningError(
                        "Relaunch aborted because active requests did not drain"
                    )
                raise ProvisioningError(
                    "Relaunch aborted because connection tracking is unavailable"
                )
            self._log(hostname, "info", "Drain complete")

            current_step = ProvisioningStep.STOPPING_LLAMACPP.value
            await self._update_state(
                hostname,
                ProvisioningStep.STOPPING_LLAMACPP,
                started_at=started_at,
            )
            self._log(hostname, "info", "Stopping the current llama.cpp server")
            stop_attempted = True
            await self._stop_llamacpp(hostname)

            current_step = ProvisioningStep.STARTING_LLAMACPP.value
            await self._update_state(
                hostname,
                ProvisioningStep.STARTING_LLAMACPP,
                started_at=started_at,
            )
            self._log(hostname, "info", "Launching the requested configuration")
            model_name, runtime = await self._launch_llamacpp_runtime(
                hostname,
                artifact,
                request,
            )
            healthy = node_with_status(
                previous_node,
                NodeStatus.HEALTHY,
                model=model_name,
                llamacpp_runtime=runtime,
                last_heartbeat=datetime.now(UTC),
            )
            current_step = ProvisioningStep.REGISTERING.value
            relaunch_record = await self._replace_node_revision(
                relaunch_record,
                relaunching,
                healthy,
                lease_id=lease_id,
            )
            relaunch_active = False
            self._registry.add(healthy)
            await self._update_state(
                hostname,
                ProvisioningStep.COMPLETE,
                started_at=started_at,
            )
            self._log(hostname, "info", "llama.cpp relaunch complete")
            return
        except BaseException as error:
            diagnostic_cancel: asyncio.CancelledError | None = None
            if isinstance(error, Exception):
                try:
                    await self._capture_failure(hostname, current_step, error)
                except asyncio.CancelledError as cancelled:
                    # Diagnostics are secondary to restoring the stopped engine.
                    diagnostic_cancel = cancelled
            if (
                stop_attempted
                and relaunch_record is not None
                and previous_node is not None
                and previous_node.llamacpp_runtime is not None
                and artifact is not None
                and lease_id is not None
            ):
                current_step = ProvisioningStep.ROLLING_BACK.value
                await self._update_state(
                    hostname,
                    ProvisioningStep.ROLLING_BACK,
                    started_at=started_at,
                )
                self._log(
                    hostname,
                    "warning",
                    f"Relaunch failed; restoring the previous policy: {error}",
                )
                try:
                    model_name, runtime = await self._launch_llamacpp_runtime(
                        hostname,
                        artifact,
                        previous_node.llamacpp_runtime.requested,
                    )
                    restored_node = node_with_status(
                        previous_node,
                        NodeStatus.HEALTHY,
                        model=model_name,
                        llamacpp_runtime=runtime,
                        last_heartbeat=datetime.now(UTC),
                    )
                    relaunch_record = await self._replace_node_revision(
                        relaunch_record,
                        node_with_status(previous_node, NodeStatus.RELAUNCHING),
                        restored_node,
                        lease_id=lease_id,
                    )
                except BaseException as rollback_error:
                    failed = node_with_status(
                        previous_node,
                        NodeStatus.RELAUNCH_FAILED,
                        llamacpp_runtime=None,
                        last_heartbeat=datetime.now(UTC),
                    )
                    terminal_written = False
                    try:
                        await self._replace_node_revision(
                            relaunch_record,
                            node_with_status(previous_node, NodeStatus.RELAUNCHING),
                            failed,
                            lease_id=0,
                        )
                    except Exception:
                        logger.warning(
                            "relaunch_terminal_state_write_failed",
                            hostname=hostname,
                            exc_info=True,
                        )
                    else:
                        terminal_written = True
                        relaunch_active = False
                        if self._registry is not None:
                            self._registry.add(failed)
                    if terminal_written and lease_id:
                        with suppress(Exception):
                            await asyncio.to_thread(
                                self._etcd_client.revoke_lease,
                                lease_id,
                            )
                    combined = (
                        f"Relaunch failed: {error}; rollback failed: {rollback_error}"
                    )
                    self._log(hostname, "error", combined)
                    await self._update_state(
                        hostname,
                        ProvisioningStep.FAILED,
                        failed_step=current_step,
                        error=combined,
                        started_at=started_at,
                    )
                    raise ProvisioningError(combined) from rollback_error
                else:
                    relaunch_active = False
                    if self._registry is not None:
                        self._registry.add(restored_node)
                    if (
                        previous_node.llamacpp_runtime.requested.sizing
                        is LlamaCppSizingMode.AUTO
                    ):
                        rollback_message = "Automatic sizing restored"
                    elif (
                        previous_node.llamacpp_runtime.requested.sizing
                        is LlamaCppSizingMode.PROFILE
                    ):
                        rollback_message = "Previous catalog profile restored"
                    else:
                        rollback_message = "Previous custom sizing restored"
                    self._log(hostname, "warning", rollback_message)
                    await self._update_state(
                        hostname,
                        ProvisioningStep.FAILED,
                        failed_step=current_step,
                        error=f"{error}; {rollback_message}",
                        started_at=started_at,
                    )
                    if diagnostic_cancel is not None:
                        raise diagnostic_cancel from error
                    raise

            if (
                relaunch_active
                and not stop_attempted
                and relaunch_record is not None
                and previous_node is not None
                and lease_id is not None
            ):
                try:
                    await self._replace_node_revision(
                        relaunch_record,
                        node_with_status(previous_node, NodeStatus.RELAUNCHING),
                        previous_node,
                        lease_id=lease_id,
                    )
                except Exception:
                    logger.warning(
                        "relaunch_pre_stop_restore_failed",
                        hostname=hostname,
                        exc_info=True,
                    )
                else:
                    relaunch_active = False
                    if self._registry is not None:
                        self._registry.add(previous_node)

            self._log(hostname, "error", f"Relaunch failed: {error}")
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step=current_step,
                error=str(error),
                started_at=started_at,
            )
            if diagnostic_cancel is not None:
                raise diagnostic_cancel from error
            raise
        finally:
            if keepalive is not None:
                keepalive.cancel()
                with suppress(asyncio.CancelledError):
                    await keepalive
            await self._finish_remote_logs(
                hostname, cancel=asyncio.current_task() in self._explicit_cancel_tasks
            )
            self._log_buffer.mark_complete(hostname)

    async def register_available(
        self, hostname: str, port: int | None = None, owner: str = ""
    ) -> None:
        """Register a hostname as available in the node pool (no provisioning).

        A custom port is rejected: a plain pool node is provisioned later on the
        configured default (``provisioning.vllm_port``), so a stored custom port
        would be silently replaced at launch time. Custom ports are supported
        only for ``register_self_setup`` adoption.
        """
        if port is not None:
            raise ValueError(
                "a custom port is only supported for self-setup adoption; "
                "plain pool registration provisions on the configured default "
                "port"
            )
        endpoint = self.validate_endpoint(hostname, port)
        node = Node(
            node_id=hostname,
            endpoint=endpoint,
            status=NodeStatus.AVAILABLE,
            managed=False,
            owner=owner,
        )
        key, value = node_to_etcd(node, self._etcd_client.prefix)
        await asyncio.to_thread(self._etcd_client.put, key, value)
        if self._registry is not None:
            self._registry.add(node)

    async def register_self_setup(
        self,
        hostname: str,
        port: int | None = None,
        owner: str = "",
        admin_only: bool = False,
        name: str = "",
    ) -> Node:
        """Adopt an already-running OpenAI-compatible server into the fleet.

        Adoption is keyed on the OpenAI-compatible contract (``GET
        /v1/models``). ``/health`` is probed best-effort, but only a missing
        health endpoint (HTTP 404/405/501) is treated as optional: an explicit
        unhealthy response (e.g. ``/health`` 503) refuses adoption because the
        model list does not establish inference readiness. On success QIIP
        registers the detected model in etcd and never owns the node's
        lifecycle.

        Requirements and limits:

        - Plain HTTP only: a backend that requires credentials on
          ``/v1/models`` is not supported.
        - The hostname and port must satisfy the routing endpoint allowlist. If
          *port* is omitted the configured ``provisioning.vllm_port`` is used;
          re-adoption that omits *port* currently selects that configured
          default rather than the port stored on the previous registration.
        - Single-model contract: a self-setup node tracks exactly one model —
          the primary id the server reports first on ``/v1/models``. Aliases
          and LoRA entries are ignored by design. Re-adoption re-probes the
          live server and reconciles the stored model, so drift from an
          external restart with another primary model is refreshed on each
          re-adoption instead of silently keeping the first reported model
          forever.
        - Circuit-breaker recovery probes ``/v1/completions``, so a self-setup
          node must expose an OpenAI-compatible completions endpoint to recover
          after its breaker opens.
        """
        endpoint = self.validate_endpoint(hostname, port)
        model = await self._discover_running_model(endpoint, hostname)
        prior = self._registry.get(hostname) if self._registry is not None else None
        if prior is not None and prior.self_setup and prior.model != model:
            logger.warning(
                "self_setup_model_drift_reconciled",
                hostname=hostname,
                previous_model=prior.model,
                model=model,
            )
        node = Node(
            node_id=hostname,
            name=name,
            endpoint=endpoint,
            status=NodeStatus.HEALTHY,
            model=model,
            engine=InferenceEngine.VLLM,
            last_heartbeat=datetime.now(UTC),
            managed=False,
            self_setup=True,
            admin_only=admin_only,
            owner=owner,
        )
        key, value = node_to_etcd(node, self._etcd_client.prefix)
        await asyncio.to_thread(self._etcd_client.put, key, value)
        if self._registry is not None:
            self._registry.add(node)
        logger.info(
            "self_setup_node_registered",
            hostname=hostname,
            model=model,
            key=key,
        )
        return node

    async def update_node_owner(self, hostname: str, owner: str) -> Node:
        """Set the owner of a registered node, preserving its etcd lease.

        The change takes the host lifecycle lease, like setup, relaunch and
        teardown. Those operations decide what to do with a node record and
        then act on it later, and a provision writes its final record from its
        own arguments, so an owner written in between would be erased. While
        one of them holds the host this raises ``ProvisioningError`` at once
        instead of queueing behind a provision that can take many minutes.

        The write runs in its own task, which releases the lease when it ends.
        Cancelling the caller (a client that disconnects) therefore cannot
        free the host while the etcd write may still land.
        """
        lease = await self._lifecycle.try_acquire(hostname)
        if lease is None:
            raise ProvisioningError(
                f"Node {hostname!r} has a lifecycle operation in progress "
                "(setup, relaunch, teardown or automatic placement); change "
                "its owner when that finishes"
            )
        task = asyncio.create_task(self._write_node_owner(hostname, owner))

        def finished(done: asyncio.Task[Node]) -> None:
            lease.release()
            self._owner_updates.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieved: an abandoned caller never reads it.

        self._owner_updates.add(task)
        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def _write_node_owner(self, hostname: str, owner: str) -> Node:
        """Revision-checked owner write. The caller holds the lifecycle lease.

        The compare-and-swap keeps a concurrent status or liveness write, which
        does not take the lease, from being clobbered; it retries a bounded
        number of times, then raises ``ProvisioningError``.
        """
        key = f"{self._etcd_client.prefix}{hostname}"
        record = await asyncio.to_thread(self._etcd_client.get_record, key)
        if record is None:
            raise KeyError(hostname)
        for _attempt in range(3):
            node = node_from_etcd(
                record.key,
                record.value,
                self._etcd_client.prefix,
                endpoint_policy=self._endpoint_policy,
            )
            if node is None:
                raise ProvisioningError(
                    f"Node {hostname!r} has an invalid etcd registration"
                )
            # Assigning an owner is a person taking the node over: it stops
            # being an automatic placement, so automation never retries,
            # clears or counts it again.
            replacement = node.model_copy(
                update={
                    "owner": owner,
                    "placement": None if owner else node.placement,
                }
            )
            _, value = node_to_etcd(replacement, self._etcd_client.prefix)
            new_revision = await asyncio.to_thread(
                self._etcd_client.replace_if_revision,
                key,
                value,
                expected_mod_revision=record.mod_revision,
                lease_id=record.lease_id,
            )
            if new_revision is not None:
                if self._registry is not None:
                    self._registry.add(replacement)
                logger.info("node_owner_updated", hostname=hostname, owner=owner)
                return replacement
            fresh = await asyncio.to_thread(self._etcd_client.get_record, key)
            if fresh is None:
                raise ProvisioningError(
                    f"Node {hostname!r} disappeared during owner update"
                )
            record = fresh
        raise ProvisioningError(f"Node {hostname!r} kept changing during owner update")

    async def _discover_running_model(self, endpoint: str, hostname: str) -> str:
        """Probe an OpenAI-compatible server and return its served model id.

        Adoption is based on the OpenAI-compatible contract (``GET
        /v1/models``). ``/health`` is probed best-effort, but only a missing
        health endpoint (HTTP 404/405/501) is optional — many OpenAI-compatible
        servers do not expose it. An authoritative unhealthy response (e.g.
        ``/health`` 503) blocks adoption because the model list does not
        establish inference readiness.
        """
        health_url = build_backend_url(endpoint, "/health")
        models_url = build_backend_url(endpoint, "/v1/models")
        timeout = httpx.Timeout(_SELF_SETUP_PROBE_TIMEOUT)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # /health is optional only when the endpoint is missing.
                try:
                    health = await client.get(health_url)
                    if health.status_code == 200:
                        pass
                    elif health.status_code in _OPTIONAL_HEALTH_STATUSES:
                        logger.debug(
                            "self_setup_health_missing",
                            hostname=hostname,
                            status_code=health.status_code,
                        )
                    else:
                        raise SelfSetupError(
                            f"health check on {hostname} returned HTTP "
                            f"{health.status_code}; refusing to adopt an "
                            "explicitly unhealthy server"
                        )
                except SelfSetupError:
                    raise
                except httpx.HTTPError as exc:
                    logger.debug(
                        "self_setup_health_probe_failed",
                        hostname=hostname,
                        error=str(exc),
                    )
                models = await client.get(models_url)
                if models.status_code != 200:
                    raise SelfSetupError(
                        f"OpenAI-compatible model discovery failed on {hostname}: "
                        f"HTTP {models.status_code}"
                    )
                try:
                    payload: object = models.json()
                except ValueError as exc:
                    raise SelfSetupError(
                        f"model list on {hostname} was not valid JSON"
                    ) from exc
        except SelfSetupError:
            raise
        except httpx.TimeoutException as exc:
            raise SelfSetupError(f"model discovery timed out on {hostname}") from exc
        except httpx.HTTPError as exc:
            raise SelfSetupError(f"{hostname} is not reachable") from exc

        model = served_model_id(payload)
        if model is None:
            raise SelfSetupError(f"{hostname} responded but reported no models")
        return model

    async def remove_available(self, hostname: str) -> None:
        """Remove a manually registered node from the fleet."""
        await asyncio.to_thread(
            self._etcd_client.delete, f"{self._etcd_client.prefix}{hostname}"
        )
        if self._registry is not None:
            self._registry.remove(hostname)
        if self._tracker is not None:
            self._tracker.remove(hostname)
        if self._cb_registry is not None:
            self._cb_registry.remove(hostname)

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
        llamacpp_request: LlamaCppRuntimeRequest | None = None,
        vllm_params: VllmParams | None = None,
        lifecycle_lease: HostLifecycleLease | None = None,
        owner: str = "",
        placement: NodePlacement | None = None,
    ) -> None:
        """Provision *hostname* under the shared host lifecycle coordinator."""
        # Validate before acquiring the lifecycle lease or touching the host.
        # The API also calls this synchronously so configuration errors become
        # immediate 400 responses instead of failed background operations.
        self.validate_setup_configuration(engine)
        self.validate_endpoint(hostname)
        artifact = await self.resolve_artifact_selection(engine, artifact_id)
        if engine == InferenceEngine.LLAMA_CPP:
            llamacpp_request = (
                llamacpp_request or self._default_llamacpp_runtime_request()
            )
        elif llamacpp_request is not None:
            raise ProvisioningError(
                "llama.cpp sizing policy is only valid for llama_cpp provisioning"
            )
        draft_artifact = await self._resolve_draft_artifact(llamacpp_request)
        if placement is not None and (
            not managed
            or owner
            or llamacpp_request is None
            or llamacpp_request.profile is None
            or (placement.profile_id, placement.profile_version)
            != (
                llamacpp_request.profile.profile_id,
                llamacpp_request.profile.profile_version,
            )
        ):
            raise ProvisioningError(
                "automatic placement provisions one managed, unowned catalog profile"
            )
        lease = lifecycle_lease
        if lease is None:
            lease = await self._lifecycle.acquire(hostname)
        elif not lease.belongs_to(self._lifecycle, hostname):
            raise ValueError("lifecycle lease does not own this host")

        try:
            if llamacpp_request is None:
                await self._provision(
                    hostname,
                    managed=managed,
                    model=model,
                    engine=engine,
                    artifact=artifact,
                    vllm_params=vllm_params,
                    owner=owner,
                )
            else:
                await self._provision(
                    hostname,
                    managed=managed,
                    model=model,
                    engine=engine,
                    artifact=artifact,
                    llamacpp_request=llamacpp_request,
                    owner=owner,
                    draft_artifact=draft_artifact,
                    placement=placement,
                )
        except asyncio.CancelledError:
            record = self._provisioning_tasks.get(hostname)
            if record is None or record.attempt_id is None:
                # Cancelled before the attempt was opened. The buffer may
                # still track a row: close it only when it is a stale running
                # row so the shutdown is visible; a finished row belongs to a
                # previous attempt and must survive.
                explicit = asyncio.current_task() in self._explicit_cancel_tasks
                message = (
                    "Provisioning cancelled by teardown"
                    if explicit
                    else "Gateway collection interrupted; remote command left running"
                )
                store = self._log_buffer.store
                attempt_id = self._log_buffer.attempts.get(hostname)
                current = (
                    store.get(attempt_id)
                    if store is not None and isinstance(attempt_id, str)
                    else {}
                )
                if current.get("status") == "running":
                    self._log(hostname, "error", message)
                    await self._update_state(
                        hostname,
                        ProvisioningStep.FAILED,
                        failed_step="cancelled" if explicit else "interrupted",
                        error=message,
                    )
                    self._mark_log_complete(hostname)
                raise
            store = self._log_buffer.store
            attempt_id = record.attempt_id
            attempt = store.get(attempt_id) if store and attempt_id else {}
            if (
                attempt.get("status") == "complete"
                and attempt.get("stage") == "complete"
            ):
                # Cancellation during final log retrieval cannot undo registration.
                self._mark_log_complete(hostname)
                raise
            if attempt.get("status") == "failed":
                # A failure recorded before cancellation must survive; never
                # rewrite it to the interrupted-collection state.
                self._mark_log_complete(hostname)
                raise
            explicit = asyncio.current_task() in self._explicit_cancel_tasks
            message = (
                "Provisioning cancelled by teardown"
                if explicit
                else "Gateway collection interrupted; remote command left running"
            )
            self._log(hostname, "error", message)
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="cancelled" if explicit else "interrupted",
                error=message,
            )
            self._mark_log_complete(hostname)
            raise
        finally:
            lease.release()

    async def _resolve_draft_artifact(
        self, request: LlamaCppRuntimeRequest | None
    ) -> ResolvedGGUFArtifact | None:
        """Resolve a profile's draft GGUF through the same artifact index."""
        if (
            request is None
            or request.profile is None
            or request.profile.draft_artifact_id is None
        ):
            return None
        return await self.resolve_artifact_selection(
            InferenceEngine.LLAMA_CPP, request.profile.draft_artifact_id
        )

    async def _provision(
        self,
        hostname: str,
        *,
        managed: bool = True,
        model: str | None = None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        artifact: ResolvedGGUFArtifact | None = None,
        llamacpp_request: LlamaCppRuntimeRequest | None = None,
        vllm_params: VllmParams | None = None,
        owner: str = "",
        draft_artifact: ResolvedGGUFArtifact | None = None,
        placement: NodePlacement | None = None,
    ) -> None:
        """Run full provisioning sequence on *hostname*.

        Sequence: preflight -> register PROVISIONING -> setup.sh ->
        start-vllm.sh -> health poll -> register HEALTHY.
        Tracks state in etcd at each step (D-05 through D-11).
        """
        provision_started_at = datetime.now(UTC)
        logger.info(
            "provisioning_start",
            hostname=hostname,
            engine=engine,
            vllm_params=vllm_params.model_dump() if vllm_params else None,
        )
        # Reachability first: a powered-off host must reach its BMC power-on
        # step before the reconcile gate can judge it as unreachable.
        await self._power_on_if_needed(hostname)
        if await self._reconcile_host(hostname):
            raise ProvisioningError(
                "A prior provisioning operation is still active on the node; "
                "wait for it to finish or tear the host down before retrying"
            )
        plan_inputs = self._setup_plan_inputs(
            engine,
            artifact=artifact,
            llamacpp_request=llamacpp_request,
            vllm_params=vllm_params,
        )
        plan_id = setup_plan_id(plan_inputs)
        self._begin_log(
            hostname,
            engine,
            model=artifact.model_alias if artifact else model,
            plan_id=plan_id,
            setup_inputs=plan_inputs,
        )
        record = self._provisioning_tasks.get(hostname)
        if record is not None:
            record.attempt_id = self._log_buffer.attempts.get(hostname)
        self._log(hostname, "info", "Provisioning started")

        await self._update_state(
            hostname, ProvisioningStep.PENDING, started_at=provision_started_at
        )
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
            await self._capture_failure(hostname, "preflight", exc)
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
                placement=placement,
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

            current_step = "setup"
            await self._run_setup_with_retries(
                hostname,
                started_at=provision_started_at,
                on_step=set_current_step,
                engine=engine,
                plan_id=plan_id,
            )
            current_step = "gpu_verify"
            await self._verify_gpu(hostname)
            gpus = await self._read_gpu_inventory(
                hostname,
                profile=(
                    llamacpp_request.profile if llamacpp_request is not None else None
                ),
            )
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
            llamacpp_runtime: LlamaCppRuntimeState | None = None
            vllm_runtime: VllmRuntimeState | None = None
            if engine == InferenceEngine.LLAMA_CPP:
                model_name = await self._run_start_vllm(
                    hostname,
                    model=model,
                    engine=engine,
                    artifact=artifact,
                    llamacpp_request=llamacpp_request,
                    vllm_params=vllm_params,
                    draft_artifact=draft_artifact,
                )
                current_step = "health_poll"
                await self._update_state(
                    hostname,
                    ProvisioningStep.HEALTH_POLL,
                    started_at=provision_started_at,
                )
                self._log(hostname, "info", "Waiting for health endpoint")
                await self._poll_health(hostname, engine=engine)
                current_step = "llamacpp_runtime_verify"
                llamacpp_runtime = await self._verify_llamacpp_runtime(
                    hostname,
                    expected_request=(
                        llamacpp_request or self._default_llamacpp_runtime_request()
                    ),
                    draft_artifact=draft_artifact,
                    gpus=gpus,
                )
            else:
                model_name, vllm_runtime = await self._start_vllm_with_retries(
                    hostname,
                    model=model,
                    vllm_params=vllm_params,
                    started_at=provision_started_at,
                    on_step=set_current_step,
                )
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
                llamacpp_runtime=llamacpp_runtime,
                vllm_runtime=vllm_runtime,
                owner=owner,
                gpus=gpus,
                placement=placement,
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
                resume_state=(
                    exc.resume_state
                    if isinstance(exc, SetupResumeRequiredError)
                    else None
                ),
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
                    placement=placement,
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
            await self._capture_failure(hostname, current_step, exc)
            raise
        finally:
            await self._finish_remote_logs(
                hostname, cancel=asyncio.current_task() in self._explicit_cancel_tasks
            )
            self._mark_log_complete(hostname)

        logger.info("provisioning_complete", hostname=hostname)

    async def _ensure_remote_recorder(self, hostname: str) -> None:
        """Idempotently place the node recorder so the active probe is valid.

        Fresh nodes have no ``common/provision-logs.py`` and nodes provisioned
        by an older gateway lack the ``active`` action; uploading first lets
        the reconcile gate distinguish "no evidence" from "unreachable".
        """
        files = bundle_files(None, self._common_scripts_dir())
        bundle = await self._publish_bundle(hostname, files)
        if self._remote_logs is not None:
            self._remote_logs.recorders[hostname] = f"{bundle}/common/provision-logs.py"

    async def _publish_bundle(self, hostname: str, files: dict[str, bytes]) -> str:
        # Unique upload paths leave partial transfers inactive and harmless.
        # Snapshot bytes before the first await so a gateway checkout change
        # cannot silently combine files from different versions.
        with tempfile.TemporaryDirectory(prefix="qiip-bundle-") as temporary:
            name = ".upload-" + uuid.uuid4().hex
            staged = Path(temporary) / name
            staged.mkdir()
            identity = write_bundle(staged, files)
            remote_staged = f".qiip/bundles/{name}"
            final = f".qiip/bundles/{identity}"
            await self._ssh_client.run(hostname, "mkdir -p .qiip/bundles")
            await self._ssh_client.upload(hostname, staged, ".qiip/bundles/")
            await self._ssh_client.run(
                hostname,
                shlex.join(
                    (
                        "python3",
                        f"{remote_staged}/common/generations.py",
                        "publish-bundle",
                        remote_staged,
                        final,
                        identity,
                    )
                ),
            )
        return final

    async def _upload_scripts(
        self, hostname: str, engine: InferenceEngine = InferenceEngine.VLLM
    ) -> None:
        """Copy provisioning scripts to the remote host via SCP."""
        scripts_dir, common_dir = self._required_script_bundles(engine)
        bundle = await self._publish_bundle(
            hostname, bundle_files(scripts_dir, common_dir)
        )
        self._remote_bundles[hostname, engine] = bundle
        if self._remote_logs is not None:
            self._remote_logs.recorders[hostname] = f"{bundle}/common/provision-logs.py"
        if self._log_buffer.store is not None and hostname in self._log_buffer.attempts:
            self._log_buffer.store.update(
                self._log_buffer.attempts[hostname],
                staged_bundle=bundle.rsplit("/", 1)[1],
                bundle_version="sha256:" + bundle.rsplit("/", 1)[1],
                recorder_path=f"{bundle}/common/provision-logs.py",
            )

    def _setup_resume_env(self, hostname: str, plan_id: str | None) -> dict[str, str]:
        """Return plan identity and verified stages for setup.sh.

        A compatible plan reuses only stages the gateway has evidence for; a
        changed plan id yields no resume stages, so setup.sh re-verifies and
        every stale checkpoint is invalidated on the node.
        """
        env: dict[str, str] = {}
        if plan_id:
            env["QIIP_PLAN_ID"] = plan_id
        store = self._log_buffer.store
        if store is not None and plan_id:
            with suppress(Exception):
                verified = store.resumable_checkpoints(hostname, plan_id)
                env["QIIP_RESUME_STAGES"] = ",".join(sorted(verified))
        return env

    def _record_step_checkpoint(
        self, hostname: str, step_name: str, status: str, line: str
    ) -> None:
        store = self._log_buffer.store
        attempt_id = self._log_buffer.attempts.get(hostname)
        if store is None or attempt_id is None:
            return
        status_value = {
            "START": "running",
            "OK": "completed",
            "SKIP": "skipped",
            "FAIL": "failed",
        }.get(status)
        if status_value is None:
            return
        with suppress(KeyError):
            store.record_checkpoint(
                attempt_id, step_name, status_value, evidence=line[:512]
            )

    async def _run_setup_with_retries(
        self,
        hostname: str,
        *,
        started_at: datetime,
        on_step: Callable[[str], None],
        engine: InferenceEngine = InferenceEngine.VLLM,
        plan_id: str | None = None,
    ) -> None:
        """Run setup.sh, retrying only recognized transient failures.

        Bounded exponential backoff with jitter. Reconcile runs before each
        retry so a still-live remote phase is never duplicated. Completed
        stages are checkpointed by :meth:`_run_setup`, so a retry resumes
        instead of repeating verified package/compile work.
        """
        policy = RetryPolicy(
            max_attempts=self._settings.stage_retry_max_attempts,
            base_delay_seconds=self._settings.stage_retry_base_delay,
            max_delay_seconds=self._settings.stage_retry_max_delay,
            jitter=self._settings.stage_retry_jitter,
        )
        store = self._log_buffer.store
        attempt_id = self._log_buffer.attempts.get(hostname)
        step_name = "setup"
        captured_failure = False
        timeout = self._setup_engine_timeout(engine)

        def track_step(step: str) -> None:
            nonlocal step_name
            step_name = step
            on_step(step)

        for attempt in range(1, policy.max_attempts + 1):
            try:
                await self._run_setup_once(
                    hostname,
                    started_at=started_at,
                    on_step=track_step,
                    engine=engine,
                    plan_id=plan_id,
                    timeout=timeout,
                )
                # A recovered transient failure is not a failure: drop the
                # provisional record while keeping the retry history.
                if captured_failure and store is not None and attempt_id is not None:
                    with suppress(KeyError):
                        store.update(
                            attempt_id,
                            failure=None,
                            failure_summary=None,
                            diagnostics={"sources": {}},
                            failure_class=None,
                            next_action=None,
                        )
                return
            except Exception as exc:
                failure_class = classify_failure(exc)
                if (
                    failure_class != FailureClass.TRANSIENT
                    or attempt >= policy.max_attempts
                ):
                    # Terminal for this attempt chain. Record the classification
                    # and the operator action the *setup* failure implies; a
                    # later launch/health failure is classified elsewhere.
                    self._record_failure_classification(hostname, exc)
                    raise
                delay = policy.delay(attempt)
                # Reconcile before retry: never duplicate a live remote phase.
                if await self._reconcile_host(hostname):
                    raise
                if not captured_failure:
                    # Capture the original error before the retry re-launches
                    # the recorder and replaces the active phase; otherwise the
                    # failure points at a phase that never ran and loses its
                    # command evidence.
                    await self._capture_failure(hostname, step_name, exc)
                    captured_failure = True
                self._record_failure_classification(hostname, exc)
                if store is not None and attempt_id is not None:
                    with suppress(KeyError):
                        retries = store.increment_transient_retry(attempt_id)
                        store.issue(
                            attempt_id,
                            f"Transient setup failure ({failure_class}); "
                            f"automatic retry {retries} in {delay:.0f}s: {exc}",
                            source="gateway",
                        )
                self._log(
                    hostname,
                    "warning",
                    f"Transient setup failure; retry {attempt}/"
                    f"{policy.max_retries} in {delay:.0f}s",
                )
                await asyncio.sleep(delay)

    async def _run_setup_once(
        self,
        hostname: str,
        *,
        started_at: datetime,
        on_step: Callable[[str], None],
        engine: InferenceEngine,
        plan_id: str | None,
        timeout: float,
    ) -> None:
        """Run one setup attempt under its stage-specific total deadline.

        ``asyncio.timeout`` raises ``TimeoutError`` on expiry, which is also
        the type of an ordinary socket timeout. ``Timeout.expired()``
        distinguishes the two: the run's own deadline is a terminal
        :class:`SetupDeadlineError`, while a socket timeout stays
        transient and eligible for the bounded retry.
        """
        try:
            async with asyncio.timeout(timeout) as deadline:
                await self._run_setup(
                    hostname,
                    started_at=started_at,
                    on_step=on_step,
                    engine=engine,
                    plan_id=plan_id,
                )
        except TimeoutError:
            if deadline.expired():
                raise SetupDeadlineError(
                    f"setup exceeded its {timeout:.0f}s stage deadline"
                ) from None
            raise

    def _setup_engine_timeout(self, engine: InferenceEngine) -> float:
        """Return the stage-specific total deadline for one setup run."""
        return (
            self._settings.llamacpp_setup_timeout
            if engine == InferenceEngine.LLAMA_CPP
            else self._settings.vllm_setup_timeout
        )

    async def _run_setup(
        self,
        hostname: str,
        *,
        started_at: datetime,
        on_step: Callable[[str], None],
        engine: InferenceEngine = InferenceEngine.VLLM,
        plan_id: str | None = None,
    ) -> None:
        """Run setup.sh and parse step markers from stdout (D-05, D-06)."""
        env = self._setup_script_env(engine)
        env.update(self._setup_resume_env(hostname, plan_id))
        command = self._remote_script_command(
            hostname,
            engine,
            "setup.sh",
            env=env,
        )
        # The node must bound the same run the gateway bounds: pass the
        # engine's stage deadline on both the durable-recorder and direct-SSH
        # paths so a stuck command cannot outlive the gateway's wait.
        timeout = self._setup_engine_timeout(engine)
        output: AsyncIterator[tuple[str, str]]
        if self._remote_logs is not None:
            output = self._remote_logs.run(
                hostname,
                command,
                stage="setup",
                timeout=timeout,
            )
        else:
            output = self._ssh_client.run_streaming(
                hostname,
                command,
                total_timeout=timeout,
            )
        resume: SetupResumeRequiredError | None = None
        try:
            async for stream, line in output:
                resume_match = RESUME_PATTERN.search(line)
                if resume_match:
                    resume = SetupResumeRequiredError(
                        ProvisioningResumeState(resume_match.group(1)),
                        resume_match.group(2),
                    )
                if stream == "stdout":
                    self._record_generation(hostname, line)
                    match = STEP_PATTERN.search(line)
                    if match:
                        step_name, status = match.group(1), match.group(2)
                        on_step(step_name)
                        self._record_step_checkpoint(hostname, step_name, status, line)
                        if status in {"START", "WARN"}:
                            with suppress(ValueError):
                                await self._update_state(
                                    hostname,
                                    ProvisioningStep(step_name),
                                    started_at=started_at,
                                )
                        if status == "FAIL":
                            logger.error(
                                "step_failed", step=step_name, hostname=hostname
                            )
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
        except Exception as exc:
            if resume is not None:
                raise resume from exc
            raise
        if resume is not None:
            raise resume

    async def _verify_gpu(self, hostname: str) -> None:
        """Verify GPUs are visible after setup.sh installs the NVIDIA driver."""
        gpu_output = await self._ssh_run_command(
            hostname, "nvidia-smi --query-gpu=name --format=csv,noheader"
        )
        gpu_lines = [ln for ln in gpu_output.strip().splitlines() if ln.strip()]
        if not gpu_lines:
            raise ProvisioningError(f"No GPUs detected on {hostname} after setup")
        self._log(hostname, "info", f"Detected {len(gpu_lines)} GPU(s)")

    async def _read_gpu_inventory(
        self, hostname: str, *, profile: LlamaCppProfileRuntime | None
    ) -> tuple[NodeGPU, ...]:
        """Read each GPU's identity from the node itself.

        QUADS reports a product string and a processor count, not which GPUs a
        booted host exposes. A catalog profile must find exactly the GPU it
        was planned for: one device, the product's ``nvidia-smi`` name, and at
        least the product's total memory. For every other setup the inventory
        is recorded when available and never blocks the setup.
        """
        required = profile is not None
        try:
            output = await self._ssh_run_command(
                hostname,
                "nvidia-smi --query-gpu=index,uuid,name,memory.total "
                "--format=csv,noheader,nounits",
            )
            gpus: list[NodeGPU] = []
            for line in output.splitlines():
                if not line.strip():
                    continue
                index, uuid, name, total = (part.strip() for part in line.split(","))
                gpus.append(
                    NodeGPU(
                        index=int(index), uuid=uuid, name=name, total_mib=int(total)
                    )
                )
            if not gpus:
                raise ValueError("nvidia-smi returned no GPUs")
        except Exception as exc:
            if required:
                raise ProvisioningError(
                    f"could not read the GPU inventory of {hostname}: {exc}"
                ) from exc
            self._log(hostname, "warning", f"GPU inventory was not recorded: {exc}")
            return ()
        if profile is not None:
            if len(gpus) != 1:
                raise ProvisioningError(
                    "catalog profiles support exactly one GPU per host; "
                    f"{hostname} has {len(gpus)}"
                )
            gpu = gpus[0]
            if gpu.name != profile.gpu_name:
                raise ProvisioningError(
                    f"profile {profile.profile_id} was planned for a "
                    f"{profile.gpu_name!r}, but {hostname} has a {gpu.name!r}"
                )
            if gpu.total_mib < profile.gpu_min_total_mib:
                raise ProvisioningError(
                    f"profile {profile.profile_id} needs a GPU with at least "
                    f"{profile.gpu_min_total_mib} MiB, but {hostname} reports "
                    f"{gpu.total_mib} MiB"
                )
        return tuple(sorted(gpus, key=lambda gpu: gpu.index))

    async def remote_lifecycle_processes(self, hostname: str) -> list[str]:
        """Return qiip setup/start commands still running on *hostname*.

        Cancelling a gateway task does not always stop its remote command.
        Anything that is about to start provisioning on a host it did not just
        provision itself asks the host first. Raises when the host cannot be
        asked: "unknown" must never be read as "nothing is running".
        """
        output = await self._ssh_run_command(hostname, "ps -eo pid=,args=")
        return [
            line.strip()
            for line in output.splitlines()
            if _REMOTE_LIFECYCLE_PATTERN.search(line)
        ]

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

    async def _stop_failed_vllm_start(self, hostname: str) -> bool:
        """Best-effort cleanup after a vLLM start or OOM retry fails.

        Returns ``False`` when the stop command failed, so an OOM retry can
        refuse to re-plan against a host that still has an engine holding its
        VRAM. The stop runs through the uploaded bundle/generation path so a
        freshly provisioned host without the legacy script directory is still
        stopped.
        """
        command = self._remote_script_command(
            hostname, InferenceEngine.VLLM, "stop-vllm.sh"
        )
        try:
            await self._ssh_run_command(hostname, command)
        except Exception as exc:
            logger.warning(
                "vllm_cleanup_failed",
                hostname=hostname,
                error=str(exc),
                exc_info=True,
            )
            self._log(hostname, "warning", f"vLLM stop script failed: {exc}")
            return False
        return True

    async def _read_vllm_device_memory(
        self, hostname: str
    ) -> tuple[VllmDeviceMemory, ...]:
        """Read selected-device total/free memory and compute capability.

        Returns an empty tuple, never a partial reading, when the node cannot
        answer. The caller then falls back to the node-side runtime heuristics
        rather than inventing memory numbers.
        """
        try:
            output = await self._ssh_run_command(
                hostname,
                "nvidia-smi --query-gpu=index,memory.total,memory.free,compute_cap "
                "--format=csv,noheader,nounits",
            )
        except Exception as exc:
            self._log(hostname, "warning", f"vLLM memory planning skipped: {exc}")
            return ()
        devices: list[VllmDeviceMemory] = []
        try:
            for line in output.splitlines():
                if not line.strip():
                    continue
                index, total, free, capability = (
                    part.strip() for part in line.split(",")
                )
                devices.append(
                    VllmDeviceMemory(
                        index=int(index),
                        total_mib=int(total),
                        free_mib=int(free),
                        compute_capability=capability,
                    )
                )
        except ValueError as exc:
            self._log(
                hostname,
                "warning",
                f"vLLM memory planning skipped: unreadable GPU memory ({exc})",
            )
            return ()
        if not devices:
            return ()
        return tuple(sorted(devices, key=lambda device: device.index))

    def _select_vllm_devices(
        self, devices: tuple[VllmDeviceMemory, ...], vllm_params: VllmParams | None
    ) -> tuple[VllmDeviceMemory, ...]:
        """Honor an explicit device subset and fail on a stale index."""
        requested = vllm_params.gpu_devices if vllm_params is not None else None
        if not requested:
            return devices
        by_index = {device.index: device for device in devices}
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise ProvisioningError(
                f"requested GPU devices {missing} are not present on the host"
            )
        return tuple(by_index[index] for index in requested)

    @staticmethod
    def _vllm_requested(model: str | None, params: VllmParams | None) -> VllmRequested:
        if params is None:
            return VllmRequested(model=model)
        return VllmRequested(
            model=model,
            tensor_parallel_size=params.tensor_parallel_size,
            max_model_len=params.max_model_len,
            max_num_batched_tokens=params.max_num_batched_tokens,
            gpu_memory_utilization=params.gpu_memory_utilization,
            dtype=params.dtype,
            gpu_devices=params.gpu_devices,
        )

    async def _plan_vllm_launch(
        self,
        hostname: str,
        *,
        model: str | None,
        params: VllmParams | None,
        attempt: int,
    ) -> tuple[VllmFitPlan, tuple[VllmDeviceMemory, ...]] | None:
        """Plan one attempt, or return None when the host cannot be measured."""
        devices = await self._read_vllm_device_memory(hostname)
        if not devices:
            return None
        selected = self._select_vllm_devices(devices, params)
        first = selected[0]
        if first.compute_capability is None:
            self._log(
                hostname,
                "warning",
                "vLLM memory planning skipped: the node reported no compute capability",
            )
            return None
        try:
            bucket = vllm_profile_bucket(first.compute_capability, first.total_mib)
        except ValueError as exc:
            self._log(hostname, "warning", f"vLLM memory planning skipped: {exc}")
            return None
        try:
            plan = await asyncio.to_thread(
                plan_vllm_launch,
                profile_bucket=bucket,
                devices=selected,
                requested=self._vllm_requested(model, params),
                cache_dir=self._hf_cache_dir,
                attempt=attempt,
            )
        except (ValueError, OverflowError) as exc:
            raise ProvisioningError(
                f"could not plan the vLLM launch for {hostname}: {exc}"
            ) from exc
        return plan, selected

    @staticmethod
    def _require_fitting_plan(hostname: str, plan: VllmFitPlan) -> None:
        if not plan.fitting:
            raise ProvisioningError(
                f"planned vLLM launch does not fit {hostname}: {plan.reason}"
            )

    @staticmethod
    def _plan_vllm_effective(
        vllm_params: VllmParams | None,
        plan: VllmFitPlan,
        selected: tuple[VllmDeviceMemory, ...],
    ) -> VllmParams:
        """Translate an authorized plan into the env the start script reads."""
        params = vllm_params or VllmParams()
        return VllmParams(
            tensor_parallel_size=plan.effective.tensor_parallel_size,
            max_model_len=plan.effective.max_model_len,
            gpu_memory_utilization=plan.effective.gpu_memory_utilization,
            max_num_batched_tokens=plan.effective.max_num_batched_tokens,
            tool_call_parser=params.tool_call_parser,
            reasoning_parser=params.reasoning_parser,
            dtype=plan.effective.dtype,
            gpu_devices=tuple(
                device.index
                for device in selected[: plan.effective.tensor_parallel_size]
            ),
        )

    async def _vllm_oom_in_engine_log(self, hostname: str) -> bool:
        """Return whether retained output shows a recognized memory failure.

        The engine log holds vLLM's own refusal or CUDA OOM. The node-side
        preflight bails before vLLM writes anything, so its lines only reach the
        retained provisioning output, which is checked as a fallback.
        """
        try:
            log_text = await self._ssh_run_command(
                hostname,
                "cat -- "
                + shlex.quote(self._engine_log_path(hostname, InferenceEngine.VLLM)),
            )
        except Exception as exc:
            logger.warning(
                "vllm_oom_probe_failed",
                hostname=hostname,
                error=str(exc),
            )
            log_text = ""
        if log_text and is_memory_failure(log_text):
            return True
        return any(
            is_memory_failure(entry["msg"])
            for entry in self._log_buffer.get_entries(hostname)
        )

    async def _vllm_maybe_retry(
        self,
        hostname: str,
        *,
        model: str | None,
        params: VllmParams | None,
        plan: VllmFitPlan | None,
        attempt: int,
        max_attempts: int,
        failure: BaseException | None = None,
    ) -> tuple[int, VllmFitPlan, tuple[VllmDeviceMemory, ...]] | None:
        """Decide, stop cleanly, and re-plan the next bounded adjustment.

        Returns ``None`` when the failure is not a recognized memory failure,
        every adjustable field is explicit, the retry budget is exhausted, or
        the incumbent engine could not be stopped. The caller must then stop
        the engine and re-raise.
        """
        if plan is None or not plan.retry_available or attempt + 1 >= max_attempts:
            return None
        if not self._vllm_failure_is_memory(failure) and not (
            await self._vllm_oom_in_engine_log(hostname)
        ):
            return None
        self._log(
            hostname,
            "warning",
            "vLLM ran out of memory during attempt "
            f"{attempt + 1}; stopping it cleanly before the bounded retry",
        )
        stopped = await self._stop_failed_vllm_start(hostname)
        if not stopped:
            # Re-planning against a host whose retained VRAM the incumbent
            # still owns would plan from a lie. Surface the original failure.
            self._log(
                hostname,
                "warning",
                "vLLM retry skipped because the previous engine could not be "
                "stopped cleanly",
            )
            return None
        planned = await self._plan_vllm_launch(
            hostname, model=model, params=params, attempt=attempt + 1
        )
        if planned is None:
            return None
        next_plan, selected = planned
        if not next_plan.fitting:
            # The adjustment cannot make the launch fit. Stop rather than
            # replace the original failure with a misleading "does not fit"
            # error; the caller re-raises the recognized OOM.
            self._log(
                hostname,
                "warning",
                "vLLM retry cannot fit the host and will not be attempted: "
                f"{next_plan.reason}",
            )
            return None
        self._log(hostname, "info", next_plan.plan_line())
        return attempt + 1, next_plan, selected

    @staticmethod
    def _vllm_failure_is_memory(failure: BaseException | None) -> bool:
        """Classify a start failure from the exception before any engine log.

        The node-side preflight memory guard bails before vLLM writes its log,
        so its refusal only exists in the command's stderr.
        """
        if failure is None:
            return False
        stderr = getattr(failure, "stderr", "")
        return is_memory_failure(f"{failure}\n{stderr}")

    def _selected_generation(self, hostname: str) -> dict[str, object] | None:
        """Return the generation the start script selected for this attempt."""
        store = self._log_buffer.store
        attempt_id = self._log_buffer.attempts.get(hostname)
        if store is None or attempt_id is None:
            return None
        with suppress(Exception):
            selected = store.get(attempt_id).get("selected_generation")
            if isinstance(selected, dict):
                return selected
        return None

    async def _read_vllm_model_revision(self, hostname: str, model: str) -> str | None:
        """Resolve the cached Hugging Face commit when the model card lacks it."""
        if model.startswith("/") or "/" not in model:
            return None
        slug = model.replace("/", "--")
        path = f"{self._settings.nfs_mount_point}/hub/models--{slug}/refs/main"
        try:
            output = await self._ssh_run_command(
                hostname, f"cat -- {shlex.quote(path)} 2>/dev/null || true"
            )
        except Exception:
            logger.debug(
                "vllm_model_revision_unavailable",
                hostname=hostname,
                model=model,
                exc_info=True,
            )
            return None
        revision = output.strip()
        return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None

    async def _verify_vllm_serving(
        self,
        hostname: str,
        *,
        expected_model: str,
        started_at: datetime,
        expected_max_model_len: int | None = None,
    ) -> VllmServingVerification:
        """Fail closed unless this attempt's server answers real inference.

        The process check proves the listener belongs to this attempt and, when
        a generation was selected, that it runs the exact published runtime.
        The server's own ``/version`` and ``/v1/models`` responses then prove
        the model and version, and one bounded completion proves the model can
        serve. A failure must never reach registration.
        """
        base_url = self.validate_endpoint(hostname)
        selected = self._selected_generation(hostname) or {}
        runtime_path = selected.get("runtime_path")
        generation_id = selected.get("generation_id")
        runtime_path = runtime_path if isinstance(runtime_path, str) else None
        generation_id = generation_id if isinstance(generation_id, str) else None
        attempt_elapsed = max(0.0, (datetime.now(UTC) - started_at).total_seconds())
        process_output = await self._ssh_run_command(
            hostname, "ps -eo pid=,etimes=,args="
        )
        try:
            process = select_process(
                parse_vllm_processes(process_output),
                port=self._settings.vllm_port,
                expected_model=expected_model,
                runtime_path=runtime_path,
                attempt_elapsed_seconds=attempt_elapsed,
                slack_seconds=self._settings.vllm_process_start_slack,
            )
            async with httpx.AsyncClient() as client:
                identity = await fetch_server_identity(
                    client,
                    base_url,
                    timeout=self._settings.vllm_verify_timeout,
                )
                card = select_model(identity.models, expected_model)
                if (
                    expected_max_model_len is not None
                    and card.max_model_len is not None
                    and card.max_model_len != expected_max_model_len
                ):
                    raise VllmServingError(
                        "the server reports max_model_len "
                        f"{card.max_model_len}, planned {expected_max_model_len}"
                    )
                probe = await probe_inference(
                    client,
                    base_url,
                    model=card.model_id,
                    timeout=self._settings.vllm_inference_probe_timeout,
                    max_tokens=self._settings.vllm_inference_probe_max_tokens,
                )
        except VllmServingError as exc:
            self._log(hostname, "error", f"vLLM serving verification failed: {exc}")
            raise ProvisioningError(str(exc)) from exc

        revision = revision_from_root(card.root)
        if revision is None:
            revision = await self._read_vllm_model_revision(hostname, card.model_id)
        evidence = VllmServingVerification(
            served_model=card.model_id,
            runtime_version=identity.version,
            generation_id=generation_id,
            runtime_path=runtime_path,
            model_revision=revision,
            model_max_model_len=card.max_model_len,
            process_id=process.pid,
            process_age_seconds=process.age_seconds,
            task=probe.task,
            endpoint=probe.endpoint,
            latency_ms=round(probe.latency_ms, 2),
            prompt_tokens=probe.prompt_tokens,
            completion_tokens=probe.completion_tokens,
            finish_reason=probe.finish_reason,
            verified_at=datetime.now(UTC),
        )
        self._log(
            hostname,
            "info",
            "vLLM serving verified: "
            f"model={evidence.served_model} "
            f"runtime_version={evidence.runtime_version} "
            f"revision={evidence.model_revision or 'unknown'} "
            f"generation={evidence.generation_id or 'legacy'} "
            f"pid={evidence.process_id} "
            f"task={evidence.task} endpoint={evidence.endpoint} "
            f"latency_ms={evidence.latency_ms:g} "
            f"completion_tokens={evidence.completion_tokens}",
        )
        return evidence

    async def _verify_vllm_runtime(
        self,
        hostname: str,
        *,
        plan: VllmFitPlan,
        vllm_params: VllmParams | None,
        selected: tuple[VllmDeviceMemory, ...],
        serving: VllmServingVerification | None = None,
    ) -> VllmRuntimeState:
        """Persist the authorized plan and observe post-load selected-device memory."""
        memory_text = await self._ssh_run_command(
            hostname,
            "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free "
            "--format=csv,noheader,nounits",
        )
        selected_indices = {device.index for device in selected}
        rows: list[VllmGPUState] = []
        try:
            for line in memory_text.splitlines():
                if not line.strip():
                    continue
                index_text, total_text, used_text, free_text = line.split(",")
                index = int(index_text.strip())
                if index not in selected_indices:
                    continue
                rows.append(
                    VllmGPUState(
                        index=index,
                        total_mib=int(total_text.strip()),
                        used_mib=int(used_text.strip()),
                        free_mib=int(free_text.strip()),
                    )
                )
        except ValueError as exc:
            raise ProvisioningError(
                "could not parse post-load NVIDIA memory telemetry"
            ) from exc
        if not rows:
            raise ProvisioningError(
                "post-load NVIDIA memory telemetry returned no selected GPUs"
            )
        params = vllm_params or VllmParams()
        request = VllmRuntimeRequest(
            model=plan.requested.model,
            params=VllmParams(
                tensor_parallel_size=plan.requested.tensor_parallel_size,
                max_model_len=plan.requested.max_model_len,
                max_num_batched_tokens=plan.requested.max_num_batched_tokens,
                gpu_memory_utilization=plan.requested.gpu_memory_utilization,
                tool_call_parser=params.tool_call_parser,
                reasoning_parser=params.reasoning_parser,
                dtype=plan.requested.dtype,
                gpu_devices=plan.requested.gpu_devices,
            ),
        )
        runtime = VllmRuntimeState(
            requested=request,
            effective=VllmRuntimeEffective(
                model=plan.effective.model,
                model_selection=VllmModelSelection(plan.model_selection),
                model_reason=plan.model_reason,
                tensor_parallel_size=plan.effective.tensor_parallel_size,
                max_model_len=plan.effective.max_model_len,
                max_num_batched_tokens=plan.effective.max_num_batched_tokens,
                gpu_memory_utilization=plan.effective.gpu_memory_utilization,
                dtype=plan.effective.dtype,
                attention_backend=plan.effective.attention_backend,
            ),
            estimate=VllmMemoryEstimate(
                weights_mib=plan.estimate.weights_mib,
                weights_per_gpu_mib=plan.estimate.weights_per_gpu_mib,
                kv_cache_mib=plan.estimate.kv_cache_mib,
                runtime_overhead_mib=plan.estimate.runtime_overhead_mib,
                reserve_mib=plan.estimate.reserve_mib,
                usable_budget_mib=plan.estimate.usable_budget_mib,
                free_per_gpu_mib=plan.estimate.free_per_gpu_mib,
            ),
            adjustments=tuple(
                VllmAdjustment(
                    field=item.field,
                    previous=item.previous,
                    value=item.value,
                    reason=item.reason,
                )
                for item in plan.adjustments
            ),
            attempt=plan.attempt,
            reason=plan.reason,
            approximate=plan.approximate,
            gpus=tuple(sorted(rows, key=lambda row: row.index)),
            serving=serving,
            observed_at=datetime.now(UTC),
        )
        used = ",".join(str(row.used_mib) for row in runtime.gpus)
        free = ",".join(str(row.free_mib) for row in runtime.gpus)
        self._log(
            hostname,
            "info",
            "vLLM fitted: "
            f"model={runtime.effective.model} "
            f"selection={runtime.effective.model_selection.value} "
            f"tp={runtime.effective.tensor_parallel_size} "
            f"max_model_len={runtime.effective.max_model_len} "
            f"max_num_batched_tokens={runtime.effective.max_num_batched_tokens} "
            f"gpu_memory_utilization={runtime.effective.gpu_memory_utilization:g} "
            f"attempt={runtime.attempt} gpu_used_mib={used} gpu_free_mib={free}",
        )
        return runtime

    async def _start_vllm_with_retries(
        self,
        hostname: str,
        *,
        model: str | None,
        vllm_params: VllmParams | None,
        started_at: datetime,
        on_step: Callable[[str], None] | None = None,
    ) -> tuple[str, VllmRuntimeState | None]:
        """Start vLLM, poll health, and retry recognized OOMs within bounds.

        Without a memory plan (a host that cannot be measured) the legacy
        node-side behavior is preserved: start once, poll health once, and
        record no runtime state.
        """
        plan: VllmFitPlan | None = None
        selected: tuple[VllmDeviceMemory, ...] = ()
        # The incumbent server holds VRAM that would otherwise look allocated to
        # a process the plan does not control, so it must be stopped before the
        # first measurement. The retry path already stops before re-planning.
        await self._stop_failed_vllm_start(hostname)
        planned = await self._plan_vllm_launch(
            hostname, model=model, params=vllm_params, attempt=0
        )
        if planned is not None:
            plan, selected = planned
            self._require_fitting_plan(hostname, plan)
            self._log(hostname, "info", plan.plan_line())
        max_attempts = self._settings.vllm_memory_retry_max_attempts
        attempt = 0
        while True:
            if on_step is not None:
                on_step("starting_engine")
            start_model = plan.effective.model if plan is not None else model
            effective_params = (
                self._plan_vllm_effective(vllm_params, plan, selected)
                if plan is not None
                else vllm_params
            )
            # Verification compares process age to this specific launch, not
            # the whole attempt, so a survivor from an earlier retry cannot
            # pass as freshly started.
            launch_started_at = datetime.now(UTC)
            try:
                model_name = await self._run_start_vllm(
                    hostname,
                    model=start_model,
                    vllm_params=effective_params,
                    vllm_plan=plan,
                )
            except Exception as exc:
                retry = await self._vllm_maybe_retry(
                    hostname,
                    model=model,
                    params=vllm_params,
                    plan=plan,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    failure=exc,
                )
                if retry is None:
                    await self._stop_failed_vllm_start(hostname)
                    raise
                attempt, plan, selected = retry
                continue
            if on_step is not None:
                on_step("health_poll")
            await self._update_state(
                hostname, ProvisioningStep.HEALTH_POLL, started_at=started_at
            )
            self._log(hostname, "info", "Waiting for health endpoint")
            try:
                await self._poll_health(hostname, engine=InferenceEngine.VLLM)
            except Exception as exc:
                retry = await self._vllm_maybe_retry(
                    hostname,
                    model=model,
                    params=vllm_params,
                    plan=plan,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    failure=exc,
                )
                if retry is None:
                    await self._stop_failed_vllm_start(hostname)
                    raise
                attempt, plan, selected = retry
                continue
            if on_step is not None:
                on_step("vllm_serving_verify")
            expected_model = plan.effective.model if plan is not None else model_name
            try:
                serving = await self._verify_vllm_serving(
                    hostname,
                    expected_model=expected_model,
                    started_at=launch_started_at,
                    expected_max_model_len=(
                        plan.effective.max_model_len if plan is not None else None
                    ),
                )
                if plan is None:
                    # A host without a memory plan keeps the legacy return
                    # shape, but inference is still proven before routing.
                    return model_name, None
                if on_step is not None:
                    on_step("vllm_runtime_verify")
                runtime = await self._verify_vllm_runtime(
                    hostname,
                    plan=plan,
                    vllm_params=vllm_params,
                    selected=selected,
                    serving=serving,
                )
            except Exception:
                # A runtime we cannot verify must not stay running and
                # registered as healthy.
                await self._stop_failed_vllm_start(hostname)
                raise
            return model_name, runtime

    @staticmethod
    def _verify_profile_evidence(
        fit: LlamaCppRuntimeFit,
        expected_request: LlamaCppRuntimeRequest,
        draft_artifact: ResolvedGGUFArtifact | None,
        gpus: tuple[NodeGPU, ...],
        memory_rows: list[LlamaCppGPUState],
    ) -> LlamaCppSpeculativeEffective | None:
        """Compare a profile launch with the requested profile, field by field.

        The profile id in the log is a label the start script echoed back. It
        proves nothing, so every value the profile fixes is checked against
        what llama-server itself reported.
        """
        profile = expected_request.profile
        evidence = fit.profile
        if profile is None:
            if evidence is not None:
                raise ProvisioningError(
                    "llama.cpp launched a catalog profile that was not requested"
                )
            return None
        if evidence is None:
            raise ProvisioningError(
                "llama.cpp startup log has no catalog profile evidence"
            )
        expected_draft_cache = (
            profile.draft_cache_type.value
            if profile.draft_cache_type is not None
            else None
        )
        expected_draft_gguf = (
            draft_artifact.node_relative_entrypoint
            if draft_artifact is not None
            else None
        )
        if (
            evidence.profile_id != profile.profile_id
            or evidence.profile_version != profile.profile_version
            or evidence.ubatch != profile.ubatch
            or evidence.spec_type != profile.speculative_type.value
            or evidence.spec_draft_n_max != profile.speculative_draft_n_max
            or evidence.draft_cache_type != expected_draft_cache
            or evidence.draft_gguf != expected_draft_gguf
            or evidence.required_free_mib != profile.required_free_mib
            or evidence.cuda_graphs_disabled != profile.disable_cuda_graphs
        ):
            raise ProvisioningError(
                "llama.cpp runtime differs from its requested catalog profile"
            )
        if (profile.draft_artifact_id is None) != (draft_artifact is None):
            raise ProvisioningError(
                "the verified draft artifact differs from the catalog profile"
            )
        if evidence.gpu_free_mib < profile.required_free_mib + fit.fit_target_mib:
            raise ProvisioningError(
                "llama.cpp launched with less free VRAM than the profile requires"
            )
        if len(memory_rows) != 1 or len(gpus) != 1:
            raise ProvisioningError("catalog profiles support exactly one GPU per host")
        if (
            gpus[0].name != profile.gpu_name
            or gpus[0].total_mib < profile.gpu_min_total_mib
            or memory_rows[0].total_mib < profile.gpu_min_total_mib
        ):
            raise ProvisioningError(
                "llama.cpp launched on a GPU other than the profile's product"
            )
        if evidence.gpu_uuid != gpus[0].uuid:
            raise ProvisioningError(
                "llama.cpp launched on a GPU other than the inventoried one"
            )
        shares_cache = (
            profile.speculative_type is LlamaCppSpeculativeType.DRAFT_MTP_ASSISTANT
        )
        if evidence.draft_shares_target_cache != shares_cache:
            raise ProvisioningError(
                "llama.cpp draft cache sharing differs from the catalog profile"
            )
        return LlamaCppSpeculativeEffective(
            type=LlamaCppSpeculativeType(evidence.spec_type),
            draft_n_max=evidence.spec_draft_n_max,
            draft_gpu_layers=evidence.draft_gpu_layers,
            draft_total_layers=evidence.draft_total_layers,
            draft_cache_type_k=(
                None
                if evidence.draft_cache_type_k is None
                else LlamaCppCacheType(evidence.draft_cache_type_k)
            ),
            draft_cache_type_v=(
                None
                if evidence.draft_cache_type_v is None
                else LlamaCppCacheType(evidence.draft_cache_type_v)
            ),
            draft_shares_target_cache=shares_cache,
        )

    async def _verify_llamacpp_runtime(
        self,
        hostname: str,
        *,
        expected_request: LlamaCppRuntimeRequest,
        draft_artifact: ResolvedGGUFArtifact | None = None,
        gpus: tuple[NodeGPU, ...] = (),
    ) -> LlamaCppRuntimeState:
        """Fail closed unless the healthy server proves the managed fit contract."""
        try:
            log_text = await self._ssh_run_command(
                hostname,
                "cat -- "
                + shlex.quote(
                    self._engine_log_path(hostname, InferenceEngine.LLAMA_CPP)
                ),
            )
            fit = _parse_llamacpp_runtime_fit(log_text)
            memory_text = await self._ssh_run_command(
                hostname,
                "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free "
                "--format=csv,noheader,nounits",
            )
            memory_rows: list[LlamaCppGPUState] = []
            try:
                for line in memory_text.splitlines():
                    if not line.strip():
                        continue
                    index_text, total_text, used_text, free_text = line.split(",")
                    memory_rows.append(
                        LlamaCppGPUState(
                            index=int(index_text.strip()),
                            total_mib=int(total_text.strip()),
                            used_mib=int(used_text.strip()),
                            free_mib=int(free_text.strip()),
                        )
                    )
            except ValueError as exc:
                raise ProvisioningError(
                    "could not parse post-load NVIDIA memory telemetry"
                ) from exc
            if not memory_rows:
                raise ProvisioningError(
                    "post-load NVIDIA memory telemetry returned no GPUs"
                )
            if len({row.index for row in memory_rows}) != len(memory_rows):
                raise ProvisioningError(
                    "post-load NVIDIA memory telemetry returned duplicate GPU indices"
                )
            if fit.sizing != expected_request.sizing.value:
                raise ProvisioningError(
                    "llama.cpp runtime sizing mode differs from its requested value"
                )
            if fit.fit_target_mib != expected_request.fit_target_mib:
                raise ProvisioningError(
                    "llama.cpp runtime fit target differs from its requested value"
                )
            speculative = self._verify_profile_evidence(
                fit, expected_request, draft_artifact, gpus, memory_rows
            )
            if expected_request.sizing is not LlamaCppSizingMode.AUTO:
                expected_cache_type = expected_request.cache_type
                assert expected_cache_type is not None
                if (
                    fit.context_per_slot != expected_request.context_per_slot
                    or fit.slots != expected_request.slots
                    or fit.cache_type_k != expected_cache_type.value
                    or fit.cache_type_v != expected_cache_type.value
                ):
                    raise ProvisioningError(
                        "llama.cpp runtime differs from its requested "
                        f"{expected_request.sizing.value} sizing"
                    )
            if (
                fit.estimator_overrun_used
                and not expected_request.allow_estimator_overrun
            ):
                raise ProvisioningError(
                    "llama.cpp used an estimator overrun without request permission"
                )
            if any(row.free_mib < fit.fit_target_mib for row in memory_rows):
                raise ProvisioningError(
                    "llama.cpp post-load free VRAM is below the requested fit target"
                )

            runtime = LlamaCppRuntimeState(
                requested=expected_request,
                effective=LlamaCppRuntimeEffective(
                    train_context=fit.train_context,
                    context_per_slot=fit.context_per_slot,
                    slot_context_limit=fit.slot_context_limit,
                    slots=fit.slots,
                    aggregate_context=fit.aggregate_context,
                    cache_type_k=LlamaCppCacheType(fit.cache_type_k),
                    cache_type_v=LlamaCppCacheType(fit.cache_type_v),
                    flash_attn=LlamaCppFlashAttention(fit.flash_attn),
                    kv_unified=fit.kv_unified,
                    gpu_layers=fit.gpu_layers,
                    total_layers=fit.total_layers,
                    estimator_overrun_used=fit.estimator_overrun_used,
                    ubatch=fit.profile.ubatch if fit.profile is not None else None,
                    speculative=speculative,
                ),
                gpus=tuple(sorted(memory_rows, key=lambda row: row.index)),
                observed_at=datetime.now(UTC),
            )
            used = ",".join(str(row.used_mib) for row in runtime.gpus)
            free = ",".join(str(row.free_mib) for row in runtime.gpus)
            self._log(
                hostname,
                "info",
                "llama.cpp fitted: "
                f"sizing={fit.sizing} "
                f"train_context={fit.train_context} "
                f"context_per_slot={fit.context_per_slot} "
                f"slot_context_limit={fit.slot_context_limit} "
                f"slots={fit.slots} "
                f"aggregate_context={fit.aggregate_context} "
                f"kv_unified={str(fit.kv_unified).lower()} "
                f"cache_type_k={fit.cache_type_k} "
                f"cache_type_v={fit.cache_type_v} "
                f"flash_attn={fit.flash_attn} "
                f"gpu_layers={fit.gpu_layers}/{fit.total_layers} "
                f"fit_target_mib={expected_request.fit_target_mib} "
                f"estimator_overrun_used={str(fit.estimator_overrun_used).lower()} "
                f"gpu_used_mib={used} gpu_free_mib={free}",
            )
            return runtime
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
        llamacpp_request: LlamaCppRuntimeRequest | None = None,
        vllm_params: VllmParams | None = None,
        draft_artifact: ResolvedGGUFArtifact | None = None,
        vllm_plan: VllmFitPlan | None = None,
    ) -> str:
        """Run the engine start script and extract model name from stdout."""
        if engine == InferenceEngine.LLAMA_CPP:
            script = "start-llamacpp.sh"
        else:
            script = "start-vllm.sh"
        command = self._remote_script_command(
            hostname,
            engine,
            script,
            env=self._start_script_env(
                model,
                engine,
                artifact,
                llamacpp_request=llamacpp_request,
                vllm_params=vllm_params,
                draft_artifact=draft_artifact,
                vllm_plan=vllm_plan,
            ),
        )
        output: AsyncIterator[tuple[str, str]]
        if self._remote_logs is not None:
            log_path = self._engine_log_path(hostname, engine)
            variable = (
                "AUTOLLAMACPP_LOG_FILE"
                if engine == InferenceEngine.LLAMA_CPP
                else "AUTOVLLM_LOG_FILE"
            )
            command = f"{variable}={shlex.quote(log_path)} " + command
            output = self._remote_logs.run(
                hostname, command, stage="start", engine_log=log_path
            )
        else:
            output = self._ssh_client.run_streaming(hostname, command)
        model_name: str | None = None
        async for stream, line in output:
            logger.debug(
                "start_vllm_output", stream=stream, line=line, hostname=hostname
            )
            self._log(hostname, "debug", line, stream=stream)
            if stream == "stdout":
                self._record_generation(hostname, line)
                match = MODEL_PATTERN.search(line)
                if match:
                    model_name = match.group(1).strip()
                    if self._log_buffer.store is not None:
                        self._log_buffer.store.update(
                            self._log_buffer.attempts[hostname], model=model_name
                        )
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
        if self._remote_logs is not None:
            await self._remote_logs.follow(self._log_buffer.attempts[hostname])
            return
        log_path = self._engine_log_path(hostname, engine)
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
        llamacpp_runtime: LlamaCppRuntimeState | None = None,
        vllm_runtime: VllmRuntimeState | None = None,
        self_setup: bool = False,
        owner: str = "",
        gpus: tuple[NodeGPU, ...] = (),
        placement: NodePlacement | None = None,
    ) -> None:
        """Register node in etcd with correct fields (D-11, D-12)."""
        node = Node(
            node_id=hostname,
            endpoint=self.validate_endpoint(hostname),
            status=NodeStatus.HEALTHY,
            model=model,
            engine=engine,
            artifact_id=artifact_id,
            llamacpp_runtime=llamacpp_runtime,
            vllm_runtime=vllm_runtime,
            last_heartbeat=datetime.now(UTC),
            managed=managed,
            self_setup=self_setup,
            owner=owner,
            gpus=gpus,
            placement=placement,
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
        operation: BackgroundOperation = BackgroundOperation.PROVISION,
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
                task_name = f"{operation.value}:{provisioning_hostname}"

        record: ProvisioningTask | None = None
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
            record = ProvisioningTask(task, provisioning_identity, operation)
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

    def active_provision(self, hostname: str) -> ProvisioningTask | None:
        """Inspect the current provisioning operation without mutating it."""
        record = self._provisioning_tasks.get(hostname)
        if (
            record is None
            or record.task.done()
            or record.operation is not BackgroundOperation.PROVISION
        ):
            return None
        return record

    async def cancel_active_provision(
        self, hostname: str
    ) -> ProvisioningIdentity | None:
        """Cancel the current provision for automatic-placement fencing.

        Manual teardown must instead inspect, persist suspension, and then
        cancel that inspected record through ``cancel_provision``.
        """
        record = self.active_provision(hostname)
        if record is None:
            return None
        return await self.cancel_provision(hostname, record)

    async def cancel_provision(
        self, hostname: str, record: ProvisioningTask
    ) -> ProvisioningIdentity:
        """Cancel only the inspected record, never its replacement.

        There is no await between checking identity and cancelling the task.
        Callers may safely persist operator intent after inspection without
        risking cancellation of another operation that took its place.
        """
        if self.active_provision(hostname) is not record:
            raise ProvisioningOperationChangedError(
                f"Provisioning operation for '{hostname}' changed; retry teardown"
            )
        task = record.task
        self._explicit_cancel_tasks.add(task)
        try:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            self._explicit_cancel_tasks.discard(task)

        if not task.cancelled():
            raise RuntimeError(
                f"Provisioning task for '{hostname}' swallowed cancellation"
            )

        if not record.started:
            # A task cancelled before its coroutine first runs cannot record
            # its own terminal state. Its attempt has not been opened yet, so
            # the buffer still tracks the previous finished attempt: only the
            # etcd state may be written, never that older history row.
            await self._update_state(
                hostname,
                ProvisioningStep.FAILED,
                failed_step="cancelled",
                error="Provisioning cancelled by teardown",
                history=False,
            )
        return record.identity

    async def _drain_wait(self, hostname: str) -> DrainOutcome:
        """Wait for active connections to reach zero or timeout (D-08, D-09)."""
        if self._tracker is None:
            logger.warning("drain_skip_no_tracker", hostname=hostname)
            return DrainOutcome.UNAVAILABLE
        deadline = asyncio.get_running_loop().time() + self._settings.drain_timeout
        while True:
            if self._tracker.get(hostname) == 0:
                return DrainOutcome.COMPLETE
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("drain_timeout_expired", hostname=hostname)
                return DrainOutcome.TIMED_OUT
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
            if await self._reconcile_host(hostname, block=False):
                logger.warning(
                    "teardown_reconcile_survivor",
                    hostname=hostname,
                    reason=(
                        "a prior remote operation is still active on the node; "
                        "kill it manually on the host before reprovisioning"
                    ),
                )
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
        self._begin_log(hostname, engine, operation="teardown")
        self._log(hostname, "info", f"Teardown started (force={force})")

        try:
            if not force:
                await self._update_state(
                    hostname, ProvisioningStep.DRAINING, started_at=teardown_started_at
                )
                self._log(hostname, "info", "Draining active connections")
                if self._registry is not None:
                    self._registry.drain(hostname)
                drain_outcome = await self._drain_wait(hostname)
                if drain_outcome is DrainOutcome.COMPLETE:
                    self._log(hostname, "info", "Drain complete")
                elif drain_outcome is DrainOutcome.TIMED_OUT:
                    self._log(
                        hostname,
                        "warning",
                        "Drain timed out; continuing teardown",
                    )
                else:
                    self._log(
                        hostname,
                        "warning",
                        "Connection tracking unavailable; continuing teardown",
                    )

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
            stop_command = self._remote_script_command(
                hostname,
                engine,
                stop_script,
                args=("--force",) if force else (),
            )
            await self._ssh_run_command(hostname, stop_command)

            await self._update_state(
                hostname, ProvisioningStep.DEREGISTERING, started_at=teardown_started_at
            )
            node = self._registry.get(hostname) if self._registry else None
            repool = node is not None and not node.managed
            self._log(hostname, "info", "Deregistering node from etcd")
            await asyncio.to_thread(
                self._etcd_client.delete, f"{self._etcd_client.prefix}{hostname}"
            )
            if self._registry is not None:
                self._registry.remove(hostname)
            if repool:
                await self.register_available(hostname)
                self._log(
                    hostname, "info", "Returned standalone node to available pool"
                )

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
            await self._capture_failure(hostname, "teardown", exc)
            if self._registry is not None:
                self._registry.update_status(
                    hostname,
                    NodeStatus.FAILED,
                    # A failed cleanup must not turn the teardown-only
                    # relaunch terminal state into setup-retryable FAILED.
                    allowed_from=set(NodeStatus) - {NodeStatus.RELAUNCH_FAILED},
                )
            raise
        finally:
            await self._finish_remote_logs(hostname)
            self._mark_log_complete(hostname)

        logger.info("teardown_complete", hostname=hostname)
