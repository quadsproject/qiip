"""Admin API for operational visibility into the gateway.

Per D-05: Endpoints under /admin namespace, separate from /v1 proxy API.
Per D-06: Separate APIRouter in api/admin.py with prefix="/admin".
Per METR-03: Node entries include identity, health, active connections,
and circuit breaker state for the operations dashboard.
"""

from __future__ import annotations

import asyncio
import json
import zlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from inference_proxy.config.dependencies import (
    get_catalog_service,
    get_download_service,
    get_llmfit_runner,
    get_placement_suspensions,
    get_provisioner,
    get_quads_client,
    get_quads_poller,
    get_redfish_client,
    get_registry,
    get_request_metrics,
    get_settings,
    get_unified_node_service,
    require_admin_auth,
)
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.catalog import (
    ModelCatalogResponse,
    ModelCatalogService,
)
from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.llmfit.errors import LLMFitParseError, LLMFitTimeoutError
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.models.admin import (
    _HOSTNAME_RE,
    AdminMetricsResponse,
    AdminNodeResponse,
    DownloadRequest,
    DownloadStatusResponse,
    LlamaCppRelaunchRequest,
    LlamaCppRelaunchResponse,
    OwnerUpdateRequest,
    PowerActionRequest,
    PowerStateResponse,
    QUADSStatusResponse,
    RecommendationResponse,
    RegisterRequest,
    SetupRequest,
    SetupResponse,
    TaskStatusResponse,
    TeardownResponse,
)
from inference_proxy.models.endpoint import EndpointValidationError
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppRuntimeRequest,
    Node,
    NodeStatus,
    VllmParams,
)
from inference_proxy.placement.suspensions import SuspensionStore
from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.provisioner import (
    BackgroundOperation,
    NodeProvisioner,
    ProvisioningCapacityError,
    ProvisioningError,
    ProvisioningIdentity,
    ProvisioningOperationChangedError,
    RelaunchPreconditionError,
    RelaunchValidationError,
    SelfSetupError,
)
from inference_proxy.provisioning.reliability import GroupBy, build_report
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHConnectionError,
)
from inference_proxy.quads.client import (
    QUADSClient,
    QUADSConnectionError,
    availability_window_end,
    canonical_hostname,
)
from inference_proxy.quads.poller import QUADSPoller
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishDestinationError, RedfishError
from inference_proxy.routing.request_metrics import RequestMetrics
from inference_proxy.services.unified_nodes import UnifiedNodeService

logger = structlog.get_logger()

_DEGRADED_DATA_HEADER = "X-Inference-Proxy-Data-Degraded"
_PROVISIONING_TASKS_DEGRADED = "provisioning-tasks"
_MODEL_CATALOG_DEGRADED = "model-catalog"

admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_auth)],
)

# D-08: module-level set to prevent duplicate setup requests
# ponytail: single-worker-only dedup guard; move to etcd CAS if workers > 1
pending_hosts: set[str] = set()

_SETUP_REJECTED_STATUSES = frozenset(
    {
        NodeStatus.HEALTHY,
        NodeStatus.UNHEALTHY,
        NodeStatus.RELAUNCHING,
        NodeStatus.RELAUNCH_FAILED,
    }
)
_SETUP_RETRYABLE_STATUSES = frozenset(
    {
        NodeStatus.AVAILABLE,
        NodeStatus.PROVISIONING,
        NodeStatus.FAILED,
        NodeStatus.UNKNOWN,
        NodeStatus.DRAINING,
    }
)

_SETUP_SELECTION_FIELDS = frozenset({"engine", "model", "artifact_id"})


@dataclass(frozen=True, slots=True)
class _SetupSelection:
    """Effective engine-specific setup identity after retry inheritance."""

    engine: InferenceEngine
    model: str | None
    artifact_id: str | None
    llamacpp_request: LlamaCppRuntimeRequest | None
    vllm_params: VllmParams | None = None


def _effective_setup_selection(
    body: SetupRequest,
    node: Node | None,
    *,
    fallback: _SetupSelection | None = None,
) -> _SetupSelection:
    """Resolve explicit setup input or inherit retry identity from a node.

    The fallback preserves the identity observed before waiting for the host
    lease if the stale record disappears while the lease is acquired.
    """
    explicit = bool(body.model_fields_set & _SETUP_SELECTION_FIELDS)
    if explicit:
        return _SetupSelection(
            body.engine,
            body.model,
            body.artifact_id,
            None,
            body.vllm_params,
        )
    if node is None:
        return fallback or _SetupSelection(
            body.engine,
            body.model,
            body.artifact_id,
            None,
            body.vllm_params,
        )
    if node.engine is InferenceEngine.LLAMA_CPP:
        request = (
            node.llamacpp_runtime.requested
            if node.llamacpp_runtime is not None
            else None
        )
        return _SetupSelection(node.engine, None, node.artifact_id, request)
    return _SetupSelection(
        node.engine,
        node.model or None,
        None,
        None,
        body.vllm_params,
    )


async def _validate_setup_selection(
    provisioner: NodeProvisioner,
    selection: _SetupSelection,
) -> None:
    """Validate one effective setup selection before any destructive work."""
    provisioner.validate_setup_configuration(selection.engine)
    await provisioner.resolve_artifact_selection(
        selection.engine, selection.artifact_id
    )


def _validated_hostname(hostname: str) -> str:
    """Normalize and validate a hostname path parameter."""
    hostname = canonical_hostname(hostname)
    if not hostname or len(hostname) > 253 or not _HOSTNAME_RE.fullmatch(hostname):
        raise HTTPException(status_code=400, detail="Invalid hostname")
    return hostname


@admin_router.get("/nodes")
async def list_nodes(
    response: Response,
    service: UnifiedNodeService = Depends(get_unified_node_service),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> list[AdminNodeResponse]:
    """Return unified node list merging QUADS hosts with etcd nodes."""
    try:
        results = await provisioner.list_tasks_raw()
    except Exception:
        logger.warning("provisioning_task_list_unavailable", exc_info=True)
        response.headers[_DEGRADED_DATA_HEADER] = _PROVISIONING_TASKS_DEGRADED
        results = []
    task_map: dict[str, TaskStatusResponse] = {}
    for value_bytes, _metadata in results:
        try:
            data = json.loads(value_bytes)
            task = TaskStatusResponse(**data)
            task_map[task.hostname] = task
        except (json.JSONDecodeError, ValidationError):
            pass  # ponytail: silently skip malformed entries
    return service.get_unified_nodes(task_map=task_map)


@admin_router.get("/metrics")
async def get_metrics(
    request_metrics: RequestMetrics = Depends(get_request_metrics),
) -> AdminMetricsResponse:
    """Return aggregate request counter data for the operations dashboard."""
    return AdminMetricsResponse(
        total_requests=request_metrics.get_total(),
        per_model=request_metrics.get_per_model(),
        per_node=request_metrics.get_per_node(),
    )


@admin_router.get("/models/catalog")
async def list_catalog(
    response: Response,
    catalog: ModelCatalogService = Depends(get_catalog_service),
) -> ModelCatalogResponse:
    """Return the list of models available in the HuggingFace NFS cache."""
    result = await catalog.list_models()
    if (
        result.incomplete_count
        or result.unverifiable_count
        or result.invalid_artifact_count
        or result.cache_warning_count
    ):
        response.headers[_DEGRADED_DATA_HEADER] = _MODEL_CATALOG_DEGRADED
    return result


@admin_router.post("/models/download", status_code=202)
async def trigger_download(
    body: DownloadRequest,
    response: Response,
    svc: DownloadService = Depends(get_download_service),
) -> DownloadStatusResponse:
    """Trigger a background model download (DL-01).

    Returns 202 for new downloads. Duplicate POSTs for an in-progress
    download return 200 with the existing status (D-10).
    """
    result = await svc.trigger_download(
        body.repo_id,
        revision=body.revision,
        engine=body.engine,
        gguf=body.gguf,
    )
    response.status_code = 202 if result.started else 200
    return result.status


@admin_router.get("/models/downloads")
async def list_downloads(
    svc: DownloadService = Depends(get_download_service),
) -> list[DownloadStatusResponse]:
    """Return status of all tracked downloads (DL-03)."""
    return svc.get_all_statuses()


@admin_router.post("/nodes/pool", status_code=201)
async def register_node(
    body: RegisterRequest,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> JSONResponse:
    """Register a node in the pool, or adopt a running OpenAI-compatible server.

    Registration is a host-mutating operation, so the hostname is reserved
    through the host lifecycle coordinator before any remote probe (self-setup
    adoption) or fleet write. The registry is re-checked while holding the
    lease so adoption cannot race with setup, teardown, or another concurrent
    registration for the same hostname.

    A custom port is accepted only for ``self_setup`` adoption. A plain pool
    node is provisioned later on the configured default
    (``provisioning.vllm_port``), so a stored custom port would be silently
    replaced at launch time and is therefore rejected.

    Re-adoption of an existing self-setup instance is allowed: it re-probes
    the live server and reconciles the tracked model with what the server
    currently serves (see ``NodeProvisioner.register_self_setup``).
    """
    hostname = canonical_hostname(body.hostname)
    if body.port is not None and not body.self_setup:
        raise HTTPException(
            status_code=400,
            detail=(
                "a custom port is only supported for self-setup adoption; "
                "plain pool registration provisions on the configured default "
                "port"
            ),
        )
    node = registry.get(hostname)
    if node is not None and (not body.self_setup or not node.self_setup):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Node '{hostname}' is already registered"
                + (
                    "; re-adoption requires the existing registration to be "
                    "a self-setup instance"
                    if body.self_setup
                    else ""
                )
            ),
        )
    try:
        provisioner.validate_endpoint(hostname, body.port)
    except EndpointValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Host lifecycle operation already in progress for '{hostname}'; "
                "wait for it to finish before retrying registration"
            ),
        )
    try:
        # Re-check under exclusive ownership: setup, teardown, or a concurrent
        # registration may have claimed the host while we validated the endpoint.
        node = registry.get(hostname)
        if node is not None and (not body.self_setup or not node.self_setup):
            raise HTTPException(
                status_code=409, detail=f"Node '{hostname}' is already registered"
            )
        if body.self_setup:
            if node is not None:
                logger.info(
                    "self_setup_re_adoption_started",
                    hostname=hostname,
                    previous_model=node.model,
                )
            # Only override an already-registered owner when the request
            # explicitly supplies one; a bare re-adoption must not wipe it.
            owner = (
                body.owner
                if "owner" in body.model_fields_set
                else (node.owner if node else "")
            )
            # The same rule applies to the admin-only flag and the display
            # name: a bare re-adoption (e.g. the dashboard "Add to Fleet"
            # form, which sends only hostname/self_setup) must not silently
            # de-classify an admin-only server or drop its name.
            admin_only = (
                body.admin_only
                if "admin_only" in body.model_fields_set
                else (node.admin_only if node else False)
            )
            name = (
                body.name
                if "name" in body.model_fields_set
                else (node.name if node else "")
            )
            try:
                adopted = await provisioner.register_self_setup(
                    hostname,
                    body.port,
                    owner=owner,
                    admin_only=admin_only,
                    name=name,
                )
            except SelfSetupError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return JSONResponse(
                status_code=201,
                content={
                    "hostname": hostname,
                    "state": adopted.status.value,
                    "model": adopted.model,
                    "self_setup": True,
                    "admin_only": adopted.admin_only,
                    "name": adopted.name,
                },
            )
        await provisioner.register_available(hostname, body.port, owner=body.owner)
        return JSONResponse(
            status_code=201,
            content={"hostname": hostname, "state": "available"},
        )
    finally:
        lease.release()


@admin_router.delete("/nodes/{node_id}/pool")
async def remove_from_pool(
    node_id: str,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> JSONResponse:
    """Remove a manually registered node from the fleet."""
    hostname = _validated_hostname(node_id)
    node = registry.get(hostname)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{hostname}' not found")

    # Removal deletes registry and etcd state, so it must hold the host
    # lifecycle lease and re-check the node before deleting: otherwise setup,
    # teardown, or another registration could own the same hostname.
    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Host lifecycle operation already in progress for '{hostname}'; "
                "wait for it to finish before retrying removal"
            ),
        )
    try:
        node = registry.get(hostname)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node '{hostname}' not found")
        if node.self_setup:
            await provisioner.remove_available(hostname)
            return JSONResponse(content={"hostname": hostname, "removed": True})
        if node.status != NodeStatus.AVAILABLE:
            raise HTTPException(
                status_code=409,
                detail=f"Node '{hostname}' is {node.status.value}; use teardown instead",
            )
        if node.managed:
            raise HTTPException(
                status_code=409,
                detail=f"Node '{hostname}' is managed; use teardown instead",
            )
        await provisioner.remove_available(hostname)
        return JSONResponse(content={"hostname": hostname, "removed": True})
    finally:
        lease.release()


@admin_router.patch("/nodes/{node_id}/owner")
async def update_node_owner(
    node_id: str,
    body: OwnerUpdateRequest,
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> JSONResponse:
    """Assign (or clear) the endpoint owner for a registered node."""
    hostname = _validated_hostname(node_id)
    try:
        node = await provisioner.update_node_owner(hostname, body.owner)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Node '{hostname}' not found"
        ) from None
    except ProvisioningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content={"node_id": node.node_id, "owner": node.owner})


@admin_router.post("/nodes/setup", status_code=202)
async def setup_node(
    body: SetupRequest,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
    quads_client: QUADSClient | None = Depends(get_quads_client),
    settings: Settings = Depends(get_settings),
) -> SetupResponse:
    """Trigger provisioning of a new node (runs in background).

    Includes dedup guard (D-08) and live QUADS re-validation (D-10/D-11).
    """
    hostname = canonical_hostname(body.hostname)
    logger.info(
        "setup_request_received",
        hostname=hostname,
        engine=body.engine,
        vllm_params=body.vllm_params.model_dump() if body.vllm_params else None,
        fields_set=sorted(body.model_fields_set),
    )
    initial_node = registry.get(hostname)
    if initial_node is not None and initial_node.self_setup:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Node '{hostname}' is a self-setup instance; "
                "remove it from the fleet instead of provisioning"
            ),
        )
    selection = _effective_setup_selection(body, initial_node)

    try:
        provisioner.validate_endpoint(hostname)
        await _validate_setup_selection(provisioner, selection)
    except (EndpointValidationError, ProvisioningError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # D-08: dedup guard
    if hostname in pending_hosts:
        raise HTTPException(
            status_code=409,
            detail=f"Setup already in progress for '{hostname}'",
        )

    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Host lifecycle operation already in progress for '{hostname}'; "
                "wait for it to finish before retrying setup"
            ),
        )

    transferred = False
    try:
        # A setup could have passed the first check before waiting on the host
        # reservation. Re-check under exclusive lifecycle ownership.
        if hostname in pending_hosts:
            raise HTTPException(
                status_code=409,
                detail=f"Setup already in progress for '{hostname}'",
            )

        node = registry.get(hostname)
        selection = _effective_setup_selection(body, node, fallback=selection)
        if node is not None and node.self_setup:
            # Re-checked under exclusive ownership: adoption can complete after
            # the initial read at the top of this handler, and teardown performs
            # the same second check. A self-setup node can never be provisioned.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Node '{hostname}' is a self-setup instance; "
                    "remove it from the fleet instead of provisioning"
                ),
            )
        if node is not None and node.status in _SETUP_REJECTED_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Host '{hostname}' is {node.status.value}; tear it down "
                    "before starting setup"
                ),
            )

        if node is not None and node.status == NodeStatus.DRAINING:
            active_connections = provisioner.connection_count(hostname)
            if active_connections:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Host '{hostname}' is draining with "
                        f"{active_connections} active request(s); wait for "
                        "requests to finish or complete teardown"
                    ),
                )

        try:
            await _validate_setup_selection(provisioner, selection)
        except ProvisioningError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if node is not None and node.status in _SETUP_RETRYABLE_STATUSES:
            try:
                await provisioner.cleanup_stale_node(hostname)
            except Exception as exc:
                logger.warning(
                    "setup_stale_cleanup_failed",
                    hostname=hostname,
                    status=node.status.value,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not clean stale state for '{hostname}'",
                ) from exc

        # D-10/D-11: live QUADS re-validation (skip for unmanaged nodes).
        if body.managed and quads_client is not None:
            lookahead_hours = settings.quads.schedule_lookahead_hours
            window_end = availability_window_end(lookahead_hours)
            try:
                available = await quads_client.get_available(end=window_end)
            except QUADSConnectionError as exc:
                raise HTTPException(
                    status_code=503, detail="QUADS unavailable"
                ) from exc
            if hostname not in available:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Host '{hostname}' is currently assigned or has an "
                        f"upcoming QUADS assignment within the configured "
                        f"{lookahead_hours}-hour scheduling window"
                    ),
                )

        # The lease already closes the async TOCTOU window. Keep the legacy
        # single-worker guard for clear duplicate-setup responses.
        pending_hosts.add(hostname)

        # Retrying setup must not wipe an owner established by registration;
        # only an explicit owner in the request overrides it. Prefer the
        # post-lease node snapshot (a concurrent register/owner PATCH may have
        # committed between the initial read and lease acquisition); fall back
        # to the initial snapshot only if the node vanished under the lease.
        setup_owner = (
            body.owner
            if "owner" in body.model_fields_set
            else (
                node.owner
                if node is not None
                else (initial_node.owner if initial_node is not None else "")
            )
        )

        async def _provision_and_cleanup() -> None:
            try:
                if selection.llamacpp_request is None:
                    await provisioner.provision(
                        hostname,
                        managed=body.managed,
                        model=selection.model,
                        engine=selection.engine,
                        artifact_id=selection.artifact_id,
                        vllm_params=selection.vllm_params,
                        lifecycle_lease=lease,
                        owner=setup_owner,
                    )
                else:
                    await provisioner.provision(
                        hostname,
                        managed=body.managed,
                        model=selection.model,
                        engine=selection.engine,
                        artifact_id=selection.artifact_id,
                        llamacpp_request=selection.llamacpp_request,
                        lifecycle_lease=lease,
                        owner=setup_owner,
                    )
            finally:
                pending_hosts.discard(hostname)
                # NodeProvisioner owns the lease after transfer. This
                # idempotent release also covers test doubles and cancellation
                # during wrapper cleanup.
                lease.release()

        background = _provision_and_cleanup()
        try:
            task = provisioner.fire_background(
                background,
                provisioning_hostname=hostname,
                provisioning_identity=ProvisioningIdentity(
                    engine=selection.engine,
                    artifact_id=selection.artifact_id,
                ),
            )
        except ProvisioningCapacityError as exc:
            background.close()
            pending_hosts.discard(hostname)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Provisioning capacity reached: {exc.active} active "
                    f"task(s), limit {exc.limit}; retry after an existing "
                    "setup finishes"
                ),
            ) from exc
        except Exception:
            background.close()
            pending_hosts.discard(hostname)
            raise

        def _setup_done(_task: object) -> None:
            # A task cancelled before its coroutine starts never reaches the
            # wrapper's finally block.
            pending_hosts.discard(hostname)
            lease.release()

        task.add_done_callback(_setup_done)
        transferred = True
        return SetupResponse(task_id=hostname)
    finally:
        if not transferred:
            lease.release()


@admin_router.post("/nodes/{hostname}/llamacpp/relaunch", status_code=202)
async def relaunch_llamacpp_node(
    hostname: str,
    body: LlamaCppRelaunchRequest,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> LlamaCppRelaunchResponse:
    """Queue a drain-safe managed llama.cpp runtime relaunch."""
    hostname = _validated_hostname(hostname)
    if not provisioner.connection_tracking_available:
        raise HTTPException(
            status_code=503,
            detail="llama.cpp relaunch requires active-connection tracking",
        )
    request = LlamaCppRuntimeRequest.model_validate(body.model_dump())
    try:
        node = provisioner.validate_llamacpp_relaunch(
            registry.get(hostname),
            request,
        )
    except RelaunchPreconditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RelaunchValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=f"Host lifecycle operation already in progress for '{hostname}'",
        )

    transferred = False
    try:
        try:
            node = provisioner.validate_llamacpp_relaunch(
                registry.get(hostname),
                request,
            )
        except RelaunchPreconditionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RelaunchValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async def _relaunch_and_cleanup() -> None:
            try:
                await provisioner.relaunch_llamacpp(
                    hostname,
                    request,
                    lifecycle_lease=lease,
                )
            finally:
                lease.release()

        background = _relaunch_and_cleanup()
        try:
            task = provisioner.fire_background(
                background,
                provisioning_hostname=hostname,
                provisioning_identity=ProvisioningIdentity(
                    engine=InferenceEngine.LLAMA_CPP,
                    artifact_id=node.artifact_id,
                ),
                operation=BackgroundOperation.RELAUNCH,
            )
        except ProvisioningCapacityError as exc:
            background.close()
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Provisioning capacity reached: {exc.active} active "
                    f"task(s), limit {exc.limit}; retry after an existing "
                    "operation finishes"
                ),
            ) from exc
        except Exception:
            background.close()
            raise

        task.add_done_callback(lambda _task: lease.release())
        transferred = True
        return LlamaCppRelaunchResponse(task_id=hostname)
    finally:
        if not transferred:
            lease.release()


@admin_router.get("/provisioning/tasks")
async def list_provisioning_tasks(
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> list[TaskStatusResponse]:
    """Return status of all provisioning/teardown operations from etcd."""
    results = await provisioner.list_tasks_raw()
    tasks: list[TaskStatusResponse] = []
    for value_bytes, _metadata in results:
        try:
            data = json.loads(value_bytes)
            tasks.append(TaskStatusResponse(**data))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("task_parse_failed", raw=value_bytes[:200], error=str(exc))
    return tasks


def _attempt_store(provisioner: NodeProvisioner) -> AttemptLogStore:
    store = provisioner.log_buffer.store
    if store is None:
        raise HTTPException(
            status_code=503, detail="Durable provisioning logs unavailable"
        )
    return store


@admin_router.get("/provisioning/reliability", response_model=None)
async def provisioning_reliability(
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: GroupBy = "signature",
    hostname: Annotated[list[str] | None, Query(max_length=100)] = None,
    download: bool = False,
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> dict[str, Any] | JSONResponse:
    """Export the same retained cohort and evidence shown in the fleet view."""
    if any(value is not None and value.tzinfo is None for value in (since, until)):
        raise HTTPException(
            status_code=422, detail="Report timestamps require a timezone"
        )
    if since and until and since >= until:
        raise HTTPException(status_code=422, detail="since must be earlier than until")
    report = await asyncio.to_thread(
        build_report,
        _attempt_store(provisioner),
        since=since,
        until=until,
        group_by=group_by,
        hostnames=[_validated_hostname(host) for host in hostname or []],
    )
    if download:
        return JSONResponse(
            report,
            headers={
                "Content-Disposition": 'attachment; filename="fleet-reliability.json"'
            },
        )
    return report


def _owned_attempt(
    store: AttemptLogStore, hostname: str, attempt_id: str
) -> dict[str, Any]:
    hostname = _validated_hostname(hostname)
    try:
        manifest = store.get(attempt_id)
        if manifest["hostname"] == hostname:
            return manifest
    except KeyError:
        pass
    raise HTTPException(
        status_code=404, detail="Attempt unavailable or evicted by retention"
    )


@admin_router.get("/provisioning/{hostname}/attempts")
async def provisioning_attempts(
    hostname: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> dict[str, object]:
    return await asyncio.to_thread(
        _attempt_store(provisioner).history,
        _validated_hostname(hostname),
        limit=limit,
        offset=offset,
    )


@admin_router.get("/provisioning/{hostname}/attempts/{attempt_id}/logs")
async def attempt_logs(
    hostname: str,
    attempt_id: str,
    after: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=500),
    source: str = Query(default="", max_length=100),
    limit: int = Query(default=500, ge=1, le=1000),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> dict[str, object]:
    store = _attempt_store(provisioner)
    _owned_attempt(store, hostname, attempt_id)
    try:
        return await asyncio.to_thread(
            store.read, attempt_id, after=after, query=q, source=source, limit=limit
        )
    except KeyError:
        raise HTTPException(
            status_code=404, detail="Attempt unavailable or evicted by retention"
        ) from None


@admin_router.post("/provisioning/{hostname}/attempts/{attempt_id}/collect")
async def collect_attempt_logs(
    hostname: str,
    attempt_id: str,
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> dict[str, object]:
    hostname = _validated_hostname(hostname)
    _owned_attempt(_attempt_store(provisioner), hostname, attempt_id)
    try:
        return await provisioner.collect_logs(hostname, attempt_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail="Attempt unavailable or evicted by retention"
        ) from None


@admin_router.get("/provisioning/{hostname}/attempts/{attempt_id}/bundle")
async def download_attempt_logs(
    hostname: str,
    attempt_id: str,
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> StreamingResponse:
    store = _attempt_store(provisioner)
    manifest = _owned_attempt(store, hostname, attempt_id)

    def generate() -> Iterator[bytes]:
        compressor = zlib.compressobj(wbits=31)
        yield compressor.compress((json.dumps({"manifest": manifest}) + "\n").encode())
        after = 0
        exported = 0
        while after < manifest["next_seq"]:
            try:
                page = store.read(attempt_id, after=after)
            except KeyError:
                break  # The export footer reports eviction during download.
            for record in page["records"]:
                if record["seq"] < manifest["next_seq"]:
                    exported += 1
                    yield compressor.compress((json.dumps(record) + "\n").encode())
            after = page["next_offset"]
            if not page["has_more"]:
                break
        expected = manifest["next_seq"] - manifest["dropped_records"]
        footer = {
            "export": {
                "exported_records": exported,
                "expected_records": expected,
                "incomplete": exported != expected,
                "warning": "Records rotated during export"
                if exported != expected
                else None,
            }
        }
        yield compressor.compress((json.dumps(footer) + "\n").encode())
        yield compressor.flush()

    return StreamingResponse(
        generate(),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="provisioning-{manifest["attempt_id"]}.jsonl.gz"',
        },
    )


@admin_router.get("/provisioning/{hostname}/logs")
async def stream_provisioning_logs(
    hostname: str,
    attempt_id: str | None = None,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> StreamingResponse:
    """Stream one attempt with stable IDs, including after a gateway restart."""
    hostname = _validated_hostname(hostname)
    buf = provisioner.log_buffer
    if buf.store is not None:
        store = buf.store
        if last_event_id:
            try:
                resumed_attempt, position = last_event_id.rsplit(":", 1)
                if attempt_id is not None and attempt_id != resumed_attempt:
                    raise ValueError("attempt mismatch")
                attempt_id, after = resumed_attempt, max(after, int(position) + 1)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid Last-Event-ID"
                ) from None
        if attempt_id is None:
            history = (await asyncio.to_thread(store.history, hostname, limit=1))[
                "attempts"
            ]
            if not history:
                raise HTTPException(
                    status_code=404, detail=f"No provisioning log for '{hostname}'"
                )
            attempt_id = history[0]["attempt_id"]
        assert attempt_id is not None
        _owned_attempt(store, hostname, attempt_id)

        async def durable() -> AsyncIterator[str]:
            cursor = after
            while True:
                try:
                    page = await asyncio.to_thread(store.read, attempt_id, after=cursor)
                except KeyError:
                    yield 'event: unavailable\ndata: {"reason":"Attempt evicted by retention"}\n\n'
                    return
                for entry in page["records"]:
                    if entry["seq"] > cursor:
                        gap = buf._gap_entry(entry["seq"] - cursor)
                        yield f"data: {json.dumps(gap)}\n\n"
                    yield f"id: {attempt_id}:{entry['seq']}\ndata: {json.dumps(entry)}\n\n"
                    cursor = entry["seq"] + 1
                if not page["has_more"]:
                    if cursor < page["next_offset"]:
                        yield f"data: {json.dumps(buf._gap_entry(page['next_offset'] - cursor))}\n\n"
                    cursor = page["next_offset"]
                    if page["attempt"]["status"] != "running":
                        yield "event: complete\ndata: {}\n\n"
                        return
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)

        return StreamingResponse(durable(), media_type="text/event-stream")

    if not buf.has(hostname):
        raise HTTPException(
            status_code=404, detail=f"No provisioning log for '{hostname}'"
        )

    async def memory() -> AsyncIterator[str]:
        async for _pos, entry in buf.iter_from(hostname, after):
            yield f"data: {json.dumps(entry)}\n\n"

    return StreamingResponse(memory(), media_type="text/event-stream")


@admin_router.delete("/nodes/{node_id}", status_code=202)
async def teardown_node(
    node_id: str,
    force: bool = False,
    recovery_engine: InferenceEngine | None = None,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
    suspensions: SuspensionStore = Depends(get_placement_suspensions),
) -> TeardownResponse:
    """Trigger teardown of a node (runs in background)."""
    node_id = canonical_hostname(node_id)
    cancelled_identity: ProvisioningIdentity | None = None
    existing = registry.get(node_id)
    if existing is not None and existing.self_setup:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Node '{node_id}' is a self-setup instance; "
                "remove it from the fleet instead of tearing it down"
            ),
        )

    async def persist_suspension() -> None:
        try:
            await suspensions.suspend(node_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("placement_suspend_failed", hostname=node_id)
            raise HTTPException(
                status_code=503,
                detail="Could not suspend automatic placement; teardown was not started",
            ) from exc

    suspended_before_cancel = False
    suspended_detail = "automatic placement is suspended; retry teardown or resume"

    if recovery_engine is not None:
        if not force:
            raise HTTPException(
                status_code=400,
                detail="recovery_engine requires force=true",
            )
        if registry.get(node_id) is not None:
            raise HTTPException(
                status_code=400,
                detail="recovery_engine cannot override a registered node identity",
            )
        try:
            provisioner.validate_endpoint(node_id)
        except EndpointValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Recovery never cancels an active operation. A busy lease means an
        # authoritative provisioning identity already exists or another host
        # lifecycle operation is in progress.
        lease = await provisioner.try_reserve_host(node_id)
        if lease is None:
            raise HTTPException(
                status_code=409,
                detail=f"Host lifecycle operation already in progress for '{node_id}'",
            )
    else:
        active = provisioner.active_provision(node_id)
        if existing is None and active is None:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        if active is not None:
            # This task holds the lifecycle lease. Persist before cancelling it;
            # a store failure must leave active provisioning completely untouched.
            await persist_suspension()
            suspended_before_cancel = True
            try:
                cancelled_identity = await provisioner.cancel_provision(node_id, active)
            except ProvisioningOperationChangedError as exc:
                raise HTTPException(
                    status_code=409, detail=f"{exc}; {suspended_detail}"
                ) from exc
            except Exception as exc:
                logger.exception("teardown_cancel_failed", hostname=node_id)
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not cancel provisioning; {suspended_detail}",
                ) from exc
        try:
            lease = await provisioner.try_reserve_host(node_id)
        except Exception as exc:
            detail = "Could not reserve host for teardown"
            if suspended_before_cancel:
                detail += f"; {suspended_detail}"
            raise HTTPException(status_code=503, detail=detail) from exc
        if lease is None:
            if suspended_before_cancel:
                detail = f"Host '{node_id}' was re-reserved after provisioning cancellation; {suspended_detail}"
            else:
                detail = f"Host lifecycle operation already in progress for '{node_id}'"
            raise HTTPException(status_code=409, detail=detail)

    transferred = False
    try:
        registered_node = registry.get(node_id)
        if registered_node is not None and registered_node.self_setup:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Node '{node_id}' is a self-setup instance; "
                    "remove it from the fleet instead of tearing it down"
                ),
            )
        if recovery_engine is not None and registered_node is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Node '{node_id}' registered while force recovery was waiting; "
                    "retry teardown without recovery_engine"
                ),
            )
        if (
            registered_node is None
            and cancelled_identity is None
            and recovery_engine is None
        ):
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

        if suspended_before_cancel:
            # Resume can win the gap between cancelling the old task and taking
            # its lease. Respect that choice rather than writing another opt-out.
            try:
                still_suspended = node_id in await suspensions.list()
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Could not verify automatic placement suspension; retry teardown or resume",
                ) from exc
            if not still_suspended:
                raise HTTPException(
                    status_code=409,
                    detail="Automatic placement was resumed during teardown; retry teardown if still intended",
                )
        else:
            # Registered nodes and force recovery are validated under the lease
            # before persisting anything. Rejected requests leave no opt-out.
            await persist_suspension()

        async def _teardown_and_cleanup() -> None:
            try:
                await provisioner.teardown(
                    node_id,
                    force=force,
                    provisioning_identity=cancelled_identity,
                    recovery_engine=recovery_engine,
                    lifecycle_lease=lease,
                )
            finally:
                lease.release()

        background = _teardown_and_cleanup()
        try:
            task = provisioner.fire_background(
                background,
                task_name=f"teardown:{node_id}",
            )
        except Exception:
            background.close()
            raise
        task.add_done_callback(lambda _task: lease.release())
        transferred = True
        return TeardownResponse(task_id=node_id)
    finally:
        if not transferred:
            lease.release()


@admin_router.get("/quads/status")
async def get_quads_status(
    poller: QUADSPoller | None = Depends(get_quads_poller),
) -> QUADSStatusResponse:
    """Return QUADS poller staleness for the dashboard status indicator."""
    if poller is None:
        return QUADSStatusResponse(
            status="unavailable", last_sync=None, consecutive_failures=0
        )
    if poller.last_sync is None or poller.consecutive_failures >= 3:
        status = "unavailable"
    elif poller.consecutive_failures >= 1:
        status = "stale"
    else:
        status = "connected"
    return QUADSStatusResponse(
        status=status,
        last_sync=poller.last_sync,
        consecutive_failures=poller.consecutive_failures,
    )


@admin_router.get("/nodes/{hostname}/power")
async def get_power_state(
    hostname: str,
    redfish: RedfishClient | None = Depends(get_redfish_client),
) -> PowerStateResponse:
    """Query current power state of a node's BMC (PWR-04)."""
    if redfish is None:
        raise HTTPException(status_code=503, detail="Redfish not configured")
    hostname = _validated_hostname(hostname)
    try:
        state = await redfish.get_power_state(hostname)
    except RedfishDestinationError as exc:
        raise HTTPException(status_code=400, detail=exc.human_message) from exc
    except RedfishError as exc:
        raise HTTPException(status_code=502, detail=exc.human_message) from exc
    return PowerStateResponse(hostname=hostname, power_state=state)


@admin_router.post("/nodes/{hostname}/power")
async def execute_power_action(
    hostname: str,
    body: PowerActionRequest,
    redfish: RedfishClient | None = Depends(get_redfish_client),
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> PowerStateResponse:
    """Execute a power action on a node's BMC (PWR-01/02/03, D-05).

    Power actions are host-mutating, so the hostname is reserved through the
    host lifecycle coordinator up front and owned through the Redfish call:
    a self-setup node is not yet registered while adoption holds the lease
    (the ``registry.get`` below would miss it), and a concurrent setup or
    teardown must not be power-cycled out from under itself.
    """
    if redfish is None:
        raise HTTPException(status_code=503, detail="Redfish not configured")
    hostname = _validated_hostname(hostname)
    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Host lifecycle operation already in progress for '{hostname}'; "
                "wait for it to finish before sending BMC power actions"
            ),
        )
    try:
        # Re-check under exclusive ownership: adoption can complete while we
        # waited for the lease, so the node may now be registered as
        # self-setup even though the earlier read would have missed it.
        node = registry.get(hostname)
        if node is not None and node.self_setup:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Node '{hostname}' is a self-setup instance; "
                    "QIIP does not own its hardware and cannot send BMC power actions"
                ),
            )
        try:
            final_state = await redfish.power_action(hostname, body.action.value)
        except RedfishDestinationError as exc:
            raise HTTPException(status_code=400, detail=exc.human_message) from exc
        except RedfishError as exc:
            raise HTTPException(status_code=502, detail=exc.human_message) from exc
        return PowerStateResponse(hostname=hostname, power_state=final_state)
    finally:
        lease.release()


@admin_router.get(
    "/nodes/{hostname}/recommendations",
    response_model=RecommendationResponse,
    responses={502: {"description": "LLMFit or SSH failure"}},
)
async def get_recommendations(
    hostname: str,
    runner: LLMFitRunner = Depends(get_llmfit_runner),
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
    poller: QUADSPoller | None = Depends(get_quads_poller),
) -> RecommendationResponse | JSONResponse:
    """Return ranked model recommendations for a node's hardware.

    Recommendations can install llmfit with sudo on the host, so the hostname
    is reserved through the host lifecycle coordinator before the ownership
    check and held through ``runner.recommend()``: a self-setup node is not
    yet registered while adoption holds the lease, so the ``registry.get``
    below would otherwise let software be installed on externally owned
    hardware.
    """
    hostname = _validated_hostname(hostname)
    try:
        provisioner.validate_endpoint(hostname)
    except EndpointValidationError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{hostname}' is not available for recommendations",
        ) from exc

    lease = await provisioner.try_reserve_host(hostname)
    if lease is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Host lifecycle operation already in progress for '{hostname}'; "
                "wait for it to finish before requesting recommendations"
            ),
        )
    try:
        # Re-check under exclusive ownership: adoption can complete while we
        # waited for the lease, so the node may now be registered as
        # self-setup even though the earlier read would have missed it.
        registered_node = registry.get(hostname)
        if registered_node is not None and registered_node.self_setup:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Node '{hostname}' is a self-setup instance; "
                    "QIIP cannot recommend or install software on externally "
                    "owned hardware"
                ),
            )
        registered = registered_node is not None
        quads_available = False
        if poller is not None:
            inventory = {canonical_hostname(host.hostname) for host in poller.hosts}
            available = {
                canonical_hostname(available_host)
                for available_host in poller.available_hostnames
            }
            quads_available = hostname in inventory and hostname in available
        if not registered and not quads_available:
            raise HTTPException(
                status_code=404,
                detail=f"Node '{hostname}' is not available for recommendations",
            )

        try:
            result = await runner.recommend(hostname)
        except LLMFitTimeoutError as exc:
            logger.warning("llmfit_timeout", host=exc.host, timeout=exc.timeout)
            return JSONResponse(
                status_code=502,
                content={"error_type": "timeout", "detail": str(exc)},
            )
        except LLMFitParseError as exc:
            logger.warning(
                "llmfit_parse_error",
                host=hostname,
                reason=exc.reason,
                raw_output=exc.raw_output,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error_type": "parse_error",
                    "detail": f"Failed to parse llmfit output: {exc.reason}",
                },
            )
        except SSHConnectionError as exc:
            logger.warning(
                "llmfit_ssh_connection_error", host=exc.host, reason=exc.reason
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error_type": "connection_error",
                    "detail": f"SSH connection failed: {exc.reason}",
                },
            )
        except RemoteCommandError as exc:
            logger.warning(
                "llmfit_remote_command_error",
                host=exc.host,
                exit_status=exc.exit_status,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error_type": "ssh_error",
                    "detail": f"llmfit exited with status {exc.exit_status}",
                },
            )
        return RecommendationResponse(
            hostname=hostname, system=result.system, models=result.models
        )
    finally:
        lease.release()
