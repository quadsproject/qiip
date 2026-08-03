"""Admin API for operational visibility into the gateway.

Per D-05: Endpoints under /admin namespace, separate from /v1 proxy API.
Per D-06: Separate APIRouter in api/admin.py with prefix="/admin".
Per METR-03: Node entries include identity, health, active connections,
and circuit breaker state for the operations dashboard.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from inference_proxy.config.dependencies import (
    get_catalog_service,
    get_download_service,
    get_llmfit_runner,
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
    AdminMetricsResponse,
    AdminNodeResponse,
    DownloadRequest,
    DownloadStatusResponse,
    PowerActionRequest,
    PowerStateResponse,
    QUADSStatusResponse,
    RecommendationResponse,
    SetupRequest,
    SetupResponse,
    TaskStatusResponse,
    TeardownResponse,
)
from inference_proxy.models.endpoint import EndpointValidationError
from inference_proxy.models.node import NodeStatus
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningCapacityError,
    ProvisioningError,
)
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
    }
)
_SETUP_RETRYABLE_STATUSES = frozenset(
    {
        NodeStatus.PROVISIONING,
        NodeStatus.FAILED,
        NodeStatus.UNKNOWN,
        NodeStatus.DRAINING,
    }
)

# Regex from SetupRequest.validate_hostname — reused for path-parameter validation
_HOSTNAME_RE = re.compile(r"[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?")


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
    if result.incomplete_count or result.unverifiable_count:
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
    result = await svc.trigger_download(body.repo_id)
    response.status_code = 202 if result.started else 200
    return result.status


@admin_router.get("/models/downloads")
async def list_downloads(
    svc: DownloadService = Depends(get_download_service),
) -> list[DownloadStatusResponse]:
    """Return status of all tracked downloads (DL-03)."""
    return svc.get_all_statuses()


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

    try:
        provisioner.validate_endpoint(hostname)
        provisioner.validate_setup_configuration()
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

        async def _provision_and_cleanup() -> None:
            try:
                await provisioner.provision(
                    hostname,
                    managed=body.managed,
                    model=body.model,
                    engine=body.engine,
                    lifecycle_lease=lease,
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


@admin_router.get("/provisioning/{hostname}/logs")
async def stream_provisioning_logs(
    hostname: str,
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> StreamingResponse:
    """Stream provisioning log entries as SSE events.

    If provisioning is in progress, keeps the connection open and
    streams live.  If complete/failed, dumps all entries and closes.
    Returns 404 if no log exists for the hostname.
    """
    hostname = _validated_hostname(hostname)
    buf = provisioner.log_buffer
    if not buf.has(hostname):
        raise HTTPException(
            status_code=404,
            detail=f"No provisioning log for '{hostname}'",
        )

    async def _generate() -> AsyncIterator[str]:
        async for _pos, entry in buf.iter_from(hostname):
            data = json.dumps(entry)
            yield f"data: {data}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@admin_router.delete("/nodes/{node_id}", status_code=202)
async def teardown_node(
    node_id: str,
    force: bool = False,
    registry: NodeRegistry = Depends(get_registry),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> TeardownResponse:
    """Trigger teardown of a node (runs in background)."""
    node_id = canonical_hostname(node_id)
    cancelled_provision = await provisioner.cancel_active_provision(node_id)
    lease = await provisioner.try_reserve_host(node_id)
    if lease is None:
        if cancelled_provision is not None:
            detail = (
                f"Host '{node_id}' was re-reserved after provisioning "
                "cancellation; wait for the current operation to finish and "
                "retry teardown"
            )
        else:
            detail = f"Host lifecycle operation already in progress for '{node_id}'"
        raise HTTPException(
            status_code=409,
            detail=detail,
        )

    transferred = False
    try:
        if registry.get(node_id) is None and cancelled_provision is None:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

        async def _teardown_and_cleanup() -> None:
            try:
                await provisioner.teardown(
                    node_id,
                    force=force,
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
) -> PowerStateResponse:
    """Execute a power action on a node's BMC (PWR-01/02/03, D-05)."""
    if redfish is None:
        raise HTTPException(status_code=503, detail="Redfish not configured")
    hostname = _validated_hostname(hostname)
    try:
        final_state = await redfish.power_action(hostname, body.action.value)
    except RedfishDestinationError as exc:
        raise HTTPException(status_code=400, detail=exc.human_message) from exc
    except RedfishError as exc:
        raise HTTPException(status_code=502, detail=exc.human_message) from exc
    return PowerStateResponse(hostname=hostname, power_state=final_state)


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
    """Return ranked model recommendations for a node's hardware."""
    hostname = _validated_hostname(hostname)
    try:
        provisioner.validate_endpoint(hostname)
    except EndpointValidationError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{hostname}' is not available for recommendations",
        ) from exc

    registered = registry.get(hostname) is not None
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
        logger.warning("llmfit_ssh_connection_error", host=exc.host, reason=exc.reason)
        return JSONResponse(
            status_code=502,
            content={
                "error_type": "connection_error",
                "detail": f"SSH connection failed: {exc.reason}",
            },
        )
    except RemoteCommandError as exc:
        logger.warning(
            "llmfit_remote_command_error", host=exc.host, exit_status=exc.exit_status
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
