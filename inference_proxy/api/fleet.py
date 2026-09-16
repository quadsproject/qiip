"""Fleet endpoints for signed-in viewers (admin-only node visibility contract).

``GET /fleet/nodes`` backs the fleet page for non-admin signed-in users
(and local-admin sessions via Basic): it returns registered nodes with
admin-only servers removed, operational actions stripped, and nodes owned
by another user excluded (RFE-107 privacy, matching ``/v1/models`` and the
endpoint picker). Admins keep the full operational view through
``GET /admin/nodes``, so the fleet endpoint never needs to return
admin-only identity — admin-only servers stay off this surface.

The per-node read-only surface (``/fleet/nodes/{node_id}`` plus its tasks
and provisioning-log stream) backs the read-only node detail page: a
signed-in non-admin can inspect a node their fleet view allows, but every
operational endpoint stays behind ``require_admin_auth``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from inference_proxy.config.dependencies import (
    get_provisioner,
    get_unified_node_service,
    require_fleet_viewer,
    require_fleet_viewer_email,
)
from inference_proxy.models.admin import AdminNodeResponse, TaskStatusResponse
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.services.unified_nodes import UnifiedNodeService

fleet_router = APIRouter(
    prefix="/fleet",
    tags=["fleet"],
    dependencies=[Depends(require_fleet_viewer)],
)

# Credential-shaped substrings that must never reach a non-admin viewer:
# provisioning failures are surfaced verbatim (task error fields, log
# entries), and engine-start commands can embed HF_TOKEN or similar. The
# provisioner shell-quotes a value that carries a trailing newline
# (e.g. ``HF_TOKEN='token\n'``), so the value may be single- or
# double-quoted; the unquoted branch keeps the older (bare) assignments.
_SECRET_ASSIGN_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_]*(?:TOKEN|PASSWORD|SECRET|CREDENTIAL|APIKEY)[A-Za-z0-9_]*)\s*[=:]\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;\"']+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_URL_CREDS_RE = re.compile(r"(?i)(://[^/\s:@]+:)[^@\s]+(@)")


def redact_secrets(text: str) -> str:
    """Mask credential-like substrings in free-form text.

    Keeps the key name for diagnosability (``HF_TOKEN=[REDACTED]``) but
    removes the value, so a failed command log cannot disclose tokens or
    passwords to a signed-in non-admin viewer of a node they may inspect.
    """
    if not text:
        return text
    text = _SECRET_ASSIGN_RE.sub(lambda m: f"{m.group('key')}=[REDACTED]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _URL_CREDS_RE.sub(r"\1[REDACTED]\2", text)
    return text


def _redact_node(node: AdminNodeResponse) -> AdminNodeResponse:
    """Redact credential-shaped node error text before exposing to viewers."""
    if node.error:
        return node.model_copy(update={"error": redact_secrets(node.error)})
    return node


def _visible_node(
    service: UnifiedNodeService,
    node_id: str,
    viewer_email: str | None,
) -> AdminNodeResponse:
    """Return the node for a non-admin viewer, or 404 when not visible.

    The same visibility contract as ``/fleet/nodes`` applies (no admin-only
    nodes, no nodes owned by someone else, no absent nodes), so the
    read-only detail surface can never disclose more than the fleet list.
    """
    node = next(
        (
            n
            for n in service.get_unified_nodes(
                viewer_admin=False, viewer_email=viewer_email
            )
            if n.node_id == node_id
        ),
        None,
    )
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@fleet_router.get("/nodes", response_model=list[AdminNodeResponse])
async def fleet_nodes(
    service: UnifiedNodeService = Depends(get_unified_node_service),
    viewer_email: str | None = Depends(require_fleet_viewer_email),
) -> list[AdminNodeResponse]:
    """Return the fleet view for a signed-in (non-admin) viewer."""
    return [
        _redact_node(node)
        for node in service.get_unified_nodes(
            viewer_admin=False, viewer_email=viewer_email
        )
    ]


@fleet_router.get("/nodes/{node_id}", response_model=AdminNodeResponse)
async def fleet_node_detail(
    node_id: str,
    service: UnifiedNodeService = Depends(get_unified_node_service),
    viewer_email: str | None = Depends(require_fleet_viewer_email),
) -> AdminNodeResponse:
    """Return the read-only node detail for a signed-in (non-admin) viewer."""
    return _redact_node(_visible_node(service, node_id, viewer_email))


@fleet_router.get("/nodes/{node_id}/tasks", response_model=list[TaskStatusResponse])
async def fleet_node_tasks(
    node_id: str,
    service: UnifiedNodeService = Depends(get_unified_node_service),
    viewer_email: str | None = Depends(require_fleet_viewer_email),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> list[TaskStatusResponse]:
    """Return provisioning tasks for a node the viewer is allowed to inspect."""
    _visible_node(service, node_id, viewer_email)
    results = await provisioner.list_tasks_raw()
    tasks: list[TaskStatusResponse] = []
    for value_bytes, _metadata in results:
        try:
            data = json.loads(value_bytes)
            task = TaskStatusResponse(**data)
        except (json.JSONDecodeError, ValueError):
            continue
        if task.hostname == node_id:
            if task.error:
                task = task.model_copy(update={"error": redact_secrets(task.error)})
            tasks.append(task)
    return tasks


@fleet_router.get("/nodes/{node_id}/logs")
async def fleet_node_logs(
    node_id: str,
    service: UnifiedNodeService = Depends(get_unified_node_service),
    viewer_email: str | None = Depends(require_fleet_viewer_email),
    provisioner: NodeProvisioner = Depends(get_provisioner),
) -> StreamingResponse:
    """Stream provisioning log entries as SSE for a visible node.

    Mirrors ``GET /admin/provisioning/{hostname}/logs`` for the read-only
    viewer: no provisioning buffer, no configuration, just the existing
    installation log — with credential-shaped content redacted, including
    retained (already-buffered) entries.
    """
    _visible_node(service, node_id, viewer_email)
    buf = provisioner.log_buffer
    if not buf.has(node_id):
        raise HTTPException(
            status_code=404,
            detail=f"No provisioning log for '{node_id}'",
        )

    async def _generate() -> AsyncIterator[str]:
        async for _pos, entry in buf.iter_from(node_id):
            redacted = dict(entry)
            redacted["msg"] = redact_secrets(str(entry.get("msg", "")))
            yield f"data: {json.dumps(redacted)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
