"""Fleet endpoint for signed-in viewers (admin-only node visibility contract).

``GET /fleet/nodes`` backs the fleet page for non-admin signed-in users
(and local-admin sessions via Basic): it returns registered nodes with
admin-only servers removed, operational actions stripped, and nodes owned
by another user excluded (RFE-107 privacy, matching ``/v1/models`` and the
endpoint picker). Admins keep the full operational view through
``GET /admin/nodes``, so the fleet endpoint never needs to return
admin-only identity — admin-only servers stay off this surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from inference_proxy.config.dependencies import (
    get_unified_node_service,
    require_fleet_viewer,
    require_fleet_viewer_email,
)
from inference_proxy.models.admin import AdminNodeResponse
from inference_proxy.services.unified_nodes import UnifiedNodeService

fleet_router = APIRouter(
    prefix="/fleet",
    tags=["fleet"],
    dependencies=[Depends(require_fleet_viewer)],
)


@fleet_router.get("/nodes", response_model=list[AdminNodeResponse])
async def fleet_nodes(
    service: UnifiedNodeService = Depends(get_unified_node_service),
    viewer_email: str | None = Depends(require_fleet_viewer_email),
) -> list[AdminNodeResponse]:
    """Return the fleet view for a signed-in (non-admin) viewer."""
    return service.get_unified_nodes(viewer_admin=False, viewer_email=viewer_email)
