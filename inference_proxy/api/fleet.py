"""Fleet endpoint for signed-in viewers (admin-only node visibility contract).

``GET /fleet/nodes`` backs the fleet page for non-admin signed-in users
(and local-admin sessions via Basic): it returns registered nodes with
admin-only servers removed and operational actions stripped. Admins keep
the full operational view through ``GET /admin/nodes``, so the fleet
endpoint never needs to return admin-only identity — admin-only servers
stay off this surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from inference_proxy.config.dependencies import (
    get_unified_node_service,
    require_fleet_viewer,
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
) -> list[AdminNodeResponse]:
    """Return the fleet view for a signed-in (non-admin) viewer."""
    return service.get_unified_nodes(viewer_admin=False)
