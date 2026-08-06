"""Dashboard route for the operations UI.

Per D-01: Dashboard served at /dashboard, separate from /admin/* JSON API.
Per D-02: Client-side fetch -- HTML shell rendered by Jinja2, JS fetches /admin/nodes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from inference_proxy.api.templating import templates
from inference_proxy.config.dependencies import get_settings, require_admin_auth
from inference_proxy.config.settings import Settings

dashboard_router = APIRouter(
    tags=["dashboard"],
    dependencies=[Depends(require_admin_auth)],
)


@dashboard_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the operations dashboard HTML shell."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
        },
    )


@dashboard_router.get("/dashboard/nodes/{node_id:path}", response_class=HTMLResponse)
async def node_detail(
    request: Request,
    node_id: str,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render per-node detail page with provisioning tasks."""
    return templates.TemplateResponse(
        request=request,
        name="node_detail.html",
        context={
            "node_id": node_id,
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
        },
    )
