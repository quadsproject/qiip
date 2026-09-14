"""Dashboard route for the operations UI.

Per D-01: Dashboard served at /dashboard, separate from /admin/* JSON API.
Per D-02: Client-side fetch -- HTML shell rendered by Jinja2, JS fetches
/admin/nodes (admins) or /fleet/nodes (signed-in non-admins).

Viewer contract (RFE hidden servers + admin roles):

- The fleet page (``/dashboard``) is available to every authenticated
  viewer: the HTTP Basic local admin, a signed-in Google user, or a Google
  user granted the admin role. Anonymous visitors get the sign-in page with
  the "Local Admin" and "Google Auth" options.
- Node detail, model catalog, token dashboards, and the admin page are
  admin-only (local admin or admin-role user); other viewers get the
  sign-in page with an access notice.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from inference_proxy.api.templating import signin_response, templates
from inference_proxy.config.dependencies import get_settings, viewer_role
from inference_proxy.config.settings import Settings

dashboard_router = APIRouter(
    tags=["dashboard"],
)

_ADMIN_REQUIRED_NOTICE = "Administrator access required to view this page."


def _admin_or_signin(
    request: Request,
    settings: Settings,
) -> tuple[bool, HTMLResponse | None]:
    """Return (allowed, response) for admin-only pages.

    Anonymous and non-admin signed-in callers receive the sign-in page;
    only the response is returned when access is denied.
    """
    role = viewer_role(request, settings)
    if role is None:
        return False, signin_response(request)
    if role != "admin":
        return False, signin_response(request, notice=_ADMIN_REQUIRED_NOTICE)
    return True, None


@dashboard_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the operations dashboard HTML shell."""
    role = viewer_role(request, settings)
    if role is None:
        return signin_response(request)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
            "viewer_role": role,
        },
    )


@dashboard_router.get("/dashboard/nodes/{node_id:path}", response_class=HTMLResponse)
async def node_detail(
    request: Request,
    node_id: str,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render per-node detail page with provisioning tasks."""
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="node_detail.html",
        context={
            "node_id": node_id,
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
        },
    )


@dashboard_router.get("/models", response_class=HTMLResponse)
async def models_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the model catalog page."""
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "models",
        },
    )


@dashboard_router.get("/dashboard/tokens", response_class=HTMLResponse)
async def tokens_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the admin token-management dashboard shell."""
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="tokens.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "tokens",
        },
    )


@dashboard_router.get("/dashboard/users/{user_id}", response_class=HTMLResponse)
async def user_detail_page(
    request: Request,
    user_id: int,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the admin per-user token and usage detail shell."""
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="user_detail.html",
        context={
            "user_id": user_id,
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "tokens",
        },
    )


@dashboard_router.get("/dashboard/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the admin page: hidden inference servers and admin users."""
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "admin",
        },
    )
