"""Dashboard route for the operations UI.

Per D-01: Dashboard served at /dashboard, separate from /admin/* JSON API.
Per D-02: Client-side fetch -- HTML shell rendered by Jinja2, JS fetches
the /admin/* endpoints.

Viewer contract: every page served here is admin-only, the same authority
as ``/admin/*`` (the HTTP Basic local admin or an admin-role session).
Anonymous visitors get the sign-in page; signed-in non-admins are sent to
``/start``, which forwards the full-access trust list on to ``/profile``.
That is the non-admin surface: ``/start`` for the token home and
``/leaderboard`` for the minimal usage board -- not this fleet page. The
read-only ``/fleet/*`` JSON still answers non-admin viewers (see
``api/fleet.py``), but no page renders it for them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from inference_proxy.api.templating import (
    signin_response,
    templates,
    user_home_redirect,
)
from inference_proxy.config.dependencies import get_settings, viewer_role
from inference_proxy.config.settings import Settings

dashboard_router = APIRouter(
    tags=["dashboard"],
)


def _admin_or_signin(
    request: Request,
    settings: Settings,
) -> tuple[bool, HTMLResponse | RedirectResponse | None]:
    """Return (allowed, response) for admin-only pages.

    Anonymous callers receive the sign-in page and signed-in non-admins are
    redirected to their own page (``/start``); only the response is returned
    when access is denied.
    """
    role = viewer_role(request, settings)
    if role is None:
        return False, signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    if role != "admin":
        return False, user_home_redirect()
    return True, None


@dashboard_router.get("/dashboard", response_class=HTMLResponse, response_model=None)
async def dashboard(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
    """Render the operations dashboard HTML shell."""
    role = viewer_role(request, settings)
    if role is None:
        return signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    if role != "admin":
        return user_home_redirect()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
        },
    )


@dashboard_router.get(
    "/dashboard/nodes/{node_id:path}", response_class=HTMLResponse, response_model=None
)
async def node_detail(
    request: Request,
    node_id: str,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
    """Render per-node detail page with provisioning tasks (admins only).

    Anonymous visitors get the sign-in page; signed-in non-admins are sent
    to their own page (``/start``).
    """
    role = viewer_role(request, settings)
    if role is None:
        return signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    if role != "admin":
        return user_home_redirect()
    return templates.TemplateResponse(
        request=request,
        name="node_detail.html",
        context={
            "node_id": node_id,
            "poll_interval": settings.dashboard.poll_interval,
            "active_page": "dashboard",
        },
    )


@dashboard_router.get("/models", response_class=HTMLResponse, response_model=None)
async def models_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
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


@dashboard_router.get(
    "/dashboard/tokens", response_class=HTMLResponse, response_model=None
)
async def tokens_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
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


@dashboard_router.get(
    "/dashboard/users/{user_id}", response_class=HTMLResponse, response_model=None
)
async def user_detail_page(
    request: Request,
    user_id: int,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
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


@dashboard_router.get(
    "/dashboard/reliability", response_class=HTMLResponse, response_model=None
)
async def reliability_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
    _, denied = _admin_or_signin(request, settings)
    if denied is not None:
        return denied
    return templates.TemplateResponse(
        request=request,
        name="reliability.html",
        context={"active_page": "admin"},
    )


@dashboard_router.get(
    "/dashboard/admin", response_class=HTMLResponse, response_model=None
)
async def admin_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse | RedirectResponse:
    """Render the admin page: admin-only inference servers and admin-role
    management (users table with grant/revoke actions)."""
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
