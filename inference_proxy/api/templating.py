"""Shared Jinja environment with content-versioned static asset URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_BASE_DIR = Path(__file__).resolve().parent.parent
_STATIC_DIR = (_BASE_DIR / "static").resolve()


def static_asset_url(request: Request, path: str) -> str:
    """Return a static URL whose query changes with the file contents.

    The dashboard JavaScript files are interdependent. Content versions keep a
    newly rendered HTML shell from running against an older browser-cached
    script generation after a deployment.
    """
    asset = (_STATIC_DIR / path).resolve()
    try:
        asset.relative_to(_STATIC_DIR)
    except ValueError as exc:
        raise ValueError("static asset path escapes the static directory") from exc
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
    return f"{request.url_for('static', path=path)}?v={digest}"


def signin_response(
    request: Request,
    notice: str = "",
    *,
    oauth_enabled: bool = True,
    sessions_available: bool = True,
) -> HTMLResponse:
    """Render the sign-in page, gating each option by what is configured.

    The local admin option is a username/password form (no native Basic
    challenge popup) and needs ``auth.session_secret``; the Google option
    starts the OAuth flow and needs the OAuth integration enabled. Options
    that would fail (404/503) are hidden and replaced by an HTTP-Basic hint.
    """
    return templates.TemplateResponse(
        request=request,
        name="signin.html",
        context={
            "notice": notice,
            "active_page": "dashboard",
            "oauth_enabled": oauth_enabled,
            "sessions_available": sessions_available,
        },
    )


def viewer_is_admin(request: Request) -> bool:
    """Jinja global: whether the current request viewer is an admin.

    Resolves the same identity chain as the page gates: HTTP Basic local
    admin, or a signed-in Google user carrying the admin role.
    """
    from inference_proxy.config.dependencies import get_settings, viewer_role

    return viewer_role(request, get_settings()) == "admin"


def viewer_signed_in(request: Request) -> bool:
    """Jinja global: whether the current request viewer is authenticated.

    True for the HTTP Basic local admin, a local-admin session, or any
    signed-in Google user — regardless of admin role.
    """
    from inference_proxy.config.dependencies import get_settings, viewer_role

    return viewer_role(request, get_settings()) is not None


def viewer_identity(request: Request) -> str | None:
    """Human-readable identity of the current viewer, for the navbar.

    Signed-in Google users see their email; the local admin (HTTP Basic or
    a local-admin session) is labelled "Local Admin". None when anonymous.
    """
    from inference_proxy.auth.session import get_session_user_id
    from inference_proxy.auth.store import AuthStore
    from inference_proxy.config.dependencies import get_settings, viewer_role

    user_id = get_session_user_id(request)
    if user_id is not None:
        store: AuthStore | None = getattr(request.app.state, "auth_store", None)
        if store is not None:
            user = store.get_user(user_id)
            if user is not None:
                return user.email
        return None
    if viewer_role(request, get_settings()) == "admin":
        return "Local Admin"
    return None


templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
templates.env.globals["static_asset_url"] = static_asset_url
templates.env.globals["viewer_is_admin"] = viewer_is_admin
templates.env.globals["viewer_signed_in"] = viewer_signed_in
templates.env.globals["viewer_identity"] = viewer_identity
