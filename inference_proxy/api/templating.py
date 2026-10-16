"""Shared Jinja environment with content-versioned static asset URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from inference_proxy.config.settings import Settings
from inference_proxy.plugins.interfaces.auth import AuthPlugin

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


def _google_auth_available(request: Request) -> bool:
    """Return True when a loaded auth plugin named ``google`` is configured.

    The Google button links to ``/auth/login?provider=google``, which
    resolves a loaded plugin by name over ``get_plugins_by_type(AuthPlugin)``
    (``_configured_auth_plugins``), so this mirrors that rule instead of the
    manager key (which equals ``auth.<name>`` only for built-ins). A plugin
    that is disabled or unconfigured is not loaded at all, or reports itself
    unconfigured. The template combines this with the OAuth integration's own
    precondition (``oauth_enabled``), which the route needs for every
    provider: without ``redirect_uri`` ``/auth/login`` answers 503.
    """
    manager = getattr(request.app.state, "plugin_manager", None)
    if manager is None:
        return False
    return any(
        isinstance(plugin, AuthPlugin)
        and plugin.name == "google"
        and plugin.is_configured()
        for plugin in manager.get_plugins_by_type(AuthPlugin)
    )


def _local_auth_available(request: Request) -> bool:
    """Return True when the opt-in local OIDC auth plugin is loaded and configured.

    The plugin is opt-in (off unless ``plugins.config`` enables it and
    supplies the provider discovery URL), so a default deployment renders
    the sign-in page exactly as before. The Local Login form's credential
    resolver reads the plugin by its qualified key, so the label follows the
    same lookup: a different plugin that merely happens to be named
    ``internal_oidc`` must not advertise a form that ignores it.
    """
    manager = getattr(request.app.state, "plugin_manager", None)
    if manager is None:
        return False
    plugin = manager.get_plugin("auth.internal_oidc")
    return isinstance(plugin, AuthPlugin) and plugin.is_configured()


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
    starts the OAuth flow and needs the Google provider plugin loaded and
    configured *and* the OAuth integration configured (the credential triple
    alone is not enough when the provider is disabled and replaced, and a
    configured replacement alone is not enough without ``redirect_uri``); the
    local OIDC option starts the local provider flow and needs the opt-in
    auth.internal_oidc plugin loaded. Options that would fail (404/503) are
    hidden and replaced by an HTTP-Basic hint.
    """
    return templates.TemplateResponse(
        request=request,
        name="signin.html",
        context={
            "notice": notice,
            "active_page": "dashboard",
            "oauth_enabled": oauth_enabled,
            "google_enabled": _google_auth_available(request),
            "local_auth_enabled": _local_auth_available(request),
            "sessions_available": sessions_available,
        },
    )


# Normal (non-admin) users get one page: the onboarding / token home. Every
# operations page sends them there instead of rendering.
USER_HOME = "/start"


def user_home_redirect() -> RedirectResponse:
    """Redirect a signed-in non-admin to their home page."""
    return RedirectResponse(USER_HOME, status_code=302)


def _request_settings(request: Request) -> Settings:
    """Return the settings the page handlers resolve for this app.

    ``create_app(settings=...)`` stores the resolved instance on
    ``app.state`` (and overrides the ``get_settings`` dependency with it),
    so the Jinja navbar globals resolve the *same* settings as the routes —
    an injected instance must never be shadowed by environment credentials
    read through the cache, and missing environment settings must not make
    template rendering fail.
    """
    from inference_proxy.config.dependencies import get_settings

    state = getattr(request.app, "state", None) if hasattr(request, "app") else None
    resolved = getattr(state, "settings", None) if state is not None else None
    if isinstance(resolved, Settings):
        return resolved
    return get_settings()


def leaderboard_redirect(request: Request) -> str | None:
    """Where a viewer who cannot render ``/leaderboard`` is sent, else None.

    One rule for the page route and the navbar link, so the two never
    disagree: an admin with a browser session sees the non-admin ranking, a
    normal user without full access sees the page, the HTTP Basic admin is
    sent to ``/dashboard`` (no cookie for the page's own requests) and the
    full-access user to ``/profile`` (their token manager lives there, and
    the page's onboarding endpoints refuse them). Anonymous visitors get the
    sign-in page from the route itself, not a redirect from here.
    """
    from inference_proxy.auth.session import (
        get_local_admin_session,
        get_session_user_id,
    )
    from inference_proxy.config.dependencies import (
        session_user_has_full_access,
        viewer_role,
    )

    settings = _request_settings(request)
    role = viewer_role(request, settings)
    if (
        role == "admin"
        and get_session_user_id(request) is None
        and not get_local_admin_session(request)
    ):
        return "/dashboard"
    if role == "user" and session_user_has_full_access(request, settings):
        return "/profile"
    return None


def viewer_is_admin(request: Request) -> bool:
    """Jinja global: whether the current request viewer is an admin.

    Resolves the same identity chain as the page gates: HTTP Basic local
    admin, or a signed-in Google user carrying the admin role.
    """
    from inference_proxy.config.dependencies import viewer_role

    return viewer_role(request, _request_settings(request)) == "admin"


def viewer_signed_in(request: Request) -> bool:
    """Jinja global: whether the current request viewer is authenticated.

    True for the HTTP Basic local admin, a local-admin session, or any
    signed-in Google user — regardless of admin role.
    """
    from inference_proxy.config.dependencies import viewer_role

    return viewer_role(request, _request_settings(request)) is not None


def viewer_identity(request: Request) -> str | None:
    """Human-readable identity of the current viewer, for the navbar.

    Signed-in Google users see their email; the local admin (HTTP Basic or
    a local-admin session) is labelled "Local Admin". None when anonymous.
    """
    from inference_proxy.auth.session import get_session_user_id
    from inference_proxy.auth.store import AuthStore
    from inference_proxy.config.dependencies import viewer_role

    user_id = get_session_user_id(request)
    if user_id is not None:
        store: AuthStore | None = getattr(request.app.state, "auth_store", None)
        if store is not None:
            user = store.get_user(user_id)
            if user is not None:
                return user.email
        return None
    if viewer_role(request, _request_settings(request)) == "admin":
        return "Local Admin"
    return None


def leaderboard_link_visible(request: Request) -> bool:
    """Jinja global: whether the Leaderboard link leads anywhere for the viewer.

    Anonymous visitors keep the link (the page itself serves them the
    sign-in form); viewers the page redirects do not, so the navbar never
    offers a link that bounces back.
    """
    return leaderboard_redirect(request) is None


templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
templates.env.globals["static_asset_url"] = static_asset_url
templates.env.globals["viewer_is_admin"] = viewer_is_admin
templates.env.globals["viewer_signed_in"] = viewer_signed_in
templates.env.globals["viewer_identity"] = viewer_identity
templates.env.globals["leaderboard_link_visible"] = leaderboard_link_visible
