"""OAuth login/logout and current-user endpoints.

Flow (RFC 6749 Authorization Code + OIDC):

    /auth/login    -> 302 to the provider's authorization endpoint
    /auth/callback -> provider redirects back with a code; the auth plugin
                      exchanges it, this router verifies/upserts the user,
                      and signs the session cookie
    /auth/logout   -> clears the session cookie
    /auth/me       -> JSON identity for the signed-in user (or 401)

All failure paths return to the profile page with a short ``error`` query
parameter that the profile UI surfaces, so a failed or denied login never
leaves the user on a bare error screen.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated
from urllib.parse import urlsplit

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ValidationError

from inference_proxy.auth.allowlist import (
    AllowlistUnavailableError,
    SSOAllowlist,
    enforce_allowlist,
)
from inference_proxy.auth.dependencies import (
    get_auth_plugin,
    get_auth_store,
    get_sso_allowlist,
    require_profile_user,
)
from inference_proxy.auth.models import PublicUser, User
from inference_proxy.auth.scopes import is_full_access
from inference_proxy.auth.session import (
    clear_local_admin_session,
    clear_session_user,
    get_session_user_id,
    set_local_admin_session,
    set_session_user,
)
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import _credentials_match, get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.plugins.interfaces.auth import AuthCallbackError, AuthPlugin

logger = structlog.get_logger()

auth_router = APIRouter(prefix="/auth", tags=["auth"])

_PROFILE_HOME = "/profile"
_DASHBOARD_HOME = "/dashboard"


def _error_redirect(error: str) -> RedirectResponse:
    """Redirect back to the profile page carrying a short error code."""
    return RedirectResponse(f"{_PROFILE_HOME}?error={error}", status_code=302)


class LocalAdminLogin(BaseModel):
    """JSON body for the local-admin sign-in form (JSON-only CSRF boundary)."""

    username: str
    password: str


def _oauth_redirect_uri(request: Request, settings: Settings) -> str | None:
    """Return the provider callback URI for this request, host-aware.

    Multi-name deployments (e.g. one wildcard certificate serving several
    hostnames) must return the user to the *same* hostname they started the
    flow from; otherwise the OAuth state nonce — stored in the host-scoped
    session cookie at ``/auth/login`` — is invisible to the callback and
    the sign-in fails. When the request host is allowlisted (including the
    ``redirect_uri`` host itself) the callback URI is therefore rebuilt
    from the request host; any other host falls back to the configured
    ``redirect_uri`` so a never-trusted ``Host`` header can never steer
    the flow.
    """
    configured = settings.oauth.redirect_uri
    if configured is None:
        return None
    configured_host = urlsplit(configured).hostname
    allowed = {host.lower() for host in settings.oauth.allowed_redirect_hosts}
    if configured_host:
        allowed.add(configured_host.lower())
    host = request.url.hostname
    if host and host.lower() in allowed:
        # Preserve the request port and the configured callback path. A bare
        # ``{scheme}://{host}/auth/callback`` rebuild dropped the port
        # (``.hostname`` omits it) and any configured path prefix, breaking
        # previously valid OAuth redirects (e.g. ``http://localhost:5000``
        # became ``http://localhost``). Use base_url semantics —
        # ``scheme://host[:port]`` + the configured path.
        path = urlsplit(configured).path or "/auth/callback"
        return f"{str(request.base_url).rstrip('/')}{path}"
    return configured


@auth_router.get("/local-admin")
async def local_admin_login_page() -> RedirectResponse:
    """Send direct visitors to the sign-in page (no Basic challenge popup)."""
    return RedirectResponse(_DASHBOARD_HOME, status_code=302)


@auth_router.post("/local-admin")
async def local_admin_login(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Sign in as the local admin through the sign-in page form.

    Accepts a JSON body (``{"username": ..., "password": ...}``) — the same
    JSON-only state-changing convention as the admin API, so the login cannot
    be CSRF'd by a cross-origin form (a browser form can only send
    ``text/plain``-style bodies that never pass preflight). On success a
    signed session cookie is set and the browser is redirected to the fleet
    page; invalid credentials return 401 with a visible error message.
    """
    if settings.auth.session_secret is None:
        raise HTTPException(
            status_code=503,
            detail="Session storage is not configured; cannot sign in via the form",
        )
    media_type = request.headers.get("content-type", "").partition(";")[0].lower()
    if media_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Login requires Content-Type: application/json",
        )
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail="Login body must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Login body must be a JSON object")
    try:
        body = LocalAdminLogin.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Login body must be a JSON object"
        ) from exc
    username = body.username.strip()
    password = body.password
    if not _credentials_match(username, password, settings):
        logger.warning("local admin login rejected", username=username)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    set_local_admin_session(request, settings.auth.session_ttl_seconds)
    logger.info("local admin signed in")
    return RedirectResponse(_DASHBOARD_HOME, status_code=302)


@auth_router.get("/login")
async def oauth_login(
    request: Request,
    auth: Annotated[AuthPlugin, Depends(get_auth_plugin)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedirectResponse:
    """Start the provider Authorization Code flow (302 to the provider).

    When a session already exists the request short-circuits to the profile
    page instead of forcing a re-authentication round-trip.
    """
    if get_session_user_id(request) is not None:
        return RedirectResponse(_PROFILE_HOME, status_code=302)
    redirect_uri = _oauth_redirect_uri(request, settings)
    if redirect_uri is None:
        raise HTTPException(
            status_code=503, detail="OAuth redirect URI is not configured"
        )
    return await auth.start_login(request, redirect_uri)


@auth_router.get("/callback")
async def oauth_callback(
    request: Request,
    auth: Annotated[AuthPlugin, Depends(get_auth_plugin)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    allowlist: Annotated[SSOAllowlist | None, Depends(get_sso_allowlist)] = None,
) -> RedirectResponse:
    """Handle the provider's post-login redirect, upsert the user, sign the session.

    The auth plugin validates the ID-token claims (authlib validates the
    JWT signature, issuer, audience, and nonce/state) and returns a
    normalized identity. Policy applied here is provider-neutral: email
    presence, optional email-verification requirement, the optional
    hosted-domain allowlist, and the optional SSO user whitelist. The
    whitelist check runs before the user row is created and before the
    session is signed, so a denied login never persists an account; an
    unavailable whitelist fails closed with a redirect.
    """
    try:
        identity = await auth.complete_login(request)
    except AuthCallbackError as exc:
        logger.warning("oauth callback rejected", code=exc.code)
        return _error_redirect(exc.code)

    email = identity.email
    if not email or not identity.sub:
        logger.warning("oauth callback empty identity")
        return _error_redirect("no_profile")

    if settings.auth.require_email_verification and not identity.email_verified:
        logger.warning("oauth callback unverified email", email=email)
        return _error_redirect("unverified_email")

    allowed_domains = [domain.lower() for domain in settings.oauth.allowed_domains]
    domain = email.rsplit("@", 1)[-1].lower() if "@" in email else ""
    if allowed_domains and domain not in allowed_domains:
        logger.warning("oauth callback domain not allowed", email=email)
        return _error_redirect("domain_not_allowed")

    if settings.auth.enforce_sso_whitelist and not is_full_access(email, settings):
        try:
            allowed = await enforce_allowlist(email, allowlist)
        except AllowlistUnavailableError:
            logger.warning("oauth callback whitelist unavailable", email=email)
            return _error_redirect("allowlist_unavailable")
        if not allowed:
            logger.warning("oauth callback user not whitelisted", email=email)
            return _error_redirect("not_whitelisted")

    # ponytail: google is the only provider; scope the stored subject by
    # issuer when a second auth provider lands (store keyed by google_sub).
    user = await asyncio.to_thread(
        store.upsert_google_user,
        google_sub=identity.sub,
        email=email,
        name=identity.name,
        picture=identity.picture,
    )
    set_session_user(request, user.id, settings.auth.session_ttl_seconds)
    logger.info("user signed in", user_id=user.id, email=email)
    return RedirectResponse(_PROFILE_HOME, status_code=302)


@auth_router.post("/logout")
async def oauth_logout(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedirectResponse:
    """Clear the session cookie (POST-only to avoid trivially CSRF'd logouts).

    When sessions are not configured (``auth.session_secret`` unset) there
    is no cookie to clear; redirect home instead of touching the session
    machinery that is absent.
    """
    if settings.auth.session_secret is not None:
        clear_session_user(request)
        clear_local_admin_session(request)
    return RedirectResponse(_DASHBOARD_HOME, status_code=302)


@auth_router.get("/me")
async def oauth_me(user: Annotated[User, Depends(require_profile_user)]) -> PublicUser:
    """Return the signed-in user's public identity, or 401 when anonymous."""
    return PublicUser(
        id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture,
        is_admin=user.is_admin,
    )
