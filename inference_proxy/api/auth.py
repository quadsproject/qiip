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

from inference_proxy.api.templating import USER_HOME
from inference_proxy.auth.allowlist import SSOAllowlist
from inference_proxy.auth.dependencies import (
    AUTH_PROVIDER_SESSION_KEY,
    get_auth_plugin,
    get_auth_store,
    get_sso_allowlist,
    require_profile_user,
)
from inference_proxy.auth.models import PublicUser, User
from inference_proxy.auth.session import (
    clear_local_admin_session,
    clear_session_user,
    get_session_user_id,
    set_local_admin_session,
    set_session_user,
)
from inference_proxy.auth.signin_policy import check_signin_policy
from inference_proxy.auth.store import AccountConflictError, AuthStore
from inference_proxy.config.dependencies import (
    _credentials_match,
    get_settings,
    viewer_role,
)
from inference_proxy.config.settings import Settings
from inference_proxy.plugins.interfaces.auth import AuthCallbackError, AuthPlugin

logger = structlog.get_logger()

auth_router = APIRouter(prefix="/auth", tags=["auth"])

# Sign-in policy rejections surface as 403 details on the local form (the
# codes come from check_signin_policy; the OIDC callback redirects instead).
_POLICY_ERROR_MESSAGES = {
    "unverified_email": "This account has an unverified email address and cannot sign in.",
    "domain_not_allowed": "This account is not in the allowed domains for this gateway.",
    "allowlist_unavailable": "The whitelist service is unavailable. Please try again later.",
    "not_whitelisted": "This account is not on the gateway whitelist.",
}

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
    store: Annotated[AuthStore, Depends(get_auth_store)],
    allowlist: Annotated[SSOAllowlist | None, Depends(get_sso_allowlist)] = None,
) -> Response:
    """Sign in through the sign-in page's local form.

    Accepts a JSON body (``{"username": ..., "password": ...}``) — the same
    JSON-only state-changing convention as the admin API, so the login cannot
    be CSRF'd by a cross-origin form (a browser form can only send
    ``text/plain``-style bodies that never pass preflight). The configured
    admin username is always an admin login: qiip's admin password wins and a
    same-named entry in the local oauth user list is ignored. Any other
    username is checked against the local oauth user list when the opt-in
    local provider is enabled (and refuses otherwise). On success a signed
    session cookie is set and the browser is redirected; invalid credentials
    return 401 with a visible error message.
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

    # Admin first: qiip's configured admin password always wins.
    if _credentials_match(username, password, settings):
        set_local_admin_session(request, settings.auth.session_ttl_seconds)
        logger.info("local admin signed in")
        return RedirectResponse(_DASHBOARD_HOME, status_code=302)

    # Local oauth user list (opt-in local provider). The local user list is
    # an operator-curated trust list, so its accounts pass the email
    # verification gate; the hosted-domain and SSO-whitelist gates still
    # apply (same policy as the OIDC callback).
    identity = _local_login_identity(request, username, password, settings)
    if identity is not None:
        policy_error = await check_signin_policy(
            identity[0], identity[1], True, settings, allowlist
        )
        if policy_error is not None:
            logger.warning(
                "local login policy rejected", email=identity[0], code=policy_error
            )
            raise HTTPException(
                status_code=403, detail=_POLICY_ERROR_MESSAGES[policy_error]
            )
        try:
            user = await asyncio.to_thread(
                store.upsert_google_user,
                google_sub=identity[0],
                issuer=identity[1],
                email=identity[0],
                name=username,
                picture="",
            )
        except AccountConflictError as exc:
            # The username is valid but its email is already taken by an
            # account from another provider; refuse rather than migrating
            # the account (admin role and tokens) across providers.
            logger.warning("local login provider conflict", email=exc.email)
            raise HTTPException(
                status_code=409,
                detail="This email is already linked to another sign-in provider",
            ) from None
        set_session_user(request, user.id, settings.auth.session_ttl_seconds)
        logger.info("local oauth user signed in", user_id=user.id, email=identity[0])
        home = _PROFILE_HOME if user.is_admin else USER_HOME
        return RedirectResponse(home, status_code=302)

    logger.warning("local login rejected", username=username)
    raise HTTPException(status_code=401, detail="Invalid username or password")


def _local_login_identity(
    request: Request,
    username: str,
    password: str,
    settings: Settings,
) -> tuple[str, str] | None:
    """Return (email, issuer) for valid local-oauth credentials, else None.

    Consults the loaded ``auth.internal_oidc`` plugin; it is the only
    provider with a local user list (``users_file`` in its config). Google
    has none, so a Google-only deployment keeps admin-only local login.
    """
    manager = getattr(request.app.state, "plugin_manager", None)
    plugin = manager.get_plugin("auth.internal_oidc") if manager is not None else None
    if not isinstance(plugin, AuthPlugin):
        return None
    email = plugin.verify_local_credentials(username, password, settings.admin.username)
    issuer = getattr(plugin, "issuer", "")
    if not email or not issuer:
        return None
    return email, issuer


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
        home = _PROFILE_HOME if viewer_role(request, settings) == "admin" else USER_HOME
        return RedirectResponse(home, status_code=302)
    redirect_uri = _oauth_redirect_uri(request, settings)
    if redirect_uri is None:
        raise HTTPException(
            status_code=503, detail="OAuth redirect URI is not configured"
        )
    # Record which provider started the flow so the provider-less callback
    # (redirect_uri carries no hint) resolves the same plugin that issued
    # the authorization request.
    request.session[AUTH_PROVIDER_SESSION_KEY] = auth.name
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

    policy_error = await check_signin_policy(
        email, identity.issuer, identity.email_verified, settings, allowlist
    )
    if policy_error is not None:
        logger.warning("oauth callback policy rejected", email=email, code=policy_error)
        return _error_redirect(policy_error)

    # User rows are keyed by (issuer, sub) so Google and a local OIDC
    # provider can issue the same opaque sub without colliding. The email
    # unique index rebinds an account that changes sub *within the same
    # issuer*; a row whose email belongs to another issuer is never moved
    # (see AuthStore.upsert_google_user) and the login is refused.
    try:
        user = await asyncio.to_thread(
            store.upsert_google_user,
            google_sub=identity.sub,
            issuer=identity.issuer,
            email=email,
            name=identity.name,
            picture=identity.picture,
        )
    except AccountConflictError as exc:
        logger.warning("oauth callback provider conflict", email=exc.email)
        return _error_redirect("account_conflict")
    set_session_user(request, user.id, settings.auth.session_ttl_seconds)
    logger.info("user signed in", user_id=user.id, email=email)
    home = _PROFILE_HOME if user.is_admin else USER_HOME
    return RedirectResponse(home, status_code=302)


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
