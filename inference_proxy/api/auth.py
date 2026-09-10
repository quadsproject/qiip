"""Google OAuth login/logout and current-user endpoints.

Flow (RFC 6749 Authorization Code + OIDC):

    /auth/login    -> 302 to Google's authorization endpoint
    /auth/callback -> Google redirects back with a code; we exchange it,
                      verify/upsert the user, and sign the session cookie
    /auth/logout   -> clears the session cookie
    /auth/me       -> JSON identity for the signed-in user (or 401)

All failure paths return to the profile page with a short ``error`` query
parameter that the profile UI surfaces, so a failed or denied login never
leaves the user on a bare error screen.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, cast

import structlog
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from inference_proxy.auth.dependencies import (
    get_auth_store,
    get_oauth_client,
    require_profile_user,
)
from inference_proxy.auth.models import PublicUser, User
from inference_proxy.auth.session import (
    clear_session_user,
    get_session_user_id,
    set_session_user,
)
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings

logger = structlog.get_logger()

auth_router = APIRouter(prefix="/auth", tags=["auth"])

_PROFILE_HOME = "/profile"


def _error_redirect(error: str) -> RedirectResponse:
    """Redirect back to the profile page carrying a short error code."""
    return RedirectResponse(f"{_PROFILE_HOME}?error={error}", status_code=302)


@auth_router.get("/login")
async def oauth_login(
    request: Request,
    oauth: Annotated[OAuth, Depends(get_oauth_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedirectResponse:
    """Start the Google Authorization Code flow (302 to Google).

    When a session already exists the request short-circuits to the profile
    page instead of forcing a re-authentication round-trip.
    """
    if get_session_user_id(request) is not None:
        return RedirectResponse(_PROFILE_HOME, status_code=302)
    redirect_uri = settings.oauth.redirect_uri
    return cast(
        RedirectResponse,
        await oauth.google.authorize_redirect(request, redirect_uri),
    )


@auth_router.get("/callback")
async def oauth_callback(
    request: Request,
    oauth: Annotated[OAuth, Depends(get_oauth_client)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedirectResponse:
    """Handle Google's post-login redirect, upsert the user, sign the session.

    Verification performed against the verified ID-token claims (authlib
    validates the JWT signature, issuer, audience, and nonce/state):
    email presence, optional email-verification requirement, and the
    optional hosted-domain allowlist.
    """
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        logger.warning(
            "oauth callback rejected",
            error=exc.error,
            description=exc.description,
        )
        return _error_redirect("login_failed")

    userinfo = token.get("userinfo") or {}
    email = userinfo.get("email")
    if not isinstance(email, str) or not email:
        logger.warning("oauth callback missing email", userinfo=userinfo)
        return _error_redirect("no_profile")

    if settings.auth.require_email_verification and not userinfo.get("email_verified"):
        logger.warning("oauth callback unverified email", email=email)
        return _error_redirect("unverified_email")

    allowed_domains = [domain.lower() for domain in settings.oauth.allowed_domains]
    domain = email.rsplit("@", 1)[-1].lower() if "@" in email else ""
    if allowed_domains and domain not in allowed_domains:
        logger.warning("oauth callback domain not allowed", email=email)
        return _error_redirect("domain_not_allowed")

    name = userinfo.get("name") or ""
    picture = userinfo.get("picture") or ""
    google_sub = userinfo.get("sub")
    if not isinstance(google_sub, str) or not google_sub:
        logger.warning("oauth callback missing subject", email=email)
        return _error_redirect("no_profile")

    user = await asyncio.to_thread(
        store.upsert_google_user,
        google_sub=google_sub,
        email=email,
        name=name if isinstance(name, str) else "",
        picture=picture if isinstance(picture, str) else "",
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
    return RedirectResponse(_PROFILE_HOME, status_code=302)


@auth_router.get("/me")
async def oauth_me(user: Annotated[User, Depends(require_profile_user)]) -> PublicUser:
    """Return the signed-in user's public identity, or 401 when anonymous."""
    return PublicUser(
        id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture,
    )
