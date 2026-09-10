"""FastAPI dependencies for user sessions and API-token authentication.

Split responsibilities:

* ``get_auth_store`` / ``get_auth_plugin`` read the prepared singletons
  from ``app.state`` (created during the application lifespan).
* ``require_profile_user`` authenticates the browser session cookie for the
  profile surface.
* ``get_api_auth`` authenticates bearer tokens on the /v1 inference API
  (AUTH-03) with config-gated enforcement.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from inference_proxy.api.errors import ApiAuthError
from inference_proxy.auth.models import TokenAuth, User
from inference_proxy.auth.session import get_session_user_id
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.plugins.interfaces.auth import AuthPlugin


def get_auth_store(request: Request) -> AuthStore:
    """Return the lifespan-created auth store from application state."""
    store = getattr(request.app.state, "auth_store", None)
    if not isinstance(store, AuthStore):
        raise HTTPException(status_code=503, detail="Auth store is unavailable")
    return store


def get_auth_plugin(request: Request) -> AuthPlugin:
    """Return the lifespan-loaded auth plugin, or 404 when unconfigured."""
    plugin = getattr(request.app.state, "auth_plugin", None)
    if not isinstance(plugin, AuthPlugin) or not plugin.is_configured():
        raise HTTPException(status_code=404, detail="OAuth is not configured")
    return plugin


async def require_profile_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> User:
    """Resolve and return the signed-in user for the profile surface.

    Sessions are available only when ``auth.session_secret`` is set, so a
    missing secret fails with 503 rather than guessing.
    """
    if settings.auth.session_secret is None:
        raise HTTPException(status_code=503, detail="Sessions are not configured")
    user_id = get_session_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    user = await asyncio.to_thread(store.get_user, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


async def get_api_auth(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> TokenAuth | None:
    """Authenticate a /v1 bearer token, applying configured enforcement.

    * A valid ``Authorization: Bearer <token>`` is always accepted and
      attributes usage (AUTH-04).
    * ``auth.enforce_api_tokens`` (AUTH-03) decides what happens when there
      is no usable token. With it False (the default) an absent or invalid
      token simply means anonymous: enforcement-off deployments already
      permit anonymous traffic, so an invalid presented token is treated as
      anonymous rather than rejected. With enforcement on, absent or
      invalid tokens are rejected with 401.

    Rejections raise ``ApiAuthError`` so the /v1 error handler can emit an
    OpenAI-compatible ``invalid_api_key`` body.
    """
    authorization = request.headers.get("authorization", "")
    scheme, _, raw_token = authorization.partition(" ")
    presented = scheme.lower() == "bearer" and bool(raw_token.strip())

    if presented:
        raw_token = raw_token.strip()
        if settings.auth.enforce_api_tokens:
            auth = await asyncio.to_thread(store.resolve_token, raw_token)
            if auth is None:
                raise ApiAuthError("Invalid API token")
            return auth
        # Enforcement off: resolve for attribution, but never hard-fail on a
        # bad token (anonymous traffic is already permitted in this mode).
        return await asyncio.to_thread(store.resolve_token, raw_token)

    if settings.auth.enforce_api_tokens:
        raise ApiAuthError("Authentication required for inference requests")
    return None
