"""Per-user profile page: API-token management and token-usage reporting.

The HTML page at ``/profile`` is intentionally public: it renders a sign-in
call-to-action when the visitor is anonymous and the full manager once
signed in (the client decides from ``/auth/me``). The JSON endpoints under
``/profile/*`` are guarded by ``require_profile_user`` (AUTH-03).

Per AUTH-02 the raw token secret is returned exactly once, at creation; the
list endpoints return only the token prefix for display.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from inference_proxy.api.templating import templates
from inference_proxy.auth.allowlist import (
    AllowlistUnavailableError,
    SSOAllowlist,
    enforce_allowlist,
)
from inference_proxy.auth.dependencies import (
    get_auth_store,
    get_sso_allowlist,
    require_profile_user,
)
from inference_proxy.auth.models import (
    CreatedToken,
    CreateTokenRequest,
    PublicToken,
    PublicUser,
    User,
)
from inference_proxy.auth.scopes import is_full_access, pickable_endpoints
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_registry, get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry

profile_router = APIRouter(prefix="/profile", tags=["profile"])


@profile_router.get("", response_class=HTMLResponse)
async def profile_page(request: Request) -> HTMLResponse:
    """Render the profile HTML shell (client decides signed-in state)."""
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={"active_page": "profile"},
    )


@profile_router.get("/me")
async def profile_me(
    user: Annotated[User, Depends(require_profile_user)],
) -> PublicUser:
    """Return the signed-in user's public identity (used by the page)."""
    return PublicUser(
        id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture,
    )


@profile_router.get("/tokens")
async def list_tokens(
    user: Annotated[User, Depends(require_profile_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> list[PublicToken]:
    """List the user's API tokens (prefix only, never the raw secret)."""
    tokens = await asyncio.to_thread(store.list_tokens, user.id)
    return [PublicToken.from_token(token) for token in tokens]


@profile_router.post("/tokens", status_code=201)
async def create_token(
    body: CreateTokenRequest,
    user: Annotated[User, Depends(require_profile_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
    allowlist: Annotated[SSOAllowlist | None, Depends(get_sso_allowlist)] = None,
) -> CreatedToken:
    """Mint an API token; returns the raw secret exactly once (AUTH-02).

    When ``auth.enforce_sso_whitelist`` is on, the signed-in user must be on
    the fetched per-domain allowlist; otherwise 403. An unavailable
    whitelist fails closed with 503. Tokens are the only credential accepted
    on /v1, so this gate plus the use-time re-check in ``get_api_auth``
    bounds inference access to allowlist members.

    An optional ``endpoints`` pin must reference registered nodes the user
    may route to: an unknown hostname is rejected (400) and a node owned by
    someone else is rejected (403). Admins may pin any registered node.
    """
    admin = is_full_access(user.email, settings)
    if settings.auth.enforce_sso_whitelist and not admin:
        try:
            allowed = await enforce_allowlist(user.email, allowlist)
        except AllowlistUnavailableError:
            raise HTTPException(
                status_code=503, detail="SSO whitelist is unavailable"
            ) from None
        if not allowed:
            raise HTTPException(
                status_code=403, detail="User is not in the SSO whitelist"
            )
    if body.endpoints is not None:
        email = user.email.lower()
        for hostname in body.endpoints:
            node = registry.get(hostname)
            if node is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Endpoint '{hostname}' is not a registered node",
                )
            if not admin and node.owner and node.owner.lower() != email:
                raise HTTPException(
                    status_code=403,
                    detail=f"Endpoint '{hostname}' is owned by another user",
                )
    created = await asyncio.to_thread(
        store.create_token, user.id, body.name, body.endpoints
    )
    return created


@profile_router.get("/endpoints")
async def list_pickable_endpoints(
    user: Annotated[User, Depends(require_profile_user)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
) -> list[dict[str, str]]:
    """List endpoints the signed-in user may pin a token to."""
    nodes = registry.get_all()
    pickable = pickable_endpoints(user.email, settings, nodes)
    by_id = {node.node_id: node for node in nodes}
    return [{"node_id": node_id, "model": by_id[node_id].model} for node_id in pickable]


@profile_router.delete("/tokens/{token_id}")
async def revoke_token(
    token_id: int,
    user: Annotated[User, Depends(require_profile_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> JSONResponse:
    """Revoke a token so it can no longer authenticate /v1 requests."""
    revoked = await asyncio.to_thread(store.revoke_token, user.id, token_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Token not found")
    return JSONResponse(content={"revoked": True})


@profile_router.get("/usage")
async def usage_summary(
    user: Annotated[User, Depends(require_profile_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> dict[str, object]:
    """Return per-token/per-model usage plus headline totals (AUTH-04)."""
    summary, totals = await asyncio.gather(
        asyncio.to_thread(store.get_usage_summary, user.id),
        asyncio.to_thread(store.get_usage_totals, user.id),
    )
    return {
        "totals": totals,
        "rows": summary,
    }
