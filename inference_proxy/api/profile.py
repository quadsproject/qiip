"""Per-user profile page: API-token management and token-usage reporting.

The HTML page at ``/profile`` is intentionally public: it renders a sign-in
call-to-action when the visitor is anonymous and the full manager once
signed in (the client decides from ``/auth/me``). The JSON endpoints under
``/profile/*`` are guarded by ``require_profile_user`` (AUTH-03).

Per AUTH-02 a minted token's raw secret is returned exactly once, at
creation; the list endpoints return only the token prefix for display. The
one exception is ``name: agent-config``: that request returns the user's
reusable derived config key (same value on every download, see
``AuthStore.get_or_create_config_token``), which is intentionally not
one-time.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from inference_proxy.api.templating import templates, user_home_redirect
from inference_proxy.auth.allowlist import (
    AllowlistUnavailableError,
    SSOAllowlist,
    enforce_allowlist,
)
from inference_proxy.auth.billing import premium_equivalent_cost
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
from inference_proxy.auth.scopes import (
    has_admin_access,
    is_full_access,
    pickable_endpoints,
)
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_registry,
    get_settings,
    viewer_role,
)
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry

profile_router = APIRouter(prefix="/profile", tags=["profile"])


async def enforce_mint_allowlist(
    user: User,
    settings: Settings,
    allowlist: SSOAllowlist | None,
) -> None:
    """Apply the SSO whitelist gate to a token mint (403/503 on denial).

    Matches the login and use-time gates exactly: only the full-access trust
    list bypasses it. Admin-role users must still pass, otherwise the use
    time check (auth/dependencies.py) would 401 every call of the new token.
    """
    if not settings.auth.enforce_sso_whitelist or is_full_access(user.email, settings):
        return
    try:
        allowed = await enforce_allowlist(user.email, allowlist)
    except AllowlistUnavailableError:
        raise HTTPException(
            status_code=503, detail="SSO whitelist is unavailable"
        ) from None
    if not allowed:
        raise HTTPException(status_code=403, detail="User is not in the SSO whitelist")


@profile_router.get("", response_class=HTMLResponse, response_model=None)
async def profile_page(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HTMLResponse | RedirectResponse:
    """Render the profile HTML shell (client decides signed-in state).

    Signed-in normal users are sent to their own page (``/start``); the
    profile stays for admins and for anonymous sign-in error display.
    """
    if viewer_role(request, settings) == "user":
        return user_home_redirect()
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
        is_admin=user.is_admin,
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

    The ``agent-config`` name is special: it returns the user's single
    reusable config token (derived on demand, never stored) so config
    downloads stay stable across servers and browsers. Ordinary names mint
    a fresh random token whose raw value is returned exactly once.

    When ``auth.enforce_sso_whitelist`` is on, the signed-in user must be on
    the fetched per-domain allowlist; otherwise 403. An unavailable
    whitelist fails closed with 503. Tokens are the only credential accepted
    on /v1, so this gate plus the use-time re-check in ``get_api_auth``
    bounds inference access to allowlist members.

    An optional ``endpoints`` pin must reference registered nodes the user
    may route to: an unknown hostname is rejected (400) and a node owned by
    someone else is rejected (403). Admins may pin any registered node.
    """
    admin = has_admin_access(user.email, settings, is_admin=user.is_admin)
    if not admin:
        # Normal users own exactly one model-scoped token, minted on /start.
        # Minting here (including the agent-config key) would hand them a
        # second, unrestricted credential the onboarding page never shows.
        raise HTTPException(
            status_code=403,
            detail="Tokens are managed from the qiip start page",
        )
    await enforce_mint_allowlist(user, settings, allowlist)
    if body.name == "agent-config" and body.endpoints is not None:
        # The agent-config key is a single stable full-access credential
        # (get_or_create_config_token always stores a NULL scope), so a pin
        # cannot be honored. Reject instead of silently discarding it.
        raise HTTPException(
            status_code=400,
            detail="The agent-config token cannot be pinned to endpoints",
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
            if not admin:
                if node.admin_only:
                    # Non-admins cannot route to admin-only servers; accepting
                    # the pin would mint a token _in_scope rejects on every use.
                    raise HTTPException(
                        status_code=403,
                        detail=f"Endpoint '{hostname}' is admin-only",
                    )
                if node.owner and node.owner.lower() != email:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Endpoint '{hostname}' is owned by another user",
                    )
    if body.name == "agent-config":
        # Agent-config downloads share one stable token per user. The raw
        # value is derived (never stored) so any browser/machine gets the
        # same key; see AuthStore.get_or_create_config_token.
        secret = settings.auth.session_secret
        if secret is None:  # pragma: no cover - require_profile_user already
            raise HTTPException(status_code=503, detail="Sessions are not configured")
        created = await asyncio.to_thread(
            store.get_or_create_config_token, user.id, secret.get_secret_value()
        )
    else:
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
    pickable = pickable_endpoints(user.email, settings, nodes, is_admin=user.is_admin)
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
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    """Return per-token/per-model usage plus headline totals (AUTH-04)."""
    summary, totals = await asyncio.gather(
        asyncio.to_thread(store.get_usage_summary, user.id),
        asyncio.to_thread(store.get_usage_totals, user.id),
    )
    return {
        "totals": totals,
        "rows": summary,
        "estimated_premium_cost_usd": premium_equivalent_cost(
            totals.prompt_tokens,
            totals.completion_tokens,
            input_rate_per_mtok=settings.pricing.input_rate_per_mtok,
            output_rate_per_mtok=settings.pricing.output_rate_per_mtok,
        ),
        "model_label": settings.pricing.model_label,
    }
