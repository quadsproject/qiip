"""Admin token-management endpoints (RFE #113).

Kept in its own module: ``api/admin.py`` already owns the node/provisioning
surface, and token administration has its own request/response shapes and
pricing logic. All routes inherit the existing Basic-auth gate
(``require_admin_auth``) applied at router level, matching ``api/admin.py``.

The JSON surface is read-mostly plus one destructive action (revoke), and
the raw token secret is never exposed here (AUTH-02): token rows return
only the display ``prefix``.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from inference_proxy.auth.billing import premium_equivalent_cost
from inference_proxy.auth.dependencies import get_auth_store
from inference_proxy.auth.models import (
    AdminTokenView,
    AdminUserDetail,
    AdminUserStats,
    BillingSummary,
    PublicUser,
    UsageTotals,
)
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings, require_admin_auth
from inference_proxy.config.settings import Settings

admin_tokens_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_auth)],
)


def _estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    settings: Settings,
) -> float:
    """Compute the premium-equivalent estimate for a token mix."""
    return premium_equivalent_cost(
        prompt_tokens,
        completion_tokens,
        input_rate_per_mtok=settings.pricing.input_rate_per_mtok,
        output_rate_per_mtok=settings.pricing.output_rate_per_mtok,
    )


@admin_tokens_router.get("/users")
async def list_users(
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[AdminUserStats]:
    """Return every user with token counts and usage totals."""
    users = await asyncio.to_thread(store.list_users_with_stats)
    return [
        user.model_copy(
            update={
                "estimated_cost_usd": _estimate_cost(
                    user.prompt_tokens, user.completion_tokens, settings
                )
            }
        )
        for user in users
    ]


@admin_tokens_router.get("/tokens")
async def list_tokens(
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> list[AdminTokenView]:
    """Return every generated token with its owner, newest first."""
    return await asyncio.to_thread(store.list_all_tokens)


@admin_tokens_router.get("/users/{user_id}")
async def user_detail(
    user_id: int,
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AdminUserDetail:
    """Return the full per-user view: tokens, usage rows, totals, timeline."""
    user = await asyncio.to_thread(store.get_user, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    rows = await asyncio.gather(
        asyncio.to_thread(store.list_tokens, user_id),
        asyncio.to_thread(store.get_usage_summary, user_id),
        asyncio.to_thread(store.get_usage_totals, user_id),
        asyncio.to_thread(store.get_user_usage_timeline, user_id),
    )
    tokens, usage, totals, timeline = rows

    views = [
        AdminTokenView(
            id=token.id,
            user_id=token.user_id,
            user_email=user.email,
            user_name=user.name,
            name=token.name,
            prefix=token.prefix,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            revoked=token.revoked,
            endpoint_scope=token.endpoint_scope,
        )
        for token in tokens
    ]
    return AdminUserDetail(
        user=PublicUser(
            id=user.id, email=user.email, name=user.name, picture=user.picture
        ),
        tokens=views,
        usage=usage,
        totals=totals,
        timeline=timeline,
        estimated_cost_usd=_estimate_cost(
            totals.prompt_tokens, totals.completion_tokens, settings
        ),
        model_label=settings.pricing.model_label,
    )


@admin_tokens_router.delete("/tokens/{token_id}")
async def revoke_any_token(
    token_id: int,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> JSONResponse:
    """Revoke any token by id, so it can no longer authenticate /v1."""
    revoked = await asyncio.to_thread(store.revoke_any_token, token_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Token not found")
    return JSONResponse(content={"revoked": True})


@admin_tokens_router.get("/billing")
async def billing_summary(
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BillingSummary:
    """Return global usage totals and the premium-equivalent cost."""
    totals: UsageTotals = await asyncio.to_thread(store.get_usage_totals_all)
    return BillingSummary(
        totals=totals,
        estimated_cost_usd=_estimate_cost(
            totals.prompt_tokens, totals.completion_tokens, settings
        ),
        model_label=settings.pricing.model_label,
    )
