"""Self-service onboarding for normal (non-admin) users.

One page (``/start``) drives everything: a first-time user is walked through
naming a token, choosing a coding harness, and choosing models; a returning
user sees their single token and its models. Either way the flow ends in a
short-lived ``curl ... | bash`` link (``/s/{id}``) whose script writes the
harness config, token included.

Rules enforced here:

* a normal user owns exactly one token; minting replaces the previous one;
* the token's model scope is enforced on /v1 (see ``routes.py``);
* a setup link stores no credential: the script is rendered on fetch from a
  re-derived token, so an expired or superseded link serves nothing useful.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import Annotated
from urllib.parse import urlsplit

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from inference_proxy.api.profile import enforce_mint_allowlist
from inference_proxy.api.routes import router as inference_router
from inference_proxy.api.templating import signin_response, templates
from inference_proxy.auth.allowlist import SSOAllowlist
from inference_proxy.auth.dependencies import (
    get_auth_store,
    get_sso_allowlist,
    require_profile_user,
)
from inference_proxy.auth.models import ApiToken, User
from inference_proxy.auth.scopes import pickable_endpoints
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_registry,
    get_settings,
    viewer_role,
)
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import NodeStatus
from inference_proxy.onboarding.harness import HARNESSES, Harness, get_harness
from inference_proxy.onboarding.script import (
    HEREDOC_DELIMITER,
    render_expired_script,
    render_setup_script,
)

logger = structlog.get_logger()

onboarding_router = APIRouter(tags=["onboarding"])

SETUP_LINK_TTL_SECONDS = 15 * 60
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_LINK_ID = re.compile(r"^[a-z0-9]{8,32}$")


class TokenRequest(BaseModel):
    """Body for minting the user's token."""

    name: str = Field(min_length=1, max_length=60)
    models: list[str] = Field(min_length=1, max_length=50)


class ModelsRequest(BaseModel):
    """Body for replacing the token's model scope."""

    models: list[str] = Field(min_length=1, max_length=50)


class SetupLinkRequest(BaseModel):
    """Body for creating a setup-script link."""

    harness: str
    models: list[str] = Field(min_length=1, max_length=50)


async def require_normal_user(
    user: Annotated[User, Depends(require_profile_user)],
) -> User:
    """Require a signed-in non-admin user.

    The onboarding flow enforces one token per user by revoking every other
    token on mint. Admins keep a portfolio of pinned and agent-config tokens
    on /profile, so the flow must never run for them.
    """
    if user.is_admin:
        raise HTTPException(
            status_code=403, detail="Admins manage tokens from the profile page"
        )
    return user


def public_base_url(request: Request, settings: Settings) -> str:
    """Return the origin users and their tools should reach qiip on.

    ``request.base_url`` is only as good as the proxy headers uvicorn trusts:
    behind a reverse proxy it does not trust (e.g. the rootless Podman nginx)
    the scheme degrades to ``http``. The configured OAuth ``redirect_uri`` is
    the one origin the operator has declared, so it anchors the result the
    same way ``_oauth_redirect_uri`` does: an allowlisted request host keeps
    its own name (multi-name deployments) but never downgrades a configured
    https origin, and any other host falls back to the configured origin.
    """
    fallback = str(request.base_url).rstrip("/")
    configured = settings.oauth.redirect_uri
    if not configured:
        return fallback
    parts = urlsplit(configured)
    if not parts.scheme or not parts.netloc:
        return fallback
    allowed = {host.lower() for host in settings.oauth.allowed_redirect_hosts}
    if parts.hostname:
        allowed.add(parts.hostname.lower())
    host = request.url.hostname
    if host and host.lower() in allowed:
        scheme = "https" if parts.scheme == "https" else request.url.scheme
        return f"{scheme}://{request.url.netloc}"
    return f"{parts.scheme}://{parts.netloc}"


def _session_secret(settings: Settings) -> str:
    secret = settings.auth.session_secret
    if secret is None:  # pragma: no cover - require_profile_user already 503s
        raise HTTPException(status_code=503, detail="Sessions are not configured")
    return secret.get_secret_value()


def _available_models(
    user: User, settings: Settings, registry: NodeRegistry
) -> list[str]:
    """Model ids the user can reach right now, sorted and de-duplicated."""
    nodes = registry.get_all()
    pickable = set(
        pickable_endpoints(user.email, settings, nodes, is_admin=user.is_admin)
    )
    models: set[str] = set()
    for node in nodes:
        if (
            node.node_id not in pickable
            or node.status != NodeStatus.HEALTHY
            or not node.model
        ):
            continue
        if not _MODEL_ID.match(node.model) or HEREDOC_DELIMITER in node.model:
            # Still served on /v1; just not safe to template into a script.
            logger.warning(
                "model id not offered for onboarding",
                model=node.model,
                node_id=node.node_id,
            )
            continue
        models.add(node.model)
    return sorted(models)


def _served_routes() -> set[str]:
    """Paths the inference router serves (a harness needs its API route)."""
    return {getattr(route, "path", "") for route in inference_router.routes}


def _harness_view(harness: Harness, served: set[str]) -> dict[str, object]:
    return {
        "id": harness.id,
        "label": harness.label,
        "tagline": harness.tagline,
        "command": harness.command,
        "multi_model": harness.multi_model,
        "available": harness.requires_route in served,
    }


def _token_view(token: ApiToken | None) -> dict[str, object] | None:
    if token is None:
        return None
    return {
        "id": token.id,
        "name": token.name,
        "prefix": token.prefix,
        "created_at": token.created_at.isoformat(),
        "last_used_at": token.last_used_at.isoformat() if token.last_used_at else None,
        "models": token.model_scope,
        # Only personal tokens can be re-derived for another setup script.
        "exportable": token.purpose == "personal",
    }


def _require_available(models: list[str], available: list[str]) -> list[str]:
    """Return *models* de-duplicated in order; 400 when any is not available."""
    chosen = list(dict.fromkeys(models))
    missing = [model for model in chosen if model not in available]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{missing[0]}' is not available right now",
        )
    return chosen


@onboarding_router.get("/start", response_class=HTMLResponse, response_model=None)
async def start_page(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HTMLResponse | RedirectResponse:
    """Render the onboarding / token home page for signed-in normal users.

    ``viewer_role`` re-reads the user row, so a cookie for a deleted user
    gets the sign-in page instead of a shell that can never load its state.
    Admins (local or Google) have no single token to manage here.
    """
    role = viewer_role(request, settings)
    if role == "admin":
        return RedirectResponse("/dashboard", status_code=302)
    if role is None:
        return signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    return templates.TemplateResponse(request=request, name="start.html", context={})


@onboarding_router.get("/onboarding/state")
async def onboarding_state(
    user: Annotated[User, Depends(require_normal_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
) -> dict[str, object]:
    """Everything the page needs: identity, token, models, harnesses."""
    token = await asyncio.to_thread(store.get_active_token, user.id)
    served = _served_routes()
    return {
        "user": {
            "name": user.name,
            "email": user.email,
            "picture": user.picture,
            "is_admin": user.is_admin,
        },
        "token": _token_view(token),
        "models": _available_models(user, settings, registry),
        "harnesses": [_harness_view(harness, served) for harness in HARNESSES],
    }


@onboarding_router.post("/onboarding/token", status_code=201)
async def create_personal_token(
    body: TokenRequest,
    user: Annotated[User, Depends(require_normal_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
    allowlist: Annotated[SSOAllowlist | None, Depends(get_sso_allowlist)] = None,
) -> dict[str, object]:
    """Mint the user's token, replacing (revoking) any previous one.

    The raw secret is not returned: the user never needs to handle it, the
    setup script delivers it straight into the harness config.
    """
    await enforce_mint_allowlist(user, settings, allowlist)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Give your token a name")
    models = _require_available(
        body.models, _available_models(user, settings, registry)
    )
    created = await asyncio.to_thread(
        store.create_personal_token, user.id, name, models, _session_secret(settings)
    )
    return {"token": _token_view(created)}


@onboarding_router.put("/onboarding/token/models")
async def update_token_models(
    body: ModelsRequest,
    user: Annotated[User, Depends(require_normal_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
) -> dict[str, object]:
    """Replace the models the user's token may use.

    Models already on the token stay selectable even when no node serves
    them at the moment, so an edit never silently drops one.
    """
    token = await asyncio.to_thread(store.get_active_token, user.id)
    if token is None:
        raise HTTPException(status_code=404, detail="You don't have a token yet")
    if token.purpose != "personal":
        # Narrowing an unrestricted legacy or agent-config key cannot be undone
        # here (a scope needs at least one model), so never do it implicitly.
        raise HTTPException(
            status_code=409,
            detail="This token was created the old way. Create a new token first.",
        )
    allowed = _available_models(user, settings, registry) + (token.model_scope or [])
    models = _require_available(body.models, allowed)
    if not await asyncio.to_thread(store.set_token_models, user.id, token.id, models):
        raise HTTPException(
            status_code=409, detail="Your token changed. Reload and try again."
        )
    token = await asyncio.to_thread(store.get_active_token, user.id)
    return {"token": _token_view(token)}


@onboarding_router.post("/onboarding/setup-link", status_code=201)
async def create_setup_link(
    request: Request,
    body: SetupLinkRequest,
    user: Annotated[User, Depends(require_normal_user)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: NodeRegistry = Depends(get_registry),
) -> dict[str, object]:
    """Create the short-lived ``curl | bash`` link for one harness.

    Models outside the token's current scope are added to it, so the config
    a user just asked for always works.
    """
    harness = get_harness(body.harness)
    if harness is None or harness.requires_route not in _served_routes():
        raise HTTPException(status_code=400, detail="That tool isn't supported yet")
    token = await asyncio.to_thread(store.get_active_token, user.id)
    if token is None:
        raise HTTPException(status_code=404, detail="You don't have a token yet")
    if token.purpose != "personal":
        raise HTTPException(
            status_code=409,
            detail="This token was created the old way. Create a new token first.",
        )
    models = _require_available(
        body.models, _available_models(user, settings, registry)
    )
    if not harness.multi_model and len(models) != 1:
        raise HTTPException(
            status_code=400, detail=f"{harness.label} works with exactly one model"
        )
    if (
        await asyncio.to_thread(
            store.reveal_personal_token, token.id, _session_secret(settings)
        )
        is None
    ):
        raise HTTPException(
            status_code=409,
            detail="This token can no longer be exported. Create a new token first.",
        )
    if token.model_scope is not None:
        widened = list(dict.fromkeys([*token.model_scope, *models]))
        if widened != token.model_scope:
            await asyncio.to_thread(store.set_token_models, user.id, token.id, widened)
    link_id, expires = await asyncio.to_thread(
        store.create_setup_link,
        user.id,
        token.id,
        harness.id,
        models,
        SETUP_LINK_TTL_SECONDS,
    )
    url = f"{public_base_url(request, settings)}/s/{link_id}"
    return {
        "url": url,
        # No -f: an expired link must still pipe its explanation into bash.
        "command": f"curl -sSL {shlex.quote(url)} | bash",
        "expires_at": expires.isoformat(),
        "harness": harness.id,
        "run_command": harness.command,
    }


@onboarding_router.get("/s/{link_id}", response_class=PlainTextResponse)
async def setup_script(
    request: Request,
    link_id: str,
    store: Annotated[AuthStore, Depends(get_auth_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PlainTextResponse:
    """Serve the setup script for a live link (the link id is the secret)."""
    headers = {
        "Cache-Control": "no-store",
        "X-Robots-Tag": "noindex",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    # 200 on purpose: the body is a script that explains and exits 1, so the
    # piped command fails visibly. A 404 would reach bash as empty stdin and
    # "succeed" with nothing configured.
    expired = PlainTextResponse(render_expired_script(), headers=headers)
    secret = settings.auth.session_secret
    if secret is None or not _LINK_ID.match(link_id):
        return expired
    link = await asyncio.to_thread(store.resolve_setup_link, link_id)
    harness = get_harness(link.harness) if link is not None else None
    if link is None or harness is None:
        return expired
    raw = await asyncio.to_thread(
        store.reveal_personal_token, link.token_id, secret.get_secret_value()
    )
    if raw is None:
        return expired
    try:
        script = render_setup_script(
            harness,
            base_url=public_base_url(request, settings),
            token=raw,
            models=link.models,
        )
    except ValueError:
        logger.warning("setup script could not be rendered", harness=harness.id)
        return expired
    return PlainTextResponse(script, headers=headers)
