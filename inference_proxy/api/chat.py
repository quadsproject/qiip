"""Chat page route for the chatbot playground UI.

Serves the chat HTML shell at ``/chat``. Client-side JS handles model
selection, message sending, and streaming responses, and runs the playground on
the signed-in viewer's own token (``/onboarding/token/reveal``). Signed-in
admins use their personal token when they own one; without one the page says
so and offers the profile token manager (issue #294).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from inference_proxy.api.templating import signin_response, templates
from inference_proxy.auth.dependencies import get_auth_store
from inference_proxy.auth.session import get_session_user_id
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_settings,
    session_user_has_full_access,
    viewer_role,
)
from inference_proxy.config.settings import Settings

chat_router = APIRouter(tags=["chat"])


@chat_router.get("/chat", response_class=HTMLResponse, response_model=None)
async def chat(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> HTMLResponse:
    """Render the chat playground HTML shell for signed-in viewers.

    Anonymous visitors get the sign-in page. A signed-in user chats on their
    own token, which the page reveals through ``/onboarding/token/reveal`` and
    sends on every ``/v1`` call, so usage is attributed and admin-only nodes
    stay unreachable. A signed-in admin does the same when they own a personal
    token; without one the page says so and points at the profile token
    manager instead of sending an unattributable request (issue #294). The
    full-access trust list and the session-less local admin (HTTP Basic or the
    local-admin form) keep the anonymous playground: the trust list already
    holds a token portfolio on /profile, and the session-less admin has no
    user row that could own a personal token.
    """
    role = viewer_role(request, settings)
    if role is None:
        return signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    anonymous_playground = session_user_has_full_access(request, settings)
    missing_personal_token = False
    if role == "admin" and not anonymous_playground:
        user_id = get_session_user_id(request)
        if user_id is None:
            anonymous_playground = True
        else:
            token = await asyncio.to_thread(store.get_active_token, user_id)
            missing_personal_token = token is None or token.purpose != "personal"
    # /start redirects admin sessions away, so the credential notices must
    # point an admin at the page that can mint a personal token.
    credential_help_href = "/start"
    credential_help_label = "Open /start"
    if role == "admin" and get_session_user_id(request) is not None:
        credential_help_href = "/profile#api-tokens"
        credential_help_label = "Open /profile"
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "active_page": "chat",
            "anonymous_playground": anonymous_playground,
            "missing_personal_token": missing_personal_token,
            "credential_help_href": credential_help_href,
            "credential_help_label": credential_help_label,
        },
    )
