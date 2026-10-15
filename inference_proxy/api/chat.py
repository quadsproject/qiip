"""Chat page route for the chatbot playground UI.

Serves the chat HTML shell at ``/chat``. Client-side JS handles model
selection, message sending, and streaming responses, and runs the playground on
the signed-in viewer's own token (``/onboarding/token/reveal``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from inference_proxy.api.templating import signin_response, templates
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
) -> HTMLResponse:
    """Render the chat playground HTML shell for signed-in viewers.

    Anonymous visitors get the sign-in page. The page shell says whether the
    viewer chats anonymously (an admin, the full-access trust list, or the
    local admin: none of them has a personal token to reveal) or on their own
    token, which the page reveals through ``/onboarding/token/reveal`` and
    sends on every ``/v1`` call, so usage is attributed and admin-only nodes
    stay unreachable.
    """
    role = viewer_role(request, settings)
    if role is None:
        return signin_response(
            request,
            oauth_enabled=settings.oauth.enabled,
            sessions_available=settings.auth.session_secret is not None,
        )
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "active_page": "chat",
            # Admins (admin role, full-access trust list, and the local admin)
            # have no personal token to reveal: they chat anonymously.
            "anonymous_playground": (
                role == "admin" or session_user_has_full_access(request, settings)
            ),
        },
    )
