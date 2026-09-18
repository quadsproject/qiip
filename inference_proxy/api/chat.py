"""Chat page route for the chatbot playground UI.

Serves the chat HTML shell at /chat. Client-side JS handles
model selection, message sending, and streaming responses.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from inference_proxy.api.templating import templates, user_home_redirect
from inference_proxy.config.dependencies import get_settings, viewer_role
from inference_proxy.config.settings import Settings

chat_router = APIRouter(tags=["chat"])


@chat_router.get("/chat", response_class=HTMLResponse, response_model=None)
async def chat(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HTMLResponse | RedirectResponse:
    """Render the chat playground HTML shell (normal users go to /start)."""
    if viewer_role(request, settings) == "user":
        return user_home_redirect()
    return templates.TemplateResponse(
        request=request, name="chat.html", context={"active_page": "chat"}
    )
