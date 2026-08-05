"""Chat page route for the chatbot playground UI.

Serves the chat HTML shell at /chat. Client-side JS handles
model selection, message sending, and streaming responses.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from inference_proxy.api.templating import templates

chat_router = APIRouter(tags=["chat"])


@chat_router.get("/chat", response_class=HTMLResponse)
async def chat(request: Request) -> HTMLResponse:
    """Render the chat playground HTML shell."""
    return templates.TemplateResponse(request=request, name="chat.html")
