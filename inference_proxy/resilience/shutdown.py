"""Graceful shutdown middleware for the inference proxy.

Intercepts incoming requests during shutdown and returns 503 to prevent
new work from being accepted while in-flight requests drain.

Per D-09: When the gateway receives a shutdown signal, new requests get
503 via middleware and in-flight requests complete.

Per D-12: The ``/health`` endpoint is exempt from shutdown rejection --
it continues to return 200 so health probes from orchestrators or load
balancers can detect the gateway is still alive during drain.

Per T-05-07: The middleware only checks ``app.state.shutting_down``
(a boolean set exclusively by the lifespan shutdown path).  No user
input influences the decision.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class ShutdownMiddleware(BaseHTTPMiddleware):
    """Return 503 for new requests when the gateway is shutting down.

    Reads ``request.app.state.shutting_down`` on every request.  When
    ``True``, returns a 503 JSON response with an OpenAI-compatible
    error body.  The ``/health`` endpoint is exempt (per D-12).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check shutdown state and either reject or pass through."""
        shutting_down: bool = getattr(request.app.state, "shutting_down", False)

        if shutting_down and request.url.path != "/health":
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Gateway is shutting down",
                        "type": "server_error",
                        "code": "shutting_down",
                    },
                },
            )

        return await call_next(request)
