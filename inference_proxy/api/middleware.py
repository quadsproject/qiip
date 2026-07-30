"""Request logging middleware for structured observability.

Produces a structured JSON log entry for every HTTP request containing
method, path, status_code, duration_ms, and target_node (per OBSV-01).

Per D-01: Single middleware, not per-route logging.
Per D-02: OBSV-01 minimum fields only.
Per D-03: Logs ALL requests; target_node is null for non-proxy routes.
Per D-04: Reads target_node from request.state (set by route handlers).
"""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, duration, and target node."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Time the request, read target_node from state, and emit a log entry."""
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        target_node: str | None = getattr(request.state, "target_node", None)

        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            target_node=target_node,
        )
        return response
