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


class NoStoreMiddleware(BaseHTTPMiddleware):
    """Forbid browser caching of dynamic responses.

    Every non-static response is either an HTML shell that embeds the
    viewer's role (``VIEWER_ROLE``) or a JSON endpoint returning
    user-specific data, so a stale cached copy can render the wrong surface
    (e.g. a cached admin shell for a non-admin session, causing admin
    fetches, Basic popups, and endless retries). Static assets are
    content-versioned by ``static_asset_url`` and may be cached, so they are
    the only responses excluded.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Add ``Cache-Control: no-store`` to every non-static response."""
        response = await call_next(request)
        if not request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response


def _loggable_path(path: str) -> str:
    """Return *path* with capability-URL secrets removed.

    ``/s/{id}`` setup links are bearer credentials for their 15-minute life
    (the id alone fetches a script containing an API token), so the id must
    not be persisted in logs.
    """
    if path.startswith("/s/"):
        return "/s/[redacted]"
    return path


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
            path=_loggable_path(request.url.path),
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            target_node=target_node,
        )
        return response
