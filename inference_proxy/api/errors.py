"""Map proxy exceptions to OpenAI-compatible error responses.

This is a pure function module that converts ``httpx`` exceptions and
generic errors into ``(status_code, ErrorResponse)`` tuples using the
existing Pydantic error models from ``inference_proxy.models.openai``.
"""

from __future__ import annotations

import httpx
import structlog

from inference_proxy.models.openai import ErrorDetail, ErrorResponse

logger = structlog.get_logger()


def map_proxy_error(exc: Exception) -> tuple[int, ErrorResponse]:
    """Map a proxy exception to an OpenAI-compatible error response.

    Args:
        exc: The exception raised during proxying.

    Returns:
        A tuple of ``(http_status_code, ErrorResponse)``.
    """
    if isinstance(exc, httpx.ConnectError):
        logger.error("backend connection failed", error=str(exc))
        return 502, ErrorResponse(
            error=ErrorDetail(
                message="Failed to connect to inference backend",
                type="upstream_error",
                code="backend_unavailable",
            )
        )

    if isinstance(exc, httpx.TimeoutException):
        logger.error("backend request timed out", error=str(exc))
        return 504, ErrorResponse(
            error=ErrorDetail(
                message="Inference backend timed out",
                type="upstream_error",
                code="backend_timeout",
            )
        )

    if isinstance(exc, httpx.TransportError):
        logger.error("backend transport failed", error=str(exc))
        return 502, ErrorResponse(
            error=ErrorDetail(
                message="Inference backend connection failed during request",
                type="upstream_error",
                code="backend_transport_error",
            )
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        response_text = exc.response.text
        logger.error(
            "backend returned error status",
            status_code=status,
            response_text=response_text,
        )
        safe_message = response_text[:200] if response_text else ""
        return status, ErrorResponse(
            error=ErrorDetail(
                message=f"Inference backend returned error: {safe_message}",
                type="upstream_error",
                code=str(status),
            )
        )

    logger.error("unexpected proxy error", error=str(exc), exc_info=True)
    return 500, ErrorResponse(
        error=ErrorDetail(
            message="Internal gateway error",
            type="server_error",
            code="internal_error",
        )
    )


def model_not_found_error(model: str) -> tuple[int, ErrorResponse]:
    """Return a 404 error when the requested model is not served by any node.

    Used when no registered node (regardless of health status) serves the
    requested model name (D-04).

    Args:
        model: The model name that was requested.

    Returns:
        A tuple of ``(404, ErrorResponse)`` with a ``model_not_found`` code.
    """
    return 404, ErrorResponse(
        error=ErrorDetail(
            message=f"The model '{model}' does not exist",
            type="invalid_request_error",
            code="model_not_found",
        )
    )


def model_unavailable_error(model: str) -> tuple[int, ErrorResponse]:
    """Return a 503 error when the model exists but all nodes are unavailable.

    Used when nodes are registered for the requested model but all are
    in UNHEALTHY or DRAINING status (D-06).

    Args:
        model: The model name that was requested.

    Returns:
        A tuple of ``(503, ErrorResponse)`` with a ``model_unavailable`` code.
    """
    return 503, ErrorResponse(
        error=ErrorDetail(
            message=f"The model '{model}' is temporarily unavailable",
            type="server_error",
            code="model_unavailable",
        )
    )


def no_nodes_error() -> tuple[int, ErrorResponse]:
    """Return a 503 error response for when no inference nodes are available.

    Returns:
        A tuple of ``(503, ErrorResponse)`` with a ``no_nodes`` code.
    """
    return 503, ErrorResponse(
        error=ErrorDetail(
            message="No inference nodes available",
            type="server_error",
            code="no_nodes",
        )
    )
