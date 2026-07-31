"""OpenAI-compatible API route handlers for the inference proxy.

Provides FastAPI route handlers for:
- POST /v1/chat/completions (streaming + non-streaming)
- POST /v1/completions (streaming + non-streaming)
- GET /v1/models (aggregated model listing from registry)

Route handlers depend on abstractions (ProxyClient, NodeRegistry) via
dependency injection, following the Dependency Inversion Principle.

Non-streaming requests use ProxyClient.forward() for JSON pass-through.
Streaming requests use httpx-sse for upstream SSE consumption and
FastAPI's EventSourceResponse for downstream re-emission.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends
from fastapi import Request as StarletteRequest
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, format_sse_event
from httpx_sse import aconnect_sse
from starlette.background import BackgroundTask

from inference_proxy.api.errors import (
    map_proxy_error,
    model_not_found_error,
    model_unavailable_error,
    no_nodes_error,
)
from inference_proxy.config.dependencies import (
    get_circuit_breaker_registry,
    get_node_selector,
    get_proxy_client,
    get_request_metrics,
    get_settings,
)
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.models.openai import ChatCompletionRequest, CompletionRequest
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics

logger = structlog.get_logger()

router = APIRouter()


def _select_error(
    model: str | None,
    node_selector: NodeSelector,
) -> tuple[int, Any]:
    """Return the appropriate error when node selection fails.

    Distinguishes between:
    - 503 no_nodes: no nodes registered at all
    - 404 model_not_found: nodes exist but none (any status) serve the model
    - 503 model_unavailable: nodes serve the model but all are draining/unhealthy
    """
    all_nodes = node_selector._registry.get_all()
    if not all_nodes:
        return no_nodes_error()
    if model and not node_selector.has_model(model):
        return model_not_found_error(model)
    if model and node_selector.has_model(model):
        return model_unavailable_error(model)
    return no_nodes_error()


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception should trigger a retry on another node.

    Retryable exceptions:
    - ConnectError: backend is unreachable
    - TimeoutException: backend timed out (includes ReadTimeout)
    - HTTPStatusError with status >= 500: backend returned a server error
    """
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


def _record_failure_and_trip(
    node: Node,
    circuit_breaker_registry: CircuitBreakerRegistry,
    node_selector: NodeSelector,
) -> None:
    """Record a failure in the circuit breaker and trip to UNHEALTHY if open.

    Per D-07: When the circuit breaker trips (is_open becomes True),
    the node is marked UNHEALTHY in the registry so it exits the
    routing pool.
    """
    breaker = circuit_breaker_registry.get_or_create(node.node_id)
    breaker.record_failure()
    if breaker.is_open and node_selector._registry.update_status(
        node.node_id,
        NodeStatus.UNHEALTHY,
        allowed_from={NodeStatus.HEALTHY},
    ):
        logger.info(
            "circuit breaker tripped, node marked unhealthy",
            node_id=node.node_id,
        )


async def _proxy_non_streaming(
    endpoint_path: str,
    body: dict[str, Any],
    node_selector: NodeSelector,
    proxy: ProxyClient,
    circuit_breaker_registry: CircuitBreakerRegistry,
    request_metrics: RequestMetrics,
    max_retries: int = 3,
    starlette_request: StarletteRequest | None = None,
) -> JSONResponse:
    """Forward a non-streaming request with retry-on-failover.

    Retries on a different node when the current node fails with a
    retryable error (ConnectError, TimeoutException, 5xx).  Each failed
    node is excluded from subsequent selection via ``exclude_node_ids``.

    Per T-05-05: retry count bounded by ``max_retries``; each retry
    goes to a different node.
    """
    model = body.get("model")
    excluded: set[str] = set()
    last_error_response: JSONResponse | None = None
    first_attempt = True

    for attempt in range(1, max_retries + 1):
        reservation = node_selector.select_and_reserve(
            model=model,
            exclude_node_ids=excluded or None,
        )
        if reservation is None:
            if last_error_response is not None:
                return last_error_response
            status, error_resp = _select_error(model, node_selector)
            return JSONResponse(content=error_resp.model_dump(), status_code=status)
        node = reservation.node

        if first_attempt:
            request_metrics.record_request(node.node_id, model)
            first_attempt = False
        else:
            request_metrics.record_node_attempt(node.node_id)

        if starlette_request is not None:
            starlette_request.state.target_node = node.endpoint

        url = f"http://{node.endpoint}{endpoint_path}"
        try:
            response = await proxy.forward("POST", url, body)
            circuit_breaker_registry.get_or_create(node.node_id).record_success()
            try:
                content = response.json()
            except (json.JSONDecodeError, ValueError):
                content = {"raw": response.text}
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception as exc:
            _record_failure_and_trip(node, circuit_breaker_registry, node_selector)
            status, error_resp = map_proxy_error(exc)
            last_error_response = JSONResponse(
                content=error_resp.model_dump(),
                status_code=status,
            )
            if _is_retryable(exc):
                excluded.add(node.node_id)
                logger.warning(
                    "retrying on different node",
                    failed_node=node.node_id,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                )
                continue
            return last_error_response
        finally:
            reservation.release()

    # All retries exhausted
    assert last_error_response is not None
    return last_error_response


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    starlette_request: StarletteRequest,
    node_selector: NodeSelector = Depends(get_node_selector),
    proxy: ProxyClient = Depends(get_proxy_client),
    circuit_breaker_registry: CircuitBreakerRegistry = Depends(
        get_circuit_breaker_registry,
    ),
    request_metrics: RequestMetrics = Depends(get_request_metrics),
) -> JSONResponse | EventSourceResponse:
    """Proxy a chat completion request to a vLLM backend.

    When ``stream`` is true, returns an SSE stream of token chunks.
    Otherwise, returns the full JSON response from the backend.
    """
    body = request.model_dump(exclude_none=True)
    settings = get_settings()
    if request.stream:
        return await _stream_completion(
            endpoint_path="/v1/chat/completions",
            body=body,
            node_selector=node_selector,
            proxy=proxy,
            circuit_breaker_registry=circuit_breaker_registry,
            request_metrics=request_metrics,
            starlette_request=starlette_request,
        )
    return await _proxy_non_streaming(
        "/v1/chat/completions",
        body,
        node_selector,
        proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        max_retries=settings.routing.max_retries,
        starlette_request=starlette_request,
    )


@router.post("/v1/completions", response_model=None)
async def text_completions(
    request: CompletionRequest,
    starlette_request: StarletteRequest,
    node_selector: NodeSelector = Depends(get_node_selector),
    proxy: ProxyClient = Depends(get_proxy_client),
    circuit_breaker_registry: CircuitBreakerRegistry = Depends(
        get_circuit_breaker_registry,
    ),
    request_metrics: RequestMetrics = Depends(get_request_metrics),
) -> JSONResponse | EventSourceResponse:
    """Proxy a text completion request to a vLLM backend.

    When ``stream`` is true, returns an SSE stream of token chunks.
    Otherwise, returns the full JSON response from the backend.
    """
    body = request.model_dump(exclude_none=True)
    settings = get_settings()
    if request.stream:
        return await _stream_completion(
            endpoint_path="/v1/completions",
            body=body,
            node_selector=node_selector,
            proxy=proxy,
            circuit_breaker_registry=circuit_breaker_registry,
            request_metrics=request_metrics,
            starlette_request=starlette_request,
        )
    return await _proxy_non_streaming(
        "/v1/completions",
        body,
        node_selector,
        proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        max_retries=settings.routing.max_retries,
        starlette_request=starlette_request,
    )


@router.get("/v1/models")
async def list_models(
    node_selector: NodeSelector = Depends(get_node_selector),
) -> JSONResponse:
    """Return an OpenAI-compatible list of available models.

    Aggregates model names from all healthy registered nodes,
    deduplicating by model name.  DRAINING nodes are excluded so
    clients only see models that can accept new requests.
    """
    nodes = node_selector._registry.get_all()
    models_seen: dict[str, dict[str, str | int]] = {}

    for node in nodes:
        if node.status != NodeStatus.HEALTHY:
            continue
        if node.model and node.model not in models_seen:
            models_seen[node.model] = {
                "id": node.model,
                "object": "model",
                "created": 0,
                "owned_by": "vllm",
            }

    return JSONResponse(
        content={
            "object": "list",
            "data": list(models_seen.values()),
        }
    )


async def _stream_completion(
    endpoint_path: str,
    body: dict[str, Any],
    node_selector: NodeSelector,
    proxy: ProxyClient,
    circuit_breaker_registry: CircuitBreakerRegistry,
    request_metrics: RequestMetrics,
    starlette_request: StarletteRequest | None = None,
) -> JSONResponse | EventSourceResponse:
    """Stream SSE events from a vLLM backend to the client.

    Consumes upstream SSE events via ``httpx-sse`` and re-emits them
    using FastAPI's ``EventSourceResponse``.

    Uses ``format_sse_event(data_str=...)`` to avoid double JSON encoding
    (upstream data is already JSON-serialised by vLLM).

    Records success/failure in the circuit breaker but does NOT retry
    mid-stream.  Per plan: streaming requests may record failures but
    do not retry once the SSE connection has started.
    """
    model = body.get("model")
    reservation = node_selector.select_and_reserve(model=model)
    if reservation is None:
        status, error_resp = _select_error(model, node_selector)
        return JSONResponse(content=error_resp.model_dump(), status_code=status)
    node = reservation.node

    if starlette_request is not None:
        starlette_request.state.target_node = node.endpoint

    url = f"http://{node.endpoint}{endpoint_path}"

    request_metrics.record_request(node.node_id, model)

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            async with aconnect_sse(
                proxy.client, "POST", url, json=body
            ) as event_source:
                event_source.response.raise_for_status()
                async for sse in event_source.aiter_sse():
                    if sse.data == "[DONE]":
                        yield format_sse_event(data_str="[DONE]")
                        circuit_breaker_registry.get_or_create(
                            node.node_id,
                        ).record_success()
                        return
                    yield format_sse_event(data_str=sse.data)
        except Exception as exc:
            logger.error("streaming proxy error", error=str(exc), url=url)
            _record_failure_and_trip(
                node,
                circuit_breaker_registry,
                node_selector,
            )
            _, error_resp = map_proxy_error(exc)
            error_json = json.dumps(error_resp.model_dump())
            yield format_sse_event(data_str=error_json)
            yield format_sse_event(data_str="[DONE]")
        finally:
            reservation.release()

    return EventSourceResponse(
        event_generator(),
        background=BackgroundTask(reservation.release),
    )
