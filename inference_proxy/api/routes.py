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

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends
from fastapi import Request as StarletteRequest
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, format_sse_event
from httpx_sse import EventSource, aconnect_sse
from starlette.background import BackgroundTask

from inference_proxy.api.errors import (
    map_proxy_error,
    model_not_found_error,
    model_unavailable_error,
    no_nodes_error,
)
from inference_proxy.auth.dependencies import get_api_auth, get_auth_store
from inference_proxy.auth.models import TokenAuth
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_circuit_breaker_registry,
    get_node_selector,
    get_proxy_client,
    get_request_metrics,
    get_settings,
)
from inference_proxy.config.settings import Settings
from inference_proxy.models.endpoint import build_backend_url
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.models.openai import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeReservation, NodeSelector
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
    - TransportError: backend connection or protocol failed
    - HTTPStatusError with status >= 500: backend returned a server error
    """
    if isinstance(exc, httpx.TransportError):
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


def _proxy_error_response(
    status: int,
    error: ErrorResponse,
    *,
    failover_exhausted: bool = False,
    attempts: int = 0,
) -> JSONResponse:
    """Build an OpenAI-compatible proxy error response.

    Exhaustion is marked here, outside ``map_proxy_error``, so the shared
    streaming HTTP-status mapping remains untouched for PR 3.
    """
    content = error.model_dump()
    headers: dict[str, str] | None = None
    if failover_exhausted:
        content["error"]["code"] = "failover_exhausted"
        headers = {
            "X-Inference-Proxy-Failover": "exhausted",
            "X-Inference-Proxy-Attempts": str(attempts),
        }
    return JSONResponse(content=content, status_code=status, headers=headers)


def _response_content(response: httpx.Response) -> Any:
    """Decode an upstream response without changing its JSON shape."""
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return {"raw": response.text}


def _extract_usage(payload: Any) -> tuple[int, int, int] | None:
    """Extract OpenAI ``usage`` token counts from a response payload.

    Returns ``(prompt, completion, total)`` or None when the payload is not
    a dict or carries no usable usage counters.
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or not isinstance(completion_tokens, int)
        or not isinstance(total_tokens, int)
    ):
        return None
    return prompt_tokens, completion_tokens, total_tokens


def _extract_usage_from_data(data: str) -> tuple[int, int, int] | None:
    """Extract usage from an SSE ``data`` JSON payload (AUTH-04)."""
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None
    return _extract_usage(payload)


async def _record_usage(
    store: AuthStore,
    auth: TokenAuth,
    *,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Persist one token-attributed usage row off the event loop."""
    await asyncio.to_thread(
        store.record_usage,
        user_id=auth.user.id,
        token_id=auth.token.id,
        model=model or "",
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def _close_streaming_attempt(
    stack: AsyncExitStack,
    reservation: NodeReservation,
    *,
    node_id: str,
) -> None:
    """Close one upstream context and always release its node reservation."""
    try:
        await stack.aclose()
    except Exception:
        logger.warning(
            "failed to close upstream streaming context",
            node_id=node_id,
            exc_info=True,
        )
    finally:
        reservation.release()


class _StreamingSession:
    """Own the successful upstream stream and its reservation."""

    def __init__(
        self,
        event_source: EventSource,
        stack: AsyncExitStack,
        reservation: NodeReservation,
    ) -> None:
        self.event_source = event_source
        self._stack = stack
        self._reservation = reservation
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        """Close the upstream and release its reservation exactly once."""
        async with self._close_lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(
                    _close_streaming_attempt(
                        self._stack,
                        self._reservation,
                        node_id=self._reservation.node.node_id,
                    )
                )
            close_task = self._close_task
        await asyncio.shield(close_task)


async def _stream_events(
    session: _StreamingSession,
    node: Node,
    url: str,
    circuit_breaker_registry: CircuitBreakerRegistry,
    node_selector: NodeSelector,
    usage_store: AuthStore | None = None,
    usage_auth: TokenAuth | None = None,
    model: str = "",
    endpoint: str = "",
) -> AsyncGenerator[bytes, None]:
    """Relay one established upstream stream without attempting failover.

    When a bearer token authenticates the request, the OpenAI ``usage``
    carried by the final streamed chunk is recorded once for usage
    tracking (AUTH-04). The recorded usage is the last non-empty usage
    observation, matching the OpenAI convention that the final chunk is
    authoritative.
    """
    usage: tuple[int, int, int] | None = None
    recorded = False

    async def _flush() -> None:
        nonlocal recorded
        if recorded or usage_store is None or usage_auth is None:
            return
        recorded = True
        prompt, completion, total = usage or (0, 0, 0)
        await _record_usage(
            usage_store,
            usage_auth,
            model=model,
            endpoint=endpoint,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    try:
        async for sse in session.event_source.aiter_sse():
            if sse.data == "[DONE]":
                yield format_sse_event(data_str="[DONE]")
                circuit_breaker_registry.get_or_create(
                    node.node_id,
                ).record_success()
                await _flush()
                return
            found = _extract_usage_from_data(sse.data)
            if found is not None:
                usage = found
            yield format_sse_event(data_str=sse.data)
        await _flush()
    except Exception as exc:
        logger.error("streaming proxy error", error=str(exc), url=url)
        _record_failure_and_trip(
            node,
            circuit_breaker_registry,
            node_selector,
        )
        _, error_resp = map_proxy_error(exc)
        error_json = json.dumps(error_resp.model_dump())
        await _flush()
        yield format_sse_event(data_str=error_json)
        yield format_sse_event(data_str="[DONE]")
    finally:
        await session.close()


async def _proxy_non_streaming(
    endpoint_path: str,
    body: dict[str, Any],
    node_selector: NodeSelector,
    proxy: ProxyClient,
    circuit_breaker_registry: CircuitBreakerRegistry,
    request_metrics: RequestMetrics,
    max_attempts: int = 3,
    starlette_request: StarletteRequest | None = None,
    usage_store: AuthStore | None = None,
    usage_auth: TokenAuth | None = None,
) -> JSONResponse:
    """Forward a non-streaming request with retry-on-failover.

    Retries on a different node when the current node fails with a
    retryable error (ConnectError, TimeoutException, 5xx).  Each failed
    node is excluded from subsequent selection via ``exclude_node_ids``.

    ``max_attempts`` includes the initial request. Each retry goes to a
    different node.

    When a bearer token authenticates the request, OpenAI usage from the
    successful backend response is recorded for usage tracking (AUTH-04).
    """
    model = body.get("model")
    excluded: set[str] = set()
    last_error: tuple[int, ErrorResponse] | None = None
    first_attempt = True
    attempts = 0

    for _ in range(max_attempts):
        reservation = node_selector.select_and_reserve(
            model=model,
            exclude_node_ids=excluded or None,
        )
        if reservation is None:
            if last_error is not None:
                status, error = last_error
                return _proxy_error_response(
                    status,
                    error,
                    failover_exhausted=True,
                    attempts=attempts,
                )
            status, error_resp = _select_error(model, node_selector)
            return JSONResponse(content=error_resp.model_dump(), status_code=status)
        node = reservation.node
        attempts += 1

        if first_attempt:
            request_metrics.record_request(node.node_id, model)
            first_attempt = False
        else:
            request_metrics.record_node_attempt(node.node_id)

        if starlette_request is not None:
            starlette_request.state.target_node = node.endpoint

        url = build_backend_url(node.endpoint, endpoint_path)
        try:
            response = await proxy.forward("POST", url, body)
            # A client error proves reachability but not successful inference.
            # Keep it neutral so it cannot erase earlier backend failures.
            if response.is_success:
                circuit_breaker_registry.get_or_create(node.node_id).record_success()
            content = _response_content(response)
            if (
                response.is_success
                and usage_store is not None
                and usage_auth is not None
            ):
                usage = _extract_usage(content)
                if usage is not None:
                    prompt_tokens, completion_tokens, total_tokens = usage
                    await _record_usage(
                        usage_store,
                        usage_auth,
                        model=body.get("model") or "",
                        endpoint=endpoint_path,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception as exc:
            retryable = _is_retryable(exc)
            _record_failure_and_trip(
                node,
                circuit_breaker_registry,
                node_selector,
            )
            status, error_resp = map_proxy_error(exc)
            last_error = (status, error_resp)
            if retryable:
                excluded.add(node.node_id)
                logger.warning(
                    "retrying on different node",
                    failed_node=node.node_id,
                    attempt=attempts,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                continue
            return _proxy_error_response(status, error_resp)
        finally:
            reservation.release()

    if last_error is None:
        raise RuntimeError("attempt budget exhausted without a backend failure")
    status, error = last_error
    return _proxy_error_response(
        status,
        error,
        failover_exhausted=True,
        attempts=attempts,
    )


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
    settings: Settings = Depends(get_settings),
    usage_store: AuthStore = Depends(get_auth_store),
    usage_auth: TokenAuth | None = Depends(get_api_auth),
) -> JSONResponse | EventSourceResponse:
    """Proxy a chat completion request to a vLLM backend.

    When ``stream`` is true, returns an SSE stream of token chunks.
    Otherwise, returns the full JSON response from the backend.

    A valid ``Authorization: Bearer <token>`` header (AUTH-03) authenticates
    the request for usage tracking; enforcement is config-gated.
    """
    body = request.model_dump(exclude_none=True)
    # Keep top-level optional parameters clean while preserving the distinction
    # between an omitted message field and an explicitly supplied null. Tool-call
    # assistant turns canonically carry ``content: null``.
    body["messages"] = [
        message.model_dump(exclude_unset=True) for message in request.messages
    ]
    if request.stream:
        return await _stream_completion(
            endpoint_path="/v1/chat/completions",
            body=body,
            node_selector=node_selector,
            proxy=proxy,
            circuit_breaker_registry=circuit_breaker_registry,
            request_metrics=request_metrics,
            starlette_request=starlette_request,
            max_attempts=settings.routing.max_attempts,
            handshake_timeout=settings.routing.timeout,
            usage_store=usage_store,
            usage_auth=usage_auth,
        )
    return await _proxy_non_streaming(
        "/v1/chat/completions",
        body,
        node_selector,
        proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        max_attempts=settings.routing.max_attempts,
        starlette_request=starlette_request,
        usage_store=usage_store,
        usage_auth=usage_auth,
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
    settings: Settings = Depends(get_settings),
    usage_store: AuthStore = Depends(get_auth_store),
    usage_auth: TokenAuth | None = Depends(get_api_auth),
) -> JSONResponse | EventSourceResponse:
    """Proxy a text completion request to a vLLM backend.

    When ``stream`` is true, returns an SSE stream of token chunks.
    Otherwise, returns the full JSON response from the backend.

    A valid ``Authorization: Bearer <token>`` header (AUTH-03) authenticates
    the request for usage tracking; enforcement is config-gated.
    """
    body = request.model_dump(exclude_none=True)
    if request.stream:
        return await _stream_completion(
            endpoint_path="/v1/completions",
            body=body,
            node_selector=node_selector,
            proxy=proxy,
            circuit_breaker_registry=circuit_breaker_registry,
            request_metrics=request_metrics,
            starlette_request=starlette_request,
            max_attempts=settings.routing.max_attempts,
            handshake_timeout=settings.routing.timeout,
            usage_store=usage_store,
            usage_auth=usage_auth,
        )
    return await _proxy_non_streaming(
        "/v1/completions",
        body,
        node_selector,
        proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        max_attempts=settings.routing.max_attempts,
        starlette_request=starlette_request,
        usage_store=usage_store,
        usage_auth=usage_auth,
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
                "owned_by": node.engine,
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
    max_attempts: int = 3,
    handshake_timeout: float = 30,
    usage_store: AuthStore | None = None,
    usage_auth: TokenAuth | None = None,
) -> JSONResponse | EventSourceResponse:
    """Establish a backend SSE stream, then expose it to the client.

    Retryable failures before upstream response headers fail over to a
    different node. The downstream 200 is committed only after a backend
    returns a successful response. The complete pre-stream retry phase is
    bounded by ``handshake_timeout``.

    Once streaming begins, events are re-emitted with their existing JSON
    payloads and failures are reported in-band without retrying.
    """
    model = body.get("model")
    excluded: set[str] = set()
    last_error: tuple[int, ErrorResponse] | None = None
    attempts = 0
    first_attempt = True
    deadline = asyncio.get_running_loop().time() + handshake_timeout

    for _ in range(max_attempts):
        if asyncio.get_running_loop().time() >= deadline:
            break

        reservation = node_selector.select_and_reserve(
            model=model,
            exclude_node_ids=excluded or None,
        )
        if reservation is None:
            if last_error is not None:
                status, error = last_error
                return _proxy_error_response(
                    status,
                    error,
                    failover_exhausted=True,
                    attempts=attempts,
                )
            status, error_resp = _select_error(model, node_selector)
            return JSONResponse(content=error_resp.model_dump(), status_code=status)

        node = reservation.node
        attempts += 1
        if first_attempt:
            request_metrics.record_request(node.node_id, model)
            first_attempt = False
        else:
            request_metrics.record_node_attempt(node.node_id)

        if starlette_request is not None:
            starlette_request.state.target_node = node.endpoint

        url = build_backend_url(node.endpoint, endpoint_path)
        stack = AsyncExitStack()
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise httpx.ReadTimeout(
                    "Streaming upstream handshake exceeded routing timeout"
                )
            try:
                async with asyncio.timeout(remaining):
                    event_source = await stack.enter_async_context(
                        aconnect_sse(proxy.client, "POST", url, json=body)
                    )
                    response = event_source.response
                    if not response.is_success:
                        await response.aread()
            except TimeoutError as exc:
                raise httpx.ReadTimeout(
                    "Streaming upstream handshake exceeded routing timeout"
                ) from exc

            if response.status_code >= 500:
                # D3: the streamed body was read above, so error mapping can
                # safely inspect response.text after this context is closed.
                response.raise_for_status()

            if not response.is_success:
                # Preserve 4xx responses verbatim without treating malformed
                # client traffic as positive circuit-breaker evidence.
                content = _response_content(response)
                await _close_streaming_attempt(
                    stack,
                    reservation,
                    node_id=node.node_id,
                )
                return JSONResponse(
                    content=content,
                    status_code=response.status_code,
                )
        except Exception as exc:
            await _close_streaming_attempt(
                stack,
                reservation,
                node_id=node.node_id,
            )
            retryable = _is_retryable(exc)
            _record_failure_and_trip(
                node,
                circuit_breaker_registry,
                node_selector,
            )
            status, error_resp = map_proxy_error(exc)
            last_error = (status, error_resp)
            if retryable:
                excluded.add(node.node_id)
                logger.warning(
                    "retrying streaming handshake on different node",
                    failed_node=node.node_id,
                    attempt=attempts,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                continue
            return _proxy_error_response(status, error_resp)

        # Only the successful context crosses the handler/generator boundary.
        # Every failed context was closed before the next selection attempt.
        session = _StreamingSession(event_source, stack, reservation)

        return EventSourceResponse(
            _stream_events(
                session,
                node,
                url,
                circuit_breaker_registry,
                node_selector,
                usage_store=usage_store,
                usage_auth=usage_auth,
                model=body.get("model") or "",
                endpoint=endpoint_path,
            ),
            background=BackgroundTask(session.close),
        )

    if last_error is None:
        raise RuntimeError("attempt budget exhausted without a backend failure")
    status, error = last_error
    return _proxy_error_response(
        status,
        error,
        failover_exhausted=True,
        attempts=attempts,
    )
