"""Inference API route handlers for the inference proxy.

Provides FastAPI route handlers for:
- POST /v1/chat/completions (streaming + non-streaming)
- POST /v1/completions (streaming + non-streaming)
- POST /v1/messages (Anthropic Messages API, streaming + non-streaming)
- POST /v1/messages/count_tokens (Anthropic token counting)
- POST /v1/responses (OpenAI Responses API, streaming + non-streaming)
- GET /v1/models (aggregated model listing from registry)

The Messages and Responses APIs are forwarded to backends that implement
them natively (vLLM and llama.cpp); ``dialects`` holds the few per-format
details the proxy itself needs.

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
from fastapi.sse import EventSourceResponse
from httpx_sse import EventSource, aconnect_sse
from starlette.background import BackgroundTask

from inference_proxy.api.dialects import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    Dialect,
    relay_frame,
)
from inference_proxy.api.errors import (
    map_proxy_error,
    model_not_found_error,
    model_not_permitted_error,
    model_unavailable_error,
    no_nodes_error,
)
from inference_proxy.api.system_messages import (
    inline_anthropic_system_messages,
    inline_responses_system_messages,
)
from inference_proxy.auth.dependencies import (
    get_api_auth,
    get_auth_store,
    get_messages_api_auth,
)
from inference_proxy.auth.models import TokenAuth
from inference_proxy.auth.scopes import auth_scope
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
    ErrorDetail,
    ErrorResponse,
)
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import (
    NodeReservation,
    NodeSelector,
    _in_scope,
)
from inference_proxy.routing.request_metrics import RequestMetrics

logger = structlog.get_logger()

router = APIRouter()


def _select_error(
    model: str | None,
    node_selector: NodeSelector,
    allowed_node_ids: frozenset[str] | None = None,
    owner: str | None = None,
) -> tuple[int, Any]:
    """Return the appropriate error when node selection fails.

    Distinguishes between:
    - 503 no_nodes: no nodes registered at all (or none within scope)
    - 404 model_not_found: nodes exist but none serve the model
    - 503 model_unavailable: nodes serve the model but all are draining/unhealthy
    """
    all_nodes = node_selector._registry.get_all()
    scoped = [node for node in all_nodes if _in_scope(node, allowed_node_ids, owner)]
    if not scoped:
        return no_nodes_error()
    if model:
        if not any(node.model == model for node in scoped):
            return model_not_found_error(model)
        return model_unavailable_error(model)
    return no_nodes_error()


def _model_scope_denied(
    auth: TokenAuth | None,
    model: str | None,
    dialect: Dialect = OPENAI_CHAT,
) -> JSONResponse | None:
    """Return a 403 response when the token's model scope excludes *model*.

    ``model_scope`` is None for unrestricted tokens. Every inference POST
    must call this before selecting a node: a route that skips it serves a
    scoped token any model. ``model`` is a required request field today, so
    the empty case is purely defensive.
    """
    if auth is None or auth.token.model_scope is None:
        return None
    if model and model in auth.token.model_scope:
        return None
    status, error = model_not_permitted_error(model or "")
    return JSONResponse(
        content=dialect.error_content(error, status), status_code=status
    )


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
    dialect: Dialect = OPENAI_CHAT,
) -> JSONResponse:
    """Build a proxy error response in the client's API format.

    Exhaustion is marked here, outside ``map_proxy_error``, so the shared
    streaming HTTP-status mapping remains untouched for PR 3.
    """
    headers: dict[str, str] | None = None
    code: str | None = None
    if failover_exhausted:
        code = "failover_exhausted"
        headers = {
            "X-Inference-Proxy-Failover": "exhausted",
            "X-Inference-Proxy-Attempts": str(attempts),
        }
    content = dialect.error_content(error, status, code=code)
    return JSONResponse(content=content, status_code=status, headers=headers)


def _invalid_request(
    dialect: Dialect,
    message: str,
    *,
    param: str | None = None,
) -> JSONResponse:
    """Reject a malformed request body with a 400 in the client's format."""
    error = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type="invalid_request_error",
            param=param,
            code="invalid_request",
        )
    )
    return JSONResponse(content=dialect.error_content(error, 400), status_code=400)


def _response_content(response: httpx.Response) -> Any:
    """Decode an upstream response without changing its JSON shape."""
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return {"raw": response.text}


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
    """Persist one token-attributed usage row off the event loop.

    A storage failure is logged, never raised: it happens after the backend
    already served the request, so it must not count against the node,
    trigger a retry on another node, or turn a delivered response into an
    error.
    """
    try:
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
    except Exception:
        logger.warning(
            "failed to record token usage",
            endpoint=endpoint,
            model=model,
            exc_info=True,
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
    dialect: Dialect = OPENAI_CHAT,
) -> AsyncGenerator[bytes, None]:
    """Relay one established upstream stream without attempting failover.

    Events are re-emitted with their payloads and event names unchanged.
    The dialect decides which event ends a successful response (``[DONE]``
    for OpenAI chat, ``message_stop`` for Anthropic, ``response.completed``
    for Responses) and where usage is reported. When a bearer token
    authenticates the request, that usage is recorded once for usage
    tracking (AUTH-04), using the latest observation.

    A failure to record usage is logged and never reported as a backend
    failure: the node served the stream, and the client may already have
    its terminal event.
    """
    tracker = dialect.stream_tracker()
    recorded = False

    async def _flush() -> None:
        nonlocal recorded
        if recorded or usage_store is None or usage_auth is None:
            return
        recorded = True
        prompt, completion, total = tracker.usage or (0, 0, 0)
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
            finished = tracker.observe(sse.event, sse.data)
            yield relay_frame(sse.event, sse.data)
            if finished:
                circuit_breaker_registry.get_or_create(
                    node.node_id,
                ).record_success()
                await _flush()
                return
        await _flush()
    except Exception as exc:
        logger.error("streaming proxy error", error=str(exc), url=url)
        _record_failure_and_trip(
            node,
            circuit_breaker_registry,
            node_selector,
        )
        status, error_resp = map_proxy_error(exc)
        await _flush()
        for frame in dialect.stream_error(error_resp, status):
            yield frame
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
    allowed_node_ids: frozenset[str] | None = None,
    owner: str | None = None,
    dialect: Dialect = OPENAI_CHAT,
) -> JSONResponse:
    """Forward a non-streaming request with retry-on-failover.

    Retries on a different node when the current node fails with a
    retryable error (ConnectError, TimeoutException, 5xx).  Each failed
    node is excluded from subsequent selection via ``exclude_node_ids``.

    ``max_attempts`` includes the initial request. Each retry goes to a
    different node.

    When a bearer token authenticates the request, usage from the
    successful backend response, read the way ``dialect`` reports it, is
    recorded for usage tracking (AUTH-04). qiip's own errors are rendered
    in the dialect's error format; backend errors pass through verbatim.
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
            allowed_node_ids=allowed_node_ids,
            owner=owner,
        )
        if reservation is None:
            if last_error is not None:
                status, error = last_error
                return _proxy_error_response(
                    status,
                    error,
                    failover_exhausted=True,
                    attempts=attempts,
                    dialect=dialect,
                )
            status, error_resp = _select_error(
                model, node_selector, allowed_node_ids, owner
            )
            return _proxy_error_response(status, error_resp, dialect=dialect)
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
                usage = dialect.usage_from_body(content)
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
            return _proxy_error_response(status, error_resp, dialect=dialect)
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
        dialect=dialect,
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
    denied = _model_scope_denied(usage_auth, body.get("model"))
    if denied is not None:
        return denied
    allowed, owner = auth_scope(usage_auth, settings)
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
            allowed_node_ids=allowed,
            owner=owner,
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
        allowed_node_ids=allowed,
        owner=owner,
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
    denied = _model_scope_denied(usage_auth, body.get("model"))
    if denied is not None:
        return denied
    allowed, owner = auth_scope(usage_auth, settings)
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
            allowed_node_ids=allowed,
            owner=owner,
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
        allowed_node_ids=allowed,
        owner=owner,
    )


async def _presented_model_scope(
    request: StarletteRequest,
    store: AuthStore,
) -> list[str] | None:
    """Return the model scope of a presented bearer token, if any.

    Listing never rejects (it stays open exactly as before); a valid scoped
    token simply narrows the list so harnesses only offer usable models.
    """
    scheme, _, raw = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not raw.strip():
        return None
    auth = await asyncio.to_thread(store.resolve_token, raw.strip())
    return auth.token.model_scope if auth is not None else None


async def _read_json_body(
    starlette_request: StarletteRequest,
    dialect: Dialect,
) -> dict[str, Any] | JSONResponse:
    """Parse a pass-through request body, checking only what routing needs.

    The Messages and Responses APIs are forwarded as sent, so only
    ``model`` (used for node selection) and ``stream`` are validated here.
    Everything else is left to the backend, which implements the API
    natively; validating it here would reject fields newer than qiip.
    """
    try:
        payload = await starlette_request.json()
    except ValueError:
        return _invalid_request(dialect, "Request body must be valid JSON")
    if not isinstance(payload, dict):
        return _invalid_request(dialect, "Request body must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return _invalid_request(
            dialect,
            "'model' must be a non-empty string",
            param="model",
        )
    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        return _invalid_request(dialect, "'stream' must be a boolean", param="stream")
    return payload


async def _forward_native(
    endpoint_path: str,
    body: dict[str, Any],
    *,
    dialect: Dialect,
    starlette_request: StarletteRequest,
    node_selector: NodeSelector,
    proxy: ProxyClient,
    circuit_breaker_registry: CircuitBreakerRegistry,
    request_metrics: RequestMetrics,
    settings: Settings,
    usage_store: AuthStore,
    usage_auth: TokenAuth | None,
    record_usage: bool = True,
) -> JSONResponse | EventSourceResponse:
    """Forward a natively supported API request to a node serving its model."""
    denied = _model_scope_denied(usage_auth, body.get("model"), dialect)
    if denied is not None:
        return denied
    allowed, owner = auth_scope(usage_auth, settings)
    store = usage_store if record_usage else None
    if body.get("stream") is True:
        return await _stream_completion(
            endpoint_path=endpoint_path,
            body=body,
            node_selector=node_selector,
            proxy=proxy,
            circuit_breaker_registry=circuit_breaker_registry,
            request_metrics=request_metrics,
            starlette_request=starlette_request,
            max_attempts=settings.routing.max_attempts,
            handshake_timeout=settings.routing.timeout,
            usage_store=store,
            usage_auth=usage_auth,
            allowed_node_ids=allowed,
            owner=owner,
            dialect=dialect,
        )
    return await _proxy_non_streaming(
        endpoint_path,
        body,
        node_selector,
        proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        max_attempts=settings.routing.max_attempts,
        starlette_request=starlette_request,
        usage_store=store,
        usage_auth=usage_auth,
        allowed_node_ids=allowed,
        owner=owner,
        dialect=dialect,
    )


@router.post("/v1/messages", response_model=None)
async def anthropic_messages(
    starlette_request: StarletteRequest,
    node_selector: NodeSelector = Depends(get_node_selector),
    proxy: ProxyClient = Depends(get_proxy_client),
    circuit_breaker_registry: CircuitBreakerRegistry = Depends(
        get_circuit_breaker_registry,
    ),
    request_metrics: RequestMetrics = Depends(get_request_metrics),
    settings: Settings = Depends(get_settings),
    usage_store: AuthStore = Depends(get_auth_store),
    usage_auth: TokenAuth | None = Depends(get_messages_api_auth),
) -> JSONResponse | EventSourceResponse:
    """Proxy an Anthropic Messages API request, as sent by Claude Code.

    The body is forwarded as sent, except that system messages inside the
    conversation become ``<system-reminder>`` user turns so strict chat
    templates accept them without breaking prompt caching (see
    ``system_messages``). The API key may arrive as ``Authorization:
    Bearer`` or as ``x-api-key``.
    """
    body = await _read_json_body(starlette_request, ANTHROPIC_MESSAGES)
    if isinstance(body, JSONResponse):
        return body
    return await _forward_native(
        "/v1/messages",
        inline_anthropic_system_messages(body),
        dialect=ANTHROPIC_MESSAGES,
        starlette_request=starlette_request,
        node_selector=node_selector,
        proxy=proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        settings=settings,
        usage_store=usage_store,
        usage_auth=usage_auth,
    )


@router.post("/v1/messages/count_tokens", response_model=None)
async def anthropic_count_tokens(
    starlette_request: StarletteRequest,
    node_selector: NodeSelector = Depends(get_node_selector),
    proxy: ProxyClient = Depends(get_proxy_client),
    circuit_breaker_registry: CircuitBreakerRegistry = Depends(
        get_circuit_breaker_registry,
    ),
    request_metrics: RequestMetrics = Depends(get_request_metrics),
    settings: Settings = Depends(get_settings),
    usage_store: AuthStore = Depends(get_auth_store),
    usage_auth: TokenAuth | None = Depends(get_messages_api_auth),
) -> JSONResponse | EventSourceResponse:
    """Proxy Anthropic token counting to a node serving the model.

    The count comes from the backend's own chat template, so the body gets
    the same system-message rewrite as ``/v1/messages``. Counting never
    streams (a ``stream`` flag copied from a messages body is dropped) and
    generates nothing, so no usage is recorded.
    """
    body = await _read_json_body(starlette_request, ANTHROPIC_MESSAGES)
    if isinstance(body, JSONResponse):
        return body
    body = {key: value for key, value in body.items() if key != "stream"}
    return await _forward_native(
        "/v1/messages/count_tokens",
        inline_anthropic_system_messages(body),
        dialect=ANTHROPIC_MESSAGES,
        starlette_request=starlette_request,
        node_selector=node_selector,
        proxy=proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        settings=settings,
        usage_store=usage_store,
        usage_auth=usage_auth,
        record_usage=False,
    )


@router.post("/v1/responses", response_model=None)
async def openai_responses(
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
    """Proxy an OpenAI Responses API request, as sent by Codex.

    The body is forwarded as sent, except that system and developer items
    after the first conversation item become ``<system-reminder>`` user
    items, and leading ones join ``instructions`` (see
    ``system_messages``). Requests are served statelessly: the client sends
    the whole conversation each turn, as Codex does with ``store: false``.
    """
    body = await _read_json_body(starlette_request, OPENAI_RESPONSES)
    if isinstance(body, JSONResponse):
        return body
    return await _forward_native(
        "/v1/responses",
        inline_responses_system_messages(body),
        dialect=OPENAI_RESPONSES,
        starlette_request=starlette_request,
        node_selector=node_selector,
        proxy=proxy,
        circuit_breaker_registry=circuit_breaker_registry,
        request_metrics=request_metrics,
        settings=settings,
        usage_store=usage_store,
        usage_auth=usage_auth,
    )


@router.get("/v1/models")
async def list_models(
    starlette_request: StarletteRequest,
    node_selector: NodeSelector = Depends(get_node_selector),
    usage_store: AuthStore = Depends(get_auth_store),
) -> JSONResponse:
    """Return an OpenAI-compatible list of available models.

    Aggregates model names from all healthy registered nodes,
    deduplicating by model name.  DRAINING nodes are excluded so
    clients only see models that can accept new requests.  Nodes owned
    by a user are private (RFE #107) and their models are not listed.
    """
    nodes = node_selector._registry.get_all()
    models_seen: dict[str, dict[str, str | int]] = {}
    model_scope = await _presented_model_scope(starlette_request, usage_store)

    for node in nodes:
        if model_scope is not None and node.model not in model_scope:
            continue
        if node.status != NodeStatus.HEALTHY:
            continue
        if node.admin_only:
            continue
        if node.owner:
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
    allowed_node_ids: frozenset[str] | None = None,
    owner: str | None = None,
    dialect: Dialect = OPENAI_CHAT,
) -> JSONResponse | EventSourceResponse:
    """Establish a backend SSE stream, then expose it to the client.

    Retryable failures before upstream response headers fail over to a
    different node. The downstream 200 is committed only after a backend
    returns a successful response. The complete pre-stream retry phase is
    bounded by ``handshake_timeout``.

    Once streaming begins, events are re-emitted with their existing JSON
    payloads and event names, and failures are reported in-band, in the
    dialect's format, without retrying.
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
            allowed_node_ids=allowed_node_ids,
            owner=owner,
        )
        if reservation is None:
            if last_error is not None:
                status, error = last_error
                return _proxy_error_response(
                    status,
                    error,
                    failover_exhausted=True,
                    attempts=attempts,
                    dialect=dialect,
                )
            status, error_resp = _select_error(
                model, node_selector, allowed_node_ids, owner
            )
            return _proxy_error_response(status, error_resp, dialect=dialect)

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
            return _proxy_error_response(status, error_resp, dialect=dialect)

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
                dialect=dialect,
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
        dialect=dialect,
    )
