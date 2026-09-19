"""Per-protocol details for the inference routes.

qiip forwards three request formats to backends that implement each one
natively: OpenAI chat and text completions, the Anthropic Messages API used
by Claude Code, and the OpenAI Responses API used by Codex. The proxy never
translates between them. Per format it only needs to know where token usage
is reported, which stream event marks a successful end, and how to shape
qiip's own errors so the client can display them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from fastapi.sse import format_sse_event

from inference_proxy.models.openai import ErrorResponse

Usage = tuple[int, int, int]
"""Token usage as ``(prompt, completion, total)``."""

_DEFAULT_SSE_EVENT = "message"
_ANTHROPIC_PATH = "/v1/messages"
_RESPONSES_PATH = "/v1/responses"


def relay_frame(event: str, data: str) -> bytes:
    """Re-encode one upstream SSE event, keeping its event name.

    The Anthropic SDK dispatches on the ``event:`` field and ignores frames
    without one, so dropping it would silently discard a Claude Code stream.
    Frames that had no event line (the SSE default ``message``) stay bare.
    """
    name = None if event == _DEFAULT_SSE_EVENT else event
    return format_sse_event(data_str=data, event=name)


def _load_json(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    """Return *value* when it is a real integer (``bool`` excluded)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _event_type(event: str, payload: Any) -> str | None:
    """Name an event by its SSE ``event:`` line, else by its JSON ``type``."""
    if event != _DEFAULT_SSE_EVENT:
        return event
    if isinstance(payload, dict):
        kind = payload.get("type")
        if isinstance(kind, str):
            return kind
    return None


class StreamTracker(ABC):
    """Watch one relayed stream for its usage and its successful end."""

    @abstractmethod
    def observe(self, event: str, data: str) -> bool:
        """Record one upstream event; return True when it ends the response."""

    @property
    @abstractmethod
    def usage(self) -> Usage | None:
        """Usage reported so far, or None when the stream carried none."""


class Dialect(ABC):
    """Usage, stream-end and error conventions of one API format."""

    name: str

    @abstractmethod
    def usage_from_body(self, payload: Any) -> Usage | None:
        """Extract usage from a non-streaming response body."""

    @abstractmethod
    def stream_tracker(self) -> StreamTracker:
        """Return fresh per-stream state."""

    @abstractmethod
    def error_content(
        self,
        error: ErrorResponse,
        status: int,
        *,
        code: str | None = None,
    ) -> dict[str, Any]:
        """Render a qiip-generated error body; *code* overrides the error code."""

    @abstractmethod
    def stream_error(self, error: ErrorResponse, status: int) -> list[bytes]:
        """Render SSE frames that report a failure after the stream started."""


def _openai_error_content(error: ErrorResponse, code: str | None) -> dict[str, Any]:
    content = error.model_dump()
    if code is not None:
        content["error"]["code"] = code
    return content


class _ChatStreamTracker(StreamTracker):
    def __init__(self) -> None:
        self._usage: Usage | None = None

    def observe(self, event: str, data: str) -> bool:
        if data == "[DONE]":
            return True
        found = OPENAI_CHAT.usage_from_body(_load_json(data))
        if found is not None:
            self._usage = found
        return False

    @property
    def usage(self) -> Usage | None:
        return self._usage


class _OpenAIChatDialect(Dialect):
    """OpenAI chat and text completions: the gateway's original format."""

    name = "openai-chat"

    def usage_from_body(self, payload: Any) -> Usage | None:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = _int(usage.get("prompt_tokens"))
        completion = _int(usage.get("completion_tokens"))
        total = _int(usage.get("total_tokens"))
        if prompt is None or completion is None or total is None:
            return None
        return prompt, completion, total

    def stream_tracker(self) -> StreamTracker:
        return _ChatStreamTracker()

    def error_content(
        self,
        error: ErrorResponse,
        status: int,
        *,
        code: str | None = None,
    ) -> dict[str, Any]:
        return _openai_error_content(error, code)

    def stream_error(self, error: ErrorResponse, status: int) -> list[bytes]:
        return [
            format_sse_event(data_str=json.dumps(error.model_dump())),
            format_sse_event(data_str="[DONE]"),
        ]


def _anthropic_prompt_tokens(usage: dict[str, Any]) -> int | None:
    """Total prompt tokens: Anthropic excludes cached tokens from input_tokens."""
    uncached = _int(usage.get("input_tokens"))
    if uncached is None:
        return None
    cached = _int(usage.get("cache_read_input_tokens")) or 0
    created = _int(usage.get("cache_creation_input_tokens")) or 0
    return uncached + cached + created


class _AnthropicStreamTracker(StreamTracker):
    def __init__(self) -> None:
        self._prompt: int | None = None
        self._completion: int | None = None

    def observe(self, event: str, data: str) -> bool:
        payload = _load_json(data)
        kind = _event_type(event, payload)
        if kind == "message_stop":
            return True
        if not isinstance(payload, dict):
            return False
        usage: Any = None
        if kind == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                usage = message.get("usage")
        elif kind == "message_delta":
            usage = payload.get("usage")
        if isinstance(usage, dict):
            # vLLM reports the prompt only in the final delta and llama.cpp
            # only in message_start, so a zero never replaces a real count.
            prompt = _anthropic_prompt_tokens(usage)
            if prompt is not None and (prompt > 0 or self._prompt is None):
                self._prompt = prompt
            # output_tokens is cumulative, so the latest value is the total.
            completion = _int(usage.get("output_tokens"))
            if completion is not None:
                self._completion = completion
        return False

    @property
    def usage(self) -> Usage | None:
        if self._prompt is None and self._completion is None:
            return None
        prompt = self._prompt or 0
        completion = self._completion or 0
        return prompt, completion, prompt + completion


_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def anthropic_error_type(status: int) -> str:
    """Map an HTTP status to the Anthropic error ``type`` clients expect."""
    if status in _ANTHROPIC_ERROR_TYPES:
        return _ANTHROPIC_ERROR_TYPES[status]
    return "invalid_request_error" if status < 500 else "api_error"


class _AnthropicDialect(Dialect):
    """Anthropic Messages API, as sent by Claude Code."""

    name = "anthropic-messages"

    def usage_from_body(self, payload: Any) -> Usage | None:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = _anthropic_prompt_tokens(usage)
        completion = _int(usage.get("output_tokens"))
        if prompt is None or completion is None:
            return None
        return prompt, completion, prompt + completion

    def stream_tracker(self) -> StreamTracker:
        return _AnthropicStreamTracker()

    def error_content(
        self,
        error: ErrorResponse,
        status: int,
        *,
        code: str | None = None,
    ) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "type": anthropic_error_type(status),
            "message": error.error.message,
        }
        resolved = code if code is not None else error.error.code
        if resolved is not None:
            detail["code"] = str(resolved)
        return {"type": "error", "error": detail}

    def stream_error(self, error: ErrorResponse, status: int) -> list[bytes]:
        body = self.error_content(error, status)
        return [format_sse_event(data_str=json.dumps(body), event="error")]


class _ResponsesStreamTracker(StreamTracker):
    def __init__(self) -> None:
        self._usage: Usage | None = None

    def observe(self, event: str, data: str) -> bool:
        payload = _load_json(data)
        kind = _event_type(event, payload)
        if isinstance(payload, dict):
            response = payload.get("response")
            if isinstance(response, dict):
                found = OPENAI_RESPONSES.usage_from_body(response)
                if found is not None:
                    self._usage = found
        # An incomplete response (for example max_output_tokens reached) is
        # still a backend that worked.
        return kind in ("response.completed", "response.incomplete")

    @property
    def usage(self) -> Usage | None:
        return self._usage


class _OpenAIResponsesDialect(Dialect):
    """OpenAI Responses API, as sent by Codex."""

    name = "openai-responses"

    def usage_from_body(self, payload: Any) -> Usage | None:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = _int(usage.get("input_tokens"))
        completion = _int(usage.get("output_tokens"))
        if prompt is None or completion is None:
            return None
        total = _int(usage.get("total_tokens"))
        return prompt, completion, total if total is not None else prompt + completion

    def stream_tracker(self) -> StreamTracker:
        return _ResponsesStreamTracker()

    def error_content(
        self,
        error: ErrorResponse,
        status: int,
        *,
        code: str | None = None,
    ) -> dict[str, Any]:
        return _openai_error_content(error, code)

    def stream_error(self, error: ErrorResponse, status: int) -> list[bytes]:
        code = error.error.code
        failed = {
            "type": "response.failed",
            "response": {
                "id": "resp_qiip_error",
                "object": "response",
                "status": "failed",
                "output": [],
                "error": {
                    "code": str(code) if code is not None else "server_error",
                    "message": error.error.message,
                },
            },
        }
        return [format_sse_event(data_str=json.dumps(failed), event="response.failed")]


OPENAI_CHAT: Dialect = _OpenAIChatDialect()
ANTHROPIC_MESSAGES: Dialect = _AnthropicDialect()
OPENAI_RESPONSES: Dialect = _OpenAIResponsesDialect()


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def dialect_for_path(path: str) -> Dialect:
    """Return the dialect whose error shape a request to *path* expects."""
    if _under(path, _ANTHROPIC_PATH):
        return ANTHROPIC_MESSAGES
    if _under(path, _RESPONSES_PATH):
        return OPENAI_RESPONSES
    return OPENAI_CHAT
