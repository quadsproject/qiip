"""OpenAI-compatible Pydantic models for the inference proxy.

Covers chat completion (D-09, D-10, D-11) and text completion (D-12) endpoints:
request models, response models, streaming chunk models, and error schema.

Design decisions:
- D-09: Preserve OpenAI message content and tool metadata without constraining
  nested extensions understood by vLLM.
- D-10: extra='allow' on request models and nested messages for forward
  compatibility with vLLM.
- D-11: Both request AND response models
- D-12: Chat and text completion models are fully separate (no shared base class)
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Chat Completion Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions.

    Uses extra='allow' (D-10) so unknown fields pass through to vLLM.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage statistics for a completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response body for POST /v1/chat/completions (non-streaming)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chat completion chunk."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chat completion chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming chat completion response (SSE event)."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# Text Completion Models (D-12 -- separate from chat, no shared base class)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions.

    Uses extra='allow' (D-10) so unknown fields pass through to vLLM.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)


class CompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """Response body for POST /v1/completions (non-streaming)."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class CompletionChunkChoice(BaseModel):
    """A single choice in a streaming text completion chunk."""

    index: int
    text: str
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    """A single chunk in a streaming text completion response (SSE event)."""

    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: list[CompletionChunkChoice]


# ---------------------------------------------------------------------------
# Error Models
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Error detail matching the OpenAI error schema."""

    message: str
    type: str
    param: str | None = None
    code: str | int | None = None


class ErrorResponse(BaseModel):
    """Error response wrapper matching the OpenAI error schema."""

    error: ErrorDetail
