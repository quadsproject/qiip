"""Tests for OpenAI-compatible Pydantic models.

Covers chat completion, text completion, streaming chunks, and error schema.
"""

import pytest
from pydantic import ValidationError

from inference_proxy.models.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorDetail,
    ErrorResponse,
    Usage,
)

# --- ChatMessage tests ---


class TestChatMessage:
    def test_chat_message_with_role_and_content(self) -> None:
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_chat_message_content_can_be_none(self) -> None:
        msg = ChatMessage(role="assistant", content=None)
        assert msg.role == "assistant"
        assert msg.content is None

    def test_chat_message_content_defaults_to_none(self) -> None:
        msg = ChatMessage(role="system")
        assert msg.content is None

    def test_tool_call_message_model_preserves_nested_fields(self) -> None:
        payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Raleigh"}',
                    },
                }
            ],
            "name": "weather-agent",
            "vendor_extension": {"trace_id": "trace-123"},
        }

        message = ChatMessage.model_validate(payload)

        assert message.model_dump(exclude_unset=True) == payload

    def test_multimodal_content_parts_round_trip(self) -> None:
        content = [
            {"type": "text", "text": "What is shown?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AA==",
                    "detail": "low",
                },
            },
        ]

        message = ChatMessage(role="user", content=content)

        assert message.model_dump()["content"] == content


# --- ChatCompletionRequest tests ---


class TestChatCompletionRequest:
    def test_valid_request_with_model_and_messages(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hello!")],
        )
        assert req.model == "llama-2-7b"
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_request_with_optional_fields(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stream=True,
            n=2,
        )
        assert req.temperature == 0.5
        assert req.max_tokens == 100
        assert req.top_p == 0.9
        assert req.stream is True
        assert req.n == 2

    def test_request_rejects_empty_messages(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="llama-2-7b",
                messages=[],
            )

    def test_request_rejects_temperature_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="llama-2-7b",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=-0.1,
            )

    def test_request_rejects_temperature_above_two(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="llama-2-7b",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=2.1,
            )

    def test_request_rejects_max_tokens_zero_or_negative(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="llama-2-7b",
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=0,
            )
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="llama-2-7b",
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=-5,
            )

    def test_request_extra_fields_pass_through(self) -> None:
        """D-10: unknown fields pass through to vLLM."""
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hello!")],
            custom_vllm_param=42,
        )
        assert req.model_extra == {"custom_vllm_param": 42}

    def test_request_stream_defaults_to_false(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.stream is False

    def test_request_stop_accepts_str(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="END",
        )
        assert req.stop == "END"

    def test_request_stop_accepts_list_of_str(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["END", "STOP"],
        )
        assert req.stop == ["END", "STOP"]

    def test_request_stop_accepts_none(self) -> None:
        req = ChatCompletionRequest(
            model="llama-2-7b",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=None,
        )
        assert req.stop is None

    def test_request_presence_penalty_range(self) -> None:
        # Valid at boundaries
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="Hi")],
            presence_penalty=-2.0,
        )
        assert req.presence_penalty == -2.0

        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="Hi")],
            presence_penalty=2.0,
        )
        assert req.presence_penalty == 2.0

        # Invalid outside range
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="Hi")],
                presence_penalty=-2.1,
            )

    def test_request_frequency_penalty_range(self) -> None:
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="Hi")],
            frequency_penalty=0.5,
        )
        assert req.frequency_penalty == 0.5

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="Hi")],
                frequency_penalty=2.1,
            )


# --- ChatCompletionResponse tests ---


class TestChatCompletionResponse:
    def test_response_creates_successfully(self) -> None:
        resp = ChatCompletionResponse(
            id="chatcmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi there!"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.id == "chatcmpl-abc123"
        assert resp.created == 1700000000
        assert resp.model == "llama-2-7b"
        assert len(resp.choices) == 1

    def test_response_object_defaults_to_chat_completion(self) -> None:
        resp = ChatCompletionResponse(
            id="chatcmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[],
        )
        assert resp.object == "chat.completion"

    def test_choice_contains_message_and_finish_reason(self) -> None:
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Reply"),
            finish_reason="stop",
        )
        assert isinstance(choice.message, ChatMessage)
        assert choice.finish_reason == "stop"

    def test_usage_validates(self) -> None:
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_response_with_usage(self) -> None:
        resp = ChatCompletionResponse(
            id="chatcmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        assert resp.usage is not None
        assert resp.usage.total_tokens == 15


# --- ChatCompletionChunk (streaming) tests ---


class TestChatCompletionChunk:
    def test_chunk_creates_successfully(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.id == "chatcmpl-abc123"
        assert len(chunk.choices) == 1

    def test_chunk_object_defaults_to_chat_completion_chunk(self) -> None:
        chunk = ChatCompletionChunk(
            id="chatcmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[],
        )
        assert chunk.object == "chat.completion.chunk"

    def test_chunk_delta_optional_role_and_content(self) -> None:
        delta = ChatCompletionChunkDelta()
        assert delta.role is None
        assert delta.content is None

        delta_with_role = ChatCompletionChunkDelta(role="assistant")
        assert delta_with_role.role == "assistant"
        assert delta_with_role.content is None

        delta_with_content = ChatCompletionChunkDelta(content="token")
        assert delta_with_content.role is None
        assert delta_with_content.content == "token"


# --- CompletionRequest (text completion) tests ---


class TestCompletionRequest:
    def test_request_with_model_and_prompt(self) -> None:
        req = CompletionRequest(model="llama-2-7b", prompt="Once upon a time")
        assert req.model == "llama-2-7b"
        assert req.prompt == "Once upon a time"

    def test_request_with_list_prompt(self) -> None:
        req = CompletionRequest(
            model="llama-2-7b",
            prompt=["Hello", "World"],
        )
        assert req.prompt == ["Hello", "World"]

    @pytest.mark.parametrize(
        "prompt",
        [
            [101, 202, 303],
            [[101, 202], [303, 404]],
        ],
    )
    def test_token_id_prompt_forms_preserve_integer_types(
        self,
        prompt: list[int] | list[list[int]],
    ) -> None:
        request = CompletionRequest(model="llama-2-7b", prompt=prompt)
        dumped_prompt = request.model_dump()["prompt"]

        assert dumped_prompt == prompt
        if prompt and isinstance(prompt[0], list):
            assert all(
                type(token) is int for sequence in dumped_prompt for token in sequence
            )
        else:
            assert all(type(token) is int for token in dumped_prompt)

    def test_request_extra_fields_pass_through(self) -> None:
        """D-10: unknown fields pass through to vLLM."""
        req = CompletionRequest(
            model="llama-2-7b",
            prompt="Hello",
            best_of=3,
        )
        assert req.model_extra == {"best_of": 3}

    def test_request_has_same_optional_fields(self) -> None:
        req = CompletionRequest(
            model="llama-2-7b",
            prompt="Hello",
            temperature=1.0,
            max_tokens=50,
            top_p=0.8,
            stream=True,
            stop=["END"],
            n=2,
            presence_penalty=0.5,
            frequency_penalty=-0.5,
        )
        assert req.temperature == 1.0
        assert req.max_tokens == 50
        assert req.top_p == 0.8
        assert req.stream is True
        assert req.stop == ["END"]
        assert req.n == 2
        assert req.presence_penalty == 0.5
        assert req.frequency_penalty == -0.5

    def test_request_rejects_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            CompletionRequest(
                model="llama-2-7b",
                prompt="Hello",
                temperature=-1.0,
            )

    def test_request_stream_defaults_to_false(self) -> None:
        req = CompletionRequest(model="llama-2-7b", prompt="Hello")
        assert req.stream is False


# --- CompletionResponse (text completion) tests ---


class TestCompletionResponse:
    def test_response_creates_successfully(self) -> None:
        resp = CompletionResponse(
            id="cmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[
                CompletionChoice(
                    index=0,
                    text="there was a dragon",
                    finish_reason="length",
                )
            ],
        )
        assert resp.id == "cmpl-abc123"
        assert len(resp.choices) == 1

    def test_response_object_defaults_to_text_completion(self) -> None:
        resp = CompletionResponse(
            id="cmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[],
        )
        assert resp.object == "text_completion"

    def test_choice_contains_text_not_message(self) -> None:
        choice = CompletionChoice(
            index=0,
            text="generated text",
            finish_reason="stop",
        )
        assert choice.text == "generated text"
        assert choice.finish_reason == "stop"
        assert not hasattr(choice, "message")


# --- CompletionChunk (text completion streaming) tests ---


class TestCompletionChunk:
    def test_chunk_creates_successfully(self) -> None:
        chunk = CompletionChunk(
            id="cmpl-abc123",
            created=1700000000,
            model="llama-2-7b",
            choices=[
                CompletionChunkChoice(
                    index=0,
                    text="token",
                    finish_reason=None,
                )
            ],
        )
        assert chunk.id == "cmpl-abc123"
        assert len(chunk.choices) == 1

    def test_chunk_choice_has_text_field(self) -> None:
        choice = CompletionChunkChoice(
            index=0,
            text="some text",
            finish_reason=None,
        )
        assert choice.text == "some text"
        assert not hasattr(choice, "delta")


# --- ErrorResponse tests ---


class TestErrorResponse:
    def test_error_response_creates_successfully(self) -> None:
        resp = ErrorResponse(
            error=ErrorDetail(
                message="Model not found",
                type="invalid_request_error",
            )
        )
        assert resp.error.message == "Model not found"
        assert resp.error.type == "invalid_request_error"

    def test_error_detail_param_and_code_optional(self) -> None:
        detail = ErrorDetail(message="error", type="server_error")
        assert detail.param is None
        assert detail.code is None

    def test_error_detail_code_accepts_str(self) -> None:
        detail = ErrorDetail(
            message="error",
            type="server_error",
            code="model_not_found",
        )
        assert detail.code == "model_not_found"

    def test_error_detail_code_accepts_int(self) -> None:
        detail = ErrorDetail(
            message="error",
            type="server_error",
            code=500,
        )
        assert detail.code == 500

    def test_error_detail_with_param(self) -> None:
        detail = ErrorDetail(
            message="Invalid temperature",
            type="invalid_request_error",
            param="temperature",
            code="invalid_value",
        )
        assert detail.param == "temperature"
        assert detail.code == "invalid_value"
