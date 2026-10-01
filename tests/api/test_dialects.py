"""Behavioral coverage for per-protocol usage, stream-end and error rules."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inference_proxy.api.dialects import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    Dialect,
    StreamTracker,
    anthropic_error_type,
    dialect_for_path,
    relay_frame,
)
from inference_proxy.models.openai import ErrorDetail, ErrorResponse


def _error(code: str = "no_nodes") -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            message="No inference nodes available",
            type="server_error",
            code=code,
        )
    )


def _feed(tracker: StreamTracker, events: list[tuple[str, Any]]) -> list[bool]:
    return [
        tracker.observe(event, data if isinstance(data, str) else json.dumps(data))
        for event, data in events
    ]


class TestRelayFrame:
    def test_default_event_stays_a_bare_data_frame(self) -> None:
        assert relay_frame("message", '{"a":1}') == b'data: {"a":1}\n\n'

    def test_named_event_keeps_its_event_line(self) -> None:
        frame = relay_frame("message_start", '{"type":"message_start"}')

        assert frame == b'event: message_start\ndata: {"type":"message_start"}\n\n'


class TestOpenAIChat:
    def test_done_ends_the_stream_and_the_last_usage_wins(self) -> None:
        tracker = OPENAI_CHAT.stream_tracker()
        usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

        finished = _feed(
            tracker,
            [
                ("message", {"usage": {**usage, "completion_tokens": 1}}),
                ("message", {"usage": usage}),
                ("message", "[DONE]"),
            ],
        )

        assert finished == [False, False, True]
        assert tracker.usage == (3, 4, 7)

    @pytest.mark.parametrize(
        "usage",
        [
            {"prompt_tokens": 1, "completion_tokens": 2},
            {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
            "not a dict",
        ],
    )
    def test_incomplete_usage_is_ignored(self, usage: Any) -> None:
        assert OPENAI_CHAT.usage_from_body({"usage": usage}) is None

    def test_stream_error_is_an_error_payload_then_done(self) -> None:
        frames = OPENAI_CHAT.stream_error(_error(), 503)

        assert frames == [
            b"data: " + json.dumps(_error().model_dump()).encode() + b"\n\n",
            b"data: [DONE]\n\n",
        ]


class TestAnthropicMessages:
    def test_llama_cpp_stream_usage_counts_cached_prompt_tokens(self) -> None:
        tracker = ANTHROPIC_MESSAGES.stream_tracker()
        start = {
            "type": "message_start",
            "message": {
                "usage": {
                    "cache_read_input_tokens": 900,
                    "input_tokens": 100,
                    "output_tokens": 0,
                }
            },
        }
        delta = {"type": "message_delta", "usage": {"output_tokens": 42}}

        finished = _feed(
            tracker,
            [
                ("message_start", start),
                ("content_block_delta", {"type": "content_block_delta"}),
                ("message_delta", delta),
                ("message_stop", {"type": "message_stop"}),
            ],
        )

        assert finished == [False, False, False, True]
        assert tracker.usage == (1000, 42, 1042)

    def test_vllm_final_delta_restates_prompt_usage(self) -> None:
        tracker = ANTHROPIC_MESSAGES.stream_tracker()
        start = {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 0, "output_tokens": 0}},
        }
        delta = {
            "type": "message_delta",
            "usage": {
                "input_tokens": 50,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
                "output_tokens": 7,
            },
        }

        _feed(tracker, [("message_start", start), ("message_delta", delta)])

        assert tracker.usage == (100, 7, 107)

    def test_zero_prompt_in_the_final_delta_keeps_the_reported_count(self) -> None:
        tracker = ANTHROPIC_MESSAGES.stream_tracker()
        start = {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 80, "output_tokens": 0}},
        }
        delta = {
            "type": "message_delta",
            "usage": {"input_tokens": 0, "output_tokens": 9},
        }

        _feed(tracker, [("message_start", start), ("message_delta", delta)])

        assert tracker.usage == (80, 9, 89)

    def test_event_type_falls_back_to_the_json_type(self) -> None:
        tracker = ANTHROPIC_MESSAGES.stream_tracker()

        assert _feed(tracker, [("message", {"type": "message_stop"})]) == [True]

    def test_stream_without_usage_reports_none(self) -> None:
        tracker = ANTHROPIC_MESSAGES.stream_tracker()

        _feed(tracker, [("ping", {"type": "ping"}), ("message", "not json")])

        assert tracker.usage is None

    def test_body_usage_counts_cached_prompt_tokens(self) -> None:
        payload = {
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 5,
                "output_tokens": 3,
            }
        }

        assert ANTHROPIC_MESSAGES.usage_from_body(payload) == (15, 3, 18)

    @pytest.mark.parametrize(
        "payload",
        [
            {"usage": {"input_tokens": 10}},
            {"usage": {"output_tokens": 3}},
            {"usage": None},
            ["not", "a", "dict"],
        ],
    )
    def test_incomplete_body_usage_is_ignored(self, payload: Any) -> None:
        assert ANTHROPIC_MESSAGES.usage_from_body(payload) is None

    def test_error_content_uses_the_anthropic_envelope(self) -> None:
        content = ANTHROPIC_MESSAGES.error_content(_error(), 503)

        assert content == {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "No inference nodes available",
                "code": "no_nodes",
            },
        }

    def test_error_code_can_be_overridden(self) -> None:
        content = ANTHROPIC_MESSAGES.error_content(
            _error(), 502, code="failover_exhausted"
        )

        assert content["error"]["type"] == "api_error"
        assert content["error"]["code"] == "failover_exhausted"

    def test_stream_error_is_an_error_event(self) -> None:
        (frame,) = ANTHROPIC_MESSAGES.stream_error(_error(), 502)

        head, data = frame.decode().rstrip("\n").split("\n")
        assert head == "event: error"
        assert json.loads(data.removeprefix("data: "))["error"]["type"] == "api_error"

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (418, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (503, "overloaded_error"),
            (504, "api_error"),
            (529, "overloaded_error"),
        ],
    )
    def test_status_maps_to_anthropic_error_type(
        self, status: int, expected: str
    ) -> None:
        assert anthropic_error_type(status) == expected


class TestOpenAIResponses:
    def test_completed_event_ends_the_stream_with_its_usage(self) -> None:
        tracker = OPENAI_RESPONSES.stream_tracker()
        usage = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}

        finished = _feed(
            tracker,
            [
                ("response.created", {"type": "response.created", "response": {}}),
                ("response.output_text.delta", {"type": "response.output_text.delta"}),
                (
                    "response.completed",
                    {"type": "response.completed", "response": {"usage": usage}},
                ),
            ],
        )

        assert finished == [False, False, True]
        assert tracker.usage == (12, 5, 17)

    def test_incomplete_response_still_ends_successfully(self) -> None:
        tracker = OPENAI_RESPONSES.stream_tracker()

        assert _feed(tracker, [("message", {"type": "response.incomplete"})]) == [True]

    def test_failed_response_keeps_usage_but_is_not_success(self) -> None:
        tracker = OPENAI_RESPONSES.stream_tracker()
        failed = {
            "type": "response.failed",
            "response": {"usage": {"input_tokens": 4, "output_tokens": 0}},
        }

        assert _feed(tracker, [("response.failed", failed)]) == [False]
        assert tracker.usage == (4, 0, 4)

    def test_body_usage_without_total_is_summed(self) -> None:
        payload = {"usage": {"input_tokens": 2, "output_tokens": 3}}

        assert OPENAI_RESPONSES.usage_from_body(payload) == (2, 3, 5)

    def test_error_content_uses_the_openai_shape(self) -> None:
        content = OPENAI_RESPONSES.error_content(
            _error(), 502, code="failover_exhausted"
        )

        assert content["error"]["code"] == "failover_exhausted"
        assert content["error"]["message"] == "No inference nodes available"

    def test_stream_error_is_a_failed_response_event(self) -> None:
        (frame,) = OPENAI_RESPONSES.stream_error(_error("backend_timeout"), 504)

        head, data = frame.decode().rstrip("\n").split("\n")
        assert head == "event: response.failed"
        payload = json.loads(data.removeprefix("data: "))
        assert payload["type"] == "response.failed"
        assert payload["response"]["status"] == "failed"
        assert payload["response"]["error"] == {
            "code": "backend_timeout",
            "message": "No inference nodes available",
        }


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v1/messages", ANTHROPIC_MESSAGES),
        ("/v1/messages/count_tokens", ANTHROPIC_MESSAGES),
        ("/v1/messagesx", OPENAI_CHAT),
        ("/v1/responses", OPENAI_RESPONSES),
        ("/v1/chat/completions", OPENAI_CHAT),
    ],
)
def test_error_format_follows_the_request_path(path: str, expected: Dialect) -> None:
    assert dialect_for_path(path) is expected
