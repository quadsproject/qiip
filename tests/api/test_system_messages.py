"""Behavioral coverage for the mid-conversation system-message rewrite.

The rewrite must keep strict chat templates happy (no system message after
the first) without moving anything that changes between turns, so each
request's rewritten conversation stays a prefix of the next one's and the
backend prompt cache keeps working.
"""

from __future__ import annotations

import copy
from itertools import pairwise
from typing import Any

from inference_proxy.api.system_messages import (
    inline_anthropic_system_messages,
    inline_responses_system_messages,
)

_ENV = "# Environment\nPrimary working directory: /work"


def _reminder(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>\n"


def _budget(remaining: int) -> dict[str, Any]:
    return {
        "role": "system",
        "content": f"<total_tokens>{remaining} tokens left</total_tokens>",
    }


def _claude_code_turn(tool_results: int) -> dict[str, Any]:
    """Mimic Claude Code: an environment block, then a note per tool result.

    Like Claude Code, only the newest system message carries a
    ``cache_control`` marker; older ones are re-sent as plain strings.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Run the tests"}]},
        {"role": "system", "content": _ENV},
    ]
    for turn in range(tool_results):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{turn}",
                        "name": "Bash",
                        "input": {"command": "pytest"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{turn}",
                        "content": f"run {turn}: ok",
                    }
                ],
            }
        )
        messages.append(_budget(1000 - turn))
    newest = messages[-1]
    newest["content"] = [
        {
            "type": "text",
            "text": newest["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return {
        "model": "qwen",
        "system": [{"type": "text", "text": "You are a coding agent."}],
        "messages": messages,
        "stream": True,
    }


def _without_cache_markers(value: Any) -> Any:
    """Drop ``cache_control`` keys: backends render text, not markers."""
    if isinstance(value, dict):
        return {
            key: _without_cache_markers(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_without_cache_markers(item) for item in value]
    return value


class TestAnthropicMessages:
    def test_body_without_system_messages_is_returned_unchanged(self) -> None:
        body = {"model": "qwen", "messages": [{"role": "user", "content": "Hi"}]}

        assert inline_anthropic_system_messages(body) is body

    def test_claude_code_system_messages_become_reminders_in_place(self) -> None:
        body = _claude_code_turn(tool_results=1)
        original = copy.deepcopy(body)

        result = inline_anthropic_system_messages(body)

        assert body == original
        assert result["system"] == original["system"]
        assert [m["role"] for m in result["messages"]] == [
            "user",
            "user",
            "assistant",
            "user",
            "user",
        ]
        assert result["messages"][1] == {
            "role": "user",
            "content": [{"type": "text", "text": _reminder(_ENV)}],
        }
        assert result["messages"][4]["content"] == [
            {
                "type": "text",
                "text": _reminder("<total_tokens>1000 tokens left</total_tokens>"),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert result["messages"][3] == original["messages"][3]

    def test_each_turn_extends_the_previous_turn_unchanged(self) -> None:
        turns = [
            _without_cache_markers(
                inline_anthropic_system_messages(_claude_code_turn(n))
            )
            for n in range(4)
        ]

        for previous, current in pairwise(turns):
            assert current["system"] == previous["system"]
            count = len(previous["messages"])
            assert current["messages"][:count] == previous["messages"]

    def test_last_cache_marker_moves_to_the_last_reminder_block(self) -> None:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "A", "cache_control": {"type": "a"}},
                        {"type": "text", "text": "B"},
                    ],
                },
            ],
        }

        blocks = inline_anthropic_system_messages(body)["messages"][1]["content"]

        assert blocks == [
            {"type": "text", "text": _reminder("A")},
            {"type": "text", "text": _reminder("B"), "cache_control": {"type": "a"}},
        ]

    def test_no_system_role_reaches_the_backend(self) -> None:
        result = inline_anthropic_system_messages(_claude_code_turn(3))

        assert all(m["role"] != "system" for m in result["messages"])

    def test_leading_system_messages_join_a_string_system_prompt(self) -> None:
        body = {
            "model": "qwen",
            "system": "Be brief.",
            "messages": [
                {"role": "system", "content": "Project rules."},
                {"role": "user", "content": "Hi"},
            ],
        }

        result = inline_anthropic_system_messages(body)

        assert result["system"] == "Be brief.\n\nProject rules."
        assert result["messages"] == [{"role": "user", "content": "Hi"}]

    def test_leading_system_messages_extend_a_block_system_prompt(self) -> None:
        body = {
            "model": "qwen",
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Rule one."},
                        {"type": "text", "text": "Rule two."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }

        result = inline_anthropic_system_messages(body)

        assert result["system"] == [
            {"type": "text", "text": "Be brief."},
            {"type": "text", "text": "\n\nRule one."},
            {"type": "text", "text": "\n\nRule two."},
        ]

    def test_leading_system_message_becomes_the_system_prompt(self) -> None:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "Project rules."},
                {"role": "user", "content": "Hi"},
            ],
        }

        assert inline_anthropic_system_messages(body)["system"] == "Project rules."

    def test_every_text_block_gets_its_own_envelope(self) -> None:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "First."},
                        {"type": "image", "source": {"type": "url", "url": "x"}},
                        {"type": "text", "text": "Second."},
                    ],
                },
            ],
        }

        result = inline_anthropic_system_messages(body)

        assert result["messages"][1]["content"] == [
            {"type": "text", "text": _reminder("First.")},
            {"type": "text", "text": _reminder("Second.")},
        ]

    def test_later_system_message_without_text_is_dropped(self) -> None:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": ""},
            ],
        }

        result = inline_anthropic_system_messages(body)

        assert result["messages"] == [{"role": "user", "content": "Hi"}]

    def test_non_list_messages_are_left_for_the_backend(self) -> None:
        body = {"model": "qwen", "messages": "not a list"}

        assert inline_anthropic_system_messages(body) is body

    def test_unexpected_system_shape_keeps_leading_messages_in_place(self) -> None:
        leading = {"role": "system", "content": "Project rules."}
        body = {
            "model": "qwen",
            "system": {"unexpected": True},
            "messages": [leading, {"role": "user", "content": "Hi"}],
        }

        result = inline_anthropic_system_messages(body)

        assert result["system"] == {"unexpected": True}
        assert result["messages"][0] == leading


def _codex_turn() -> dict[str, Any]:
    return {
        "model": "qwen",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Sandbox: read-only."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix the test"}],
            },
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Run it first."}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd": "pytest"}',
                "call_id": "call_1",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ],
        "store": False,
        "stream": True,
    }


class TestResponsesInput:
    def test_leading_developer_item_joins_instructions(self) -> None:
        body = _codex_turn()
        original = copy.deepcopy(body)

        result = inline_responses_system_messages(body)

        assert body == original
        assert result["instructions"] == "You are Codex.\n\nSandbox: read-only."
        assert result["input"] == original["input"][1:]

    def test_leading_items_become_instructions_when_none_exist(self) -> None:
        body = _codex_turn()
        del body["instructions"]

        result = inline_responses_system_messages(body)

        assert result["instructions"] == "Sandbox: read-only."

    def test_later_system_items_become_user_reminders_in_place(self) -> None:
        body = _codex_turn()
        body["input"].insert(
            3,
            {"role": "system", "content": "Approval mode changed."},
        )

        result = inline_responses_system_messages(body)

        assert result["input"][2] == {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": _reminder("Approval mode changed.")}
            ],
        }
        assert all(
            item.get("role") not in ("system", "developer") for item in result["input"]
        )

    def test_string_input_is_left_alone(self) -> None:
        body = {"model": "qwen", "input": "Hello", "instructions": "Be brief."}

        assert inline_responses_system_messages(body) is body

    def test_input_without_system_items_is_returned_unchanged(self) -> None:
        body = _codex_turn()
        body["input"] = body["input"][1:]

        assert inline_responses_system_messages(body) is body

    def test_non_string_instructions_keep_leading_items_in_place(self) -> None:
        body = _codex_turn()
        body["instructions"] = [{"type": "input_text", "text": "You are Codex."}]

        result = inline_responses_system_messages(body)

        assert result["instructions"] == body["instructions"]
        assert result["input"][0] == body["input"][0]
