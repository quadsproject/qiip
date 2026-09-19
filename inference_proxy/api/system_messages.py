"""Keep mid-conversation system messages out of the system prompt.

Claude Code sends system messages between conversation turns: its
environment block after the first user turn, and a token-budget note after
every tool result. Codex sends a developer message next to its instructions.
Some chat templates (Qwen 3.x, for example) accept a system message only as
the very first message, so on backends that pass these through unchanged
(llama.cpp) the request fails.

Folding them into the system prompt would render, but the system prompt sits
in front of every message: any growth there invalidates the backend's prompt
cache for the whole conversation, every turn. vLLM does exactly that for such
templates. Instead, following bifrost's approach, only system and developer
messages that come before the conversation starts join the system prompt.
Every later one becomes a user message at the same position, wrapped in the
``<system-reminder>`` envelope Claude Code itself uses. Earlier turns then
render identically from one request to the next, so the cached prefix keeps
growing, and the backend never sees a system message after the first.

System and developer content is text; any non-text part of such a message
is dropped by the rewrite. Both rewrites are pure: they return a new body and
never mutate the input.
"""

from __future__ import annotations

from typing import Any

_SYSTEM_ROLES = frozenset({"system", "developer"})


def _reminder(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>\n"


def _is_system(item: Any) -> bool:
    return isinstance(item, dict) and item.get("role") in _SYSTEM_ROLES


def _text_parts(content: Any) -> list[dict[str, Any]]:
    """Return the text parts of a ``content`` value that carry any text."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return []
    parts = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(part)
    return parts


def _content_texts(content: Any) -> list[str]:
    return [part["text"] for part in _text_parts(content)]


def _join_text(existing: str, additions: list[str]) -> str:
    return "\n\n".join([existing, *additions] if existing else additions)


def _anthropic_system_with(system: str | list[Any] | None, additions: list[str]) -> Any:
    """Append *additions* to an Anthropic top-level ``system`` value."""
    if system is None:
        return "\n\n".join(additions)
    if isinstance(system, str):
        return _join_text(system, additions)
    separator = "\n\n" if _content_texts(system) else ""
    blocks = [{"type": "text", "text": f"{separator}{additions[0]}"}]
    blocks += [{"type": "text", "text": f"\n\n{text}"} for text in additions[1:]]
    return [*system, *blocks]


def _anthropic_reminder(content: Any) -> dict[str, Any] | None:
    """Render a later system message as a user turn, one envelope per part.

    The client's last ``cache_control`` marker moves to the last block, as in
    bifrost, so a backend that honors Anthropic cache breakpoints keeps it.
    """
    parts = _text_parts(content)
    if not parts:
        return None
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": _reminder(part["text"])} for part in parts
    ]
    markers = [part["cache_control"] for part in parts if part.get("cache_control")]
    if markers:
        blocks[-1]["cache_control"] = markers[-1]
    return {"role": "user", "content": blocks}


def inline_anthropic_system_messages(body: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``role: system`` entries inside an Anthropic ``messages`` list.

    Entries before the first user or assistant message join the top-level
    ``system`` prompt; if ``system`` has an unexpected shape they are left in
    place for the backend to judge. Each later entry becomes a user message
    in place, with every text part wrapped in its own ``<system-reminder>``
    envelope. A later entry with no text is dropped.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(_is_system(m) for m in messages):
        return body
    system = body.get("system")
    can_extend = system is None or isinstance(system, str | list)

    leading: list[str] = []
    rewritten: list[Any] = []
    started = False
    for message in messages:
        if not _is_system(message):
            started = True
            rewritten.append(message)
        elif not started:
            if can_extend:
                leading.extend(_content_texts(message.get("content")))
            else:
                rewritten.append(message)
        elif (reminder := _anthropic_reminder(message.get("content"))) is not None:
            rewritten.append(reminder)

    result = {**body, "messages": rewritten}
    if leading:
        result["system"] = _anthropic_system_with(system, leading)
    return result


def _responses_reminder(content: Any) -> dict[str, Any] | None:
    texts = _content_texts(content)
    if not texts:
        return None
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": _reminder(text)} for text in texts],
    }


def inline_responses_system_messages(body: dict[str, Any]) -> dict[str, Any]:
    """Rewrite system and developer items inside a Responses ``input`` list.

    Items before the first other input item join ``instructions`` when it is
    a string or absent; any other ``instructions`` shape leaves them in place
    for the backend to judge. Each later item becomes a user message item in
    place, with every text part wrapped in its own ``<system-reminder>``
    envelope. A later item with no text is dropped. A string ``input`` has no
    items and is left alone.
    """
    items = body.get("input")
    if not isinstance(items, list) or not any(_is_system(i) for i in items):
        return body
    instructions = body.get("instructions")
    can_extend = instructions is None or isinstance(instructions, str)

    leading: list[str] = []
    rewritten: list[Any] = []
    started = False
    for item in items:
        if not _is_system(item):
            started = True
            rewritten.append(item)
        elif not started:
            if can_extend:
                leading.extend(_content_texts(item.get("content")))
            else:
                rewritten.append(item)
        elif (reminder := _responses_reminder(item.get("content"))) is not None:
            rewritten.append(reminder)

    result = {**body, "input": rewritten}
    if leading:
        result["instructions"] = _join_text(instructions or "", leading)
    return result
