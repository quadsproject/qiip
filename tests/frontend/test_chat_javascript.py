"""Behavioral tests for the shipped chat JavaScript."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CHAT_JS = _ROOT / "inference_proxy/static/js/chat.js"


def _run_node(harness: str, scenario: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "Node.js is required for chat JavaScript regressions; "
            "CI must install it explicitly"
        )

    result = subprocess.run(
        [node, "-e", harness, str(_CHAT_JS), scenario],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


_TRANSACTION_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const scenario = process.argv[2];
const requests = [];
let completionCall = 0;

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.value = "";
    this.disabled = false;
    this.className = "";
    this._text = "";
    this._html = "";
    const classes = new Set();
    this.classList = {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name) {
        if (classes.has(name)) classes.delete(name);
        else classes.add(name);
      },
    };
  }
  get textContent() { return this._text; }
  set textContent(value) {
    this._text = String(value);
    if (value === "") this.children = [];
  }
  get innerHTML() { return this._html; }
  set innerHTML(value) { this._html = String(value); this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelector(selector) {
    return this.children.find(function (child) {
      return selector === ".streaming-cursor" && child.className === "streaming-cursor";
    }) || null;
  }
  focus() {}
  remove() {}
}

function streamResponse(events) {
  let position = 0;
  return {
    ok: true,
    status: 200,
    body: {
      getReader() {
        return {
          async read() {
            const event = events[position++];
            if (event instanceof Error) throw event;
            return event || { done: true };
          },
        };
      },
    },
  };
}

function contentEvent(content) {
  return {
    done: false,
    value: Buffer.from(
      'data: {"choices":[{"delta":{"content":' + JSON.stringify(content) + '}}]}\n\n',
    ),
  };
}

function firstResponse() {
  if (scenario === "http_error") {
    return {
      ok: false,
      status: 400,
      json: async function () {
        return { error: { message: "Conversation roles must alternate" } };
      },
    };
  }
  if (scenario === "sse_error") {
    return streamResponse([{
      done: false,
      value: Buffer.from(
        'data: {"error":{"message":"Conversation roles must alternate"}}\n\n',
      ),
    }]);
  }
  if (scenario === "network_empty") throw new Error("connection lost");
  if (scenario === "network_partial") {
    return streamResponse([contentEvent("partial"), new Error("connection lost")]);
  }
  if (scenario === "success") {
    return streamResponse([contentEvent("complete")]);
  }
  if (scenario === "decoy_message") {
    return streamResponse([{
      done: false,
      value: Buffer.from(
        'data: {"message":{"id":"metadata"},"choices":[{"delta":{"content":"hello world"}}]}\n\n',
      ),
    }]);
  }
  if (scenario === "empty_success") return streamResponse([]);
  throw new Error("unknown scenario: " + scenario);
}

const elements = new Map();
function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
}

const sandbox = {
  showToast: function () {},
  console,
  Buffer,
  TextDecoder,
  ANONYMOUS_PLAYGROUND: true,
  document: {
    getElementById: byId,
    createElement() { return new Element(); },
    addEventListener() {},
  },
  localStorage: { getItem() { return null; }, setItem() {} },
  requestAnimationFrame(callback) { callback(); },
  setTimeout() { return 1; },
  DOMPurify: { sanitize(value) { return value; } },
  marked: { parse(value) { return value; } },
  fetch: async function (url, options) {
    if (url !== "/v1/chat/completions") throw new Error("unexpected URL: " + url);
    requests.push(JSON.parse(options.body));
    completionCall += 1;
    return completionCall === 1 ? firstResponse() : streamResponse([]);
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);
sandbox.messageAreaInner = new Element();
sandbox.messageArea = { scrollHeight: 100, scrollTop: 0, clientHeight: 100 };
sandbox.emptyState = { style: { display: "none" } };
sandbox.chatInput = new Element();
sandbox.sendBtn = new Element();
sandbox.modelSelect = { value: "org/model" };
sandbox.modelsAvailable = true;
sandbox.systemPromptTextarea = { value: "" };

(async function () {
  sandbox.chatInput.value = "first";
  await sandbox.sendMessage();
  sandbox.chatInput.value = "second";
  await sandbox.sendMessage();
  process.stdout.write(JSON.stringify({ requests, history: sandbox.messages }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""


@pytest.mark.parametrize(
    ("scenario", "expected_roles", "expected_contents"),
    [
        ("http_error", ["user"], ["second"]),
        ("sse_error", ["user"], ["second"]),
        ("network_empty", ["user"], ["second"]),
        (
            "network_partial",
            ["user", "assistant", "user"],
            ["first", "partial", "second"],
        ),
        (
            "success",
            ["user", "assistant", "user"],
            ["first", "complete", "second"],
        ),
        (
            "decoy_message",
            ["user", "assistant", "user"],
            ["first", "hello world", "second"],
        ),
        (
            "empty_success",
            ["user", "assistant", "user"],
            ["first", "", "second"],
        ),
    ],
)
def test_chat_history_commits_only_complete_turns(
    scenario: str,
    expected_roles: list[str],
    expected_contents: list[str],
) -> None:
    """The second request contains only prior turns committed by the matrix."""
    result = _run_node(_TRANSACTION_HARNESS, scenario)
    requests = result["requests"]
    assert isinstance(requests, list)
    assert len(requests) == 2
    second_request = requests[1]
    assert isinstance(second_request, dict)
    request_messages = second_request["messages"]
    assert isinstance(request_messages, list)

    roles = [message["role"] for message in request_messages]
    contents = [message["content"] for message in request_messages]
    assert roles == expected_roles
    assert contents == expected_contents
    assert all(left != right for left, right in zip(roles, roles[1:], strict=False))


_SYSTEM_PROMPT_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const scenario = process.argv[2];
const savedPrompt = scenario === "no_prompt" ? "" : "Be concise";
const requests = [];
let domReady = null;

class Element {
  constructor(id) {
    this.id = id || "";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    this.value = "";
    this.disabled = false;
    this.className = "";
    this._text = "";
    this._html = "";
    const classes = new Set();
    this.classList = {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name) {
        if (classes.has(name)) classes.delete(name);
        else classes.add(name);
      },
    };
  }
  get textContent() { return this._text; }
  set textContent(value) {
    this._text = String(value);
    if (value === "") this.children = [];
  }
  get innerHTML() { return this._html; }
  set innerHTML(value) { this._html = String(value); this.children = []; }
  appendChild(child) {
    this.children.push(child);
    if (this.id === "model-select" && !this.value) this.value = child.value;
    return child;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  getAttribute(name) { return this.attributes[name] || null; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  querySelector(selector) {
    if (this.id === "message-area" && selector === ".message-area-inner") {
      return byId("message-area-inner");
    }
    return this.children.find(function (child) {
      return selector === ".streaming-cursor" && child.className === "streaming-cursor";
    }) || null;
  }
  focus() {}
  remove() {}
}

const elements = new Map();
function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element(id));
  return elements.get(id);
}
byId("system-prompt-toggle").setAttribute("aria-expanded", "false");

const templateMessage = "Conversation roles must alternate user/assistant/user/assistant/...";
const sandbox = {
  showToast: function () {},
  console,
  Buffer,
  TextDecoder,
  document: {
    getElementById: byId,
    querySelector(selector) {
      return selector === ".system-prompt-toggle" ? byId("system-prompt-toggle") : null;
    },
    createElement() { return new Element(); },
    addEventListener(name, callback) {
      if (name === "DOMContentLoaded") domReady = callback;
    },
  },
  localStorage: {
    getItem(key) { return key === "systemPrompt" ? savedPrompt : null; },
    setItem() {},
  },
  requestAnimationFrame(callback) { callback(); },
  setTimeout() { return 1; },
  DOMPurify: { sanitize(value) { return value; } },
  marked: { parse(value) { return value; } },
  ANONYMOUS_PLAYGROUND: true,
  fetch: async function (url, options) {
    if (url === "/onboarding/token/reveal") {
      throw new Error("an anonymous-playground viewer must not reveal a token");
    }
    if (url === "/v1/models") {
      return { ok: true, json: async function () { return { data: [{ id: "org/model" }] }; } };
    }
    requests.push(JSON.parse(options.body));
    const body = scenario === "flat"
      ? { object: "error", message: templateMessage }
      : { error: { message: templateMessage } };
    return { ok: false, status: 400, json: async function () { return body; } };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  domReady();
  await sandbox.loadModels();
  byId("model-select").value = "org/model";
  byId("chat-input").value = "hello";
  await sandbox.sendMessage();
  const bubbles = byId("message-area-inner").children;
  const assistantBubble = bubbles[bubbles.length - 1];
  process.stdout.write(JSON.stringify({
    expanded: byId("system-prompt-panel").classList.contains("expanded"),
    ariaExpanded: byId("system-prompt-toggle").getAttribute("aria-expanded"),
    prompt: byId("system-prompt").value,
    requestMessages: requests[0].messages,
    errorText: assistantBubble.textContent,
    history: sandbox.messages,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""


@pytest.mark.parametrize("shape", ["nested", "flat"])
def test_restored_system_prompt_is_visible_with_actionable_template_error(
    shape: str,
) -> None:
    result = _run_node(_SYSTEM_PROMPT_HARNESS, shape)

    assert result["expanded"] is True
    assert result["ariaExpanded"] == "true"
    assert result["prompt"] == "Be concise"
    assert [message["role"] for message in result["requestMessages"]] == [
        "system",
        "user",
    ]
    assert result["requestMessages"][0]["content"] == "Be concise"
    assert "Conversation roles must alternate" in result["errorText"]
    assert "Clear the System Prompt and try again" in result["errorText"]
    assert result["history"] == []


def test_alternation_error_without_system_prompt_has_no_misleading_guidance() -> None:
    result = _run_node(_SYSTEM_PROMPT_HARNESS, "no_prompt")

    assert result["expanded"] is False
    assert result["ariaExpanded"] == "false"
    assert [message["role"] for message in result["requestMessages"]] == ["user"]
    assert result["errorText"] == (
        "Conversation roles must alternate user/assistant/user/assistant/..."
    )
    assert result["history"] == []


_CREDENTIAL_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const scenario = process.argv[2];
const calls = [];
let domReady = null;
let revealCount = 0;

class Element {
  constructor(id) {
    this.id = id || "";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    this.value = "";
    this.disabled = false;
    this.href = "";
    this.className = "";
    this._text = "";
    const classes = new Set();
    this.classList = {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name) {
        if (classes.has(name)) classes.delete(name);
        else classes.add(name);
      },
    };
  }
  get textContent() { return this._text; }
  set textContent(value) {
    // Like the DOM: assigning textContent replaces every child node.
    this._text = String(value);
    this.children = [];
  }
  appendChild(child) {
    this.children.push(child);
    if (this.id === "model-select" && !this.value) this.value = child.value;
    return child;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  getAttribute(name) { return this.attributes[name] || null; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  querySelector(selector) {
    if (this.id === "message-area" && selector === ".message-area-inner") {
      return byId("message-area-inner");
    }
    if (selector === ".streaming-cursor") {
      return this.children.find(function (child) {
        return child.className === "streaming-cursor";
      }) || null;
    }
    return null;
  }
  focus() {}
  remove() {}
}

const elements = new Map();
function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element(id));
  return elements.get(id);
}

function jsonResponse(body) {
  return { ok: true, status: 200, json: async function () { return body; } };
}

function failureResponse(status, body) {
  return { ok: false, status: status, json: async function () { return body; } };
}

const completionStream = {
  ok: true,
  status: 200,
  body: {
    getReader() {
      let position = 0;
      return {
        async read() {
          position += 1;
          if (position === 1) {
            return {
              done: false,
              value: Buffer.from(
                'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
              ),
            };
          }
          if (position === 2) {
            return { done: false, value: Buffer.from("data: [DONE]\n\n") };
          }
          return { done: true };
        },
      };
    },
  },
};

const sandbox = {
  showToast: function () {},
  console,
  Buffer,
  TextDecoder,
  ANONYMOUS_PLAYGROUND: scenario === "anonymous",
  document: {
    getElementById: byId,
    querySelector() { return null; },
    createElement() { return new Element(); },
    addEventListener(name, callback) {
      if (name === "DOMContentLoaded") domReady = callback;
    },
  },
  localStorage: { getItem() { return null; }, setItem() {} },
  requestAnimationFrame(callback) { callback(); },
  setTimeout() { return 1; },
  DOMPurify: { sanitize(value) { return value; } },
  marked: { parse(value) { return value; } },
  fetch: async function (url, options) {
    calls.push({
      url: url,
      method: (options && options.method) || "GET",
      headers: (options && options.headers) || {},
      body: options && options.body ? JSON.parse(options.body) : null,
    });
    if (url === "/onboarding/token/reveal") {
      revealCount += 1;
      if (scenario === "anonymous") {
        throw new Error("an anonymous-playground viewer must not reveal a token");
      }
      if (scenario === "unavailable") {
        return failureResponse(404, { detail: "You don't have a token yet" });
      }
      if (scenario === "unexpected_403") {
        return failureResponse(403, { detail: "Not allowed" });
      }
      if (scenario === "error") return failureResponse(500, {});
      // The page reveals before every turn: a token replaced elsewhere comes
      // back as the new value, and a revoked one as a refusal.
      if (scenario === "replaced_token" && revealCount > 2) {
        return jsonResponse({ token: "qiip_second" });
      }
      if (scenario === "stale" && revealCount > 2) {
        return failureResponse(404, { detail: "You don't have a token yet" });
      }
      return jsonResponse({ token: "qiip_s3cret" });
    }
    if (url === "/v1/models") {
      return jsonResponse({ data: [{ id: "org/model" }] });
    }
    if (url === "/v1/chat/completions") return completionStream;
    throw new Error("unexpected URL: " + url);
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  domReady();
  for (let tick = 0; tick < 10; tick += 1) await Promise.resolve();
  byId("model-select").value = "org/model";
  if (scenario === "long_history") {
    // Seed an oversized transcript (40 turns of 4k chars) before the turn.
    for (let turn = 0; turn < 40; turn += 1) {
      sandbox.messages.push({ role: "user", content: "u".repeat(2000) });
      sandbox.messages.push({ role: "assistant", content: "a".repeat(2000) });
    }
  }
  byId("chat-input").value = "first";
  await sandbox.sendMessage();
  for (let tick = 0; tick < 10; tick += 1) await Promise.resolve();
  byId("chat-input").value = "second";
  await sandbox.sendMessage();
  for (let tick = 0; tick < 10; tick += 1) await Promise.resolve();
  const notice = byId("empty-state-text");
  process.stdout.write(JSON.stringify({
    calls,
    disabled: byId("chat-input").disabled,
    sendDisabled: byId("send-btn").disabled,
    modelOptions: byId("model-select").children.map(function (child) {
      return child.textContent;
    }),
    noticeText: notice.textContent,
    noticeLinks: notice.children.map(function (child) { return child.href; }),
    bubbles: byId("message-area-inner").children.map(function (child) {
      return {
        error: child.classList.contains("bubble-error"),
        text: child.textContent,
        links: child.children.map(function (grandchild) {
          return grandchild.href;
        }),
      };
    }),
    history: sandbox.messages,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""


def _v1_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    calls = result["calls"]
    assert isinstance(calls, list)
    return [call for call in calls if str(call["url"]).startswith("/v1/")]


def _reveal_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    calls = result["calls"]
    assert isinstance(calls, list)
    return [call for call in calls if call["url"] == "/onboarding/token/reveal"]


def _completions(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for call in _v1_calls(result) if call["url"] == "/v1/chat/completions"]


def test_playground_reveals_per_turn_and_asks_for_streamed_usage() -> None:
    result = _run_node(_CREDENTIAL_HARNESS, "user_token")

    reveal = _reveal_calls(result)
    assert [(call["method"], call["url"]) for call in reveal] == [
        ("POST", "/onboarding/token/reveal")
    ] * 3
    calls = _v1_calls(result)
    assert [call["url"] for call in calls] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all(
        call["headers"]["Authorization"] == "Bearer qiip_s3cret" for call in calls
    )
    # A caller that means to be attributed never lets an unresolvable token
    # fall back to anonymous.
    assert all(call["headers"]["X-Qiip-Require-Token"] == "1" for call in calls)
    assert all(
        call["body"]["stream"] is True
        and call["body"]["stream_options"] == {"include_usage": True}
        for call in _completions(result)
    )
    assert result["disabled"] is False
    assert result["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "hello"},
    ]


def test_replaced_token_is_followed_between_turns() -> None:
    """The second turn must not reuse the token cached by the first one."""
    result = _run_node(_CREDENTIAL_HARNESS, "replaced_token")

    completions = _completions(result)
    assert [call["headers"]["Authorization"] for call in completions] == [
        "Bearer qiip_s3cret",
        "Bearer qiip_second",
    ]


def test_revoked_token_mid_session_refuses_the_turn() -> None:
    """A reveal that fails on the turn path disables the page: the conversation
    never silently continues as an anonymous request, and the dead end still
    carries the way out."""
    result = _run_node(_CREDENTIAL_HARNESS, "stale")

    assert len(_completions(result)) == 1
    assert result["disabled"] is True
    assert result["sendDisabled"] is True
    failure = [bubble for bubble in result["bubbles"] if bubble["error"] is True]
    assert len(failure) == 1
    assert "You don't have a token yet" in str(failure[0]["text"])
    assert failure[0]["links"] == ["/start"]


def test_playground_stays_disabled_without_a_token() -> None:
    result = _run_node(_CREDENTIAL_HARNESS, "unavailable")

    assert _v1_calls(result) == []
    assert result["disabled"] is True
    assert result["sendDisabled"] is True
    assert result["modelOptions"] == ["Chat unavailable"]
    assert "You don't have a token yet" in result["noticeText"]
    assert result["noticeLinks"] == ["/start"]


def test_credential_failure_disables_the_playground() -> None:
    result = _run_node(_CREDENTIAL_HARNESS, "error")

    assert _v1_calls(result) == []
    assert result["disabled"] is True
    assert "Reload the page" in result["noticeText"]


def test_unexpected_reveal_refusal_disables_the_playground() -> None:
    """A refusal the page does not expect must not become anonymous calls."""
    result = _run_node(_CREDENTIAL_HARNESS, "unexpected_403")

    assert _v1_calls(result) == []
    assert result["disabled"] is True
    assert result["noticeText"].startswith("Not allowed")


def test_anonymous_playground_viewer_calls_v1_anonymously() -> None:
    result = _run_node(_CREDENTIAL_HARNESS, "anonymous")

    assert _reveal_calls(result) == []
    calls = _v1_calls(result)
    assert [call["url"] for call in calls] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all("Authorization" not in call["headers"] for call in calls)
    assert all("X-Qiip-Require-Token" not in call["headers"] for call in calls)
    assert result["disabled"] is False


def test_history_is_bounded_before_it_is_resent() -> None:
    """A long session must not re-send an oversized transcript forever."""
    result = _run_node(_CREDENTIAL_HARNESS, "long_history")

    history = result["history"]
    # The seeded 80 messages (160k chars) are trimmed to the older turns that
    # fit the budget plus the two newest turns; alternation is preserved.
    assert len(history) == 14
    assert [message["role"] for message in history] == ["user", "assistant"] * 7
    assert history[-2:] == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "hello"},
    ]
    total = sum(len(message["content"]) for message in history)
    assert total <= 24000 + 10
    resent = _completions(result)[-1]["body"]["messages"]
    assert sum(len(message["content"]) for message in resent) <= (
        24000 + len("first") + len("hello") + len("second")
    )
