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
  console,
  Buffer,
  TextDecoder,
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
  fetch: async function (url, options) {
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
