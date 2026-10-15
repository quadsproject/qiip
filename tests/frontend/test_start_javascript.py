"""Behavioral tests for the onboarding page JavaScript (start.js).

The token home shows the token abbreviated (`prefix...`), so the clipboard icon
next to it must exist for an exportable token, reveal the raw value through
the page's API helper and toast the result; a token from before the onboarding
flow keeps the note that explains why it cannot be copied or exported. The
wizard's setup-command copy reuses the same helper and keeps its select-the-line
fallback, which the driving test exercises end to end.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_JS_DIR = _ROOT / "inference_proxy/static/js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

const calls = [];
const clipboardWrites = [];
const elements = new Map();
const timers = [];
const selectedNodes = [];

class EventStub {
  constructor(type) { this.type = type; }
}

function element(tag) {
  return {
    tagName: (tag || "div").toUpperCase(),
    children: [],
    _handlers: {},
    _attrs: {},
    dataset: {},
    hidden: false,
    inert: false,
    tabIndex: 0,
    isConnected: true,
    textContent: "",
    className: "",
    disabled: false,
    type: "",
    style: { setProperty() {} },
    addEventListener(name, fn) { this._handlers[name] = fn; },
    dispatchEvent(event) {
      if (this._handlers[event.type]) this._handlers[event.type](event);
      return true;
    },
    focus() {}, select() {}, click() { return this._handlers.click ? this._handlers.click() : undefined; },
    getAttribute(name) { return this._attrs[name] || null; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    appendChild(child) { this.children.push(child); child.parent = this; return child; },
    append() { for (const child of arguments) this.appendChild(child); },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    remove() { if (this.parent) this.parent.removeChild(this); },
    querySelectorAll() { return []; },
    querySelector() { return element(); },
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}

function find(node, predicate) {
  if (predicate(node)) return node;
  for (const child of node.children) {
    if (child && child.children) {
      const hit = find(child, predicate);
      if (hit) return hit;
    }
  }
  return null;
}

function currentStep() {
  return byId("ob-stage").children[byId("ob-stage").children.length - 1] || element();
}

// Buttons carry their label in a child span, so match on the descendant text.
function findByText(node, text) {
  return find(node, function (candidate) {
    return candidate.tagName === "BUTTON" &&
      Boolean(find(candidate, function (child) { return child.textContent === text; }));
  });
}

const sandbox = {
  console,
  Event: EventStub,
  navigator: {
    clipboard: {
      writeText: async function (text) {
        if (!COPYABLE) throw new Error("denied");
        clipboardWrites.push(text);
      },
    },
  },
  localStorage: { getItem() { return null; }, setItem() {} },
  sessionStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  document: {
    documentElement: { dataset: {} },
    getElementById: byId,
    addEventListener() {},
    createElement(tag) { return element(tag); },
    createElementNS(ns, tag) { return element(tag); },
    createTextNode(text) { return { textContent: text }; },
    createRange() {
      return { selectNodeContents(node) { selectedNodes.push(node); } };
    },
    execCommand() { return COPYABLE; },
    body: element("body"),
  },
  getSelection() { return { removeAllRanges() {}, addRange() {} }; },
  history: { replaceState() {} },
  matchMedia() { return { matches: true }; },
  // Step transitions use short delays and must run; UI state resets (toast
  // dismissal, the "Copied!" label) are long and are flushed on demand.
  setTimeout(fn, ms) {
    if (ms >= 1000) { timers.push(fn); return; }
    fn();
  },
  clearTimeout() {}, clearInterval() {},
  requestAnimationFrame(fn) { fn(); },
  fetch: async function (url, options) {
    calls.push({ url, method: (options && options.method) || "GET" });
    const body = {
      "/onboarding/state": STATE,
      "/onboarding/token/reveal": { token: "qiip_revealed_value" },
      "/onboarding/token": { token: MINTED },
      "/onboarding/setup-link": LINK,
    }[url];
    return { ok: true, status: 200, json: async function () { return body; } };
  },
};

// A next tick that runs the harness code (outside the vm) while the page's own
// promises settle inside it.
const settle = function () { return new Promise(function (resolve) { setImmediate(resolve); }); };

// In a browser `window` IS the global object.
sandbox.window = sandbox;

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
sandbox.window.QiipModelInfo = {
  close() {},
  wrap(chip) { return chip; },
  sizeChoices() {},
};
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);

(async function () {
  await settle();
  await settle();

  let wizard = null;
  if (DRIVE) {
    // welcome -> name
    currentStep().primary.click();
    await settle();
    // name -> harness
    const input = find(currentStep(), function (node) { return node.tagName === "INPUT"; });
    input.value = "ci-laptop";
    input.dispatchEvent(new sandbox.Event("input"));
    findByText(currentStep(), "Continue").click();
    await settle();
    // harness -> models
    find(currentStep(), function (node) { return node.className === "ob-choice"; }).click();
    await settle();
    // models -> done (the single online model is preselected; Create mints,
    // then fetches the setup link)
    findByText(currentStep(), "Create my token").click();
    await settle();
    await settle();
    const copy = find(currentStep(), function (node) { return node.className === "ob-copy"; });
    await copy.click();
    const copiedLabel = copy.textContent;
    while (timers.length) timers.shift()();
    wizard = {
      stepId: currentStep().dataset.step,
      copiedLabel: copiedLabel,
      resetLabel: copy.textContent,
      selected: selectedNodes.map(function (node) { return node.tagName; }),
      renderedCommand: find(currentStep(), function (node) {
        return node.tagName === "CODE";
      }).textContent,
    };
  }

  const stage = byId("ob-stage");
  const step = currentStep();
  const copy = DRIVE ? null : find(step, function (node) { return node.title === "Copy token"; });
  const key = DRIVE ? null : find(step, function (node) { return node.className === "ob-token-key"; });
  const revealBeforeClick = calls.filter(function (call) {
    return call.url === "/onboarding/token/reveal";
  }).length;
  if (copy) await copy.click();

  process.stdout.write(JSON.stringify({
    wizard: wizard,
    stepId: step.dataset.step,
    keyText: key ? key.textContent : null,
    keyHasIcon: Boolean(key && key.children.length > 0),
    copyTitle: copy ? copy.title : null,
    copyAriaLabel: copy ? copy.getAttribute("aria-label") : null,
    copyClassName: copy ? copy.className : null,
    revealBeforeClick: revealBeforeClick,
    copyIconShapes: copy ? copy.children[0].children.map(function (c) { return c.tagName; }) : [],
    noteText: step.children.map(function (child) { return child.textContent; }).join(" "),
    calls: calls,
    clipboardWrites: clipboardWrites,
    toastText: byId("ob-toast").textContent,
    stageCount: stage.children.length,
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""

_TOKEN = {
    "id": 7,
    "name": "laptop",
    "prefix": "qiip_abc123",
    "created_at": "2026-10-01T10:00:00Z",
    "last_used_at": None,
    "models": ["org/model-a"],
    "exportable": True,
}

_HARNESSES = [
    {
        "id": "pi",
        "label": "Pi",
        "tagline": "Pi coding agent",
        "multi_model": True,
        "available": True,
        "config_path": "~/.pi/agent/models.json",
    }
]

_STATE = {
    "user": {
        "name": "Test User",
        "email": "user@example.com",
        "picture": None,
        "is_admin": False,
    },
    "token": _TOKEN,
    "models": ["org/model-a"],
    "model_names": {"org/model-a": "Model A"},
    "model_details": {},
    "harnesses": _HARNESSES,
}

_MINTED = dict(_TOKEN, name="ci-laptop")
_LINK = {
    "url": "http://testserver/s/link42",
    "harness": "pi",
    "models": ["org/model-a"],
    "command": "curl -sSL http://testserver/s/link42 | bash",
    "expires_at": "2099-01-01T00:00:00Z",
}


def _run(
    exportable: bool = True,
    drive: bool = False,
    copyable: bool = True,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for frontend JS regressions")
    state = json.loads(json.dumps(_STATE))
    state["token"] = None if drive else dict(_TOKEN, exportable=exportable)
    script = (
        _HARNESS.replace("STATE", json.dumps(state))
        .replace("MINTED", json.dumps(_MINTED))
        .replace("LINK", json.dumps(_LINK))
        .replace("DRIVE", "true" if drive else "false")
        .replace("COPYABLE", "true" if copyable else "false")
    )
    result = subprocess.run(
        [node, "-e", script, str(_JS_DIR / "clipboard.js"), str(_JS_DIR / "start.js")],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_home_shows_the_copy_icon_beside_the_abbreviated_token() -> None:
    result = _run()
    assert result["stepId"] == "home"
    assert result["keyText"].startswith("qiip_abc123")
    assert result["keyHasIcon"] is True
    assert result["copyTitle"] == "Copy token"
    assert result["copyAriaLabel"] == "Copy token"
    assert result["copyClassName"] == "ob-token-copy"
    assert result["copyIconShapes"] == ["PATH", "RECT"]


def test_copy_icon_reveals_the_token_and_toasts() -> None:
    result = _run()
    urls = [call["url"] for call in result["calls"]]
    assert result["revealBeforeClick"] == 0
    assert urls.count("/onboarding/token/reveal") == 1
    assert {"url": "/onboarding/token/reveal", "method": "POST"} in result["calls"]
    assert result["clipboardWrites"] == ["qiip_revealed_value"]
    assert result["toastText"] == "Token copied to clipboard."


def test_legacy_token_gets_the_explaining_note_instead_of_the_icon() -> None:
    result = _run(exportable=False)
    assert result["copyTitle"] is None
    assert result["keyText"].startswith("qiip_abc123")
    assert "made before the new setup flow" in result["noteText"]
    assert result["calls"] == [{"url": "/onboarding/state", "method": "GET"}]


def test_wizard_setup_command_copies_through_the_shared_helper() -> None:
    result = _run(drive=True)
    assert result["stepId"] == "done"
    assert result["wizard"]["stepId"] == "done"
    assert result["wizard"]["renderedCommand"] == _LINK["command"]
    assert result["clipboardWrites"] == [_LINK["command"]]
    assert result["wizard"]["copiedLabel"] == "Copied!"
    assert result["wizard"]["resetLabel"] == "Copy"


def test_wizard_setup_command_selects_the_line_when_the_clipboard_refuses() -> None:
    result = _run(drive=True, copyable=False)
    assert result["clipboardWrites"] == []
    assert result["wizard"]["copiedLabel"] == "Copy"
    assert result["wizard"]["selected"] == ["CODE"]
    assert "line is selected" in result["toastText"]
