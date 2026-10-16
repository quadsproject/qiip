"""Behavioral tests for the shared clipboard helpers (clipboard.js).

The helpers back every copy action in the UI: an immediate write for a value
already in hand, a promise-backed ClipboardItem for a value still being
fetched (so a slow reveal cannot outlive the click's user activation), an
execCommand fallback for insecure contexts, and the clipboard icon button
shown next to an abbreviated user token (`Copy token` title, a confirmation
toast, a retry click when the browser refuses the write, and focus restored
to the trigger).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CLIPBOARD_JS = _ROOT / "inference_proxy/static/js/clipboard.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");

const log = [];
const focuses = [];
const textareasCreated = [];
const selectedValues = [];
const notifyCalls = [];
const writeTextCalls = [];
const writeCalls = [];
let execCommandCalls = 0;
let revealCalls = 0;
let result = null;
let clock = 1000000;
let execOk = EXEC_OK;
let execThrows = THROWS;
let writeTextOk = WRITETEXT_OK;
let writeOk = WRITE_OK;
let documentHasFocus = true;

function element(tag) {
  const el = {
    tagName: tag || "div",
    id: "",
    children: [],
    _handlers: {},
    _attrs: {},
    textContent: "",
    value: "",
    className: "",
    title: "",
    style: {},
    type: "",
    _disabled: false,
    get disabled() { return this._disabled; },
    set disabled(value) {
      this._disabled = !!value;
      // Chromium blurs a focused element when it becomes disabled.
      if (this._disabled && sandbox.document.activeElement === this) {
        sandbox.document.activeElement = sandbox.document.body;
      }
    },
    addEventListener(name, fn) { this._handlers[name] = fn; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    getAttribute(name) { return this._attrs[name] || null; },
    appendChild(child) { this.children.push(child); return child; },
    append() { for (const child of arguments) this.children.push(child); },
    remove() {},
    select() {
      selectedValues.push(this.value);
      sandbox.document.activeElement = this;
    },
    focus() {
      focuses.push(this.id || this.tagName);
      sandbox.document.activeElement = this;
    },
    click() { return this._handlers.click ? this._handlers.click() : undefined; },
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
  return el;
}

const other = element("button");
other.id = "other";
const body = element("body");
const trigger = element("button");
trigger.id = "trigger";

const sandbox = {
  console,
  Blob: function (parts, options) {
    this.parts = parts;
    this.type = options && options.type;
    this.text = parts.join("");
  },
  ClipboardItem: CLIPBOARD_ITEM
    ? function (data) {
        if (ITEM_THROWING) throw new Error("ClipboardItem refuses promises");
        this.data = data;
      }
    : undefined,
  Date: { now: function () { return clock; } },
  document: {
    createElement(tag) {
      const el = element(tag);
      if ((tag || "").toLowerCase() === "textarea") textareasCreated.push(el);
      return el;
    },
    createElementNS(ns, tag) { return element(tag); },
    body: body,
    activeElement: trigger,
    hidden: false,
    hasFocus() { return documentHasFocus; },
    _listeners: {},
    addEventListener(name, fn) { this._listeners[name] = fn; },
    execCommand(cmd) {
      execCommandCalls += 1;
      log.push("execCommand(" + cmd + ")");
      if (execThrows) throw new Error("copy is unsupported");
      return execOk;
    },
  },
  navigator: NO_API
    ? {}
    : {
        clipboard: {
          writeText(text) {
            writeTextCalls.push(text);
            log.push("writeText called");
            return writeTextOk ? Promise.resolve() : Promise.reject(new Error("denied"));
          },
          write(items) {
            writeCalls.push(items);
            log.push("clipboard.write called");
            // A real write resolves only once the item's data promise does,
            // and rejects when that promise rejects (bad payload).
            return Promise.resolve(items[0].data["text/plain"]).then(function () {
              if (!writeOk) throw new Error("denied");
            });
          },
        },
      },
};

// In a browser `window` IS the global object; mirror that so helpers assigned
// to `window` are reachable by their bare names, exactly like in the pages.
sandbox.window = sandbox;

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

// A promise for the value that logs when it becomes available, so the tests
// can prove the write was initiated before the value arrived.
function pendingValue() {
  return Promise.resolve().then(function () {
    log.push("value ready");
    return "qiip_pending";
  });
}

function revealValue() {
  return function () {
    revealCalls += 1;
    log.push("reveal requested");
    if (REJECT_REVEAL) return Promise.reject(new Error("You don't have a token yet"));
    return Promise.resolve().then(function () {
      log.push("reveal resolved");
      if (MOVED_FOCUS) sandbox.document.activeElement = other;
      return REVEAL_PAYLOAD;
    });
  };
}

function copyButton() {
  const button = sandbox.qiipTokenCopyButton({
    className: "token-copy",
    reveal: revealValue(),
    notify: function (message, type) { notifyCalls.push([message, type]); },
  });
  button.id = "trigger";
  // A keyboard activation starts on the button; the disable-blur model then
  // moves focus away exactly as Chromium does.
  sandbox.document.activeElement = button;
  sandbox.__button = button;
  return button;
}

async function settle() {
  for (let i = 0; i < 8; i += 1) await Promise.resolve();
}

(async function () {
  if (SCENARIO.indexOf("string-") === 0 && SCENARIO !== "string-empty") {
    result = await sandbox.qiipCopyToClipboard("qiip_secret");
  } else if (
    SCENARIO === "promise-item" ||
    SCENARIO === "promise-item-denied" ||
    SCENARIO === "promise-no-item"
  ) {
    result = await sandbox.qiipCopyToClipboard(pendingValue());
    await settle();
  } else if (SCENARIO === "string-empty") {
    result = await sandbox.qiipCopyToClipboard("");
  } else if (SCENARIO === "promise-non-string") {
    result = await sandbox.qiipCopyToClipboard(Promise.resolve(42));
    await settle();
  } else if (SCENARIO === "promise-empty") {
    result = await sandbox.qiipCopyToClipboard(Promise.resolve(""));
    await settle();
  } else if (SCENARIO === "promise-rejects") {
    try {
      result = await sandbox.qiipCopyToClipboard(Promise.reject(new Error("no token for you")));
    } catch (err) {
      result = "rejected: " + err.message;
    }
    await settle();
  } else if (SCENARIO === "button-appearance") {
    copyButton();
  } else if (SCENARIO === "button-copy" || SCENARIO === "button-warning") {
    await copyButton().click();
  } else if (SCENARIO === "button-item-order") {
    await copyButton().click();
    await settle();
  } else if (
    SCENARIO === "button-reveal-error" ||
    SCENARIO === "button-non-string" ||
    SCENARIO === "button-empty-token"
  ) {
    await copyButton().click();
  } else if (SCENARIO === "button-retry") {
    const button = copyButton();
    await button.click();
    await settle();
    execOk = true; // the browser stops refusing clipboard writes
    await button.click();
  } else if (SCENARIO === "button-retry-expired") {
    const button = copyButton();
    await button.click(); // fetch + refused write: the value is cached
    await settle();
    await button.click(); // inside the window: cached value, no new reveal
    await settle();
    clock += 300000; // past the retry window: the next click re-fetches
    execOk = true;
    await button.click();
  } else if (SCENARIO === "button-focus-kept") {
    await copyButton().click();
  } else if (SCENARIO === "button-focus-outside") {
    documentHasFocus = false;
    await copyButton().click();
  } else if (SCENARIO === "button-retry-tab-hidden") {
    const button = copyButton();
    await button.click();
    await settle();
    sandbox.document.hidden = true;
    if (sandbox.document._listeners.visibilitychange) {
      sandbox.document._listeners.visibilitychange({});
    }
    execOk = true;
    await button.click();
  }

  const button = sandbox.__button || null;
  const icon = button ? button.children[0] : null;
  process.stdout.write(JSON.stringify({
    result: result,
    log: log,
    focuses: focuses,
    textareasCreated: textareasCreated.length,
    selectedValues: selectedValues,
    execCommandCalls: execCommandCalls,
    writeTextCalls: writeTextCalls,
    writeCalls: writeCalls.length,
    writeItemType: writeCalls.length ? Object.keys(writeCalls[0][0].data)[0] : null,
    writeItemText: writeCalls.length
      ? await writeCalls[0][0].data["text/plain"].then(
          function (blob) { return blob.text; },
          function () { return "rejected"; }
        )
      : null,
    revealCalls: revealCalls,
    notify: notifyCalls,
    activeElement: sandbox.document.activeElement
      ? sandbox.document.activeElement.id || sandbox.document.activeElement.tagName
      : null,
    button: button
      ? {
          type: button.type,
          className: button.className,
          title: button.title,
          ariaLabel: button.getAttribute("aria-label"),
          disabled: button.disabled,
          shapes: icon ? icon.children.map(function (child) { return child.tagName; }) : [],
        }
      : null,
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""

_SCENARIOS: dict[str, dict[str, Any]] = {
    "string-api": {"write_text_ok": True},
    "string-fallback": {"write_text_ok": False, "exec_ok": True},
    "string-no-api": {"no_api": True, "exec_ok": True},
    "string-denied": {"write_text_ok": False, "exec_ok": False},
    "string-throws": {"no_api": True, "exec_throws": True},
    "string-empty": {"write_text_ok": True},
    "promise-item": {"clipboard_item": True, "write_ok": True},
    "promise-item-denied": {
        "clipboard_item": True,
        "write_ok": False,
        "write_text_ok": True,
    },
    "promise-no-item": {"write_text_ok": True},
    "promise-non-string": {"clipboard_item": True, "write_ok": True},
    "promise-empty": {"write_text_ok": True},
    "promise-rejects": {"clipboard_item": True, "clipboard_item_throws": True},
    "button-appearance": {"write_text_ok": True},
    "button-copy": {"write_text_ok": True},
    "button-item-order": {"clipboard_item": True, "write_ok": True},
    "button-warning": {"no_api": True, "write_text_ok": False, "exec_ok": False},
    "button-reveal-error": {"reject_reveal": True},
    "button-non-string": {"non_string": True},
    "button-empty-token": {"empty_token": True},
    "button-retry": {"no_api": True, "write_text_ok": False, "exec_ok": False},
    "button-retry-expired": {"no_api": True, "write_text_ok": False, "exec_ok": False},
    "button-focus-kept": {"write_text_ok": True, "moved_focus": True},
    "button-focus-outside": {"write_text_ok": True},
    "button-retry-tab-hidden": {
        "no_api": True,
        "write_text_ok": False,
        "exec_ok": False,
    },
}


def _run(scenario: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for frontend JS regressions")
    flags = _SCENARIOS[scenario]
    payload: Any = {"token": "qiip_revealed_value"}
    if flags.get("empty_token"):
        payload = {"token": ""}
    if flags.get("non_string"):
        payload = {"token": {"id": 1}}
    script = (
        _HARNESS.replace("SCENARIO", json.dumps(scenario))
        .replace("CLIPBOARD_ITEM", "true" if flags.get("clipboard_item") else "null")
        .replace(
            "ITEM_THROWING",
            "true" if flags.get("clipboard_item_throws") else "false",
        )
        .replace("NO_API", "true" if flags.get("no_api") else "false")
        .replace("WRITE_OK", "true" if flags.get("write_ok") else "false")
        .replace("WRITETEXT_OK", "true" if flags.get("write_text_ok") else "false")
        .replace("EXEC_OK", "true" if flags.get("exec_ok") else "false")
        .replace("THROWS", "true" if flags.get("exec_throws") else "false")
        .replace("MOVED_FOCUS", "true" if flags.get("moved_focus") else "false")
        .replace("REJECT_REVEAL", "true" if flags.get("reject_reveal") else "false")
        .replace("REVEAL_PAYLOAD", json.dumps(payload))
    )
    result = subprocess.run(
        [node, "-e", script, str(_CLIPBOARD_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_clipboard_api_is_preferred() -> None:
    result = _run("string-api")
    assert result["result"] is True
    assert result["writeTextCalls"] == ["qiip_secret"]
    assert result["textareasCreated"] == 0
    assert result["execCommandCalls"] == 0


def test_insecure_context_falls_back_to_exec_command() -> None:
    result = _run("string-fallback")
    assert result["result"] is True
    assert result["textareasCreated"] == 1
    assert result["selectedValues"] == ["qiip_secret"]
    assert result["execCommandCalls"] == 1


def test_missing_clipboard_api_uses_the_fallback() -> None:
    result = _run("string-no-api")
    assert result["result"] is True
    assert result["textareasCreated"] == 1


def test_a_blocked_fallback_reports_failure() -> None:
    result = _run("string-denied")
    assert result["result"] is False
    assert result["execCommandCalls"] == 1


def test_a_throwing_fallback_reports_failure() -> None:
    """execCommand throws where the command is unsupported; never propagate."""
    result = _run("string-throws")
    assert result["result"] is False
    assert result["execCommandCalls"] == 1


def test_promise_value_is_written_through_a_clipboard_item() -> None:
    """The write starts inside the click, not after the value arrives."""
    result = _run("promise-item")
    assert result["result"] is True
    assert result["log"] == ["clipboard.write called", "value ready"]
    assert result["writeItemType"] == "text/plain"
    assert result["writeItemText"] == "qiip_pending"
    assert result["writeTextCalls"] == []


def test_clipboard_item_refusal_falls_back_to_the_resolved_value() -> None:
    result = _run("promise-item-denied")
    assert result["result"] is True
    assert result["log"] == [
        "clipboard.write called",
        "value ready",
        "writeText called",
    ]
    assert result["writeTextCalls"] == ["qiip_pending"]


def test_promise_value_without_clipboard_item_support() -> None:
    result = _run("promise-no-item")
    assert result["result"] is True
    assert result["log"] == ["value ready", "writeText called"]
    assert result["writeTextCalls"] == ["qiip_pending"]


def test_token_button_is_an_icon_with_a_tooltip() -> None:
    result = _run("button-appearance")
    button = result["button"]
    assert button["type"] == "button"
    assert button["className"] == "token-copy"
    assert button["title"] == "Copy token"
    assert button["ariaLabel"] == "Copy token"
    assert button["disabled"] is False
    assert button["shapes"] == ["path", "rect"]


def test_token_button_copies_and_restores_focus() -> None:
    result = _run("button-copy")
    assert result["notify"] == [["Token copied to clipboard.", "success"]]
    assert result["writeTextCalls"] == ["qiip_revealed_value"]
    assert result["activeElement"] == "trigger"


def test_token_button_warns_and_restores_focus() -> None:
    result = _run("button-warning")
    assert result["notify"] == [
        ["Couldn't copy automatically. Press the copy icon again.", "warning"]
    ]
    assert result["activeElement"] == "trigger"


def test_token_button_reports_reveal_errors() -> None:
    result = _run("button-reveal-error")
    assert result["notify"] == [["You don't have a token yet", "error"]]


def test_token_button_refuses_a_non_string_payload() -> None:
    result = _run("button-non-string")
    assert result["notify"] == [["The server did not return a token.", "error"]]


def test_retry_click_copies_without_asking_the_server_again() -> None:
    result = _run("button-retry")
    assert result["notify"] == [
        ["Couldn't copy automatically. Press the copy icon again.", "warning"],
        ["Token copied to clipboard.", "success"],
    ]
    assert result["revealCalls"] == 1
    assert result["execCommandCalls"] == 2
    assert result["activeElement"] == "trigger"


def test_retry_value_expires_so_a_replaced_token_is_refetched() -> None:
    result = _run("button-retry-expired")
    # Click 2 rides the cached value; click 3, past the window, re-fetches.
    assert result["revealCalls"] == 2
    assert result["execCommandCalls"] == 3
    assert result["notify"][-1] == ["Token copied to clipboard.", "success"]


def test_focus_is_not_stolen_when_the_user_moved_away() -> None:
    result = _run("button-focus-kept")
    assert result["activeElement"] == "other"
    assert "trigger" not in result["focuses"]


def test_non_string_promise_value_copies_nothing() -> None:
    result = _run("promise-non-string")
    assert result["result"] is False
    assert result["writeCalls"] == 1
    assert result["writeTextCalls"] == []


def test_empty_promise_value_copies_nothing() -> None:
    result = _run("promise-empty")
    assert result["result"] is False
    assert result["writeCalls"] == 0
    assert result["writeTextCalls"] == []


def test_rejected_promise_value_rejects_the_helper() -> None:
    """The button turns the rejection into its error toast, with no stray
    unhandled rejection when the ClipboardItem constructor throws first."""
    result = _run("promise-rejects")
    assert result["result"] == "rejected: no token for you"


def test_token_button_refuses_an_empty_token() -> None:
    result = _run("button-empty-token")
    assert result["notify"] == [["The server did not return a token.", "error"]]


def test_an_empty_string_copies_nothing() -> None:
    result = _run("string-empty")
    assert result["result"] is False
    assert result["writeTextCalls"] == []
    assert result["textareasCreated"] == 0


def test_button_starts_the_write_inside_the_click() -> None:
    """The reveal is still in flight when the item-backed write is issued."""
    result = _run("button-item-order")
    assert result["log"][:3] == [
        "reveal requested",
        "clipboard.write called",
        "reveal resolved",
    ]
    assert result["writeCalls"] == 1
    assert result["writeItemText"] == "qiip_revealed_value"
    assert result["writeTextCalls"] == []
    assert result["notify"] == [["Token copied to clipboard.", "success"]]


def test_focus_is_not_pulled_back_from_outside_the_page() -> None:
    result = _run("button-focus-outside")
    assert result["activeElement"] != "trigger"
    assert "trigger" not in result["focuses"]


def test_hiding_the_tab_drops_the_retry_value() -> None:
    """A token fetched in focus is not replayed after the tab was hidden."""
    result = _run("button-retry-tab-hidden")
    assert result["revealCalls"] == 2
    assert result["notify"][-1] == ["Token copied to clipboard.", "success"]
