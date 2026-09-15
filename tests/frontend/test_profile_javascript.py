"""Behavioral tests for the profile page JavaScript (profile.js).

Regression: revoked tokens must be dropped from the user-facing token list
(they are hash-disabled and only add noise). The endpoint pin picker is a
popover: a summary button ("All Endpoints (0 selected)" / "N Endpoints
Selected") that expands into a scrollable checkbox list, mirroring its
selections into the hidden multi-select used for token creation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_JS = _ROOT / "inference_proxy/static/js/profile.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");

const calls = [];
const elements = new Map();

function element() {
  const el = {
    children: [],
    _handlers: {},
    _attrs: {},
    hidden: false, textContent: "", value: "", className: "", style: {},
    disabled: false, src: "", alt: "", colSpan: 0,
    checked: false, selected: false, type: "",
    addEventListener(name, fn) { this._handlers[name] = fn; },
    focus() {}, select() {}, click() {},
    getAttribute(name) { return this._attrs[name] || null; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    contains(target) { return this === target || this.children.indexOf(target) >= 0; },
    appendChild(child) { this.children.push(child); child.parent = this; return child; },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    remove() { if (this.parent) this.parent.removeChild(this); },
    classList: { add() {}, remove() {}, contains() { return false; } },
    get firstChild() { return this.children[0] || null; },
  };
  return el;
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}

const sandbox = {
  showToast: function () {},
  console,
  URL: URL,
  URLSearchParams: URLSearchParams,
  navigator: { clipboard: { writeText: async function () {} } },
  document: {
    readyState: "complete",
    getElementById: byId,
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return element(); },
    createElement() { return element(); },
    createTextNode(text) { return { textContent: text }; },
    execCommand() { return true; },
    body: element(),
  },
  window: { location: { search: "", reload() {} }, confirm() { return true; } },
  history: { replaceState() {} },
  requestAnimationFrame() {},
  setTimeout() {},
  clearTimeout() {},
  fetch: async function (url, options) {
    calls.push({ url, options });
    const body = {
      "/auth/me": USER,
      "/profile/tokens": TOKENS,
      "/profile/endpoints": ENDPOINTS,
      "/profile/usage": { totals: { request_count: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }, rows: [] },
    }[url];
    return {
      ok: url !== "/auth/me" || USER !== null,
      status: USER === null && url === "/auth/me" ? 401 : 200,
      json: async function () { return body; },
    };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  // init() runs on load; wait for its fetches to settle.
  await new Promise(function (r) { setTimeout(r, 20); });
  // The template renders the create form hidden; mirror that so the
  // toggle actually opens it (which is what triggers loadEndpoints).
  byId("token-form").hidden = true;
  const toggle = byId("token-toggle");
  if (toggle && toggle._handlers.click) toggle._handlers.click();
  await new Promise(function (r) { setTimeout(r, 20); });

  if (CHECK_INDEX >= 0) {
    const rows = byId("endpoint-list").children;
    const checkbox = rows[CHECK_INDEX] && rows[CHECK_INDEX].children[0];
    if (checkbox) {
      checkbox.checked = true;
      checkbox._handlers.change();
    }
  }

  const body = byId("tokens-table-body");
  const list = byId("endpoint-list");
  const select = byId("token-endpoints");
  process.stdout.write(JSON.stringify({
    summary: byId("endpoint-summary").textContent,
    emptyHidden: byId("endpoint-empty").hidden,
    rowNames: body.children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : row.textContent;
    }),
    rowScopes: body.children.map(function (row) {
      return row.children[5] ? row.children[5].textContent : "";
    }),
    listValues: list.children.map(function (label) {
      return label.children[0] ? label.children[0].value : "";
    }),
    selectValues: select.children.map(function (option) {
      return { value: option.value, selected: !!option.selected };
    }),
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""


def _run(
    tokens: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    check_index: int = -1,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for profile JS regressions")
    script = (
        _HARNESS.replace("TOKENS", json.dumps(tokens))
        .replace("ENDPOINTS", json.dumps(endpoints))
        .replace("CHECK_INDEX", str(check_index))
        .replace(
            "USER",
            json.dumps(
                {"id": 1, "email": "kambiz@redhat.com", "name": "K", "picture": None}
            ),
        )
    )
    result = subprocess.run(
        [node, "-e", script, str(_PROFILE_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _token(token_id: int, revoked: bool) -> dict[str, Any]:
    return {
        "id": token_id,
        "name": f"token-{token_id}",
        "prefix": "qiip_abc",
        "created_at": "2026-09-14T12:00:00Z",
        "last_used_at": None,
        "revoked": revoked,
        "endpoint_scope": None,
    }


def test_revoked_tokens_are_dropped_from_display() -> None:
    result = _run([_token(1, False), _token(2, True), _token(3, False)], [])
    assert "token-1" in result["rowNames"]
    assert "token-2" not in result["rowNames"]
    assert "token-3" in result["rowNames"]


def test_agent_config_token_is_labeled_distinctly() -> None:
    agent = _token(1, False)
    agent["name"] = "agent-config"
    result = _run([_token(2, False), agent], [])
    assert result["rowScopes"][0] == "Full access"
    assert result["rowScopes"][1] == "Config (agent)"


def test_picker_empty_shows_message_and_zero_summary() -> None:
    result = _run([_token(1, False)], [])
    assert result["summary"] == "All Endpoints (0 selected)"
    assert result["emptyHidden"] is False
    assert result["listValues"] == []
    assert result["selectValues"] == []


def test_picker_lists_endpoints_with_zero_selected() -> None:
    result = _run(
        [_token(1, False)],
        [{"node_id": "gpu01", "model": "llama"}, {"node_id": "gpu02", "model": "qwen"}],
    )
    assert result["summary"] == "All Endpoints (0 selected)"
    assert result["emptyHidden"] is True
    assert result["listValues"] == ["gpu01", "gpu02"]
    assert all(not item["selected"] for item in result["selectValues"])


def test_picker_selection_updates_summary_and_select() -> None:
    result = _run(
        [_token(1, False)],
        [{"node_id": "gpu01", "model": "llama"}, {"node_id": "gpu02", "model": "qwen"}],
        check_index=0,
    )
    assert result["summary"] == "1 Endpoints Selected"
    assert result["selectValues"][0] == {"value": "gpu01", "selected": True}
    assert result["selectValues"][1] == {"value": "gpu02", "selected": False}
