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
_TABLE_SORT_JS = _ROOT / "inference_proxy/static/js/table_sort.js"
_CLIPBOARD_JS = _ROOT / "inference_proxy/static/js/clipboard.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const sortSource = fs.readFileSync(process.argv[2], "utf8");
const clipboardSource = fs.readFileSync(process.argv[3], "utf8");

const calls = [];
const toasts = [];
const clipboardWrites = [];
const elements = new Map();

function element() {
  const el = {
    children: [],
    _handlers: {},
    _attrs: {},
    hidden: false, textContent: "", value: "", className: "", style: {},
    disabled: false, src: "", alt: "", colSpan: 0,
    dataset: {},
    checked: false, selected: false, type: "", selectedOptions: [],
    addEventListener(name, fn) { this._handlers[name] = fn; },
    focus() {}, select() {}, click() {},
    getAttribute(name) { return this._attrs[name] || null; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    removeAttribute(name) { delete this._attrs[name]; },
    contains(target) { return this === target || this.children.indexOf(target) >= 0; },
    appendChild(child) {
      if (child.parent && child.parent.children) {
        const index = child.parent.children.indexOf(child);
        if (index !== -1) child.parent.children.splice(index, 1);
      }
      this.children.push(child);
      child.parent = this;
      return child;
    },
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
  showToast: function (message, type) { toasts.push([message, type]); },
  console,
  URL: URL,
  URLSearchParams: URLSearchParams,
  navigator: { clipboard: { writeText: async function (text) { clipboardWrites.push(text); } } },
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
  location: { search: "", reload() {} },
  confirm() { return true },
  history: { replaceState() {} },
  requestAnimationFrame() {},
  setTimeout() {},
  clearTimeout() {},
  fetch: async function (url, options) {
    calls.push({ url, options });
    const method = (options && options.method) || "GET";
    if (url === "/profile/tokens" && method === "POST") {
      return { ok: true, status: 200, json: async function () { return CREATED_TOKEN; } };
    }
    const body = {
      "/auth/me": USER,
      "/profile/tokens": TOKENS,
      "/profile/endpoints": ENDPOINTS,
      "/profile/usage": { totals: { request_count: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }, rows: USAGE_ROWS },
    }[url];
    return {
      ok: url !== "/auth/me" || USER !== null,
      status: USER === null && url === "/auth/me" ? 401 : 200,
      json: async function () { return body; },
    };
  },
};

function buildHead(id, kinds) {
  const head = byId(id);
  const row = element();
  kinds.forEach(function (kind) {
    const cell = element();
    if (kind) cell.dataset.sort = kind;
    row.appendChild(cell);
  });
  head.appendChild(row);
}

buildHead("tokens-head", ["text", "text", "date", "date", "text", "text", null]);
buildHead("usage-head", ["text", "text", "text", "number", "number", "number", "number"]);

// In a browser `window` IS the global object; mirror that so helpers assigned
// to `window` are reachable by their bare names, exactly like in the pages.
sandbox.window = sandbox;

vm.createContext(sandbox);
vm.runInContext(sortSource, sandbox);
vm.runInContext(clipboardSource, sandbox);
vm.runInContext(source, sandbox);

(async function () {
  // Mirror the template: #endpoint-empty is the first child of
  // #endpoint-list. renderEndpointList keeps it and only prunes generated
  // rows, so a reopen test reproduces the structure that made the old
  // clearing loop stop and leave stale checked rows behind.
  byId("endpoint-list").appendChild(byId("endpoint-empty"));
  // init() runs on load; wait for its fetches to settle.
  await new Promise(function (r) { setTimeout(r, 20); });
  // The template renders the create form hidden; mirror that so the
  // toggle actually opens it (which is what triggers loadEndpoints).
  byId("token-form").hidden = true;
  const toggle = byId("token-toggle");
  if (toggle && toggle._handlers.click) toggle._handlers.click();
  await new Promise(function (r) { setTimeout(r, 20); });

  if (CHECK_INDEX >= 0) {
    const empty = byId("endpoint-empty");
    const rows = byId("endpoint-list").children.filter(function (child) {
      return child !== empty;
    });
    const checkbox = rows[CHECK_INDEX] && rows[CHECK_INDEX].children[0];
    if (checkbox) {
      checkbox.checked = true;
      checkbox._handlers.change();
    }
  }

  if (REOPEN) {
    // Close and reopen the form: loadEndpoints re-renders the endpoint list.
    toggle._handlers.click();
    await new Promise(function (r) { setTimeout(r, 20); });
    toggle._handlers.click();
    await new Promise(function (r) { setTimeout(r, 20); });
  }

  if (SHOULD_CREATE) {
    // Mint a token, then use the Copy button beside the one-time reveal.
    byId("token-name").value = "laptop";
    byId("token-form")._handlers.submit({ preventDefault() {} });
    await new Promise(function (r) { setTimeout(r, 20); });
    byId("token-copy-btn")._handlers.click();
    await new Promise(function (r) { setTimeout(r, 20); });
  }

  let usageAscending = null;
  if (SORT_USAGE) {
    const usageHead = byId("usage-head").children[0].children;
    const header = usageHead[usageHead.length - 1];
    header._handlers.click();
    usageAscending = byId("usage-table-body").children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : "";
    });
    header._handlers.click();
  }

  const body = byId("tokens-table-body");
  const list = byId("endpoint-list");
  const select = byId("token-endpoints");
  process.stdout.write(JSON.stringify({
    summary: byId("endpoint-summary").textContent,
    emptyHidden: byId("endpoint-empty").hidden,
    secret: {
      hidden: byId("token-secret").hidden,
      value: byId("token-secret-value").textContent,
    },
    clipboardWrites: clipboardWrites,
    toasts: toasts,
    posted: calls
      .filter(function (call) { return call.options && call.options.method === "POST"; })
      .map(function (call) { return call.url; }),
    rowNames: body.children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : row.textContent;
    }),
    rowScopes: body.children.map(function (row) {
      return row.children[5] ? row.children[5].textContent : "";
    }),
    usageRows: byId("usage-table-body").children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : "";
    }),
    usageAscending: usageAscending,
    usageAria: byId("usage-head").children[0].children.map(function (cell) {
      return cell.getAttribute("aria-sort");
    }),
    listValues: list.children
      .filter(function (child) { return child !== byId("endpoint-empty"); })
      .map(function (label) {
        return label.children[0] ? label.children[0].value : "";
      }),
    checkedValues: list.children
      .filter(function (child) { return child !== byId("endpoint-empty"); })
      .map(function (label) {
        return label.children[0] ? !!label.children[0].checked : false;
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
    reopen: bool = False,
    usage: list[dict[str, Any]] | None = None,
    sort_usage: bool = False,
    created_token: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for profile JS regressions")
    script = (
        _HARNESS.replace("TOKENS", json.dumps(tokens))
        .replace("ENDPOINTS", json.dumps(endpoints))
        .replace("CREATED_TOKEN", json.dumps(created_token or {}))
        .replace("SHOULD_CREATE", "true" if created_token is not None else "false")
        .replace("CHECK_INDEX", str(check_index))
        .replace("REOPEN", "true" if reopen else "false")
        .replace("USAGE_ROWS", json.dumps(usage or []))
        .replace("SORT_USAGE", "true" if sort_usage else "false")
        .replace(
            "USER",
            json.dumps(
                {"id": 1, "email": "kambiz@example.com", "name": "K", "picture": None}
            ),
        )
    )
    result = subprocess.run(
        [node, "-e", script, str(_PROFILE_JS), str(_TABLE_SORT_JS), str(_CLIPBOARD_JS)],
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


def test_picker_reopen_removes_stale_rows() -> None:
    """Regression (sjug review): reopening the form kept the previous
    checkbox rows (the clear loop stopped at the first-child paragraph), so
    a visible selection could disagree with the hidden select and mint an
    unpinned token. After reopen only fresh, unchecked rows remain."""
    result = _run(
        [_token(1, False)],
        [{"node_id": "gpu01", "model": "llama"}, {"node_id": "gpu02", "model": "qwen"}],
        check_index=0,
        reopen=True,
    )
    assert result["listValues"] == ["gpu01", "gpu02"]
    assert result["checkedValues"] == [False, False]
    assert "1 Endpoints Selected" not in result["summary"]


def test_picker_selection_updates_summary_and_select() -> None:
    result = _run(
        [_token(1, False)],
        [{"node_id": "gpu01", "model": "llama"}, {"node_id": "gpu02", "model": "qwen"}],
        check_index=0,
    )
    assert result["summary"] == "1 Endpoints Selected"
    assert result["selectValues"][0] == {"value": "gpu01", "selected": True}
    assert result["selectValues"][1] == {"value": "gpu02", "selected": False}


def test_usage_table_sorts_by_total_tokens() -> None:
    usage = [
        {
            "token_name": "ci",
            "model": "llama-3",
            "endpoint": "/v1/chat/completions",
            "request_count": 3,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        {
            "token_name": "laptop",
            "model": "qwen3",
            "endpoint": "/v1/messages",
            "request_count": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    ]

    result = _run([_token(1, False)], [], usage=usage, sort_usage=True)

    assert result["usageAscending"] == ["laptop", "ci"]
    assert result["usageRows"] == ["ci", "laptop"]
    assert result["usageAria"][-1] == "descending"
    assert result["usageAria"][0] is None


def test_unsorted_usage_table_keeps_server_order() -> None:
    usage = [
        {
            "token_name": "laptop",
            "model": "qwen3",
            "endpoint": "/v1/messages",
            "request_count": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        {
            "token_name": "ci",
            "model": "llama-3",
            "endpoint": "/v1/chat/completions",
            "request_count": 3,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    ]

    result = _run([_token(1, False)], [], usage=usage)

    assert result["usageRows"] == ["laptop", "ci"]
    assert set(result["usageAria"]) == {None}


def test_created_token_is_copied_from_the_reveal_box() -> None:
    """The one-time reveal's Copy button uses the shared clipboard helper."""
    created = {
        "token": "qiip_fresh_secret",
        "id": 9,
        "name": "laptop",
        "prefix": "qiip_fresh",
        "created_at": "2026-10-01T10:00:00Z",
        "last_used_at": None,
        "revoked": False,
        "endpoint_scope": None,
    }
    result = _run([_token(1, False)], [], created_token=created)
    assert result["posted"] == ["/profile/tokens"]
    assert result["secret"] == {"hidden": False, "value": "qiip_fresh_secret"}
    assert result["clipboardWrites"] == ["qiip_fresh_secret"]
    assert result["toasts"] == [
        ["Token created.", "success"],
        ["Token copied to clipboard.", "success"],
    ]
