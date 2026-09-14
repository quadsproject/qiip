"""Behavioral tests for the admin token dashboard JavaScript (RFE #113)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ADMIN_TOKENS_JS = _ROOT / "inference_proxy/static/js/admin_tokens.js"
_ADMIN_USER_DETAIL_JS = _ROOT / "inference_proxy/static/js/admin_user_detail.js"


def _run_node(source: Path, harness: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for admin token JS regressions")

    result = subprocess.run(
        [node, "-e", harness, str(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _render_harness(extra: str, scripts: str = "") -> str:
    return f"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const elements = new Map();
let domReady = null;

class Element {{
  constructor(tagName) {{
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {{}};
    this.className = "";
    this.style = {{}};
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this._text = "";
  }}
  get textContent() {{ return this._text; }}
  set textContent(value) {{
    this._text = String(value);
    if (value === "") this.children = [];
  }}
  appendChild(child) {{ this.children.push(child); return child; }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  setAttribute(name, value) {{ this.attributes = this.attributes || {{}}; this.attributes[name] = String(value); }}
  focus() {{ this.focused = true; }}
  remove() {{}}
}}

function byId(id) {{
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}}

const sandbox = {{
  console,
  POLL_INTERVAL_MS: 10000,
  document: {{
    getElementById: byId,
    addEventListener(name, callback) {{ if (name === "DOMContentLoaded") domReady = callback; }},
    querySelectorAll() {{ return []; }},
    createElement(tagName) {{ return new Element(tagName); }},
    createTextNode(text) {{ const node = new Element("text"); node.textContent = text; return node; }},
  }},
  confirmDialog: async function () {{ return true; }},
  requestAnimationFrame() {{}},
  setTimeout() {{ return 1; }},
  setInterval() {{ return 1; }},
  localStorage: {{ getItem() {{ return null; }}, setItem() {{}} }},
  fetch: async function (url, options) {{
    {extra}
  }},
}};

vm.createContext(sandbox);
{scripts}
vm.runInContext(source, sandbox);

(async function () {{
  domReady();
  await new Promise(function (resolve) {{ setImmediate(resolve); }});
  process.stdout.write(JSON.stringify({{
    usersRows: byId("users-table-body") ? byId("users-table-body").children.length : 0,
    usersFirst: byId("users-table-body") && byId("users-table-body").children[0]
      ? byId("users-table-body").children[0].children.map(function (c) {{ return c.textContent || (c.children[0] && c.children[0].textContent) || ""; }})
      : [],
    tokensRows: byId("tokens-table-body") ? byId("tokens-table-body").children.length : 0,
    tokensFirst: byId("tokens-table-body") && byId("tokens-table-body").children[0]
      ? byId("tokens-table-body").children[0].children.map(function (c) {{ return c.textContent || (c.children[0] && c.children[0].textContent) || ""; }})
      : [],
    tokenCount: byId("token-count") ? byId("token-count").textContent : "",
    warning: byId("poll-warning") ? byId("poll-warning").textContent : "",
  }}));
}})().catch(function (error) {{ console.error(error); process.exit(1); }});
"""


def test_admin_tokens_renders_users_and_tokens() -> None:
    harness = _render_harness(
        """
    if (url === "/admin/users") {
      return { ok: true, json: async function () { return [{
        id: 1, email: "alice@example.com", name: "Alice", picture: "",
        token_count: 1, active_token_count: 1, request_count: 2,
        prompt_tokens: 100, completion_tokens: 50, total_tokens: 150,
        estimated_cost_usd: 1.75,
      }]; } };
    }
    if (url === "/admin/tokens") {
      return { ok: true, json: async function () { return [{
        id: 5, user_id: 1, user_email: "alice@example.com", user_name: "Alice",
        name: "ci-job", prefix: "qiip_abc12345",
        created_at: "2026-01-01T00:00:00Z", last_used_at: null,
        revoked: false, endpoint_scope: null,
        request_count: 2, prompt_tokens: 100, completion_tokens: 50,
        total_tokens: 150,
      }]; } };
    }
    return { ok: false, json: async function () { return {}; } };
    """
    )

    result = _run_node(_ADMIN_TOKENS_JS, harness)

    assert result["usersRows"] == 1
    assert result["usersFirst"][0] == "Alice"
    assert result["usersFirst"][1] == "alice@example.com"
    assert result["usersFirst"][3] == "2"
    assert result["usersFirst"][6] == "150"
    assert result["usersFirst"][7] == "$1.75"
    assert result["tokensRows"] == 1
    assert "ci-job" in result["tokensFirst"]
    assert "Full access" in result["tokensFirst"]
    assert result["tokensFirst"][7] == "2"
    assert result["tokensFirst"][10] == "150"
    assert result["tokenCount"] == "1 tokens across 1 users"


def test_admin_tokens_renders_pinned_to_nothing_not_full_access() -> None:
    harness = _render_harness(
        """
    if (url === "/admin/users") {
      return { ok: true, json: async function () { return []; } };
    }
    if (url === "/admin/tokens") {
      return { ok: true, json: async function () { return [{
        id: 5, user_id: 1, user_email: "alice@example.com", user_name: "Alice",
        name: "locked", prefix: "qiip_arrrrrrrr",
        created_at: "2026-01-01T00:00:00Z", last_used_at: null,
        revoked: false, endpoint_scope: [],
        request_count: 0, prompt_tokens: 0, completion_tokens: 0,
        total_tokens: 0,
      }]; } };
    }
    return { ok: false, json: async function () { return {}; } };
    """
    )

    result = _run_node(_ADMIN_TOKENS_JS, harness)

    assert "None" in result["tokensFirst"]
    assert "Full access" not in result["tokensFirst"]


def test_admin_tokens_failed_users_fetch_sets_warning() -> None:
    harness = _render_harness(
        """
    return { ok: false, status: 500, json: async function () { return {}; } };
    """
    )

    result = _run_node(_ADMIN_TOKENS_JS, harness)

    assert result["warning"] != ""


def test_admin_user_detail_renders_full_view() -> None:
    harness = """
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const elements = new Map();
let domReady = null;

class Element {
  constructor(tagName) {
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {};
    this.className = "";
    this.style = {};
    this.value = "";
    this.hidden = false;
    this._text = "";
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value); if (value === "") this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  remove() {}
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}

const sandbox = {
  console,
  POLL_INTERVAL_MS: 10000,
  USER_ID: 1,
  document: {
    getElementById: byId,
    addEventListener(name, callback) { if (name === "DOMContentLoaded") domReady = callback; },
    querySelectorAll() { return []; },
    createElement(tagName) { return new Element(tagName); },
    createTextNode(text) { const node = new Element("text"); node.textContent = text; return node; },
  },
  confirmDialog: async function () { return true; },
  requestAnimationFrame() {},
  setTimeout() { return 1; },
  setInterval() { return 1; },
  fetch: async function (url) {
    if (url === "/admin/users/1") {
      return { ok: true, json: async function () { return {
        user: { id: 1, email: "alice@example.com", name: "Alice", picture: "" },
        tokens: [{
          id: 5, user_id: 1, user_email: "alice@example.com", user_name: "Alice",
          name: "ci-job", prefix: "qiip_abc12345",
          created_at: "2026-01-01T00:00:00Z", last_used_at: null,
          revoked: false, endpoint_scope: ["gpu01"],
          request_count: 2, prompt_tokens: 100, completion_tokens: 50,
          total_tokens: 150,
        }],
        usage: [{
          token_id: 5, token_name: "ci-job", model: "llama-3",
          endpoint: "/v1/chat/completions", request_count: 2,
          prompt_tokens: 100, completion_tokens: 50, total_tokens: 150,
        }],
        totals: { request_count: 2, prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        timeline: [{
          day: "2026-01-01", request_count: 2, prompt_tokens: 100,
          completion_tokens: 50, total_tokens: 150,
        }],
        estimated_cost_usd: 1.75,
        model_label: "claude-opus-4.8",
      }; } };
    }
    return { ok: false, json: async function () { return {}; } };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  domReady();
  await new Promise(function (resolve) { setImmediate(resolve); });
  process.stdout.write(JSON.stringify({
    title: byId("user-title").textContent,
    savings: byId("user-savings").textContent,
    tokenRows: byId("tokens-table-body").children.length,
    usageRows: byId("usage-table-body").children.length,
    timelineRows: byId("timeline-table-body").children.length,
    usageTotals: byId("usage-totals").textContent,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    result = _run_node(_ADMIN_USER_DETAIL_JS, harness)

    assert result["title"] == "Alice"
    assert "1.75" in result["savings"]
    assert "claude-opus-4.8" in result["savings"]
    assert result["tokenRows"] == 1
    assert result["usageRows"] == 1
    assert result["timelineRows"] == 1
    assert "150 total tokens" in result["usageTotals"]
