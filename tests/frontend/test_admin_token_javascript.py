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
_TABLE_SORT_JS = _ROOT / "inference_proxy/static/js/table_sort.js"
_FORMAT_USD_JS = _ROOT / "inference_proxy/static/js/format_usd.js"


def _run_node(source: Path, harness: str) -> dict[str, Any]:
    """Run a harness against ``source`` with the shared sorter loaded first.

    The module is handed to the harness as ``process.argv[1]``, the shared
    sorter at ``process.argv[2]``, and the USD formatter at
    ``process.argv[3]``, mirroring the template order.
    """
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for admin token JS regressions")

    result = subprocess.run(
        [
            node,
            "-e",
            harness,
            str(source),
            str(_TABLE_SORT_JS),
            str(_FORMAT_USD_JS),
        ],
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
const sortSource = fs.readFileSync(process.argv[2], "utf8");
const formatSource = fs.readFileSync(process.argv[3], "utf8");
const elements = new Map();
let domReady = null;

class Element {{
  constructor(tagName) {{
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {{}};
    this.dataset = {{}};
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
  removeAttribute(name) {{ if (this.attributes) delete this.attributes[name]; }}
  getAttribute(name) {{ return this.attributes && name in this.attributes ? this.attributes[name] : null; }}
  focus() {{ this.focused = true; }}
  remove() {{}}
}}

function byId(id) {{
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}}

const sandbox = {{
  showToast: function () {{}},
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
sandbox.window = sandbox;
vm.runInContext(sortSource, sandbox);
vm.runInContext(formatSource, sandbox);
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
    assert result["usersFirst"][2] == "no"
    assert result["usersFirst"][3] == "1 / 1"
    assert result["usersFirst"][4] == "2"
    assert result["usersFirst"][7] == "150"
    assert result["usersFirst"][8] == "$1.75"
    assert result["tokensRows"] == 1
    assert "ci-job" in result["tokensFirst"]
    assert "Full access" in result["tokensFirst"]
    assert result["tokensFirst"][7] == "2"
    assert result["tokensFirst"][10] == "150"
    assert result["tokenCount"] == "1 tokens across 1 users"


def _role_harness(user: dict[str, Any]) -> str:
    """Render the users table and click the Grant/Revoke Admin button.

    Regression: the role-change requests must send
    ``Content-Type: application/json`` on POST/DELETE; ``require_admin_auth``
    rejects state-changing admin requests without a JSON media type (415).
    """
    return f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const sortSource = fs.readFileSync(process.argv[2], "utf8");
const formatSource = fs.readFileSync(process.argv[3], "utf8");
const elements = new Map();
const calls = [];
let domReady = null;

class Element {{
  constructor(tagName) {{
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {{}};
    this.dataset = {{}};
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
  removeAttribute(name) {{ if (this.attributes) delete this.attributes[name]; }}
  focus() {{ this.focused = true; }}
  remove() {{}}
}}

function byId(id) {{
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}}

function ok(body) {{ return {{ ok: true, status: 200, json: async function () {{ return body; }} }}; }}

const sandbox = {{
  showToast: function () {{}},
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
    if (url === "/admin/users") return ok([{json.dumps(user)}]);
    if (url === "/admin/users/1/admin") {{ calls.push({{ url, options }}); return ok({{}}); }}
    if (url === "/admin/tokens") return ok([]);
    return {{ ok: false, status: 500, json: async function () {{ return {{}}; }} }};
  }},
}};

vm.createContext(sandbox);
sandbox.window = sandbox;
vm.runInContext(sortSource, sandbox);
vm.runInContext(formatSource, sandbox);
vm.runInContext(source, sandbox);

(async function () {{
  domReady();
  await new Promise(function (resolve) {{ setImmediate(resolve); }});
  const row = byId("users-table-body").children[0];
  const toggle = row.children[9].children[1];
  await toggle.listeners.click();
  await new Promise(function (resolve) {{ setImmediate(resolve); }});
  process.stdout.write(JSON.stringify({{
    calls: calls.map(function (c) {{
      return {{
        url: c.url,
        method: c.options ? c.options.method : null,
        contentType: c.options && c.options.headers
          ? c.options.headers["Content-Type"]
          : null,
      }};
    }})
  }}));
}})().catch(function (error) {{
  console.error(error);
  process.exit(1);
}});
"""


def test_grant_admin_sends_json_content_type() -> None:
    result = _run_node(
        _ADMIN_TOKENS_JS,
        _role_harness(
            {"id": 1, "email": "kambiz@example.com", "name": "K", "is_admin": False}
        ),
    )
    assert result["calls"][0] == {
        "url": "/admin/users/1/admin",
        "method": "POST",
        "contentType": "application/json",
    }


def test_revoke_admin_sends_json_content_type() -> None:
    result = _run_node(
        _ADMIN_TOKENS_JS,
        _role_harness(
            {"id": 1, "email": "kambiz@example.com", "name": "K", "is_admin": True}
        ),
    )
    assert result["calls"][0] == {
        "url": "/admin/users/1/admin",
        "method": "DELETE",
        "contentType": "application/json",
    }


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
const sortSource = fs.readFileSync(process.argv[2], "utf8");
const formatSource = fs.readFileSync(process.argv[3], "utf8");
const elements = new Map();
let domReady = null;

class Element {
  constructor(tagName) {
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {};
    this.dataset = {};
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
  showToast: function () {},
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
        model_label: "claude-sonnet-5",
      }; } };
    }
    return { ok: false, json: async function () { return {}; } };
  },
};

vm.createContext(sandbox);
sandbox.window = sandbox;
vm.runInContext(sortSource, sandbox);
vm.runInContext(formatSource, sandbox);
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
    assert "claude-sonnet-5" in result["savings"]
    assert result["tokenRows"] == 1
    assert result["usageRows"] == 1
    assert result["timelineRows"] == 1
    assert "150 total tokens" in result["usageTotals"]


# Column lists copied from the ``<th data-sort>`` rows in tokens.html and
# user_detail.html; the shared sorter reads exactly these attributes.
_USERS_COLUMNS = [
    ["User", "text"],
    ["Email", "text"],
    ["Admin", "text"],
    ["Tokens", "number"],
    ["Requests", "number"],
    ["Prompt tokens", "number"],
    ["Completion tokens", "number"],
    ["Total tokens", "number"],
    ["Est. premium cost", "number"],
    ["Actions", ""],
]

_USAGE_COLUMNS = [
    ["Token", "text"],
    ["Model", "text"],
    ["Endpoint", "text"],
    ["Requests", "number"],
    ["Prompt tokens", "number"],
    ["Completion tokens", "number"],
    ["Total tokens", "number"],
]

_TOKENS_COLUMNS = [
    ["User", "text"],
    ["Name", "text"],
    ["Prefix", "text"],
    ["Created", "date"],
    ["Last used", "date"],
    ["Status", "text"],
    ["Scope", "text"],
    ["Requests", "number"],
    ["Prompt tokens", "number"],
    ["Completion tokens", "number"],
    ["Total tokens", "number"],
    ["Actions", ""],
]

_TOKENS = [
    {
        "id": 11,
        "user_id": 1,
        "user_name": "Alice",
        "user_email": "alice@example.com",
        "name": "ci-job",
        "prefix": "qiip_aaa",
        "created_at": "2026-09-02T10:00:00Z",
        "last_used_at": "2026-09-03T10:00:00Z",
        "revoked": False,
        "endpoint_scope": None,
        "request_count": 4,
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "total_tokens": 60,
    },
    {
        "id": 12,
        "user_id": 1,
        "user_name": "Alice",
        "user_email": "alice@example.com",
        "name": "old-job",
        "prefix": "qiip_bbb",
        "created_at": "2026-08-20T10:00:00Z",
        "last_used_at": None,
        "revoked": True,
        "endpoint_scope": ["gpu01"],
        "request_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
]

_USERS = [
    {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
        "is_admin": False,
        "token_count": 1,
        "active_token_count": 1,
        "request_count": 2,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 300,
        "estimated_cost_usd": 1.75,
    },
    {
        "id": 2,
        "name": "Bob",
        "email": "bob@example.com",
        "is_admin": True,
        "token_count": 2,
        "active_token_count": 1,
        "request_count": 5,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 100,
        "estimated_cost_usd": 0.25,
    },
    {
        "id": 3,
        "name": "Carol",
        "email": "carol@example.com",
        "is_admin": False,
        "token_count": 1,
        "active_token_count": 0,
        "request_count": 9,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 200,
        "estimated_cost_usd": 3.5,
    },
]

_USAGE = [
    {
        "token_id": 5,
        "token_name": "ci-job",
        "model": "llama-3",
        "endpoint": "/v1/chat/completions",
        "request_count": 2,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    },
    {
        "token_id": 6,
        "token_name": "batch",
        "model": "qwen-3",
        "endpoint": "/v1/embeddings",
        "request_count": 9,
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    },
    {
        "token_id": 7,
        "token_name": "nightly",
        "model": "mistral",
        "endpoint": "/v1/completions",
        "request_count": 1,
        "prompt_tokens": 700,
        "completion_tokens": 200,
        "total_tokens": 900,
    },
]


def _sort_harness(
    columns: list[list[str]],
    index: int,
    head_id: str,
    body_id: str,
    sandbox_extra: str,
    fetch_body: str,
) -> str:
    """Render a sortable admin table, then click one header twice.

    ``columns`` mirrors the template's ``<th data-sort>`` row and ``index`` is
    the clicked column. The stub has no event bubbling, so a click goes
    straight to the ``<th>`` listener the shared sorter installed on it; the
    result reports the rendered order and ``aria-sort`` after each click.
    """
    return f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const sortSource = fs.readFileSync(process.argv[2], "utf8");
const formatSource = fs.readFileSync(process.argv[3], "utf8");
const elements = new Map();
let domReady = null;

class Element {{
  constructor(tagName) {{
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {{}};
    this.dataset = {{}};
    this.attributes = {{}};
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
  appendChild(child) {{
    const at = this.children.indexOf(child);
    if (at !== -1) this.children.splice(at, 1); // as in the DOM, re-appending moves the row
    this.children.push(child);
    return child;
  }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  setAttribute(name, value) {{ this.attributes[name] = String(value); }}
  removeAttribute(name) {{ delete this.attributes[name]; }}
  getAttribute(name) {{ return name in this.attributes ? this.attributes[name] : null; }}
  click() {{ if (this.listeners.click) this.listeners.click(); }}
  focus() {{ this.focused = true; }}
  get firstChild() {{ return this.children[0] || null; }}
  removeChild(child) {{
    const at = this.children.indexOf(child);
    if (at !== -1) this.children.splice(at, 1);
    return child;
  }}
  remove() {{}}
}}

function byId(id) {{
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}}

function makeHead(columns) {{
  const row = new Element("tr");
  columns.forEach(function (column) {{
    const th = new Element("th");
    if (column[1]) th.dataset.sort = column[1];
    row.appendChild(th);
  }});
  const head = new Element("thead");
  head.appendChild(row);
  return head;
}}

elements.set({json.dumps(head_id)}, makeHead({json.dumps(columns)}));

const sandbox = {{
  showToast: function () {{}},
  console,
  POLL_INTERVAL_MS: 10000,
  {sandbox_extra}
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
  setInterval(callback) {{ sandbox.poll = callback; return 1; }},
  localStorage: {{ getItem() {{ return null; }}, setItem() {{}} }},
  fetch: async function (url, options) {{
    {fetch_body}
  }},
}};

vm.createContext(sandbox);
sandbox.window = sandbox;
vm.runInContext(sortSource, sandbox);
vm.runInContext(formatSource, sandbox);
vm.runInContext(source, sandbox);

(async function () {{
  domReady();
  await new Promise(function (resolve) {{ setImmediate(resolve); }});
  const head = byId({json.dumps(head_id)});
  const body = byId({json.dumps(body_id)});
  const th = head.children[0].children[{index}];
  const cells = function (row) {{
    return row.children.map(function (cell) {{
      return cell.textContent || (cell.children[0] ? cell.children[0].textContent : "");
    }});
  }};
  const column = function () {{
    return body.children.map(function (row) {{ return cells(row)[{index}]; }});
  }};
  const renderOrder = column();
  th.click();
  const orderAsc = column();
  const ariaAsc = th.getAttribute("aria-sort");
  const firstAsc = cells(body.children[0]);
  th.click();
  const orderDesc = column();
  const ariaDesc = th.getAttribute("aria-sort");
  const firstDesc = cells(body.children[0]);
  await sandbox.poll();
  const orderAfterPoll = column();
  const ariaAfterPoll = th.getAttribute("aria-sort");
  process.stdout.write(JSON.stringify({{
    renderOrder: renderOrder,
    orderAsc: orderAsc,
    orderDesc: orderDesc,
    ariaAsc: ariaAsc,
    ariaDesc: ariaDesc,
    firstAsc: firstAsc,
    firstDesc: firstDesc,
    orderAfterPoll: orderAfterPoll,
    ariaAfterPoll: ariaAfterPoll,
    rowsAfterPoll: body.children.length,
  }}));
}})().catch(function (error) {{ console.error(error); process.exit(1); }});
"""


def test_admin_tokens_users_table_sorts_by_total_tokens() -> None:
    """Clicking "Total tokens" reorders the rows and flips aria-sort."""
    result = _run_node(
        _ADMIN_TOKENS_JS,
        _sort_harness(
            _USERS_COLUMNS,
            7,
            "users-head",
            "users-table-body",
            "",
            f"""
    if (url === "/admin/users") {{
      return {{ ok: true, json: async function () {{ return {json.dumps(_USERS)}; }} }};
    }}
    if (url === "/admin/tokens") {{
      return {{ ok: true, json: async function () {{ return []; }} }};
    }}
    return {{ ok: false, status: 500, json: async function () {{ return {{}}; }} }};
    """,
        ),
    )

    assert result["renderOrder"] == ["300", "100", "200"]
    assert result["orderAsc"] == ["100", "200", "300"]
    assert result["ariaAsc"] == "ascending"
    assert result["firstAsc"][0] == "Bob"
    assert result["orderDesc"] == ["300", "200", "100"]
    assert result["ariaDesc"] == "descending"
    assert result["firstDesc"][0] == "Alice"
    # The poll render replaces every row; the sorter re-applies the choice.
    assert result["orderAfterPoll"] == ["300", "200", "100"]
    assert result["ariaAfterPoll"] == "descending"
    assert result["rowsAfterPoll"] == 3


def test_emptied_table_clears_aria_sort() -> None:
    """A poll that empties a sorted table drops the announced state."""
    result = _run_node(
        _ADMIN_TOKENS_JS,
        _sort_harness(
            _USERS_COLUMNS,
            7,
            "users-head",
            "users-table-body",
            "",
            f"""
    if (url === "/admin/users") {{
      sandbox.usersFetches = (sandbox.usersFetches || 0) + 1;
      const empty = sandbox.usersFetches > 1;
      return {{ ok: true, json: async function () {{ return empty ? [] : {json.dumps(_USERS)}; }} }};
    }}
    if (url === "/admin/tokens") {{
      return {{ ok: true, json: async function () {{ return []; }} }};
    }}
    return {{ ok: false, status: 500, json: async function () {{ return {{}}; }} }};
    """,
        ),
    )

    assert result["ariaDesc"] == "descending"
    # The empty-state row replaces the data rows and the sort is no longer
    # announced; the chosen column is retained for the next non-empty render.
    assert result["orderAfterPoll"] == [None]
    assert result["ariaAfterPoll"] is None
    assert result["rowsAfterPoll"] == 1


def test_admin_tokens_created_sort_treats_revoked_rows_as_missing() -> None:
    """A revoked token shows "-" for Created and sorts as the oldest."""
    result = _run_node(
        _ADMIN_TOKENS_JS,
        _sort_harness(
            _TOKENS_COLUMNS,
            3,
            "tokens-head",
            "tokens-table-body",
            "",
            f"""
    if (url === "/admin/users") {{
      return {{ ok: true, json: async function () {{ return []; }} }};
    }}
    if (url === "/admin/tokens") {{
      return {{ ok: true, json: async function () {{ return {json.dumps(_TOKENS)}; }} }};
    }}
    return {{ ok: false, status: 500, json: async function () {{ return {{}}; }} }};
    """,
        ),
    )

    # The active token renders its date; the revoked one renders "-".
    assert result["renderOrder"][0] != "-"
    assert result["renderOrder"][1] == "-"
    # A revoked row has no displayed date, so it sorts as the oldest.
    assert result["orderAsc"][0] == "-"
    assert result["ariaAsc"] == "ascending"
    assert result["orderDesc"][-1] == "-"
    assert result["ariaDesc"] == "descending"


def test_admin_user_detail_usage_table_sorts_by_total_tokens() -> None:
    """Clicking "Total tokens" on the usage table reorders the rows."""
    detail = {
        "user": {"id": 1, "email": "alice@example.com", "name": "Alice", "picture": ""},
        "tokens": [],
        "usage": _USAGE,
        "totals": {
            "request_count": 12,
            "prompt_tokens": 830,
            "completion_tokens": 260,
            "total_tokens": 1090,
        },
        "timeline": [],
        "estimated_cost_usd": 1.75,
        "model_label": "claude-sonnet-5",
    }
    result = _run_node(
        _ADMIN_USER_DETAIL_JS,
        _sort_harness(
            _USAGE_COLUMNS,
            6,
            "usage-head",
            "usage-table-body",
            "USER_ID: 1,",
            f"""
    if (url === "/admin/users/1") {{
      return {{ ok: true, json: async function () {{ return {json.dumps(detail)}; }} }};
    }}
    return {{ ok: false, status: 500, json: async function () {{ return {{}}; }} }};
    """,
        ),
    )

    assert result["renderOrder"] == ["150", "40", "900"]
    assert result["orderAsc"] == ["40", "150", "900"]
    assert result["ariaAsc"] == "ascending"
    assert result["firstAsc"][0] == "batch"
    assert result["firstAsc"][6] == "40"
    assert result["orderDesc"] == ["900", "150", "40"]
    assert result["ariaDesc"] == "descending"
    assert result["firstDesc"][0] == "nightly"
    assert result["orderAfterPoll"] == ["900", "150", "40"]
    assert result["ariaAfterPoll"] == "descending"
    assert result["rowsAfterPoll"] == 3
