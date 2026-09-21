"""Frontend JS harness tests for the dashboard role gate (regression).

The dashboard template declares ``const VIEWER_ROLE`` at top level, which does
NOT create a ``window`` property. dashboard.js must resolve the role through
the lexical binding -- the bug this suite guards against read
``window.VIEWER_ROLE``, which is always ``undefined`` in a browser, so every
visitor took the admin code path and fetched ``/admin/nodes`` (401 + Basic
challenge) instead of ``/fleet/nodes``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_JS = _ROOT / "inference_proxy/static/js/dashboard.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const setupSelectionSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);

let captured = [];
const elements = new Map();

function element() {
  return {
    addEventListener() {}, appendChild(child) { this.children.push(child); return child; },
    remove() {},
    setAttribute() {}, removeAttribute() {}, textContent: "", innerHTML: "",
    value: "", className: "", style: {}, dataset: {}, hidden: false,
    tagName: "", children: [],
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}

const sandbox = {
  showToast: function () {},
  console,
  POLL_INTERVAL_MS: 10000,
  // Rows render a config download dropdown from config_download.js, which is
  // not loaded by the harness; a stub keeps the row construction path real.
  createConfigDropdown: function () { return element(); },
  document: {
    getElementById: byId,
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return element(); },
    createElement(tagName) { const el = element(); el.tagName = tagName; return el; },
    createTextNode(text) { return { textContent: text }; },
  },
  window: { location: { origin: "http://test" }, confirm() { return true; } },
  requestAnimationFrame() {},
  setTimeout() { return 0; },
  setInterval() { return 0; },
  clearInterval() {},
  EventSource: function () {},
  fetch: async function (url, options) {
    captured.push({ url, options });
    const body = {
      "/admin/nodes": ADMIN_NODES,
      "/fleet/nodes": FLEET_NODES,
      "/admin/metrics": { per_node: {} },
      "/admin/quads/status": { status: "connected", last_sync: new Date().toISOString() },
      "/admin/billing": { totals: null },
    }[url] ?? { detail: "not found" };
    return {
      ok: true,
      status: 200,
      json: async function () { return body; },
      headers: { get: function () { return null; } },
    };
  },
};

vm.createContext(sandbox);
vm.runInContext(setupSelectionSource, sandbox);
__PRESCRIPT__
vm.runInContext(source, sandbox);

(async function () {
  await sandbox.refreshDashboard();
  process.stdout.write(JSON.stringify({
    captured: captured.map(function (c) { return c.url; }),
    nodeCount: byId("node-count").textContent,
    badges: (function () {
      var row = byId("node-table-body").children[0];
      if (!row) return [];
      return row.children.flatMap(function (cell) { return cell.children || []; })
        .filter(function (el) { return (el.className || "").startsWith("badge "); })
        .map(function (el) { return { text: el.textContent, title: el.title || "" }; });
    })(),
    nodeIdLink: (function () {
      var row = byId("node-table-body").children[0];
      if (!row || !row.children[0] || !row.children[0].children[0]) return null;
      var el = row.children[0].children[0];
      return { tag: el.tagName, href: el.href || null, text: el.textContent };
    })(),
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""


def _run_harness(
    lexical_role: str | None = None,
    window_role: str | None = None,
    admin_nodes: list[dict[str, Any]] | None = None,
    fleet_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "Node.js is required for dashboard JavaScript regressions; "
            "CI must install it explicitly"
        )

    prescript = ""
    if lexical_role is not None:
        prescript += f"vm.runInContext('const VIEWER_ROLE = {json.dumps(lexical_role)};', sandbox);\n"
    if window_role is not None:
        prescript += f"vm.runInContext('window.VIEWER_ROLE = {json.dumps(window_role)};', sandbox);\n"
    harness = (
        _HARNESS.replace("__PRESCRIPT__", prescript)
        .replace("ADMIN_NODES", json.dumps(admin_nodes or []))
        .replace("FLEET_NODES", json.dumps(fleet_nodes or []))
    )

    result = subprocess.run(
        [node, "-e", harness, str(_DASHBOARD_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _node_payload() -> dict[str, Any]:
    return {
        "node_id": "gpu01",
        "name": "gpu01",
        "endpoint": "10.0.0.1:8000",
        "model": "llama-3",
        "state": "healthy",
        "managed": True,
        "self_setup": False,
        "admin_only": False,
        "engine": None,
        "gpu_vendor": None,
        "gpu_model": None,
        "actions": [],
    }


def test_unreachable_available_host_displays_blocker() -> None:
    node = _node_payload()
    reason = "blocked: SSH connection timed out"
    node.update(state="available", placement_blocker=reason)
    result = _run_harness(lexical_role="admin", admin_nodes=[node])
    assert {"text": "blocked", "title": reason} in result["badges"]
    assert all(badge["text"] != "available" for badge in result["badges"])


def test_user_role_fetches_fleet_endpoint() -> None:
    """The dashboard template renders `const VIEWER_ROLE` -- Lexical only."""
    result = _run_harness(lexical_role="user")
    # Regression: `window.VIEWER_ROLE` is undefined for top-level `const`, so
    # the old code took the admin branch and fetched /admin/nodes.
    assert result["captured"] == ["/fleet/nodes"]
    assert result["nodeCount"] == "0 nodes"


def test_non_admin_node_id_links_to_read_only_detail() -> None:
    """A non-admin clicking a node id lands on the read-only node detail page
    (node info, provisioning tasks, installation log) -- never the admin
    sign-in page, never a native Basic pop-up."""
    result = _run_harness(lexical_role="user", fleet_nodes=[_node_payload()])
    assert result["nodeIdLink"]["tag"] == "a"
    assert result["nodeIdLink"]["href"].endswith("/dashboard/nodes/gpu01")
    assert result["nodeIdLink"]["text"] == "gpu01"


def test_admin_node_id_links_to_node_detail() -> None:
    result = _run_harness(lexical_role="admin", admin_nodes=[_node_payload()])
    assert result["nodeIdLink"]["tag"] == "a"
    assert result["nodeIdLink"]["href"].endswith("/dashboard/nodes/gpu01")
    assert result["nodeIdLink"]["text"] == "gpu01"


def test_admin_role_fetches_admin_endpoints() -> None:
    result = _run_harness(lexical_role="admin")
    assert result["captured"] == [
        "/admin/nodes",
        "/admin/metrics",
        "/admin/quads/status",
        "/admin/billing",
    ]
    # The render path must actually run: with the real array-shaped response
    # the admin branch iterates nodes (the old {nodes: []} mock threw
    # 'not iterable' and landed in the retry catch without failing these
    # assertions).
    assert result["nodeCount"] == "0 nodes"


def test_window_role_fallback_keeps_legacy_shells_working() -> None:
    result = _run_harness(window_role="user")
    assert result["captured"] == ["/fleet/nodes"]


def test_missing_role_defaults_to_admin_legacy() -> None:
    result = _run_harness()
    assert result["captured"][0] == "/admin/nodes"
    assert result["nodeCount"] == "0 nodes"
