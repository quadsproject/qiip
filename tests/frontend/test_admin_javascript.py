"""Behavioral tests for the shipped admin JavaScript."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NODE_DETAIL_JS = _ROOT / "inference_proxy/static/js/node_detail.js"
_DASHBOARD_JS = _ROOT / "inference_proxy/static/js/dashboard.js"


def _run_node(source: Path, harness: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "Node.js is required for admin JavaScript regressions; "
            "CI must install it explicitly"
        )

    result = subprocess.run(
        [node, "-e", harness, str(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("managed", [True, False])
def test_node_detail_retry_preserves_managed(managed: bool) -> None:
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
let captured = null;

function element() {{
  return {{
    addEventListener() {{}}, appendChild() {{}}, remove() {{}},
    setAttribute() {{}}, textContent: "", innerHTML: "", value: "",
    className: "", style: {{}}, dataset: {{}},
    classList: {{ add() {{}}, remove() {{}}, contains() {{ return false; }} }},
  }};
}}

const sandbox = {{
  console,
  NODE_ID: "gpu01",
  POLL_INTERVAL_MS: 10000,
  document: {{
    getElementById() {{ return element(); }},
    addEventListener() {{}},
    querySelectorAll() {{ return []; }},
    querySelector() {{ return element(); }},
    createElement() {{ return element(); }},
    createTextNode(text) {{ return {{ textContent: text }}; }},
  }},
  window: {{ confirm() {{ return true; }} }},
  requestAnimationFrame() {{}},
  setTimeout() {{ return 0; }},
  setInterval() {{ return 0; }},
  clearInterval() {{}},
  EventSource: function () {{}},
  fetch: async function (url, options) {{
    captured = {{ url, options }};
    return {{ ok: true, json: async function () {{ return {{}}; }} }};
  }},
}};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {{}};

(async function () {{
  await sandbox.handleAction("retry", "gpu01", {{ managed: {json.dumps(managed)} }});
  const body = JSON.parse(captured.options.body);
  process.stdout.write(JSON.stringify(body));
}})().catch(function (error) {{
  console.error(error);
  process.exit(1);
}});
"""

    body = _run_node(_NODE_DETAIL_JS, harness)

    assert body == {"hostname": "gpu01", "managed": managed}


def test_dashboard_renders_task_data_degradation() -> None:
    harness = """
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const elements = new Map();

function element() {
  return {
    addEventListener() {}, appendChild() {}, remove() {},
    setAttribute() {}, textContent: "", innerHTML: "", value: "",
    className: "", colSpan: 0, style: {}, dataset: {},
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}

const sandbox = {
  console,
  POLL_INTERVAL_MS: 10000,
  document: {
    getElementById: byId,
    addEventListener() {},
    querySelectorAll() { return []; },
    createElement() { return element(); },
    createTextNode(text) { return { textContent: text }; },
  },
  window: { confirm() { return true; } },
  requestAnimationFrame() {},
  setTimeout() { return 0; },
  setInterval() { return 0; },
  fetch: async function (url) {
    if (url === "/admin/nodes") {
      return {
        ok: true,
        headers: { get(name) {
          return name === "X-Inference-Proxy-Data-Degraded"
            ? "provisioning-tasks" : null;
        } },
        json: async function () { return []; },
      };
    }
    if (url === "/admin/metrics") {
      return { ok: true, json: async function () { return { per_node: {} }; } };
    }
    return { ok: false, json: async function () { return {}; } };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  await sandbox.refreshDashboard();
  const warning = byId("poll-warning");
  process.stdout.write(JSON.stringify({
    text: warning.textContent,
    className: warning.className,
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""

    warning = _run_node(_DASHBOARD_JS, harness)

    assert warning == {
        "text": (
            "Provisioning task details are unavailable; "
            "failed step and error data may be incomplete."
        ),
        "className": "poll-warning",
    }
