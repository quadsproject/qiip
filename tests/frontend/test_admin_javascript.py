"""Behavioral tests for the shipped admin JavaScript."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NODE_DETAIL_JS = _ROOT / "inference_proxy/static/js/node_detail.js"
_DASHBOARD_JS = _ROOT / "inference_proxy/static/js/dashboard.js"


def _run_node(source: Path, harness: str) -> dict[str, Any]:
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
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.parametrize("managed", [True, False])
def test_node_detail_retry_preserves_managed(managed: bool) -> None:
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const setupSelectionSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);
let captured = null;
const elements = new Map();

function element() {{
  return {{
    addEventListener() {{}}, appendChild() {{}}, remove() {{}},
    setAttribute() {{}}, textContent: "", innerHTML: "", value: "",
    className: "", style: {{}}, dataset: {{}},
    classList: {{ add() {{}}, remove() {{}}, contains() {{ return false; }} }},
  }};
}}

function byId(id) {{
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}}

const sandbox = {{
  console,
  NODE_ID: "gpu01",
  POLL_INTERVAL_MS: 10000,
  document: {{
    getElementById: byId,
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
vm.runInContext(setupSelectionSource, sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {{}};

(async function () {{
  await sandbox.handleAction("retry", "gpu01", {{
    managed: {json.dumps(managed)}, engine: "vllm", model: "org/model",
  }});
  const body = JSON.parse(captured.options.body);
  process.stdout.write(JSON.stringify(body));
}})().catch(function (error) {{
  console.error(error);
  process.exit(1);
}});
"""

    body = _run_node(_NODE_DETAIL_JS, harness)

    assert body == {"hostname": "gpu01", "managed": managed}


def test_failed_node_setup_leaves_retry_identity_to_server() -> None:
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const setupSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);
const elements = new Map();
let captured = null;
function element() {
  return {
    children: [], listeners: {}, style: {}, value: "", textContent: "",
    addEventListener(name, callback) { this.listeners[name] = callback; },
    appendChild(child) { this.children.push(child); return child; },
    remove() {}, setAttribute() {},
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
}
function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}
const sandbox = {
  console, NODE_ID: "gpu01", POLL_INTERVAL_MS: 10000,
  document: {
    getElementById: byId, addEventListener() {}, querySelectorAll() { return []; },
    querySelector() { return element(); }, createElement() { return element(); },
    createTextNode(text) { return { textContent: text }; },
  },
  window: { confirm() { return true; } }, requestAnimationFrame() {},
  setTimeout() { return 0; }, setInterval() { return 0; }, clearInterval() {},
  EventSource: function () {},
  fetch: async function (url, options) {
    captured = { url, options };
    return { ok: true, json: async function () { return {}; } };
  },
};
vm.createContext(sandbox);
vm.runInContext(setupSource, sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {};
sandbox.setupSelection.setPreferredNode({
  engine: "llama_cpp", artifact_id: "a".repeat(64), model: "alias",
});
sandbox.setupSelection.setCatalog({
  models: [], gguf_artifacts: [{
    artifact_id: "a".repeat(64), repo_id: "org/model", model_alias: "alias",
  }],
});
(async function () {
  await sandbox.handleAction("setup", "gpu01", {
    state: "failed", managed: true, engine: "llama_cpp",
    artifact_id: "a".repeat(64),
  });
  process.stdout.write(captured.options.body);
})().catch(function (error) { console.error(error); process.exit(1); });
"""

    assert _run_node(_NODE_DETAIL_JS, harness) == {
        "hostname": "gpu01",
        "managed": True,
    }


def test_task_data_degradation_survives_poll_rewrite() -> None:
    harness = """
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const setupSelectionSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);
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
  requestAnimationFrame(callback) { callback(); },
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
vm.runInContext(setupSelectionSource, sandbox);
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


_NODE_DETAIL_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const setupSelectionSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);
const elements = new Map();
const allElements = [];
const eventSources = [];
const timers = [];
let fakeNow = 0;

class Element {
  constructor(tagName) {
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.className = "";
    this.style = {};
    this.dataset = {};
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.title = "";
    this._textContent = "";
    this.scrollHeight = 100;
    this.scrollTop = 0;
    this.clientHeight = 100;
    allElements.push(this);
  }
  get textContent() { return this._textContent; }
  set textContent(value) {
    this._textContent = String(value);
    if (value === "") this.children = [];
  }
  appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name] || null; }
  remove() {}
  querySelector() { return null; }
  getBoundingClientRect() { return { top: 0, right: 0 }; }
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.listeners = {};
    this.closed = false;
    eventSources.push(this);
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  emit(name, event) { this.listeners[name](event || {}); }
  close() { this.closed = true; }
}

class FakeDate extends Date {
  static now() { return fakeNow; }
}

const sandbox = {
  console,
  NODE_ID: "gpu01",
  POLL_INTERVAL_MS: 10000,
  Date: FakeDate,
  document: {
    getElementById: byId,
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector(selector) {
      if (selector === "#power-state span") return byId("power-badge");
      return new Element("div");
    },
    createElement(tagName) { return new Element(tagName); },
    createTextNode(text) { const node = new Element("text"); node.textContent = text; return node; },
  },
  window: { confirm() { return true; } },
  confirmDialog: async function () { return true; },
  requestAnimationFrame() {},
  setTimeout(callback, delay) {
    const timer = { callback, delay, active: true };
    timers.push(timer);
    return timers.length;
  },
  clearTimeout(id) { if (timers[id - 1]) timers[id - 1].active = false; },
  setInterval() { return 1; },
  clearInterval() {},
  EventSource: FakeEventSource,
  fetch: async function () { return { ok: false, status: 404, json: async function () { return {}; } }; },
};

vm.createContext(sandbox);
vm.runInContext(setupSelectionSource, sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {};

function runNextTimer() {
  const timer = timers.find(function (candidate) { return candidate.active; });
  if (!timer) throw new Error("no active timer");
  timer.active = false;
  timer.callback();
  return timer.delay;
}

function installDetailFetch(tasks, tasksOk, engine, runtime) {
  sandbox.fetch = async function (url) {
    if (url === "/admin/nodes") {
      return {
        ok: true,
        headers: { get() { return null; } },
        json: async function () { return [{
          node_id: "gpu01", state: "provisioning", model: "org/model",
          engine: engine || null, artifact_id: null,
          llamacpp_runtime: runtime || null,
          endpoint: "gpu01:8000", active_connections: 0,
          circuit_breaker_state: "closed", actions: [],
        }]; },
      };
    }
    if (url === "/admin/metrics") {
      return { ok: true, json: async function () { return { per_node: {} }; } };
    }
    if (url === "/admin/provisioning/tasks") {
      return {
        ok: tasksOk !== false,
        json: async function () { return tasks; },
      };
    }
    throw new Error("unexpected URL " + url);
  };
}

__SCENARIO__
"""


def _run_node_detail_scenario(scenario: str) -> dict[str, Any]:
    return _run_node(
        _NODE_DETAIL_JS,
        _NODE_DETAIL_HARNESS.replace("__SCENARIO__", scenario),
    )


def test_log_stream_reconnects_after_transient_drop() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([{
  hostname: "gpu01", current_step: "health_poll", started_at: "2026-07-31T12:00:00Z",
  updated_at: "2026-07-31T12:00:00Z", failed_step: null, error: null,
}], true);
(async function () {
  await sandbox.refreshDetail();
  const first = eventSources[0];
  const entry = { ts: "2026-07-31T12:00:00Z", level: "info", msg: "loading", stream: "stdout" };
  first.emit("message", { data: JSON.stringify(entry) });
  first.emit("error");
  const statusAfterDrop = byId("logs-status").textContent;
  const activeTimersAfterDrop = timers.filter(function (timer) { return timer.active; }).length;
  const delay = activeTimersAfterDrop > 0 ? runNextTimer() : null;
  const second = eventSources[1];
  if (second) second.emit("message", { data: JSON.stringify(entry) });
  process.stdout.write(JSON.stringify({
    statusAfterDrop,
    activeTimersAfterDrop,
    delay,
    sourceCount: eventSources.length,
    done: sandbox.logStreamDone,
    renderedLines: byId("logs-output").children.length,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "statusAfterDrop": "reconnecting",
        "activeTimersAfterDrop": 1,
        "delay": 1000,
        "sourceCount": 2,
        "done": False,
        "renderedLines": 1,
    }


def test_log_stream_stops_after_terminal_task_state() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([{
  hostname: "gpu01", current_step: "health_poll", started_at: "2026-07-31T12:00:00Z",
  updated_at: "2026-07-31T12:00:00Z", failed_step: null, error: null,
}], true);
(async function () {
  await sandbox.refreshDetail();
  const sourceBeforeCompletion = eventSources[0];
  installDetailFetch([{
    hostname: "gpu01", current_step: "failed", started_at: "2026-07-31T12:00:00Z",
    updated_at: "2026-07-31T12:01:00Z", failed_step: "health_poll", error: "failed",
  }], true);
  await sandbox.refreshDetail();
  const closedBeforeStreamEnded = sourceBeforeCompletion.closed;
  sourceBeforeCompletion.emit("error");
  process.stdout.write(JSON.stringify({
    closedBeforeStreamEnded,
    closed: sourceBeforeCompletion.closed,
    sourceCount: eventSources.length,
    done: sandbox.logStreamDone,
    status: byId("logs-status").textContent,
    activeTimers: timers.filter(function (timer) { return timer.active; }).length,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "closedBeforeStreamEnded": False,
        "closed": True,
        "sourceCount": 1,
        "done": True,
        "status": "ended",
        "activeTimers": 0,
    }


def test_completed_task_replays_buffered_logs_once() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([{
  hostname: "gpu01", current_step: "failed", started_at: "2026-07-31T12:00:00Z",
  updated_at: "2026-07-31T12:01:00Z", failed_step: "health_poll", error: "failed",
}], true);
(async function () {
  await sandbox.refreshDetail();
  const source = eventSources[0];
  source.emit("message", { data: JSON.stringify({
    ts: "2026-07-31T12:01:00Z", level: "error", msg: "failed", stream: "stderr",
  }) });
  source.emit("error");
  process.stdout.write(JSON.stringify({
    sourceCount: eventSources.length,
    renderedLines: byId("logs-output").children.length,
    done: sandbox.logStreamDone,
    status: byId("logs-status").textContent,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "sourceCount": 1,
        "renderedLines": 1,
        "done": True,
        "status": "ended",
    }


def test_log_reconnect_stops_when_task_status_stays_unavailable() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([{
  hostname: "gpu01", current_step: "health_poll", started_at: "2026-07-31T12:00:00Z",
  updated_at: "2026-07-31T12:00:00Z", failed_step: null, error: null,
}], true);
(async function () {
  await sandbox.refreshDetail();
  installDetailFetch([], false);
  await sandbox.refreshDetail();
  eventSources[0].emit("error");
  if (timers.some(function (timer) { return timer.active; })) runNextTimer();
  fakeNow = sandbox.LOG_RECONNECT_MAX_ELAPSED_MS + 1;
  if (eventSources[1]) eventSources[1].emit("error");
  process.stdout.write(JSON.stringify({
    sourceCount: eventSources.length,
    done: sandbox.logStreamDone,
    status: byId("logs-status").textContent,
    activeTimers: timers.filter(function (timer) { return timer.active; }).length,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "sourceCount": 2,
        "done": True,
        "status": "status unavailable, reload to retry",
        "activeTimers": 0,
    }


def test_catalog_refreshes_after_download() -> None:
    result = _run_node_detail_scenario(
        r"""
const requested = [];
sandbox.fetch = async function (url) {
  requested.push(url);
  if (url === "/admin/models/downloads") {
    return { ok: true, json: async function () {
      return [{ repo_id: "org/model", status: "complete" }];
    } };
  }
  if (url === "/admin/models/catalog") {
    return { ok: true, json: async function () {
      return { models: [{ repo_id: "org/model" }] };
    } };
  }
  if (url === "/admin/nodes") {
    return {
      ok: true,
      headers: { get() { return null; } },
      json: async function () { return [{
        node_id: "gpu01", state: "available", model: null, endpoint: null,
        active_connections: 0, circuit_breaker_state: "closed",
        actions: ["setup"],
      }]; },
    };
  }
  if (url === "/admin/metrics") {
    return { ok: true, json: async function () { return { per_node: {} }; } };
  }
  if (url === "/admin/provisioning/tasks") {
    return { ok: true, json: async function () { return []; } };
  }
  throw new Error("unexpected URL " + url);
};

(async function () {
  await sandbox.pollDownloadStatuses();
  process.stdout.write(JSON.stringify({
    requested,
    selectorValues: byId("model-select").children.map(function (child) { return child.value; }),
    selectorVisible: byId("model-select").style.display !== "none",
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result["requested"].count("/admin/models/catalog") == 1
    assert result["selectorValues"] == ["org/model"]
    assert result["selectorVisible"] is True


def test_catalog_degradation_is_visible_when_no_models_are_verified() -> None:
    result = _run_node_detail_scenario(
        r"""
sandbox.fetch = async function (url) {
  if (url !== "/admin/models/catalog") throw new Error("unexpected URL " + url);
  return { ok: true, json: async function () {
    return { models: [], incomplete_count: 2, unverifiable_count: 3 };
  } };
};

(async function () {
  await sandbox.fetchCatalog();
  process.stdout.write(JSON.stringify({
    selectorVisible: byId("model-select").style.display !== "none",
    statusVisible: byId("model-status").style.display !== "none",
    status: byId("model-status").textContent,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "selectorVisible": False,
        "statusVisible": True,
        "status": (
            "No verified models. 3 cached models lack manifest metadata and "
            "were hidden; re-download them to migrate the cache. 2 incomplete "
            "cached models were hidden; re-download them."
        ),
    }


def test_recommendations_render_runtime_source_and_exact_artifact_availability() -> (
    None
):
    result = _run_node_detail_scenario(
        r"""
const requested = [];
const model = function (name, runtime, sources) {
  return {
    name, runtime, gguf_sources: sources, category: "chat", score: 90,
    fit_level: "good", estimated_tps: 12, memory_required_gb: 10,
  };
};
sandbox.fetch = async function (url, options) {
  requested.push({ url, method: options && options.method });
  if (url === "/admin/nodes/gpu01/recommendations") {
    return { ok: true, json: async function () { return {
      system: { gpu_name: "GPU", gpu_vram_gb: 80, backend: "CUDA" },
      models: [
        model("no-source", "llama_cpp", []),
        model("exact", "llama_cpp", [{
          repo: "org/exact", provider: "<img src=x onerror=attack()>",
        }]),
        model("case-mismatch", "llama_cpp", [{ repo: "Org/Exact", provider: "safe" }]),
        model("mlx-model", "mlx", []),
        model("future-model", "unknown", []),
        model("org/vllm", "vllm", []),
      ],
    }; } };
  }
  if (url === "/admin/models/catalog") {
    return { ok: true, json: async function () { return {
      models: [],
      gguf_artifacts: [{
        artifact_id: "artifact-exact", repo_id: "org/exact",
        model_alias: "exact", resolved_revision: "abcdef0123456789",
      }],
    }; } };
  }
  if (url === "/admin/models/downloads") {
    return { ok: true, json: async function () { return []; } };
  }
  throw new Error("unexpected URL " + url);
};

(async function () {
  await sandbox.loadRecommendations();
  process.stdout.write(JSON.stringify({
    texts: allElements.map(function (element) { return element.textContent; }),
    downloadButtons: allElements.filter(function (element) {
      return element.tagName === "button" && element.textContent === "Download";
    }).length,
    downloadPosts: requested.filter(function (request) {
      return request.url === "/admin/models/download" && request.method === "POST";
    }).length,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert "llama.cpp" in result["texts"]
    assert "MLX" in result["texts"]
    assert "No GGUF source" in result["texts"]
    assert "Available (1 generation)" in result["texts"]
    assert "Not downloaded" in result["texts"]
    assert "Unsupported" in result["texts"]
    assert "Unknown runtime" in result["texts"]
    assert "<img src=x onerror=attack()> - org/exact" in result["texts"]
    assert result["downloadButtons"] == 1
    assert result["downloadPosts"] == 0


def test_node_detail_renders_registered_engine() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([], true, "llama_cpp");
(async function () {
  await sandbox.refreshDetail();
  process.stdout.write(JSON.stringify({
    texts: allElements.map(function (element) { return element.textContent; }),
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert "llama.cpp" in result["texts"]


def test_node_detail_renders_verified_llamacpp_runtime_snapshot() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([], true, "llama_cpp", {
  requested: { sizing: "auto", fit_target_mib: 512 },
  effective: {
    train_context: 262144, context_per_slot: 12544,
    slot_context_limit: 12544, slots: 1, aggregate_context: 12544,
    cache_type_k: "q8_0", cache_type_v: "q8_0", flash_attn: "on",
    kv_unified: true, gpu_layers: 31, total_layers: 31,
  },
  gpus: [
    { index: 0, total_mib: 14911, used_mib: 14089, free_mib: 822 },
    { index: 1, total_mib: 14911, used_mib: 14207, free_mib: 704 },
  ],
  observed_at: "2026-08-05T20:54:07Z",
});
(async function () {
  await sandbox.refreshDetail();
  process.stdout.write(JSON.stringify({
    panelHidden: byId("llamacpp-runtime-panel").hidden,
    statusHidden: byId("llamacpp-runtime-status").hidden,
    valuesHidden: byId("llamacpp-runtime-values").hidden,
    minimumFree: byId("llamacpp-runtime-min-free").textContent,
    minimumHeadroom: byId("llamacpp-runtime-min-headroom").textContent,
    sizing: byId("llamacpp-runtime-sizing").textContent,
    context: byId("llamacpp-runtime-context").textContent,
    trainContext: byId("llamacpp-runtime-train-context").textContent,
    slots: byId("llamacpp-runtime-slots").textContent,
    aggregate: byId("llamacpp-runtime-aggregate").textContent,
    kv: byId("llamacpp-runtime-kv").textContent,
    reserve: byId("llamacpp-runtime-reserve").textContent,
    gpu: byId("llamacpp-runtime-gpus").children[0].textContent,
    gpuCount: byId("llamacpp-runtime-gpus").children.length,
    observed: byId("llamacpp-runtime-observed").textContent,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "panelHidden": False,
        "statusHidden": True,
        "valuesHidden": False,
        "minimumFree": "704 MiB",
        "minimumHeadroom": "192 MiB above target",
        "sizing": "Automatic",
        "context": "12,544 tokens",
        "trainContext": "262,144 tokens",
        "slots": "1",
        "aggregate": "12,544 tokens",
        "kv": "Q8_0 / Q8_0",
        "reserve": "512 MiB per GPU",
        "gpu": (
            "GPU 0: 14,089 MiB used, 822 MiB free of 14,911 MiB (310 MiB above target)"
        ),
        "gpuCount": 2,
        "observed": result["observed"],
    }
    assert result["observed"].startswith("Post-load snapshot at ")
    assert result["observed"] != "Post-load snapshot at —"


def test_node_detail_explains_missing_llamacpp_runtime_state() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([], true, "llama_cpp", null);
(async function () {
  await sandbox.refreshDetail();
  process.stdout.write(JSON.stringify({
    panelHidden: byId("llamacpp-runtime-panel").hidden,
    statusHidden: byId("llamacpp-runtime-status").hidden,
    valuesHidden: byId("llamacpp-runtime-values").hidden,
    status: byId("llamacpp-runtime-status").textContent,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "panelHidden": False,
        "statusHidden": False,
        "valuesHidden": True,
        "status": (
            "Runtime configuration is unavailable until the next successful "
            "managed llama.cpp setup."
        ),
    }


def test_node_detail_hides_llamacpp_runtime_card_for_vllm() -> None:
    result = _run_node_detail_scenario(
        r"""
installDetailFetch([], true, "vllm", null);
(async function () {
  await sandbox.refreshDetail();
  process.stdout.write(JSON.stringify({
    panelHidden: byId("llamacpp-runtime-panel").hidden,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"panelHidden": True}


def _power_controls_result(status_code: int) -> object:
    return _run_node_detail_scenario(
        f"""
sandbox.fetch = async function () {{
  return {{ ok: false, status: {status_code}, json: async function () {{ return {{}}; }} }};
}};
(async function () {{
  await sandbox.refreshPowerState();
  const controls = byId("power-actions");
  const group = controls.children[0];
  const trigger = group ? group.children[0] : null;
  const menu = group ? group.children[1] : null;
  process.stdout.write(JSON.stringify({{
    hidden: controls.hidden,
    triggerDisabled: trigger ? trigger.disabled : null,
    triggerTitle: trigger ? trigger.title : null,
    menuItemCount: menu ? menu.children.length : 0,
  }}));
}})().catch(function (error) {{ console.error(error); process.exit(1); }});
"""
    )


def test_power_controls_hidden_when_redfish_unconfigured() -> None:
    assert _power_controls_result(503) == {
        "hidden": True,
        "triggerDisabled": None,
        "triggerTitle": None,
        "menuItemCount": 0,
    }


def test_power_controls_disabled_when_state_unknown() -> None:
    assert _power_controls_result(502) == {
        "hidden": False,
        "triggerDisabled": True,
        "triggerTitle": "Power state is temporarily unavailable; controls are disabled.",
        "menuItemCount": 4,
    }


@pytest.mark.parametrize("action", ["ForceOff", "ForceRestart"])
def test_force_power_actions_always_require_confirmation(action: str) -> None:
    result = _run_node_detail_scenario(
        f"""
let requestCount = 0;
sandbox.confirmDialog = async function () {{ return false; }};
sandbox.fetch = async function () {{ requestCount += 1; return {{ ok: true }}; }};
(async function () {{
  await sandbox.handlePowerAction({json.dumps(action)});
  process.stdout.write(JSON.stringify({{ requestCount }}));
}})().catch(function (error) {{ console.error(error); process.exit(1); }});
"""
    )

    assert result == {"requestCount": 0}


_DASHBOARD_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const source = fs.readFileSync(process.argv[1], "utf8");
const configDownloadSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "config_download.js"), "utf8"
);
const setupSelectionSource = fs.readFileSync(
  path.join(path.dirname(process.argv[1]), "setup_selection.js"), "utf8"
);
const elements = new Map();
const allElements = [];

class Element {
  constructor(tagName) {
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this._classes = new Set();
    this._className = "";
    this._textContent = "";
    this.style = {};
    this.dataset = {};
    this.disabled = false;
    this.offsetHeight = 10;
    this.offsetWidth = 10;
    this.classList = {
      add: (...names) => names.forEach((name) => this._classes.add(name)),
      remove: (...names) => names.forEach((name) => this._classes.delete(name)),
      contains: (name) => this._classes.has(name),
    };
    allElements.push(this);
  }
  get className() { return this._className; }
  set className(value) {
    this._className = String(value);
    this._classes = new Set(this._className.split(/\s+/).filter(Boolean));
  }
  get textContent() { return this._textContent; }
  set textContent(value) {
    this._textContent = String(value);
    if (value === "") this.children = [];
  }
  appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name] || null; }
  getBoundingClientRect() { return { top: 20, right: 20 }; }
  async click() {
    if (this.disabled || !this.listeners.click) return;
    return this.listeners.click({ stopPropagation() {}, preventDefault() {} });
  }
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}

const sandbox = {
  console,
  POLL_INTERVAL_MS: 10000,
  document: {
    getElementById: byId,
    addEventListener() {},
    querySelectorAll(selector) {
      if (selector === ".action-menu.open") {
        return allElements.filter(function (element) {
          return element.classList.contains("action-menu") && element.classList.contains("open");
        });
      }
      return [];
    },
    createElement(tagName) { return new Element(tagName); },
    createTextNode(text) { const node = new Element("text"); node.textContent = text; return node; },
    createDocumentFragment() { const f = new Element("fragment"); return f; },
    body: { appendChild(child) { return child; }, removeChild(child) {} },
  },
  window: { confirm() { return true; }, location: { origin: "http://localhost:8080" } },
  confirmDialog: async function () { return true; },
  URL: { createObjectURL() { return "blob:test"; }, revokeObjectURL() {} },
  Blob: function () {},
  requestAnimationFrame(callback) { callback(); },
  setTimeout() { return 1; },
  setInterval() { return 1; },
  fetch: async function () { throw new Error("fetch stub not installed"); },
};

vm.createContext(sandbox);
vm.runInContext(configDownloadSource, sandbox);
vm.runInContext(setupSelectionSource, sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {};

function response(data, headers) {
  return {
    ok: true,
    headers: { get(name) { return headers && headers[name] || null; } },
    json: async function () { return data; },
  };
}

function node(id, state, actions, engine) {
  return {
    node_id: id,
    state,
    actions: actions || [],
    managed: true,
    gpu_vendor: "NVIDIA",
    gpu_model: "GPU",
    engine: engine || null,
    model: "org/model",
    failed_step: state === "failed" ? "health_poll" : null,
    error: state === "failed" ? "backend failed" : null,
  };
}

__SCENARIO__
"""


def _run_dashboard_scenario(scenario: str) -> object:
    return _run_node(
        _DASHBOARD_JS,
        _DASHBOARD_HARNESS.replace("__SCENARIO__", scenario),
    )


def test_teardown_button_stays_disabled_across_refresh() -> None:
    result = _run_dashboard_scenario(
        r"""
let deleteCount = 0;
const deleteResolvers = [];
sandbox.fetch = async function (url, options) {
  if (url === "/admin/nodes" && !options) return response([node("gpu01", "healthy", ["teardown"])]);
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  if (url === "/admin/nodes/gpu01" && options && options.method === "DELETE") {
    deleteCount += 1;
    return new Promise(function (resolve) { deleteResolvers.push(resolve); });
  }
  throw new Error("unexpected request " + url);
};

(async function () {
  await sandbox.refreshDashboard();
  const first = allElements.filter(function (el) { return el.textContent === "Teardown"; }).pop();
  const action = first.click();
  await Promise.resolve();
  await sandbox.refreshDashboard();
  const replacement = allElements.filter(function (el) { return el.textContent === "Teardown"; }).pop();
  const replacementDisabled = replacement.disabled;
  const replacementAction = replacement.click();
  deleteResolvers.forEach(function (resolve) { resolve(response({})); });
  await Promise.all([action, replacementAction]);
  process.stdout.write(JSON.stringify({ replacementDisabled, deleteCount }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"replacementDisabled": True, "deleteCount": 1}


def test_dashboard_renders_registered_and_unknown_engines() -> None:
    result = _run_dashboard_scenario(
        r"""
sandbox.fetch = async function (url) {
  if (url === "/admin/nodes") return response([
    node("vllm-node", "healthy", [], "vllm"),
    node("llama-node", "healthy", [], "llama_cpp"),
    node("quads-only", "available", [], null),
  ]);
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  throw new Error("unexpected request " + url);
};

(async function () {
  await sandbox.refreshDashboard();
  const rows = byId("node-table-body").children;
  process.stdout.write(JSON.stringify({
    engines: rows.map(function (row) { return row.children[3].textContent; }),
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"engines": ["vLLM", "llama.cpp", "—"]}


def test_dashboard_quick_setup_sends_exact_selected_artifact() -> None:
    result = _run_dashboard_scenario(
        r"""
let captured = null;
vm.runInContext(`
  setupSelection.setCatalog({
    models: [{ repo_id: "org/vllm" }],
    gguf_artifacts: [{
      artifact_id: "a".repeat(64), repo_id: "org/model-GGUF",
      model_alias: "model-q4", resolved_revision: "b".repeat(40),
      entrypoint: "model-Q4_K_M.gguf",
    }],
  });
  setupSelection.selectEngine("llama_cpp");
  document.getElementById("artifact-select").value = "a".repeat(64);
  document.getElementById("artifact-select").listeners.change();
`, sandbox);
sandbox.fetch = async function (url, options) {
  captured = { url, options };
  return response({});
};
(async function () {
  await sandbox.handleAction(
    "setup", "gpu01", node("gpu01", "available", ["setup"], null)
  );
  process.stdout.write(JSON.stringify({
    url: captured.url,
    body: JSON.parse(captured.options.body),
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "url": "/admin/nodes/setup",
        "body": {
            "hostname": "gpu01",
            "managed": True,
            "engine": "llama_cpp",
            "artifact_id": "a" * 64,
        },
    }


def test_dashboard_preserves_expanded_error_state_across_refresh() -> None:
    result = _run_dashboard_scenario(
        r"""
sandbox.fetch = async function (url) {
  if (url === "/admin/nodes") return response([node("gpu01", "failed", ["retry"])]);
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  throw new Error("unexpected request " + url);
};

(async function () {
  await sandbox.refreshDashboard();
  const badge = allElements.filter(function (el) {
    return el.textContent === "failed" && el.attributes.role === "button";
  }).pop();
  await badge.click();
  await sandbox.refreshDashboard();
  const subrow = allElements.filter(function (el) { return el.className === "error-subrow"; }).pop();
  process.stdout.write(JSON.stringify({ display: subrow.style.display }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"display": "table-row"}


def test_dashboard_preserves_open_action_menu_across_refresh() -> None:
    result = _run_dashboard_scenario(
        r"""
sandbox.fetch = async function (url) {
  if (url === "/admin/nodes") return response([node("gpu01", "healthy", ["teardown", "force_teardown"])]);
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  throw new Error("unexpected request " + url);
};

(async function () {
  await sandbox.refreshDashboard();
  const caret = allElements.filter(function (el) { return el.textContent === "▾"; }).pop();
  await caret.click();
  await sandbox.refreshDashboard();
  const menu = allElements.filter(function (el) { return el.classList.contains("action-menu"); }).pop();
  process.stdout.write(JSON.stringify({
    open: menu.classList.contains("open"),
    top: menu.style.top,
    left: menu.style.left,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"open": True, "top": "10px", "left": "10px"}


def test_dashboard_skips_overlapping_poll() -> None:
    result = _run_dashboard_scenario(
        r"""
let nodeRequests = 0;
const nodeResolvers = [];
sandbox.fetch = async function (url) {
  if (url === "/admin/nodes") {
    nodeRequests += 1;
    return new Promise(function (resolve) { nodeResolvers.push(resolve); });
  }
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  throw new Error("unexpected request " + url);
};

(async function () {
  const first = sandbox.refreshDashboard();
  const second = sandbox.refreshDashboard();
  nodeResolvers.forEach(function (resolve) { resolve(response([])); });
  const [firstResult, secondResult] = await Promise.all([first, second]);
  process.stdout.write(JSON.stringify({ nodeRequests, firstResult, secondResult }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {
        "nodeRequests": 1,
        "firstResult": True,
        "secondResult": False,
    }


def test_stale_poll_response_does_not_overwrite_newer() -> None:
    result = _run_dashboard_scenario(
        r"""
let nodeRequests = 0;
let resolveOldNodes;
sandbox.fetch = async function (url) {
  if (url === "/admin/nodes") {
    nodeRequests += 1;
    if (nodeRequests === 1) {
      return new Promise(function (resolve) { resolveOldNodes = resolve; });
    }
    return response([node("new-node", "healthy", [])]);
  }
  if (url === "/admin/metrics") return response({ per_node: {} });
  if (url === "/admin/quads/status") return response({ status: "connected" });
  throw new Error("unexpected request " + url);
};

(async function () {
  const oldRequest = sandbox.refreshDashboard();
  vm.runInContext("dashboardPollInFlight = false", sandbox);
  await sandbox.refreshDashboard();
  resolveOldNodes(response([node("old-node", "failed", [])]));
  await oldRequest;
  const links = allElements.filter(function (el) { return el.tagName === "a"; });
  process.stdout.write(JSON.stringify({
    nodeRequests,
    renderedTitles: links.map(function (link) { return link.title; }),
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    )

    assert result == {"nodeRequests": 2, "renderedTitles": ["new-node"]}
