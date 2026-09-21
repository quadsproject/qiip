"""Execute the shipped attempt-history UI against controlled API responses."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_history_search_download_and_retrieval_contract() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "inference_proxy/static/js/provisioning_history.js"
    )
    harness = r"""
const fs = require("fs"), vm = require("vm");
const elements = new Map(), requests = [];
function element(id) {
  let text = "";
  return {
    id, value: "", hidden: false, disabled: false, dataset: {}, children: [], listeners: {},
    set textContent(v) { text = v; this.children = []; if (id === "attempt-select") this.value = ""; },
    get textContent() { return text; },
    get options() { return this.children; },
    appendChild(child) { this.children.push(child); if (id === "attempt-select" && !this.value) this.value = child.value; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
  };
}
let boot;
const attempt = { attempt_id: "abc", status: "failed", operation: "provision", engine: "vllm",
  started_at: "2026-09-16T12:00:00Z", model: "org/model", stage: "driver", bundle_version: "sha256:abc",
  sources: {"setup.stderr": "collected"}, issues: ["source unavailable"], dropped_records: 2,
  failure_summary: "driver rejected",
  failure: {failed_at: "2026-09-16T12:01:00Z", duration_seconds: 60, exit_code: 7,
    command: {stage: "setup", phase_id: "phase-1"}},
  diagnostics: {sources: {gpu: {status: "timed_out", deferred: true, reason: "<untrusted source>"},
    kernel_gpu_oom: {status: "collected", collected_at: "2026-09-16T12:01:02Z"}}} };
const context = {
  NODE_ID: "host1", console,
  document: {
    addEventListener(_name, callback) { boot = callback; },
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement(tag) { return element(tag); },
  },
  async fetch(url, options) {
    requests.push({url, options});
    let data = url.endsWith("/collect") ? attempt : url.includes("/logs?") ? {
      attempt, records: [{seq: 4, ts: attempt.started_at, stage: "driver", source: "setup.stderr", level: "error", msg: "<img src=x onerror=alert(1)>"}],
      next_offset: 5, has_more: false,
    } : { attempts: [attempt], total: 1, evicted_attempts: 3 };
    return {ok: true, json: async () => data};
  },
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
async function settle() { for (let n=0;n<15;n++) await Promise.resolve(); }
(async () => {
  boot(); await settle();
  const get = id => elements.get(id);
  const initial = { summary: get("attempt-summary").textContent, issues: get("attempt-issues").textContent,
    details: get("attempt-failure-details").textContent,
    diagnostics: get("attempt-diagnostic-sources").children.map(c => c.textContent),
    literal: get("attempt-output").children[0].textContent, download: get("attempt-download").href };
  get("attempt-query").value = "failure & detail?";
  get("attempt-search").listeners.submit({preventDefault(){}}); await settle();
  await get("attempt-collect").listeners.click(); await settle();
  console.log(JSON.stringify({initial, requests, enabled: !get("attempt-collect").disabled}));
})().catch(e => {console.error(e); process.exit(1)});
"""
    result = subprocess.run(
        ["node", "-e", harness, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["initial"]["summary"].startswith("driver rejected\n")
    assert "exit 7" in output["initial"]["details"]
    assert "phase-1" in output["initial"]["details"]
    assert (
        "gpu: timed out · <untrusted source> · retry available"
        in output["initial"]["diagnostics"]
    )
    assert "2 gateway records evicted" in output["initial"]["issues"]
    assert "3 older attempt manifests evicted" in output["initial"]["issues"]
    assert "<img src=x onerror=alert(1)>" in output["initial"]["literal"]
    assert output["initial"]["download"].endswith("/abc/bundle")
    assert any("q=failure%20%26%20detail%3F" in r["url"] for r in output["requests"])
    post = next(r for r in output["requests"] if r["url"].endswith("/collect"))
    assert post["options"] == {
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": "{}",
    }
    assert output["enabled"]
