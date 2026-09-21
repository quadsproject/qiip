"""Exercise filters, evidence drill-down, safe text and failed refresh behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.reliability import build_report
from tests.provisioning.test_reliability import fleet as fleet


def test_reliability_ui(fleet: AttemptLogStore) -> None:
    fleet.update(
        "a1", failure_summary="<img src=x onerror=alert(1)> CUDA out of memory"
    )
    report = build_report(fleet)
    script = (
        Path(__file__).resolve().parents[2] / "inference_proxy/static/js/reliability.js"
    )
    harness = r"""
const fs = require('fs'), vm = require('vm');
const elements = new Map(), requests = [];
function element(id) {
  let text = '';
  return {id, hidden: false, value: '', children: [], listeners: {},
    set innerHTML(value) { throw new Error('Unsafe HTML write'); },
    set textContent(value) { text = value; this.children = []; },
    get textContent() { return text; },
    appendChild(child) { this.children.push(child); },
    addEventListener(name, fn) { this.listeners[name] = fn; }};
}
const get = id => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); };
let boot, fail = false;
const context = {console, Map, URLSearchParams,
  document: {getElementById: get, createElement: element, addEventListener: (_n, fn) => boot = fn},
  fetch: async url => {requests.push(url); return {ok: !fail, status: 503, json: async () => JSON.parse(process.argv[2])};}
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
async function settle() { for (let n=0; n<20; n++) await Promise.resolve(); }
function contents(node) { return [node.textContent, ...node.children.flatMap(contents)]; }
(async () => {
  get('reliability-group').value = 'signature'; boot(); await settle();
  const initial = {visible: !get('reliability-report').hidden,
    text: contents(get('reliability-groups')), metrics: contents(get('reliability-metrics')),
    link: get('reliability-groups').children[0].children[1].children.at(-1).children[2].href};
  get('reliability-since').value = '2026-09-21T12:30';
  get('reliability-until').value = '2026-09-22T12:30';
  get('reliability-hostnames').value = ' host1, host2 ';
  get('reliability-group').value = 'bundle_version';
  get('reliability-filters').listeners.submit({preventDefault(){}}); await settle();
  const download = get('reliability-download').href;
  fail = true; get('reliability-filters').listeners.submit({preventDefault(){}}); await settle();
  console.log(JSON.stringify({initial, requests, download, error: get('reliability-status').textContent,
    hidden: get('reliability-report').hidden && get('reliability-download').hidden}));
})().catch(e => { console.error(e); process.exit(1); });
"""
    result = subprocess.run(
        ["node", "-e", harness, str(script), json.dumps(report)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    output = json.loads(result.stdout)
    assert output["initial"]["visible"]
    assert any(
        "<img src=x onerror=alert(1)>" in line for line in output["initial"]["text"]
    )
    assert "20.0% · 1/5" in output["initial"]["metrics"]
    assert output["initial"]["link"].endswith("/bundle")
    assert "since=2026-09-21T12%3A30%3A00Z" in output["requests"][1]
    assert "hostname=host1&hostname=host2" in output["requests"][1]
    assert "group_by=bundle_version&download=true" in output["download"]
    assert output["hidden"]
    assert "503" in output["error"]
