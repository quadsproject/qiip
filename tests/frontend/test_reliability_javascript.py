"""Exercise filters, evidence drill-down, safe text and failed refresh behavior."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from inference_proxy.provisioning.log_store import AttemptLogStore
from inference_proxy.provisioning.reliability import build_report
from tests.provisioning.test_reliability import fleet as fleet


@pytest.mark.parametrize(
    ("timezone", "since", "until"),
    [
        ("UTC", "2026-01-21T12:30:00.000Z", "2026-09-22T12:30:45.000Z"),
        ("Europe/Prague", "2026-01-21T11:30:00.000Z", "2026-09-22T10:30:45.000Z"),
        ("America/New_York", "2026-01-21T17:30:00.000Z", "2026-09-22T16:30:45.000Z"),
    ],
)
def test_reliability_ui(
    fleet: AttemptLogStore, timezone: str, since: str, until: str
) -> None:
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
let boot, status = 200, body = JSON.parse(process.argv[2]), invalidJSON = false;
const context = {console, Map, URLSearchParams,
  document: {getElementById: get, createElement: element, addEventListener: (_n, fn) => boot = fn},
  fetch: async url => {requests.push(url); return {ok: status === 200, status,
    json: async () => {if (invalidJSON) throw new Error('Invalid JSON'); return body;}};}
};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
async function settle() { for (let n=0; n<20; n++) await Promise.resolve(); }
function contents(node) { return [node.textContent, ...node.children.flatMap(contents)]; }
(async () => {
  get('reliability-group').value = 'signature'; boot(); await settle();
  const initial = {visible: !get('reliability-report').hidden,
    text: contents(get('reliability-groups')), metrics: contents(get('reliability-metrics')),
    link: get('reliability-groups').children[0].children[1].children.at(-1).children[2].href};
  get('reliability-since').value = '2026-01-21T12:30';
  get('reliability-until').value = '2026-09-22T12:30:45';
  get('reliability-hostnames').value = ' host1, host2 ';
  get('reliability-group').value = 'bundle_version';
  get('reliability-filters').listeners.submit({preventDefault(){}}); await settle();
  const download = get('reliability-download').href;
  const errors = [];
  for (const failure of [
    [401, {detail: 'Admin authentication required'}],
    [422, {detail: 'since must be earlier than until'}],
    [422, {detail: [{loc: ['query', 'since'], msg: 'Input should be a valid datetime'}]}],
    [503, {}], [502, null]
  ]) {
    [status, body] = failure; invalidJSON = body === null;
    get('reliability-filters').listeners.submit({preventDefault(){}}); await settle();
    errors.push({message: get('reliability-status').textContent,
      hidden: get('reliability-report').hidden && get('reliability-download').hidden});
  }
  status = 200; body = JSON.parse(process.argv[3]); invalidJSON = false;
  get('reliability-filters').listeners.submit({preventDefault(){}}); await settle();
  const empty = {visible: !get('reliability-report').hidden,
    metrics: contents(get('reliability-metrics'))};
  console.log(JSON.stringify({initial, requests, download, errors, empty}));
})().catch(e => { console.error(e); process.exit(1); });
"""
    result = subprocess.run(
        [
            "node",
            "-e",
            harness,
            str(script),
            json.dumps(report),
            json.dumps(build_report(fleet, hostnames=["absent"])),
        ],
        env={**os.environ, "TZ": timezone},
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
    query = parse_qs(urlsplit(output["requests"][1]).query)
    assert query["since"] == [since]
    assert query["until"] == [until]
    assert "hostname=host1&hostname=host2" in output["requests"][1]
    assert "group_by=bundle_version&download=true" in output["download"]
    assert output["download"] == output["requests"][1] + "&download=true"
    assert all(error["hidden"] for error in output["errors"])
    assert "Admin authentication required" in output["errors"][0]["message"]
    assert "since must be earlier than until" in output["errors"][1]["message"]
    assert "Input should be a valid datetime" in output["errors"][2]["message"]
    assert "503" in output["errors"][3]["message"]
    assert "502" in output["errors"][4]["message"]
    assert output["empty"]["visible"]
    assert output["empty"]["metrics"].count("Not measured · 0/0") == 4
    assert "Not measured · 0/0 successes timed" in output["empty"]["metrics"]
