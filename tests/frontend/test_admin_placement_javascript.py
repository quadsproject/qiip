"""Behavioral tests for the admin placement card's view logic."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[2] / "inference_proxy/static/js/admin_placement.js"
)
_HARNESS = """
const fs = require("fs");
const vm = require("vm");
const context = { console };
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
const data = JSON.parse(process.argv[2]);
console.log(JSON.stringify({
  summary: context.placementSummary(data),
  profiles: context.placementProfileRows(data),
  issues: context.placementIssueRows(data),
}));
"""


def _view(data: dict[str, Any]) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for admin JavaScript regressions")
    result = subprocess.run(
        [node, "-e", _HARNESS, str(_SOURCE), json.dumps(data)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


def test_unavailable_placement_shows_the_reason() -> None:
    view = _view({"available": False, "error": "needs QUADS"})

    assert view == {"summary": "needs QUADS", "profiles": [], "issues": []}


def test_profiles_and_everything_needing_attention_are_listed() -> None:
    view = _view(
        {
            "available": True,
            "enabled": True,
            "last_run_at": "2026-09-19T12:00:00Z",
            "error": "",
            "profiles": [
                {
                    "display_name": "Qwen3.8-27B",
                    "model": "unsloth/Qwen3.8-27B-GGUF",
                    "weight": 60,
                    "serving": 3,
                    "pending": 1,
                    "failed": 0,
                    "target": 5,
                    "artifacts_present": True,
                    "qualified_gpus": ["l4", "a30"],
                },
                {
                    "display_name": "Muse",
                    "model": "unsloth/Muse-Glimmer-30B-GGUF",
                    "weight": 20,
                    "serving": 0,
                    "pending": 0,
                    "failed": 0,
                    "target": 1,
                    "artifacts_present": False,
                    "qualified_gpus": [],
                },
            ],
            "usage": [
                {
                    "model": "unsloth/Qwen3.8-27B-GGUF",
                    "recorded_requests": 12,
                    "total_tokens": 3400,
                }
            ],
            "missing_artifacts": [
                {
                    "profile_id": "muse-glimmer-30b-24g",
                    "repo_id": "z-lab/x",
                    "revision": "880882627431093d99d3b2368efb4a6fcf12d4cb",
                    "filename": "draft.gguf",
                }
            ],
            "claims": [
                {"state": "active", "hostname": "l4-00", "profile_id": "p"},
                {
                    "state": "exhausted",
                    "hostname": "l4-01",
                    "profile_id": "p",
                    "attempts": 3,
                    "last_error": "<b>boom</b>",
                },
                {
                    "state": "provisioning",
                    "hostname": "l4-02",
                    "profile_id": "p",
                    "attempts": 1,
                    "last_error": "",
                },
            ],
            "unreadable_claims": ["l4-03"],
            "skipped_hosts": [{"hostname": "l4-multi", "reason": "4 GPUs"}],
        }
    )

    assert view["summary"] == "Enabled · last pass 2026-09-19T12:00:00Z"
    assert view["profiles"] == [
        [
            "Qwen3.8-27B",
            "60",
            "3 serving, 1 pending, 0 failed / target 5",
            "present",
            "L4, A30",
            "12",
            "3400",
        ],
        [
            "Muse",
            "20",
            "0 serving, 0 pending, 0 failed / target 1",
            "missing",
            "none yet",
            "0",
            "0",
        ],
    ]
    assert [row[:2] for row in view["issues"]] == [
        ["missing file", "muse-glimmer-30b-24g"],
        ["failed placement", "l4-01"],
        ["in progress", "l4-02"],
        ["unreadable claim", "l4-03"],
        ["skipped host", "l4-multi"],
    ]
    # Error text is data: it is carried verbatim and rendered with textContent.
    assert "<b>boom</b>" in view["issues"][1][2]


def test_the_card_never_builds_markup_from_api_text() -> None:
    source = _SOURCE.read_text(encoding="utf-8")

    assert "innerHTML" not in source
    assert "insertAdjacentHTML" not in source


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_resume_action_requires_confirmation_and_preserves_errors(outcome: str) -> None:
    node = shutil.which("node")
    assert node is not None
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const elements = {};
function element() {
  return {children: [], textContent: '', disabled: false,
    appendChild(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    setAttribute(name, value) { this[name] = value; },
    addEventListener(name, handler) { this[name] = handler; }};
}
context.document = {createElement: element,
  getElementById(id) { return elements[id] ||= element(); }};
const outcome = process.argv[2];
const requests = [];
let confirmations = 0, refreshed = false;
context.confirmDialog = async () => { confirmations++; return outcome !== 'cancel'; };
context.fetch = async (url, options) => {
  requests.push([url, options.method]);
  return {ok: outcome === 'success', status: 409,
    json: async () => ({detail: 'Host lifecycle operation in progress'})};
};
context.refreshPlacement = async () => { refreshed = true; context.fillSuspendedHosts([]); };
(async () => {
  context.fillSuspendedHosts(['host <unsafe>']);
  const row = elements['placement-suspension-body'].children[0];
  const button = row.children[1].children[0];
  await button.click();
  console.log(JSON.stringify({requests, confirmations, refreshed,
    name: row.children[0].textContent, disabled: button.disabled,
    status: elements['placement-resume-status']?.textContent || ''}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", harness, str(_SOURCE), outcome],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["confirmations"] == 1
    assert data["name"] == "host <unsafe>"
    assert data["requests"] == (
        []
        if outcome == "cancel"
        else [["/admin/placement/suspensions/host%20%3Cunsafe%3E", "DELETE"]]
    )
    assert data["refreshed"] is (outcome == "success")
    if outcome == "failure":
        assert data["disabled"] is False
        assert "Host lifecycle operation in progress" in data["status"]
