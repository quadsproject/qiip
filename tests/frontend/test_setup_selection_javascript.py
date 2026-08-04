"""Behavioral tests for the shared admin setup-selection controller."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SETUP_SELECTION_JS = _ROOT / "inference_proxy/static/js/setup_selection.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const elements = new Map();

class Element {
  constructor(tagName) {
    this.tagName = tagName || "div";
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this._textContent = "";
  }
  get textContent() { return this._textContent; }
  set textContent(value) {
    this._textContent = String(value);
    if (value === "") {
      this.children = [];
      if (this.tagName === "select") this.value = "";
    }
  }
  set innerHTML(_value) { throw new Error("HTML sink used for setup selection"); }
  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    if (this.tagName === "select" && !this.value && !child.disabled) {
      this.value = child.value;
    }
    return child;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
}

function byId(id) {
  if (!elements.has(id)) {
    const tag = id.indexOf("select") !== -1 ? "select" : "span";
    elements.set(id, new Element(tag));
  }
  return elements.get(id);
}

const sandbox = {
  console,
  document: {
    getElementById: byId,
    createElement(tagName) { return new Element(tagName); },
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);
const controller = sandbox.createSetupSelectionController();

__SCENARIO__
"""


def _run_scenario(scenario: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for setup-selection regressions")
    result = subprocess.run(
        [
            node,
            "-e",
            _HARNESS.replace("__SCENARIO__", scenario),
            str(_SETUP_SELECTION_JS),
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


def test_catalog_selectors_send_mutually_exclusive_engine_payloads() -> None:
    result = _run_scenario(
        r"""
controller.setCatalog({
  models: [{ repo_id: "org/vllm-model" }],
  gguf_artifacts: [{
    artifact_id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    repo_id: "org/model-GGUF", resolved_revision: "b".repeat(40),
    model_alias: "model-q4",
  }],
});
const vllmBody = controller.buildBody({ hostname: "gpu01" });
controller.selectEngine("llama_cpp");
const llamaBody = controller.buildBody({ hostname: "gpu01" });
process.stdout.write(JSON.stringify({
  vllmBody,
  llamaBody,
  engineOptions: byId("setup-engine-select").children.map(function (child) {
    return { value: child.value, label: child.textContent };
  }),
  modelValues: byId("model-select").children.map(function (child) { return child.value; }),
  artifactValues: byId("artifact-select").children.map(function (child) { return child.value; }),
}));
"""
    )

    assert result == {
        "vllmBody": {
            "hostname": "gpu01",
            "engine": "vllm",
            "model": "org/vllm-model",
        },
        "llamaBody": {
            "hostname": "gpu01",
            "engine": "llama_cpp",
            "artifact_id": "a" * 64,
        },
        "engineOptions": [
            {"value": "vllm", "label": "vLLM"},
            {"value": "llama_cpp", "label": "llama.cpp"},
        ],
        "modelValues": ["org/vllm-model"],
        "artifactValues": ["a" * 64],
    }


@pytest.mark.parametrize("preferred_first", [True, False])
def test_persisted_llamacpp_artifact_wins_regardless_of_fetch_order(
    preferred_first: bool,
) -> None:
    order = (
        "controller.setPreferredNode(node); controller.setCatalog(catalog);"
        if preferred_first
        else "controller.setCatalog(catalog); controller.setPreferredNode(node);"
    )
    result = _run_scenario(
        f"""
const artifactId = "c".repeat(64);
const node = {{ engine: "llama_cpp", artifact_id: artifactId, model: "alias" }};
const catalog = {{ models: [{{ repo_id: "org/vllm" }}], gguf_artifacts: [{{
  artifact_id: artifactId, repo_id: "org/model-GGUF",
  resolved_revision: "d".repeat(40), model_alias: "alias",
}}] }};
{order}
process.stdout.write(JSON.stringify({{
  selection: controller.getSelection(),
  engine: byId("setup-engine-select").value,
}}));
"""
    )

    assert result == {
        "selection": {"engine": "llama_cpp", "artifact_id": "c" * 64},
        "engine": "llama_cpp",
    }


def test_missing_persisted_artifact_never_falls_back_to_vllm() -> None:
    result = _run_scenario(
        r"""
controller.setPreferredNode({
  engine: "llama_cpp", artifact_id: "e".repeat(64), model: "old-alias",
});
controller.setCatalog({
  models: [{ repo_id: "org/vllm-model" }],
  gguf_artifacts: [{
    artifact_id: "f".repeat(64), repo_id: "org/other-GGUF",
    resolved_revision: "a".repeat(40), model_alias: "other",
  }],
});
process.stdout.write(JSON.stringify({
  engine: byId("setup-engine-select").value,
  valid: controller.isValid(),
  selection: controller.getSelection(),
  status: byId("model-status").textContent,
}));
"""
    )

    assert result["engine"] == "llama_cpp"
    assert result["valid"] is False
    assert result["selection"] is None
    assert "no longer available" in result["status"]


@pytest.mark.parametrize(
    ("engine", "select_id", "selection_key", "initial_value", "chosen_value"),
    [
        ("vllm", "model-select", "model", "org/old", "org/new"),
        ("llama_cpp", "artifact-select", "artifact_id", "a" * 64, "b" * 64),
    ],
)
def test_catalog_refresh_preserves_operator_selection(
    engine: str,
    select_id: str,
    selection_key: str,
    initial_value: str,
    chosen_value: str,
) -> None:
    result = _run_scenario(
        f"""
const catalog = {{
  models: [{{ repo_id: "org/old" }}, {{ repo_id: "org/new" }}],
  gguf_artifacts: [
    {{ artifact_id: "a".repeat(64), repo_id: "org/a", model_alias: "a" }},
    {{ artifact_id: "b".repeat(64), repo_id: "org/b", model_alias: "b" }},
  ],
}};
controller.setPreferredNode({{
  engine: {json.dumps(engine)},
  model: {json.dumps(initial_value if engine == "vllm" else "")},
  artifact_id: {json.dumps(initial_value if engine == "llama_cpp" else "")},
}});
controller.setCatalog(catalog);
byId({json.dumps(select_id)}).value = {json.dumps(chosen_value)};
byId({json.dumps(select_id)}).listeners.change();
controller.setCatalog(catalog);
process.stdout.write(JSON.stringify({{
  selected: byId({json.dumps(select_id)}).value,
  selection: controller.getSelection(),
}}));
"""
    )

    assert result["selected"] == chosen_value
    assert result["selection"] == {"engine": engine, selection_key: chosen_value}


def test_artifact_labels_render_untrusted_catalog_values_as_text() -> None:
    result = _run_scenario(
        r"""
controller.setCatalog({
  models: [],
  gguf_artifacts: [{
    artifact_id: "a".repeat(64),
    repo_id: "<img src=https://attacker.invalid/collect>",
    resolved_revision: "b".repeat(40),
    model_alias: "<script>attack()</script>",
  }],
});
const option = byId("artifact-select").children[0];
process.stdout.write(JSON.stringify({
  label: option.textContent,
  childCount: option.children.length,
}));
"""
    )

    assert result == {
        "label": (
            "<script>attack()</script> — "
            "<img src=https://attacker.invalid/collect>@bbbbbbbbbbbb"
        ),
        "childCount": 0,
    }
