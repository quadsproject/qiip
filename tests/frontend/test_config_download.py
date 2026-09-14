"""Behavioral tests for the config download JavaScript utility."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DOWNLOAD_JS = _ROOT / "inference_proxy/static/js/config_download.js"
_DASHBOARD_JS = _ROOT / "inference_proxy/static/js/dashboard.js"
_NODE_DETAIL_JS = _ROOT / "inference_proxy/static/js/node_detail.js"

TOKEN_PLACEHOLDER = "<paste-qiip-token-here>"


def _run_node_raw(harness: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "Node.js is required for config download regressions; "
            "CI must install it explicitly"
        )

    result = subprocess.run(
        [node, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run_node(harness: str) -> dict[str, Any]:
    parsed = _run_node_raw(harness)
    assert isinstance(parsed, dict)
    return parsed


def _run_node_yaml(harness: str) -> str:
    parsed = _run_node_raw(harness)
    assert isinstance(parsed, str)
    return parsed


def _harness(base_url: str, model_id: str, func: str) -> str:
    """Build a Node.js harness that calls a config generator and prints JSON."""
    js_path = json.dumps(str(_CONFIG_DOWNLOAD_JS))
    js_base = json.dumps(base_url)
    js_model = json.dumps(model_id)
    return (
        "const fs = require('fs');\n"
        "const vm = require('vm');\n"
        f"const source = fs.readFileSync({js_path}, 'utf8');\n"
        "const sandbox = { console };\n"
        "vm.createContext(sandbox);\n"
        "vm.runInContext(source, sandbox);\n"
        f"const result = sandbox.{func}({js_base}, {js_model});\n"
        "console.log(JSON.stringify(result));\n"
    )


class TestGenerateOpenCodeConfig:
    """generateOpenCodeConfig produces valid OpenCode CLI configuration."""

    def test_structure(self) -> None:
        result = _run_node(
            _harness(
                "http://proxy.example.com:8080",
                "meta-llama/Llama-3-8B",
                "generateOpenCodeConfig",
            )
        )
        assert result["$schema"] == "https://opencode.ai/config.json"
        provider = result["provider"]["qiip"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["name"] == "QIIP Inference Proxy"
        assert provider["options"]["baseURL"] == ("http://proxy.example.com:8080/v1")
        assert "meta-llama/Llama-3-8B" in provider["models"]
        assert provider["models"]["meta-llama/Llama-3-8B"]["name"] == (
            "meta-llama/Llama-3-8B"
        )

    def test_model_prefixed_with_qiip(self) -> None:
        result = _run_node(
            _harness("http://localhost:8080", "my-model", "generateOpenCodeConfig")
        )
        assert result["model"] == "qiip/my-model"

    def test_base_url_includes_v1(self) -> None:
        result = _run_node(
            _harness(
                "http://gpu01.example.com:8000",
                "test-model",
                "generateOpenCodeConfig",
            )
        )
        assert result["provider"]["qiip"]["options"]["baseURL"].endswith("/v1")

    def test_trailing_slash_stripped(self) -> None:
        result = _run_node(
            _harness(
                "http://proxy.example.com:8080/",
                "test-model",
                "generateOpenCodeConfig",
            )
        )
        base_url = result["provider"]["qiip"]["options"]["baseURL"]
        assert "//" not in base_url.split("://", 1)[1]
        assert base_url.endswith("/v1")


class TestGeneratePiConfig:
    """generatePiConfig produces valid Pi coding agent configuration."""

    def test_structure(self) -> None:
        result = _run_node(
            _harness(
                "http://proxy.example.com:8080",
                "meta-llama/Llama-3-8B",
                "generatePiConfig",
            )
        )
        provider = result["providers"]["qiip"]
        assert provider["baseUrl"] == "http://proxy.example.com:8080/v1"
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "none"
        assert len(provider["models"]) == 1
        assert provider["models"][0]["id"] == "meta-llama/Llama-3-8B"

    def test_compat_flags_set(self) -> None:
        result = _run_node(
            _harness("http://localhost:8080", "test-model", "generatePiConfig")
        )
        compat = result["providers"]["qiip"]["compat"]
        assert compat["supportsDeveloperRole"] is False
        assert compat["supportsReasoningEffort"] is False

    def test_base_url_includes_v1(self) -> None:
        result = _run_node(
            _harness(
                "http://gpu01.example.com:8000",
                "test-model",
                "generatePiConfig",
            )
        )
        assert result["providers"]["qiip"]["baseUrl"].endswith("/v1")

    def test_trailing_slash_stripped(self) -> None:
        result = _run_node(
            _harness(
                "http://proxy.example.com:8080/",
                "test-model",
                "generatePiConfig",
            )
        )
        base_url = result["providers"]["qiip"]["baseUrl"]
        assert "//" not in base_url.split("://", 1)[1]
        assert base_url.endswith("/v1")


class TestGenerateOmpConfig:
    """generateOmpConfig produces valid OMP models.yaml configuration."""

    def test_structure(self) -> None:
        result = _run_node_yaml(
            _harness(
                "http://proxy.example.com:8080",
                "meta-llama/Llama-3-8B",
                "generateOmpConfig",
            )
        )
        assert "providers:" in result
        assert "  qiip:" in result
        assert "    baseUrl: http://proxy.example.com:8080/v1" in result
        assert "    auth: none" in result
        assert "    api: openai-completions" in result
        assert "      - id: meta-llama/Llama-3-8B" in result
        assert "        name: meta-llama/Llama-3-8B (qiip)" in result

    def test_base_url_includes_v1(self) -> None:
        result = _run_node_yaml(
            _harness(
                "http://gpu01.example.com:8000",
                "test-model",
                "generateOmpConfig",
            )
        )
        assert "/v1" in result

    def test_trailing_slash_stripped(self) -> None:
        result = _run_node_yaml(
            _harness(
                "http://proxy.example.com:8080/",
                "test-model",
                "generateOmpConfig",
            )
        )
        assert "baseUrl: http://proxy.example.com:8080/v1" in result

    def test_special_chars_quoted(self) -> None:
        result = _run_node_yaml(
            _harness(
                "http://proxy.example.com:8080",
                "model: evil #comment",
                "generateOmpConfig",
            )
        )
        assert '"model: evil #comment"' in result


def _harness_opts(base_url: str, model_id: str, func: str, opts_json: str) -> str:
    """Like _harness but passes an opts object (node info) to the generator."""
    js_path = json.dumps(str(_CONFIG_DOWNLOAD_JS))
    js_base = json.dumps(base_url)
    js_model = json.dumps(model_id)
    return (
        "const fs = require('fs');\n"
        "const vm = require('vm');\n"
        f"const source = fs.readFileSync({js_path}, 'utf8');\n"
        "const sandbox = { console };\n"
        "vm.createContext(sandbox);\n"
        "vm.runInContext(source, sandbox);\n"
        f"const result = sandbox.{func}({js_base}, {js_model}, {opts_json});\n"
        "console.log(JSON.stringify(result));\n"
    )


class TestHiddenServerConfigs:
    """Hidden server configs declare token auth with a placeholder."""

    _BASE = "https://inference-proxy-dev.rdu2.scalelab.redhat.com"
    _MODEL = "DeepSeek-V4-Flash-Vision-Exp"
    _OPTS = '{"name": "DeepSeek-V4-Flash-Vision-Exp (qiip)", "hidden": true}'

    def test_omp_config_requires_api_key(self) -> None:
        result = _run_node_yaml(
            _harness_opts(self._BASE, self._MODEL, "generateOmpConfig", self._OPTS)
        )
        assert "    auth: apiKey" in result
        assert "    apiKey: <paste-qiip-token-here>" in result
        assert "    auth: none" not in result
        assert "        name: DeepSeek-V4-Flash-Vision-Exp (qiip)" in result

    def test_pi_config_uses_token_placeholder(self) -> None:
        result = _run_node(
            _harness_opts(self._BASE, self._MODEL, "generatePiConfig", self._OPTS)
        )
        provider = result["providers"]["qiip"]
        assert provider["apiKey"] == "<paste-qiip-token-here>"

    def test_opencode_config_uses_token_placeholder(self) -> None:
        result = _run_node(
            _harness_opts(self._BASE, self._MODEL, "generateOpenCodeConfig", self._OPTS)
        )
        options = result["provider"]["qiip"]["options"]
        assert options["apiKey"] == "<paste-qiip-token-here>"

    def test_configs_use_minted_token_when_available(self) -> None:
        token_opts = (
            '{"name": "DeepSeek-V4-Flash-Vision-Exp (qiip)", "hidden": true, '
            '"token": "qiip_abcdef123"}'
        )
        omp = _run_node_yaml(
            _harness_opts(self._BASE, self._MODEL, "generateOmpConfig", token_opts)
        )
        assert "    apiKey: qiip_abcdef123" in omp
        assert TOKEN_PLACEHOLDER not in omp

        pi = _run_node(
            _harness_opts(self._BASE, self._MODEL, "generatePiConfig", token_opts)
        )
        assert pi["providers"]["qiip"]["apiKey"] == "qiip_abcdef123"

        opencode = _run_node(
            _harness_opts(self._BASE, self._MODEL, "generateOpenCodeConfig", token_opts)
        )
        assert opencode["provider"]["qiip"]["options"]["apiKey"] == "qiip_abcdef123"

    def test_public_configs_stay_anonymous(self) -> None:
        result = _run_node_yaml(
            _harness(
                self._BASE,
                self._MODEL,
                "generateOmpConfig",
            )
        )
        assert "    auth: none" in result
        assert "    apiKey:" not in result


class TestConfigFileContents:
    """config_download.js is present and contains expected functions."""

    def test_file_exists(self) -> None:
        assert _CONFIG_DOWNLOAD_JS.is_file()

    @pytest.mark.parametrize(
        "name",
        [
            "generateOpenCodeConfig",
            "generatePiConfig",
            "generateOmpConfig",
            "downloadConfigFile",
            "createConfigDropdown",
        ],
    )
    def test_contains_function(self, name: str) -> None:
        source = _CONFIG_DOWNLOAD_JS.read_text()
        assert f"function {name}(" in source


class TestBaseUrlUsage:
    """Dashboard and node detail both use the proxy origin for config."""

    def test_dashboard_uses_window_location_origin(self) -> None:
        source = _DASHBOARD_JS.read_text()
        assert "createConfigDropdown(" in source
        assert "window.location.origin" in source

    def test_node_detail_uses_window_location_for_config(self) -> None:
        source = _NODE_DETAIL_JS.read_text()
        assert "createConfigDropdown(window.location.origin," in source

    def test_dashboard_does_not_use_node_endpoint_for_config(self) -> None:
        source = _DASHBOARD_JS.read_text()
        assert "createConfigDropdown(node.endpoint" not in source

    def test_node_detail_does_not_use_node_endpoint_for_config(self) -> None:
        source = _NODE_DETAIL_JS.read_text()
        assert "createConfigDropdown(node.endpoint" not in source


class TestMintTokenOnDownload:
    """Downloading a hidden-server config mints the shared config token."""

    def test_hidden_download_mints_stable_config_token(self) -> None:
        harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(SOURCE_PATH, "utf8");
let captured = [];
const created = [];
function element() {
  const el = {
    children: [], _handlers: {}, textContent: "", className: "", type: "",
    href: "", download: "",
    addEventListener(name, fn) { this._handlers[name] = fn; },
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) { return child; },
    remove() {}, setAttribute() {}, click() {},
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
  created.push(el);
  return el;
}
const sandbox = {
  console,
  Blob: function () {},
  URL: { createObjectURL: function () { return "blob:test"; }, revokeObjectURL: function () {} },
  document: {
    createElement: function () { return element(); },
    body: element(),
    addEventListener() {},
    querySelectorAll() { return []; },
  },
  window: { showToast: null, location: { origin: "http://test" } },
  fetch: async function (url, options) {
    captured.push({ url, options });
    return { ok: true, status: 201, json: async function () { return { token: "qiip_minted123" }; } };
  },
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
(async function () {
  sandbox.createConfigDropdown(
    "http://proxy:5000", "deepseek-model", function () {}, function () {},
    { hidden: true, name: "DeepSeek (qiip)" }
  );
  const formatButtons = created.filter(function (el) { return el._handlers.click && el.textContent; });
  const omp = formatButtons.find(function (el) { return el.textContent === "OMP Agent"; });
  await omp._handlers.click();
  const mint = captured.find(function (c) { return c.url === "/profile/tokens"; });
  process.stdout.write(JSON.stringify(mint ? JSON.parse(mint.options.body) : null));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""
        result = _run_node_raw(
            harness.replace("SOURCE_PATH", json.dumps(str(_CONFIG_DOWNLOAD_JS)))
        )
        assert result == {"name": "agent-config"}
