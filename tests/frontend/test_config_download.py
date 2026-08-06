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
    """Dashboard uses proxy origin; node detail uses node.endpoint."""

    def test_dashboard_uses_window_location_origin(self) -> None:
        source = _DASHBOARD_JS.read_text()
        assert "createConfigDropdown(" in source
        assert "window.location.origin" in source

    def test_node_detail_uses_node_endpoint(self) -> None:
        source = _NODE_DETAIL_JS.read_text()
        assert "createConfigDropdown(node.endpoint," in source

    def test_dashboard_does_not_use_node_endpoint_for_config(self) -> None:
        source = _DASHBOARD_JS.read_text()
        assert "createConfigDropdown(node.endpoint" not in source

    def test_node_detail_does_not_use_window_location_for_config(self) -> None:
        source = _NODE_DETAIL_JS.read_text()
        assert "createConfigDropdown(window.location" not in source
