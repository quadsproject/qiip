"""Behavioral tests for the admin page JavaScript (admin.js).

Regression: the Grant/Revoke Admin buttons must send
``Content-Type: application/json`` on state-changing POST/DELETE requests.
``require_admin_auth`` rejects any state-changing admin request whose body
media type is not JSON (415), so a header-less fetch made the admin role
grant fail with "Admin state-changing requests must use application/json".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ADMIN_JS = _ROOT / "inference_proxy/static/js/admin.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");

const calls = [];
const created = [];
const elements = new Map();

function element() {
  const el = {
    addEventListener(name, fn) { if (name === "click") this._click = fn; },
    appendChild() {}, remove() {},
    setAttribute() {}, textContent: "", innerHTML: "", value: "",
    className: "", style: {}, dataset: {},
    classList: { add() {}, remove() {}, contains() { return false; } },
  };
  created.push(el);
  return el;
}

function byId(id) {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}

const sandbox = {
  console,
  URL: URL,
  document: {
    getElementById: byId,
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return element(); },
    createElement() { return element(); },
    createTextNode(text) { return { textContent: text }; },
  },
  window: { confirm() { return true; } },
  requestAnimationFrame() {},
  setTimeout() {},
  clearTimeout() {},
  fetch: async function (url, options) {
    calls.push({ url, options });
    return { ok: true, status: 204, json: async function () { return {}; } };
  },
};

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

(async function () {
  sandbox.renderAdminUsers(USERS);
  const buttons = created.filter(function (el) { return el._click && el.textContent; });
  if (buttons.length === 0) throw new Error("no clickable button created");
  await buttons[buttons.length - 1]._click();
  process.stdout.write(JSON.stringify(
    calls.map(function (c) {
      return {
        url: c.url,
        method: c.options ? c.options.method : null,
        contentType: c.options && c.options.headers
          ? c.options.headers["Content-Type"]
          : null,
      };
    })
  ));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""


def _run(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for admin page JS regressions")
    script = _HARNESS.replace("USERS", json.dumps(users))
    result = subprocess.run(
        [node, "-e", script, str(_ADMIN_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, list)
    return parsed


def _role_call(calls: list[dict[str, Any]]) -> dict[str, Any]:
    call = next(c for c in calls if c["url"] == "/admin/users/1/admin")
    return call


def test_grant_admin_sends_json_content_type() -> None:
    result = _run(
        [{"id": 1, "email": "kambiz@redhat.com", "name": "K", "is_admin": False}]
    )
    assert _role_call(result) == {
        "url": "/admin/users/1/admin",
        "method": "POST",
        "contentType": "application/json",
    }


def test_revoke_admin_sends_json_content_type() -> None:
    result = _run(
        [{"id": 1, "email": "kambiz@redhat.com", "name": "K", "is_admin": True}]
    )
    assert _role_call(result) == {
        "url": "/admin/users/1/admin",
        "method": "DELETE",
        "contentType": "application/json",
    }
