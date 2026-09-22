"""Exercise model information and multi-selection in a real local browser."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_configured_browser_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = (
        os.environ.get("CHROME_BIN")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    if browser is None:
        pytest.skip("Chrome or Chromium not installed")
    for name in ("google-chrome", "chromium"):
        shadow = tmp_path / name
        shadow.write_text(
            "#!/bin/sh\necho 'Wrong browser: CHROME_BIN was ignored' >&2\nexit 42\n"
        )
        shadow.chmod(0o755)
    monkeypatch.setenv("CHROME_BIN", browser)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    test_model_tooltips_do_not_change_selection(tmp_path)


def test_model_tooltips_do_not_change_selection(tmp_path: Path) -> None:
    # Hosted runners configure stable Chrome separately from Chromium snapshots.
    browser = (
        os.environ.get("CHROME_BIN")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    if browser is None:
        pytest.skip("Chrome or Chromium not installed")
    root = Path(__file__).resolve().parents[2]
    page = (root / "inference_proxy/templates/start.html").read_text()
    page = re.sub(
        r"\{\{ static_asset_url\(request, '([^']+)'\) \}\}",
        lambda m: (root / "inference_proxy/static" / m[1]).as_uri(),
        page,
    )
    # Only local assets and controlled responses. No gateway or third-party calls.
    page = re.sub(r"<link[^>]*https://[^>]*>", "", page)
    fixture = r"""
<script>
window.matchMedia = () => ({matches: true});
const token = {name: 'Laptop', models: ['org/qwen-GGUF'], exportable: true, created_at: '2026-09-22T00:00:00Z'};
const state = {user: {email: 'test@example.com'}, token, models: ['org/qwen-GGUF', 'org/gemma-GGUF', 'custom'],
  model_details: {'org/qwen-GGUF': {context_tokens: 262144, input_modalities: ['text']},
    'org/gemma-GGUF': {context_tokens: 131072, input_modalities: ['text']}},
  harnesses: [{id: 'opencode', label: 'OpenCode', available: true, multi_model: true}]};
let writes = 0;
window.fetch = async (url, options) => {
  if (options.method === 'PUT') { writes++; token.models = JSON.parse(options.body).models; }
  return {ok: true, status: 200, json: async () => options.method === 'PUT' ? {token} : state};
};
</script>
"""
    checks = r"""
<script>
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const check = (value, message) => { if (!value) throw Error(message); };
const selected = () => [...document.querySelectorAll('button[data-model][aria-pressed="true"]')].map(b => b.dataset.model).sort().join(',');
async function step(name) {
  for (let i = 0; i < 50; i++) {
    if (document.querySelector('.ob-step[data-step="' + name + '"]:not([inert])')) { await wait(30); return; }
    await wait(20);
  }
  throw Error('Missing step ' + name);
}
(async () => {
  await step('home');
  check(document.querySelector('button[data-model="org/qwen-GGUF"]').textContent === 'qwen', 'Home label is cleaned, ID is unchanged');
  check(window.QiipModelInfo.displayName('unsloth/Qwen3.8-27B-GGUF') === 'Qwen3.8-27B', 'Qwen name');
  check(window.QiipModelInfo.displayName('unsloth/Qwen3.6-35B-A3B-MTP-GGUF') === 'Qwen3.6-35B-A3B', 'MTP packaging suffix');
  check(window.QiipModelInfo.displayName('unsloth/Muse-Glimmer-30B-GGUF') === 'Muse-Glimmer-30B', 'Muse name');
  check(window.QiipModelInfo.displayName('unsloth/gemma-4-31B-it-GGUF') === 'gemma-4-31B-it', 'Gemma variant retained');
  const tip = document.getElementById('ob-model-tooltip');
  let info = document.querySelector('.ob-model-info');
  check(document.querySelectorAll('.ob-model-info').length === 2, 'Unknown model has no tooltip');
  info.focus();
  check(!tip.hidden && tip.textContent.includes('262,144'), 'Keyboard focus shows context');
  check(tip.children.length === 2 && tip.children[0].textContent === 'Inputs: text' && tip.children[1].textContent === 'Context: 262,144 tokens', 'Inputs and context are on separate lines');
  check(info.getAttribute('aria-describedby') === tip.id, 'Accessible tooltip relationship');
  info.click();
  check(!tip.hidden && writes === 0 && selected() === 'org/qwen-GGUF', 'Info click must not select or save');
  info.click();
  check(tip.hidden, 'Second tap dismisses');
  info.dispatchEvent(new MouseEvent('mouseenter'));
  check(!tip.hidden, 'Hover opens');
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  check(tip.hidden, 'Escape dismisses');
  document.querySelector('button[data-model="org/gemma-GGUF"]').click();
  await wait(20);
  check(selected() === 'org/gemma-GGUF,org/qwen-GGUF' && writes === 1, 'Home multi-selection still works');
  [...document.querySelectorAll('button')].find(b => b.textContent === 'Set up a tool').click();
  await step('harness');
  document.querySelector('.ob-choice').click();
  await step('models');
  check(document.querySelector('button[data-model="org/qwen-GGUF"] .ob-choice-label').textContent === 'qwen', 'Wizard label is cleaned');
  check(token.models.includes('org/gemma-GGUF'), 'Saved API scope uses original ID');
  check(selected() === 'org/gemma-GGUF,org/qwen-GGUF', 'Wizard retains both selections');
  info = document.querySelectorAll('.ob-model-info')[1];
  info.focus(); info.click();
  check(tip.textContent.includes('131,072') && selected() === 'org/gemma-GGUF,org/qwen-GGUF', 'Per-model details without toggling');
  document.querySelector('button[data-model="org/qwen-GGUF"]').click();
  check(selected() === 'org/gemma-GGUF' && tip.hidden, 'Model click selects and dismisses tooltip');
  document.querySelector('button[data-model="org/qwen-GGUF"]').click();
  check(selected() === 'org/gemma-GGUF,org/qwen-GGUF', 'Multiple wizard selections remain possible');
  info.focus();
  document.getElementById('ob-back').click();
  await step('harness');
  check(tip.hidden, 'Navigation dismisses tooltip');
  document.body.dataset.testResult = 'passed';
})().catch(error => { document.body.dataset.testResult = 'FAILED: ' + error.message; });
</script>
"""
    page = page.replace("<script src=", fixture + "<script src=", 1)
    page = page.replace("</body>", checks + "</body>")
    html = tmp_path / "start.html"
    html.write_text(page)
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-first-run",
            f"--user-data-dir={tmp_path / 'browser'}",
            "--virtual-time-budget=5000",
            "--dump-dom",
            html.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    result_marker = re.search(r'data-test-result="([^"]+)"', result.stdout)
    assert result_marker and result_marker[1] == "passed", result.stdout
