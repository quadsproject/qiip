"""Exercise model information and multi-selection in a real local browser."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_LONG_IDENTITY = "firstname.lastname.department@example.com"

# A 320px iframe is the only way to give the page a real phone viewport here:
# this headless Chrome floors --window-size at 500px and ignores
# --force-device-scale-factor for innerWidth. --allow-file-access-from-files
# lets the harness read the frame's layout.
_PHONE_HARNESS = """<!doctype html>
<body>
<iframe id="page" src="__CHILD__" width="320" height="800" style="border:0"></iframe>
<script>
const check = (value, message) => { if (!value) throw Error(message); };
document.getElementById('page').addEventListener('load', function () {
  try {
    const frame = document.getElementById('page');
    const doc = frame.contentDocument;
    const viewport = frame.contentWindow.innerWidth;
    const style = frame.contentWindow.getComputedStyle;
    const header = doc.querySelector('.ob-top');
    const bar = header.getBoundingClientRect();
    const identity = doc.querySelector('.nav-identity').getBoundingClientRect();
    const theme = doc.querySelector('.theme-toggle').getBoundingClientRect();
    check(viewport === 320, 'Harness viewport must be 320px, saw ' + viewport);
    check(style(header).display === 'flex', 'start.css did not apply');
    check(bar.width <= viewport, 'Header exceeds the viewport at ' + bar.width);
    check(header.scrollWidth <= viewport, 'Header content overflows the viewport at ' + header.scrollWidth);
    check(identity.width > 0 && identity.right <= viewport, 'Identity left the screen');
    check(theme.right <= viewport, 'Theme toggle left the screen');
    check(theme.width >= 32, 'Theme toggle shrank below its 2rem target, saw ' + theme.width);
    document.body.dataset.testResult = 'passed';
  } catch (error) {
    document.body.dataset.testResult = 'FAILED: ' + error.message;
  }
});
</script>
</body>
"""


def _configured_browser() -> str | None:
    """Return the Chrome/Chromium binary to drive, or None when absent."""
    return (
        os.environ.get("CHROME_BIN")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )


def _render_start_page(root: Path, identity: str = "test@example.com") -> str:
    """Return start.html with the shared partial and local assets inlined."""
    page = (root / "inference_proxy/templates/start.html").read_text()
    cluster = (
        root / "inference_proxy/templates/partials/account_actions.html"
    ).read_text()
    # The shared account cluster renders as literal Jinja text otherwise.
    cluster = re.sub(r"\{#.*?#\}", "", cluster, flags=re.S)
    cluster = re.sub(r"\{%.*?%\}", "", cluster, flags=re.S)
    cluster = cluster.replace("{{ viewer_identity(request) }}", identity)
    page = page.replace('{% include "partials/account_actions.html" %}', cluster)
    page = re.sub(
        r"\{\{ static_asset_url\(request, '([^']+)'\) \}\}",
        lambda m: (root / "inference_proxy/static" / m[1]).as_uri(),
        page,
    )
    # Only local assets and controlled responses. No gateway or third-party calls.
    return re.sub(r"<link[^>]*https://[^>]*>", "", page)


def _dump_page(
    browser: str,
    tmp_path: Path,
    url: str,
    extra_args: tuple[str, ...] = (),
) -> str:
    """Load *url* headlessly and return the dumped DOM.

    Headless Chrome occasionally hangs once on CI runners (the virtual-time
    budget never fires). That is a transient runner flake, not a page
    defect: one retry with a fresh profile absorbs it, a second failure is
    real and still fails the test.
    """
    last_error: subprocess.TimeoutExpired | None = None
    for attempt in range(2):
        try:
            result = subprocess.run(
                [
                    browser,
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-background-networking",
                    "--no-first-run",
                    f"--user-data-dir={tmp_path / f'browser-{attempt}'}",
                    "--virtual-time-budget=5000",
                    *extra_args,
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            continue
        assert result.returncode == 0, result.stderr
        return result.stdout
    assert last_error is not None
    raise last_error


def _test_result(dumped: str) -> str:
    """Return the in-page result marker, or "" when the page never reported."""
    marker = re.search(r'data-test-result="([^"]+)"', dumped)
    return marker[1] if marker else ""


def test_configured_browser_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = _configured_browser()
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


def test_long_identity_keeps_the_account_group_on_a_phone(tmp_path: Path) -> None:
    """A long email must not widen the /start header past a 320px viewport."""
    browser = _configured_browser()
    if browser is None:
        pytest.skip("Chrome or Chromium not installed")
    root = Path(__file__).resolve().parents[2]
    child = tmp_path / "start-phone.html"
    child.write_text(_render_start_page(root, identity=_LONG_IDENTITY))
    harness = tmp_path / "start-phone-harness.html"
    harness.write_text(_PHONE_HARNESS.replace("__CHILD__", child.name))

    dumped = _dump_page(
        browser,
        tmp_path,
        harness.as_uri(),
        extra_args=("--allow-file-access-from-files",),
    )

    assert _test_result(dumped) == "passed", dumped


def test_model_tooltips_do_not_change_selection(tmp_path: Path) -> None:
    # Hosted runners configure stable Chrome separately from Chromium snapshots.
    browser = _configured_browser()
    if browser is None:
        pytest.skip("Chrome or Chromium not installed")
    root = Path(__file__).resolve().parents[2]
    page = _render_start_page(root)
    fixture = r"""
<script>
window.matchMedia = () => ({matches: true});
const token = {name: 'Laptop', models: ['org/qwen-GGUF'], exportable: true, created_at: '2026-09-22T00:00:00Z'};
const state = {user: {email: 'test@example.com'}, token, models: ['org/qwen-GGUF', 'org/gemma-GGUF', 'custom'],
  model_names: {'org/qwen-GGUF': 'Qwen label from API', 'org/gemma-GGUF': 'gemma'},
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
  check(document.querySelector('button[data-model="org/qwen-GGUF"]').textContent === 'Qwen label from API', 'Home label comes from API, ID is unchanged');
  check(document.querySelector('button[data-model="custom"]').textContent === 'custom', 'Home falls back to raw ID');
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
  check(document.querySelector('button[data-model="org/qwen-GGUF"] .ob-choice-label').textContent === 'Qwen label from API', 'Wizard label comes from API');
  check(document.querySelector('button[data-model="custom"] .ob-choice-label').textContent === 'custom', 'Wizard falls back to raw ID');
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

    dumped = _dump_page(browser, tmp_path, html.as_uri())

    assert _test_result(dumped) == "passed", dumped
