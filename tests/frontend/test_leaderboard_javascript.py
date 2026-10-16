"""Behavioral tests for the leaderboard page JavaScript (leaderboard.js).

The usage leaderboard and the signed-in user's token table sort by column
through the shared table_sort.js module; the ranking the server sends stays
the initial order until a header is activated. The token row shows only the
prefix, so the raw value is fetched on click by the clipboard icon next to it
(`Copy token`), and a token from before the onboarding flow (not exportable)
shows the note explaining why it cannot be copied or downloaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEADERBOARD_JS = _ROOT / "inference_proxy/static/js/leaderboard.js"
_TABLE_SORT_JS = _ROOT / "inference_proxy/static/js/table_sort.js"
_CLIPBOARD_JS = _ROOT / "inference_proxy/static/js/clipboard.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const sortSource = fs.readFileSync(process.argv[1], "utf8");
const clipboardSource = fs.readFileSync(process.argv[2], "utf8");
const source = fs.readFileSync(process.argv[3], "utf8");

const elements = new Map();
const calls = [];
const toasts = [];
const clipboardWrites = [];

class Element {
  constructor(tag) {
    this.tagName = tag || "div";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.parentNode = null;
    this.hidden = false;
    this.value = "";
    this.disabled = false;
    this._text = "";
  }
  // ponytail: own text first, then children in document order; the assignment
  // sites always set their own text before appending children, matching a real
  // browser closely enough for these tables.
  get textContent() {
    let text = this._text;
    for (const child of this.children) text += child.textContent;
    return text;
  }
  set textContent(value) {
    while (this.firstChild) this.removeChild(this.firstChild);
    this._text = String(value);
  }
  appendChild(child) {
    if (child.parentNode && child.parentNode.children) {
      const index = child.parentNode.children.indexOf(child);
      if (index !== -1) child.parentNode.children.splice(index, 1);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  get firstChild() { return this.children[0] || null; }
  removeChild(child) {
    const at = this.children.indexOf(child);
    if (at !== -1) this.children.splice(at, 1);
    child.parentNode = null;
    return child;
  }
  append() {
    for (const node of arguments) {
      if (typeof node === "string") this._text += node;
      else this.appendChild(node);
    }
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name] === undefined ? null : this.attributes[name]; }
  click() { if (this.listeners.click) this.listeners.click({}); }
}

function byId(id) {
  if (ABSENT_IDS.indexOf(id) !== -1) return null;
  if (!elements.has(id)) elements.set(id, new Element("div"));
  return elements.get(id);
}

// The admin view renders the ranking alone: the template omits the token
// cards, so their elements do not exist on the page. Kept in sync with the
// absent id set asserted by tests/api/test_onboarding.py
// (test_admin_sees_the_ranking_without_the_token_manager).
const ADMIN_VIEW = IS_ADMIN;
const ABSENT_IDS = ADMIN_VIEW
  ? ["tokens-head", "tokens-table-body", "token-legacy-note", "token-form",
     "token-model", "token-harness", "token-create-btn", "config-download-btn"]
  : [];

function buildHead(id, kinds) {
  const head = byId(id);
  const row = new Element("tr");
  kinds.forEach(function (kind) {
    const cell = new Element("th");
    if (kind) cell.dataset.sort = kind;
    row.appendChild(cell);
  });
  head.appendChild(row);
}

buildHead("leaderboard-head", [null, "text", "number", "number", "number", "number", "number"]);
// The template ships the server's ranking state on Total tokens.
byId("leaderboard-head").children[0].children[6].setAttribute("aria-sort", "descending");
if (!ADMIN_VIEW) buildHead("tokens-head", ["text", "text", "date", "date", "text", "text", null]);

const payloads = {
  "/auth/me": { id: 1, email: "me@example.com", name: "Me", picture: null },
  "/onboarding/state": {
    models: ["llama-3"],
    model_names: { "llama-3": "Llama 3" },
    harnesses: [],
    token: TOKEN,
  },
  "/onboarding/leaderboard": LEADERBOARD_ROWS,
  "/onboarding/token": { name: "default" },
  "/onboarding/token/reveal": { token: "qiip_revealed_value" },
};

const sandbox = {
  console,
  showToast: function (message, type) { toasts.push([message, type]); },
  confirmDialog: async function () { return true; },
  navigator: {
    clipboard: {
      writeText: async function (text) { clipboardWrites.push(text); },
    },
  },
  document: {
    readyState: "complete",
    getElementById: byId,
    addEventListener() {},
    createElement(tag) { return new Element(tag); },
    createElementNS(ns, tag) { return new Element(tag); },
    body: new Element("body"),
  },
  location: { href: "", search: "" },
  confirm: function () { return true; },
  fetch: async function (url, options) {
    calls.push({ url: url, method: (options && options.method) || "GET" });
    const body = payloads[url];
    return {
      ok: body !== undefined,
      status: body !== undefined ? 200 : 404,
      json: async function () { return body === undefined ? {} : body; },
    };
  },
  setTimeout,
  setInterval() { return 1; },
  clearInterval() {},
  localStorage: { getItem() { return null; }, setItem() {} },
  URLSearchParams,
};

// In a browser `window` IS the global object; mirror that so helpers assigned
// to `window` are reachable by their bare names, exactly like in the pages.
sandbox.window = sandbox;

vm.createContext(sandbox);
vm.runInContext(sortSource, sandbox);
vm.runInContext(clipboardSource, sandbox);
vm.runInContext(source, sandbox);

(async function () {
  // init() runs on load; wait for its fetches to settle.
  await new Promise(function (resolve) { setTimeout(resolve, 20); });

  // The token row's prefix cell carries the clipboard icon. Click it and
  // record the reveal call, the clipboard write and the toast; then check the
  // legacy-token note and the Download config gating below.
  const tokenBody = ADMIN_VIEW ? null : byId("tokens-table-body");
  const prefixCell = tokenBody && tokenBody.children[0]
    ? tokenBody.children[0].children[1]
    : null;
  const copyButton = prefixCell && prefixCell.children[0] ? prefixCell.children[0] : null;
  const revealBeforeClick = calls.filter(function (call) {
    return call.url === "/onboarding/token/reveal";
  }).length;
  if (!ADMIN_VIEW) {
    byId("token-harness").value = "pi";
    if (byId("token-harness").listeners.change) byId("token-harness").listeners.change({});
  }
  const downloadDisabled = ADMIN_VIEW ? null : byId("config-download-btn").disabled;
  if (copyButton) {
    await copyButton.listeners.click({});
    await new Promise(function (resolve) { setTimeout(resolve, 20); });
  }
  const copy = {
    title: copyButton ? copyButton.title : null,
    ariaLabel: copyButton ? copyButton.getAttribute("aria-label") : null,
    className: copyButton ? copyButton.className : null,
    iconShapes: copyButton && copyButton.children[0]
      ? copyButton.children[0].children.map(function (child) { return child.tagName; })
      : [],
    prefixText: prefixCell ? prefixCell.textContent : null,
    revealBeforeClick: revealBeforeClick,
    revealAfterClick: calls.filter(function (call) {
      return call.url === "/onboarding/token/reveal";
    }).length,
    legacyNoteHidden: ADMIN_VIEW ? null : byId("token-legacy-note").hidden,
    downloadDisabled: downloadDisabled,
    clipboardWrites: clipboardWrites.slice(),
    toasts: toasts.slice(),
  };

  const headers = byId("leaderboard-head").children[0].children;
  if (SORT_INDEX >= 0) {
    const header = headers[SORT_INDEX];
    for (let click = 0; click < CLICK_COUNT; click += 1) header.click();
  }

  const tokenHeaders = ADMIN_VIEW ? [] : byId("tokens-head").children[0].children;
  if (!ADMIN_VIEW && TOKENS_SORT_INDEX >= 0) tokenHeaders[TOKENS_SORT_INDEX].click();

  const ranks = function () {
    return byId("leaderboard-body").children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : "";
    });
  };
  const users = function () {
    return byId("leaderboard-body").children.map(function (row) {
      return row.children[1] ? row.children[1].textContent : "";
    });
  };
  const snapshot = {
    ranks: ranks(),
    users: users(),
    tokens: byId("leaderboard-body").children.map(function (row) {
      return row.children[2] ? row.children[2].textContent : "";
    }),
    aria: headers.map(function (cell) { return cell.getAttribute("aria-sort"); }),
    tokensAria: tokenHeaders.map(function (cell) { return cell.getAttribute("aria-sort"); }),
  };

  // Creating a token reloads the leaderboard, so the table renders again and
  // the sorter has to re-apply the active choice over the fresh rows.
  if (!ADMIN_VIEW) {
    byId("token-form").listeners.submit({ preventDefault: function () {} });
    await new Promise(function (resolve) { setTimeout(resolve, 20); });
  }

  process.stdout.write(JSON.stringify({
    ranks: snapshot.ranks,
    users: snapshot.users,
    tokens: snapshot.tokens,
    aria: snapshot.aria,
    tokensAria: snapshot.tokensAria,
    ranksAfterRerender: ranks(),
    usersAfterRerender: users(),
    ariaAfterRerender: headers.map(function (cell) {
      return cell.getAttribute("aria-sort");
    }),
    rowsAfterRerender: byId("leaderboard-body").children.length,
    leaderboardFetches: calls.filter(function (call) {
      return call.url === "/onboarding/leaderboard";
    }).length,
    fetches: calls.map(function (call) { return call.url; }),
    errorBannerText: byId("error-banner").textContent,
    copy: copy,
    tokenRows: ADMIN_VIEW ? [] : byId("tokens-table-body").children.map(function (row) {
      return row.children[0] ? row.children[0].textContent : "";
    }),
  }));
})().catch(function (error) {
  console.error(error);
  process.exit(1);
});
"""


def _run(
    users: list[dict[str, Any]],
    sort_index: int = -1,
    click_count: int = 1,
    token: dict[str, Any] | None = None,
    tokens_sort_index: int = -1,
    admin_view: bool = False,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for leaderboard JS regressions")
    script = (
        _HARNESS.replace("IS_ADMIN", "true" if admin_view else "false")
        .replace("TOKENS_SORT_INDEX", str(tokens_sort_index))
        .replace("SORT_INDEX", str(sort_index))
        .replace("CLICK_COUNT", str(click_count))
        .replace("LEADERBOARD_ROWS", json.dumps(users))
        .replace("TOKEN", json.dumps(token))
    )
    result = subprocess.run(
        [
            node,
            "-e",
            script,
            str(_TABLE_SORT_JS),
            str(_CLIPBOARD_JS),
            str(_LEADERBOARD_JS),
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


def _user(user_id: int, name: str, tokens: int, requests: int = 1) -> dict[str, Any]:
    return {
        "id": user_id,
        "name": name,
        "email": f"{name.lower()}@example.com",
        "active_token_count": 1,
        "token_count": tokens,
        "request_count": requests,
        "prompt_tokens": tokens * 10,
        "completion_tokens": tokens * 5,
        "total_tokens": tokens * 15,
    }


def test_unsorted_leaderboard_keeps_server_ranking() -> None:
    # The server ranks by total tokens DESC: Bo (450) before Ann (150).
    result = _run([_user(1, "Bo", 30), _user(2, "Ann", 10)])

    assert result["users"] == ["Bo (you)", "Ann"]
    # The template carries the server's ranking state on Total tokens; every
    # other header starts unsorted.
    assert result["aria"][6] == "descending"
    assert set(result["aria"][:6]) == {None}


def test_leaderboard_sorts_by_total_tokens_descending() -> None:
    bo = _user(1, "Bo", 10)
    bo["total_tokens"] = 900
    ann = _user(2, "Ann", 30)
    ann["total_tokens"] = 1500
    cy = _user(3, "Cy", 20)
    cy["total_tokens"] = 300

    result = _run([ann, bo, cy], sort_index=6, click_count=2)

    # Lexicographic order would put "900" first; the number kind must not.
    assert result["users"] == ["Ann", "Bo (you)", "Cy"]
    assert result["aria"][6] == "descending"
    assert result["aria"][0] is None


def test_tokens_column_sorts_by_total_count_not_display_text() -> None:
    active_heavy = _user(1, "Bo", 1)
    active_heavy["active_token_count"] = 9
    total_heavy = _user(2, "Ann", 20)
    total_heavy["active_token_count"] = 1

    result = _run([total_heavy, active_heavy], sort_index=2)

    assert result["tokens"] == ["9 / 1", "1 / 20"]
    assert result["aria"][2] == "ascending"


def test_rank_column_is_fixed_and_keeps_each_users_rank() -> None:
    bo = _user(1, "Bo", 10, requests=1)
    bo["total_tokens"] = 900
    ann = _user(2, "Ann", 30, requests=9)
    ann["total_tokens"] = 300
    cy = _user(3, "Cy", 20, requests=5)
    cy["total_tokens"] = 600

    # Server order is the total-token ranking: Bo 900, Cy 600, Ann 300.
    unsorted = _run([bo, cy, ann])

    assert unsorted["ranks"] == ["1", "2", "3"]
    assert unsorted["aria"][6] == "descending"
    assert set(unsorted["aria"][:6]) == {None}

    sorted_by_requests = _run([bo, cy, ann], sort_index=3, click_count=2)

    # Rank is the total-token standing, not the row position: it travels with
    # the user when another column drives the order, and a reload of the table
    # (creating a token re-renders it) keeps both the order and the ranks.
    assert sorted_by_requests["users"] == ["Ann", "Cy", "Bo (you)"]
    assert sorted_by_requests["ranks"] == ["3", "2", "1"]
    assert sorted_by_requests["usersAfterRerender"] == ["Ann", "Cy", "Bo (you)"]
    assert sorted_by_requests["ranksAfterRerender"] == ["3", "2", "1"]
    assert sorted_by_requests["rowsAfterRerender"] == 3
    assert sorted_by_requests["leaderboardFetches"] == 2
    assert sorted_by_requests["ariaAfterRerender"][3] == "descending"


def test_you_marker_does_not_join_the_user_sort_key() -> None:
    # Same display name and the same standing; the signed-in row (id 1) comes
    # first from the server.
    signed_in = _user(1, "Sam", 20)
    other = _user(2, "Sam", 20)

    result = _run([signed_in, other], sort_index=1)

    # Equal keys keep the server order: the " (you)" suffix must not become
    # part of the key and push the signed-in row after its namesake.
    assert result["ranks"] == ["1", "2"]
    assert result["users"][0] == "Sam (you)"
    assert result["users"][1] == "Sam"
    assert result["aria"][1] == "ascending"


def test_admin_view_renders_the_ranking_alone() -> None:
    # The admin shell omits the token cards (admins manage tokens on their
    # profile page), so the ranking must render and sort without touching
    # the onboarding endpoints, which answer 403 to admins.
    bo = _user(1, "Bo", 30)
    ann = _user(2, "Ann", 10)

    result = _run([bo, ann], admin_view=True, sort_index=6, click_count=2)

    assert result["users"] == ["Bo", "Ann"]
    assert result["aria"][6] == "descending"
    assert result["leaderboardFetches"] == 1
    # The ranking alone loads: no identity, state, token or config request.
    assert result["fetches"] == ["/onboarding/leaderboard"]
    assert result["errorBannerText"] == ""
    assert result["tokenRows"] == []


def test_your_tokens_table_sorts_by_last_used() -> None:
    # Without a token the body is the empty state, so there is nothing to sort;
    # with one, activating "Last used" keeps the row and marks the column. One
    # token can never show a reorder, so the observable is the surviving row.
    empty = _run([_user(1, "Bo", 10)])
    assert empty["tokenRows"] == ["No token yet. Create one below."]
    assert empty["tokensAria"][3] is None

    token = {
        "name": "default",
        "prefix": "sk-abc",
        "created_at": "2026-01-02T03:04:05Z",
        "last_used_at": None,
    }
    result = _run([_user(1, "Bo", 10)], token=token, tokens_sort_index=3)

    assert result["tokenRows"] == ["default"]
    assert "No token yet" not in result["tokenRows"][0]
    assert result["tokensAria"][3] == "ascending"


def _copy_token(exportable: bool = True) -> dict[str, Any]:
    return {
        "id": 7,
        "name": "default",
        "prefix": "qiip_abc123",
        "created_at": "2026-10-01T10:00:00Z",
        "last_used_at": None,
        "models": ["llama-3"],
        "exportable": exportable,
    }


def test_prefix_cell_offers_the_copy_icon() -> None:
    result = _run([_user(1, "Bo", 10)], token=_copy_token())

    copy = result["copy"]
    assert copy["prefixText"] == "qiip_abc123"
    assert copy["title"] == "Copy token"
    assert copy["ariaLabel"] == "Copy token"
    assert copy["className"] == "token-copy"
    assert copy["iconShapes"] == ["path", "rect"]
    assert copy["legacyNoteHidden"] is True
    assert copy["downloadDisabled"] is False


def test_copy_icon_reveals_the_token_and_toasts() -> None:
    result = _run([_user(1, "Bo", 10)], token=_copy_token())

    copy = result["copy"]
    assert copy["revealBeforeClick"] == 0
    assert copy["revealAfterClick"] == 1
    assert copy["clipboardWrites"] == ["qiip_revealed_value"]
    assert copy["toasts"] == [["Token copied to clipboard.", "success"]]


def test_legacy_token_has_no_copy_icon_and_explains_itself() -> None:
    result = _run([_user(1, "Bo", 10)], token=_copy_token(exportable=False))

    copy = result["copy"]
    assert copy["title"] is None
    assert copy["prefixText"] == "qiip_abc123"
    assert copy["legacyNoteHidden"] is False
    assert copy["downloadDisabled"] is True
    assert copy["toasts"] == []


def test_no_token_hides_the_note_and_offers_no_copy_icon() -> None:
    result = _run([_user(1, "Bo", 10)])

    copy = result["copy"]
    assert copy["title"] is None
    assert copy["legacyNoteHidden"] is True
    assert result["tokenRows"] == ["No token yet. Create one below."]


def test_delete_still_sits_in_the_actions_cell() -> None:
    result = _run([_user(1, "Bo", 10)], token=_copy_token())

    # The actions cell holds Delete alone; the copy icon belongs to the prefix.
    assert result["copy"]["prefixText"] == "qiip_abc123"
    assert result["tokenRows"] == ["default"]
