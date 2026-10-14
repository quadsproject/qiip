"""Regression tests for the shared table_sort.js column sorter."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "inference_proxy" / "static" / "js" / "table_sort.js"

_PRELUDE = r"""
const fs = require("fs");
const vm = require("vm");
const moduleSource = fs.readFileSync(process.argv[1], "utf8");

class Element {
  constructor(tag) {
    this.tagName = tag || "div";
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.parentNode = null;
    this._text = "";
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value); }
  appendChild(child) {
    if (child.parentNode && child.parentNode.children) {
      const index = child.parentNode.children.indexOf(child);
      if (index !== -1) child.parentNode.children.splice(index, 1);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  addEventListener(name, callback) {
    (this.listeners[name] = this.listeners[name] || []).push(callback);
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name] === undefined ? null : this.attributes[name]; }
  click() { (this.listeners.click || []).forEach(function (callback) { callback({}); }); }
}

function th(label, kind) {
  const cell = new Element("th");
  cell.textContent = label;
  if (kind) cell.dataset.sort = kind;
  return cell;
}

function td(text, sortValue) {
  const cell = new Element("td");
  cell.textContent = text;
  if (sortValue !== undefined) cell.dataset.sortValue = String(sortValue);
  return cell;
}

function table(heads, rows) {
  const head = new Element("thead");
  const headRow = new Element("tr");
  heads.forEach((cell) => headRow.appendChild(cell));
  head.appendChild(headRow);
  const body = new Element("tbody");
  rows.forEach((cells) => {
    const row = new Element("tr");
    cells.forEach((cell) => row.appendChild(cell));
    body.appendChild(row);
  });
  return { head: head, body: body };
}

function order(body) {
  return body.children.map((row) => row.children.map((cell) => cell.textContent));
}

function aria(head) {
  return head.children[0].children.map((cell) => cell.getAttribute("aria-sort"));
}

const sandbox = {
  console: console,
  window: {},
  document: { createElement: function (tag) { return new Element(tag); } },
};
vm.createContext(sandbox);
vm.runInContext(moduleSource, sandbox);
"""


def _run_node(harness: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "Node.js is required for table sort regressions; CI must install it explicitly"
        )
    result = subprocess.run(
        [node, "-e", _PRELUDE + harness, str(_MODULE)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


class TestTableSortValues:
    """Sort keys come from data-sort-value when the display text is formatted."""

    def test_number_cells_use_parsed_text(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("1,234")], [td("$7.00")], [td("12")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body), aria: aria(grid.head) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["$7.00"], ["12"], ["1,234"]]
        assert result["aria"] == ["ascending"]

    def test_sort_value_overrides_display_text(self) -> None:
        harness = r"""
const grid = table(
  [th("Tokens", "number")],
  [[td("3 / 5", "5")], [td("10 / 2", "2")], [td("0 / 9", "9")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["10 / 2"], ["3 / 5"], ["0 / 9"]]

    def test_date_cells_use_raw_values_and_empty_sorts_as_oldest(self) -> None:
        harness = r"""
const grid = table(
  [th("Last used", "date")],
  [
    [td("Jan 2, 2026", "2026-01-02T00:00:00Z")],
    [td("never", "")],
    [td("Oct 14, 2026", "2026-10-14T00:00:00Z")],
  ],
);
const header = grid.head.children[0].children[0];
sandbox.window.tableSort.enable(grid.head, grid.body);
header.click();
const oldest = order(grid.body);
header.click();
process.stdout.write(JSON.stringify({ oldest: oldest, newest: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["oldest"] == [["never"], ["Jan 2, 2026"], ["Oct 14, 2026"]]
        assert result["newest"] == [["Oct 14, 2026"], ["Jan 2, 2026"], ["never"]]

    def test_text_cells_ignore_case(self) -> None:
        harness = r"""
const grid = table(
  [th("User", "text")],
  [[td("bob")], [td("Alice")], [td("carol")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["Alice"], ["bob"], ["carol"]]

    def test_composite_number_text_without_override_sorts_as_missing(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("3 / 5")], [td("15")], [td("1,234")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        # Digits are not concatenated ("35" would land between 15 and 1,234):
        # composite text without data-sort-value counts as missing.
        assert result["order"] == [["3 / 5"], ["15"], ["1,234"]]

    def test_plain_and_formatted_numbers_still_parse_without_override(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("$7.00")], [td("1,234")], [td("-3")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["-3"], ["$7.00"], ["1,234"]]


class TestTableSortActivation:
    """Activation toggles direction, aria-sort, and nothing else."""

    def test_second_activation_reverses(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("10")], [td("30")], [td("20")]],
);
const header = grid.head.children[0].children[0];
sandbox.window.tableSort.enable(grid.head, grid.body);
header.click();
const first = { order: order(grid.body), aria: aria(grid.head) };
header.click();
process.stdout.write(JSON.stringify({
  first: first,
  second: { order: order(grid.body), aria: aria(grid.head) },
}));
"""
        result = _run_node(harness)

        assert result["first"] == {
            "order": [["10"], ["20"], ["30"]],
            "aria": ["ascending"],
        }
        assert result["second"] == {
            "order": [["30"], ["20"], ["10"]],
            "aria": ["descending"],
        }

    def test_unsortable_column_stays_inert(self) -> None:
        harness = r"""
const grid = table(
  [th("User", "text"), th("Actions")],
  [[td("bob"), td("revoke")], [td("Alice"), td("revoke")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[1].click();
process.stdout.write(JSON.stringify({ order: order(grid.body), aria: aria(grid.head) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["bob", "revoke"], ["Alice", "revoke"]]
        assert result["aria"] == [None, None]

    def test_other_sortable_headers_clear_aria_sort(self) -> None:
        harness = r"""
const grid = table(
  [th("User", "text"), th("Total tokens", "number")],
  [[td("bob"), td("10")], [td("Alice"), td("30")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[1].click();
process.stdout.write(JSON.stringify({ order: order(grid.body), aria: aria(grid.head) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["bob", "10"], ["Alice", "30"]]
        assert result["aria"] == [None, "ascending"]

    def test_ties_keep_server_order_in_both_directions(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("b", "5")], [td("a", "5")], [td("c", "1")]],
);
const header = grid.head.children[0].children[0];
sandbox.window.tableSort.enable(grid.head, grid.body);
header.click();
const ascending = order(grid.body);
header.click();
process.stdout.write(JSON.stringify({ ascending: ascending, descending: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["ascending"] == [["c"], ["b"], ["a"]]
        assert result["descending"] == [["b"], ["a"], ["c"]]

    def test_empty_state_row_keeps_its_index(self) -> None:
        harness = r"""
const grid = table(
  [th("User", "text"), th("Total tokens", "number")],
  [
    [td("bob"), td("10")],
    [td("No other users yet.")],
    [td("carol"), td("30")],
  ],
);
const header = grid.head.children[0].children[1];
sandbox.window.tableSort.enable(grid.head, grid.body);
header.click();
header.click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [
            ["carol", "30"],
            ["No other users yet."],
            ["bob", "10"],
        ]

    def test_enable_without_a_header_row_is_inert(self) -> None:
        harness = r"""
const grid = table([th("User", "text")], [[td("bob")], [td("Alice")]]);
const empty = new Element("thead");
sandbox.window.tableSort.enable(undefined, grid.body);
sandbox.window.tableSort.enable(empty, grid.body);
sandbox.window.tableSort.reapply(undefined, grid.body);
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["bob"], ["Alice"]]

    def test_enable_is_idempotent(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("10")], [td("30")], [td("20")]],
);
const header = grid.head.children[0].children[0];
sandbox.window.tableSort.enable(grid.head, grid.body);
sandbox.window.tableSort.enable(grid.head, grid.body);
header.click();
const afterOne = { order: order(grid.body), aria: aria(grid.head) };
header.click();
process.stdout.write(JSON.stringify({
  arrows: header.children.filter(function (child) {
    return child.className === "th-sort-arrow";
  }).length,
  listeners: header.listeners.click.length,
  afterOne: afterOne,
  afterTwo: { order: order(grid.body), aria: aria(grid.head) },
}));
"""
        result = _run_node(harness)

        assert result["arrows"] == 1
        assert result["listeners"] == 1
        assert result["afterOne"] == {
            "order": [["10"], ["20"], ["30"]],
            "aria": ["ascending"],
        }
        assert result["afterTwo"] == {
            "order": [["30"], ["20"], ["10"]],
            "aria": ["descending"],
        }

    def test_template_sort_state_seeds_the_direction(self) -> None:
        harness = r"""
const seeded = th("Total tokens", "number");
seeded.setAttribute("aria-sort", "ascending");
const ascending = table([seeded], [[td("10")], [td("30")], [td("20")]]);
const descendingHeader = th("Total tokens", "number");
descendingHeader.setAttribute("aria-sort", "descending");
const descending = table(
  [descendingHeader],
  [[td("10")], [td("30")], [td("20")]],
);
sandbox.window.tableSort.enable(ascending.head, ascending.body);
sandbox.window.tableSort.enable(descending.head, descending.body);
ascending.head.children[0].children[0].click();
descending.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({
  seededAscending: { order: order(ascending.body), aria: aria(ascending.head) },
  seededDescending: { order: order(descending.body), aria: aria(descending.head) },
}));
"""
        result = _run_node(harness)

        # A template-declared order flips on the first activation instead of
        # repeating itself.
        assert result["seededAscending"] == {
            "order": [["30"], ["20"], ["10"]],
            "aria": ["descending"],
        }
        assert result["seededDescending"] == {
            "order": [["10"], ["20"], ["30"]],
            "aria": ["ascending"],
        }

    def test_skeleton_rows_keep_their_index(self) -> None:
        harness = r"""
const grid = table([th("Total tokens", "number")], []);
function addRow(cell) {
  const row = new Element("tr");
  row.appendChild(cell);
  return grid.body.appendChild(row);
}
addRow(td("10"));
const placeholder = addRow(td(""));
placeholder.setAttribute("aria-hidden", "true");
addRow(td("2"));
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
process.stdout.write(JSON.stringify({ order: order(grid.body) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["2"], [""], ["10"]]

    def test_empty_table_clears_aria_and_refill_restores_it(self) -> None:
        harness = r"""
const grid = table([th("Total tokens", "number")], [[td("10")], [td("20")]]);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
const sorted = aria(grid.head);
grid.body.children = [];
sandbox.window.tableSort.reapply(grid.head, grid.body);
const emptied = aria(grid.head);
const row = new Element("tr");
row.appendChild(td("5"));
grid.body.appendChild(row);
sandbox.window.tableSort.reapply(grid.head, grid.body);
process.stdout.write(JSON.stringify({
  sorted: sorted,
  emptied: emptied,
  refilled: aria(grid.head),
}));
"""
        result = _run_node(harness)

        assert result["sorted"] == ["ascending"]
        assert result["emptied"] == [None]
        assert result["refilled"] == ["ascending"]


class TestTableSortReapply:
    """The stored sort survives a poll render that replaces every row."""

    def test_reapply_restores_order_after_rerender(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("10")], [td("30")], [td("20")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
grid.head.children[0].children[0].click();
grid.head.children[0].children[0].click();

const fresh = [td("40"), td("5"), td("25")];
grid.body.children = [];
fresh.forEach((cell) => {
  const row = new Element("tr");
  row.appendChild(cell);
  grid.body.appendChild(row);
});
sandbox.window.tableSort.reapply(grid.head, grid.body);
process.stdout.write(JSON.stringify({ order: order(grid.body), aria: aria(grid.head) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["40"], ["25"], ["5"]]
        assert result["aria"] == ["descending"]

    def test_reapply_before_any_activation_keeps_server_order(self) -> None:
        harness = r"""
const grid = table(
  [th("Total tokens", "number")],
  [[td("10")], [td("30")], [td("20")]],
);
sandbox.window.tableSort.enable(grid.head, grid.body);
sandbox.window.tableSort.reapply(grid.head, grid.body);
process.stdout.write(JSON.stringify({ order: order(grid.body), aria: aria(grid.head) }));
"""
        result = _run_node(harness)

        assert result["order"] == [["10"], ["30"], ["20"]]
        assert result["aria"] == [None]
