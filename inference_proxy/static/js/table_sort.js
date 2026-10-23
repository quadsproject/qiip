// ponytail: one shared client-side sorter. Headers opt in with
// data-sort="text|number|date" and a <button class="th-sort"> label; cells may
// carry data-sort-value when their display text is formatted (locale dates,
// USD, "active / total" counts). The button click bubbles to the <th>, which
// owns the single activation listener, so no DOM traversal is needed and page
// scripts can re-apply the sort after every poll render.
//
// The number fallback without data-sort-value reads one plain or formatted
// number ("12", "1,234", "$7.00") and sorts anything else, composite text
// ("3 / 5") included, as missing. Number cells with other formats must carry
// data-sort-value.

(function () {
  "use strict";

  var sorts = new WeakMap(); // <thead> element -> { index, kind, direction }
  var enabled = new WeakSet(); // <thead> elements already wired, for idempotence
  var NUMBER_TEXT =
    /^[^0-9+-]*([-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)[^0-9]*$/;

  function cellValue(cell, kind) {
    var override =
      cell && cell.dataset && cell.dataset.sortValue !== undefined
        ? cell.dataset.sortValue
        : undefined;
    var raw = override !== undefined ? override : cell ? cell.textContent : "";
    if (kind === "number") {
      var match = NUMBER_TEXT.exec(String(raw).trim());
      if (!match) return -Infinity;
      var value = Number(match[1].replace(/,/g, ""));
      return Number.isNaN(value) ? -Infinity : value;
    }
    if (kind === "date") {
      var stamp = Date.parse(String(raw));
      return Number.isNaN(stamp) ? -Infinity : stamp;
    }
    return String(raw).trim().toLowerCase();
  }

  function compare(a, b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }

  // Skipped rows keep their index: empty states span every column with one cell
  // and skeleton placeholders are marked aria-hidden by the templates.
  function applySort(head, body) {
    var state = sorts.get(head);
    var headRow = head ? head.children[0] : null;
    if (!state || !body || !headRow) return;
    var children = Array.prototype.slice.call(body.children);
    var data = new Set();
    var rows = [];
    children.forEach(function (row) {
      if (row.children.length !== headRow.children.length) return;
      if (row.getAttribute("aria-hidden") === "true") return;
      data.add(row);
      rows.push(row);
    });
    if (rows.length === 0) {
      clearAria(head); // nothing sortable on screen: announce no sort
      return;
    }
    setAria(head, state.index, state.direction);
    var keyed = rows.map(function (row, position) {
      return {
        row: row,
        value: cellValue(row.children[state.index], state.kind),
        position: position,
      };
    });
    keyed.sort(function (a, b) {
      var result = compare(a.value, b.value);
      if (result === 0) return a.position - b.position; // stable in both directions
      return state.direction === "descending" ? -result : result;
    });
    var next = 0;
    children
      .map(function (row) {
        return data.has(row) ? keyed[next++].row : row;
      })
      .forEach(function (row) {
        body.appendChild(row); // re-appending an existing row moves it
      });
  }

  function setAria(head, index, direction) {
    Array.prototype.forEach.call(head.children[0].children, function (th, position) {
      if (!th.dataset || !th.dataset.sort) return;
      if (position === index) th.setAttribute("aria-sort", direction);
      // APG: only the sorted column carries aria-sort.
      else th.removeAttribute("aria-sort");
    });
  }

  function clearAria(head) {
    Array.prototype.forEach.call(head.children[0].children, function (th) {
      if (th.dataset && th.dataset.sort) th.removeAttribute("aria-sort");
    });
  }

  function activate(head, body, index, kind) {
    var current = sorts.get(head);
    var direction =
      current && current.index === index && current.direction === "ascending"
        ? "descending"
        : "ascending";
    sorts.set(head, { index: index, kind: kind, direction: direction });
    applySort(head, body);
  }

  function enable(head, body) {
    var headRow = head ? head.children[0] : null;
    if (!headRow || enabled.has(head)) return; // inert without a row; idempotent
    enabled.add(head);
    Array.prototype.forEach.call(headRow.children, function (th, index) {
      var kind = th.dataset ? th.dataset.sort : null;
      if (!kind) return;
      var initial = th.getAttribute("aria-sort");
      if (initial === "ascending" || initial === "descending") {
        // The template marks a sort the server already applied; the first
        // activation flips it instead of repeating it.
        sorts.set(head, { index: index, kind: kind, direction: initial });
      }
      // The glyph lives in an aria-hidden span so it stays out of the
      // button's accessible name (AccName includes generated content).
      var arrow = document.createElement("span");
      arrow.className = "th-sort-arrow";
      arrow.setAttribute("aria-hidden", "true");
      th.appendChild(arrow);
      th.addEventListener("click", function () {
        activate(head, body, index, kind);
      });
    });
  }

  function reapply(head, body) {
    applySort(head, body);
  }

  window.tableSort = { enable: enable, reapply: reapply };
})();
