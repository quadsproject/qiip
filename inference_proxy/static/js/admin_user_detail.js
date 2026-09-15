// ponytail: vanilla fetch + DOM, same pattern as dashboard.js/profile.js.
// Admin per-user view: identity with premium-equivalent cost estimate,
// tokens with revoke, usage rows, and the daily usage timeline.

(function () {
  "use strict";

  let pollInFlight = false;
  let requestSequence = 0;
  let lastRenderedSequence = 0;

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function tdCell(text) {
    const td = document.createElement("td");
    td.textContent = text;
    return td;
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function formatUsd(amount) {
    return "$" + Number(amount).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  async function revokeToken(token) {
    const ok = await confirmDialog({
      title: "Revoke token",
      message: "Revoke token \"" + token.name + "\"? It can no longer authenticate /v1 requests.",
      confirmLabel: "Revoke",
      danger: true,
    });
    if (!ok) return;
    try {
      const resp = await fetch("/admin/tokens/" + token.id, { method: "DELETE" });
      if (resp.ok) {
        showToast("Token revoked.", "success");
        refresh();
      } else {
        const data = await resp.json().catch(() => ({}));
        showToast(data.detail || "Failed to revoke token.", "error");
      }
    } catch (err) {
      showToast("Failed to revoke token: " + err.message, "error");
    }
  }

  function revokeButton(token) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-neutral btn-sm";
    button.textContent = token.revoked ? "Revoked" : "Revoke";
    button.disabled = !!token.revoked;
    button.addEventListener("click", () => revokeToken(token));
    return button;
  }

  function renderIdentity(detail) {
    const user = detail.user;
    document.getElementById("user-title").textContent = user.name || user.email;
    document.getElementById("user-subtitle").textContent =
      user.email + " \u00b7 " + detail.tokens.length + " tokens \u00b7 " +
      detail.totals.request_count + " requests";
    document.getElementById("user-name").textContent = user.name || user.email;
    document.getElementById("user-email").textContent = user.email;
    const avatar = document.getElementById("user-avatar");
    if (user.picture) {
      avatar.src = user.picture;
      avatar.alt = user.name || "user";
      avatar.hidden = false;
    } else {
      avatar.hidden = true;
    }
    document.getElementById("user-savings").textContent =
      "Estimated premium-equivalent cost: " + formatUsd(detail.estimated_cost_usd) +
      " \u00b7 model: " + (detail.model_label || "n/a");
  }

  function renderTokens(tokens, body) {
    clearChildren(body);
    if (tokens.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No tokens created yet.";
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    for (const token of tokens) {
      const row = document.createElement("tr");
      row.appendChild(tdCell(token.name));
      row.appendChild(tdCell(token.prefix));
      row.appendChild(tdCell(token.revoked ? "-" : formatDate(token.created_at)));
      row.appendChild(tdCell(token.last_used_at ? formatDate(token.last_used_at) : "never"));
      const statusCell = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "badge " + (token.revoked ? "badge-unhealthy" : "badge-healthy");
      badge.textContent = token.revoked ? "revoked" : "active";
      statusCell.appendChild(badge);
      row.appendChild(statusCell);
      row.appendChild(
        tdCell(
          token.endpoint_scope === null
            ? "Full access"
            : token.endpoint_scope.length
              ? token.endpoint_scope.join(", ")
              : "None"
        )
      );
      const actions = document.createElement("td");
      actions.appendChild(revokeButton(token));
      row.appendChild(actions);
      body.appendChild(row);
    }
  }

  function renderUsage(usage, totals, body) {
    clearChildren(body);
    const totalsEl = document.getElementById("usage-totals");
    if (usage.length === 0) {
      totalsEl.textContent = "No usage recorded yet.";
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No usage recorded yet.";
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    totalsEl.textContent =
      totals.request_count + " requests \u00b7 " +
      totals.prompt_tokens + " prompt tokens \u00b7 " +
      totals.completion_tokens + " completion tokens \u00b7 " +
      totals.total_tokens + " total tokens";
    for (const row of usage) {
      const tr = document.createElement("tr");
      tr.appendChild(tdCell(row.token_name || "deleted token"));
      tr.appendChild(tdCell(row.model));
      tr.appendChild(tdCell(row.endpoint));
      tr.appendChild(tdCell(String(row.request_count)));
      tr.appendChild(tdCell(String(row.prompt_tokens)));
      tr.appendChild(tdCell(String(row.completion_tokens)));
      tr.appendChild(tdCell(String(row.total_tokens)));
      body.appendChild(tr);
    }
  }

  function renderTimeline(timeline, body) {
    clearChildren(body);
    if (timeline.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 5;
      td.textContent = "No usage in the last 30 days.";
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    for (const entry of timeline) {
      const tr = document.createElement("tr");
      tr.appendChild(tdCell(entry.day));
      tr.appendChild(tdCell(String(entry.request_count)));
      tr.appendChild(tdCell(String(entry.prompt_tokens)));
      tr.appendChild(tdCell(String(entry.completion_tokens)));
      tr.appendChild(tdCell(String(entry.total_tokens)));
      body.appendChild(tr);
    }
  }

  async function refresh() {
    if (pollInFlight) return;
    pollInFlight = true;
    const sequence = ++requestSequence;
    const lastUpdatedEl = document.getElementById("last-updated");
    const warningEl = document.getElementById("poll-warning");
    try {
      const resp = await fetch("/admin/users/" + USER_ID);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const detail = await resp.json();

      if (sequence < lastRenderedSequence) return;
      lastRenderedSequence = sequence;

      renderIdentity(detail);
      renderTokens(detail.tokens, document.getElementById("tokens-table-body"));
      renderUsage(detail.usage, detail.totals, document.getElementById("usage-table-body"));
      renderTimeline(detail.timeline, document.getElementById("timeline-table-body"));

      lastUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString();
      lastUpdatedEl.className = "last-updated";
      warningEl.textContent = "";
      warningEl.className = "";
    } catch (err) {
      if (sequence >= lastRenderedSequence) {
        warningEl.textContent = "Update failed. Retrying\u2026";
        warningEl.className = "poll-warning";
      }
    } finally {
      pollInFlight = false;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    refresh();
    setInterval(refresh, POLL_INTERVAL_MS);
  });
})();
