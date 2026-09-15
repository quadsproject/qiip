// ponytail: vanilla fetch + DOM, same pattern as dashboard.js/profile.js.
// Admin token management: per-user usage table plus the full token list,
// with revoke confirmation via the shared confirmDialog.

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

  // ponytail: only a null scope means full access; [] is pinned to nothing.
  function scopeLabel(scope) {
    if (scope === null) return "Full access";
    return scope.length ? scope.join(", ") : "None";
  }

  function userLink(userId, label) {
    const link = document.createElement("a");
    link.href = "/dashboard/users/" + encodeURIComponent(userId);
    link.textContent = label;
    return link;
  }

  async function revokeToken(token) {
    const ok = await confirmDialog({
      title: "Revoke token",
      message: "Revoke token \"" + token.name + "\" for " + token.user_email +
        "? It can no longer authenticate /v1 requests.",
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

  function renderUsers(users, body) {
    clearChildren(body);
    if (users.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 9;
      td.textContent = "No users yet.";
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    for (const user of users) {
      const row = document.createElement("tr");
      const nameCell = document.createElement("td");
      nameCell.appendChild(userLink(user.id, user.name || user.email));
      row.appendChild(nameCell);
      row.appendChild(tdCell(user.email));
      row.appendChild(tdCell(user.is_admin ? "yes" : "no"));
      row.appendChild(
        tdCell(user.active_token_count + " / " + user.token_count)
      );
      row.appendChild(tdCell(String(user.request_count)));
      row.appendChild(tdCell(String(user.prompt_tokens)));
      row.appendChild(tdCell(String(user.completion_tokens)));
      row.appendChild(tdCell(String(user.total_tokens)));
      row.appendChild(tdCell(formatUsd(user.estimated_cost_usd)));
      const actions = document.createElement("td");
      const view = document.createElement("a");
      view.href = "/dashboard/users/" + encodeURIComponent(user.id);
      view.className = "btn btn-neutral btn-sm";
      view.textContent = "View";
      actions.appendChild(view);
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className =
        "btn btn-sm " + (user.is_admin ? "btn-danger" : "btn-primary");
      toggle.textContent = user.is_admin ? "Revoke Admin" : "Grant Admin";
      toggle.addEventListener("click", async () => {
        toggle.disabled = true;
        try {
          const method = user.is_admin ? "DELETE" : "POST";
          const resp = await fetch("/admin/users/" + user.id + "/admin", {
            method: method,
            headers: { "Content-Type": "application/json" },
          });
          if (resp.ok) {
            showToast(
              user.is_admin
                ? `Admin revoked for ${user.email}`
                : `Admin granted to ${user.email}`,
              "success"
            );
            refresh();
          } else {
            const data = await resp.json().catch(() => ({}));
            showToast(data.detail || `HTTP ${resp.status}`, "error");
            toggle.disabled = false;
          }
        } catch (err) {
          showToast(`Role change failed: ${err.message}`, "error");
          toggle.disabled = false;
        }
      });
      actions.appendChild(toggle);
      row.appendChild(actions);
      body.appendChild(row);
    }
  }

  function renderTokens(tokens, body) {
    clearChildren(body);
    if (tokens.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 12;
      td.textContent = "No tokens generated yet.";
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    for (const token of tokens) {
      const row = document.createElement("tr");
      const userCell = document.createElement("td");
      userCell.appendChild(userLink(token.user_id, token.user_name || token.user_email));
      row.appendChild(userCell);
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
      row.appendChild(tdCell(scopeLabel(token.endpoint_scope)));
      row.appendChild(tdCell(String(token.request_count)));
      row.appendChild(tdCell(String(token.prompt_tokens)));
      row.appendChild(tdCell(String(token.completion_tokens)));
      row.appendChild(tdCell(String(token.total_tokens)));
      const actions = document.createElement("td");
      actions.appendChild(revokeButton(token));
      row.appendChild(actions);
      body.appendChild(row);
    }
  }

  async function refresh() {
    if (pollInFlight) return;
    pollInFlight = true;
    const sequence = ++requestSequence;
    const lastUpdatedEl = document.getElementById("last-updated");
    const warningEl = document.getElementById("poll-warning");
    try {
      const [usersResp, tokensResp] = await Promise.all([
        fetch("/admin/users"),
        fetch("/admin/tokens"),
      ]);
      if (!usersResp.ok || !tokensResp.ok) {
        throw new Error("HTTP " + (usersResp.ok ? tokensResp.status : usersResp.status));
      }
      const users = await usersResp.json();
      const tokens = await tokensResp.json();

      if (sequence < lastRenderedSequence) return;
      lastRenderedSequence = sequence;

      document.getElementById("token-count").textContent =
        tokens.length + " tokens across " + users.length + " users";
      renderUsers(users, document.getElementById("users-table-body"));
      renderTokens(tokens, document.getElementById("tokens-table-body"));

      lastUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString();
      lastUpdatedEl.className = "last-updated";
      warningEl.textContent = "";
      warningEl.className = "";
    } catch (err) {
      if (sequence >= lastRenderedSequence) {
        warningEl.textContent = "Update failed. Retrying…";
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
