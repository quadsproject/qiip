// ponytail: vanilla fetch + DOM, same pattern as profile.js/admin_tokens.js.
// Leaderboard page: usage ranking of non-admin users plus the signed-in
// user's own token manager (create replaces the single personal token,
// delete revokes it) and a harness config download.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let state = null; // /onboarding/state payload
  let me = null; // /auth/me payload

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

  async function api(method, url, body) {
    const resp = await fetch(url, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (resp.status === 401) {
      window.location.href = "/leaderboard";
      throw new Error("Signed out");
    }
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(payload.detail || "HTTP " + resp.status);
    }
    return payload;
  }

  function modelName(model) {
    return (state && state.model_names && state.model_names[model]) || model;
  }

  // ------------------------------------------------------------------
  // Leaderboard
  // ------------------------------------------------------------------

  function renderLeaderboard(rows) {
    const body = $("leaderboard-body");
    clearChildren(body);
    if (rows.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No other users yet.";
      td.className = "muted-status";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    rows.forEach((row, index) => {
      const tr = document.createElement("tr");
      tr.appendChild(tdCell(String(index + 1)));
      const user = document.createElement("td");
      user.append(row.name || row.email);
      if (me && row.id === me.id) {
        const you = document.createElement("span");
        you.className = "muted-status";
        you.textContent = " (you)";
        user.appendChild(you);
      }
      tr.appendChild(user);
      tr.appendChild(tdCell(row.active_token_count + " / " + row.token_count));
      tr.appendChild(tdCell(String(row.request_count)));
      tr.appendChild(tdCell(String(row.prompt_tokens)));
      tr.appendChild(tdCell(String(row.completion_tokens)));
      tr.appendChild(tdCell(String(row.total_tokens)));
      body.appendChild(tr);
    });
    $("board-note").textContent =
      rows.length + " users ranked by total tokens (admins excluded)";
  }

  // ------------------------------------------------------------------
  // Your tokens
  // ------------------------------------------------------------------

  function revokeButton(token) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-neutral btn-sm";
    button.textContent = token.revoked ? "Deleted" : "Delete";
    button.disabled = !!token.revoked;
    button.addEventListener("click", async () => {
      const ok = await confirmDialog({
        title: "Delete token",
        message:
          'Delete token "' +
          token.name +
          '"? It can no longer authenticate /v1 requests.',
        confirmLabel: "Delete",
        danger: true,
      });
      if (!ok) return;
      try {
        await api("DELETE", "/profile/tokens/" + token.id);
        showToast("Token deleted.", "success");
        await loadState();
      } catch (err) {
        showToast(err.message, "error");
      }
    });
    return button;
  }

  function renderTokens() {
    const body = $("tokens-table-body");
    clearChildren(body);
    const token = state.token;
    if (!token) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No token yet. Create one below.";
      td.className = "muted-status";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    const tr = document.createElement("tr");
    tr.appendChild(tdCell(token.name));
    tr.appendChild(tdCell(token.prefix));
    tr.appendChild(tdCell(formatDate(token.created_at)));
    tr.appendChild(tdCell(token.last_used_at ? formatDate(token.last_used_at) : "never"));
    const status = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "badge badge-healthy";
    badge.textContent = "active";
    status.appendChild(badge);
    tr.appendChild(status);
    tr.appendChild(
      tdCell(token.models && token.models.length ? token.models.map(modelName).join(", ") : "Every model")
    );
    const actions = document.createElement("td");
    actions.appendChild(revokeButton(token));
    tr.appendChild(actions);
    body.appendChild(tr);
  }

  // ------------------------------------------------------------------
  // Create token + config download
  // ------------------------------------------------------------------

  function fillSelect(select, options, placeholder) {
    clearChildren(select);
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = placeholder;
    select.appendChild(empty);
    for (const option of options) {
      const el = document.createElement("option");
      el.value = option.value;
      el.textContent = option.label;
      if (option.disabled) el.disabled = true;
      select.appendChild(el);
    }
  }

  function selectedModels() {
    const model = $("token-model").value;
    if (model) return [model];
    const tokenModels = (state.token && state.token.models) || [];
    return tokenModels.length ? tokenModels : (state.models || []);
  }

  function updateDownloadButton() {
    const harness = $("token-harness").value;
    $("config-download-btn").disabled = !harness || !state.token;
  }

  function wireTokenForm() {
    $("token-harness").addEventListener("change", updateDownloadButton);

    $("token-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = $("token-create-btn");
      button.disabled = true;
      try {
        const model = $("token-model").value;
        await api("POST", "/onboarding/token", {
          name: "default",
          models: model ? [model] : [],
        });
        showToast("Token created.", "success");
        await loadState();
        await loadLeaderboard();
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        button.disabled = false;
      }
    });

    $("config-download-btn").addEventListener("click", async () => {
      const harness = $("token-harness").value;
      if (!harness || !state.token) return;
      let models = selectedModels();
      const harnessDef =
        (state.harnesses || []).find((h) => h.id === harness) || null;
      if (harnessDef && !harnessDef.multi_model && models.length !== 1) {
        models = models.slice(0, 1);
      }
      if (models.length === 0) {
        showToast("Select a model first.", "warning");
        return;
      }
      const button = $("config-download-btn");
      button.disabled = true;
      try {
        const resp = await fetch("/onboarding/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ harness: harness, models: models }),
        });
        if (resp.status === 401) {
          window.location.href = "/leaderboard";
          return;
        }
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || "HTTP " + resp.status);
        }
        const disposition = resp.headers.get("content-disposition") || "";
        const match = disposition.match(/filename="?([^";]+)"?/);
        const filename = match ? match[1] : harness + "-config";
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showToast("Config downloaded.", "success");
        await loadState();
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        button.disabled = false;
        updateDownloadButton();
      }
    });
  }

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  async function loadState() {
    state = await api("GET", "/onboarding/state");
    renderTokens();
    fillSelect(
      $("token-model"),
      (state.models || []).map((model) => ({
        value: model,
        label: modelName(model),
      })),
      "All models (optional)"
    );
    fillSelect(
      $("token-harness"),
      (state.harnesses || []).map((harness) => ({
        value: harness.id,
        label: harness.label + (harness.available ? "" : " (soon)"),
        disabled: !harness.available,
      })),
      "None (skip config)"
    );
    updateDownloadButton();
  }

  async function loadLeaderboard() {
    const rows = await api("GET", "/onboarding/leaderboard");
    renderLeaderboard(rows);
  }

  async function init() {
    wireTokenForm();
    try {
      me = await api("GET", "/auth/me");
      await loadState();
      await loadLeaderboard();
      $("last-updated").textContent =
        "Updated " + new Date().toLocaleTimeString();
    } catch (err) {
      $("error-banner").textContent = "Could not load the page: " + err.message;
      $("error-banner").hidden = false;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
