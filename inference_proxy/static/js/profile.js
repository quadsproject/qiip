// ponytail: vanilla fetch + DOM, same pattern as dashboard.js/models.js.
// Profile page: checks /auth/me, renders the token manager and per-token
// usage table, and drives token creation/revocation and sign-out.

(function () {
  "use strict";

  const ERROR_MESSAGES = {
    login_failed: "Sign-in was not completed. Please try again.",
    no_profile: "Google did not return a profile. Please try again.",
    unverified_email:
      "This Google account has an unverified email address and cannot sign in.",
    domain_not_allowed:
      "This Google account is not in the allowed domains for this gateway.",
    not_whitelisted: "This account is not on the gateway whitelist.",
    allowlist_unavailable:
      "The whitelist service is unavailable. Please try again later.",
  };

  const $ = (id) => document.getElementById(id);
  let pendingToken = "";

  function showToast(message, type) {
    const container = $("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast toast-" + (type || "info");
    toast.textContent = message;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("toast-visible"));
    setTimeout(() => {
      toast.classList.remove("toast-visible");
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function tr(cells) {
    const row = document.createElement("tr");
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      row.appendChild(td);
    }
    return row;
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function parseErrorParam() {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("error");
    const banner = $("error-banner");
    if (code) {
      banner.textContent = ERROR_MESSAGES[code] || "Sign-in could not be completed.";
      banner.hidden = false;
      history.replaceState(null, "", "/profile");
    }
  }

  function setGate(message, showSignInButton) {
    $("auth-gate").hidden = false;
    $("profile-content").hidden = true;
    $("auth-gate-message").textContent = message;
    $("sign-in-btn").hidden = !showSignInButton;
  }

  function renderUser(user) {
    $("user-name").textContent = user.name || user.email;
    $("user-email").textContent = user.email;
    if (user.picture) {
      $("user-avatar").src = user.picture;
      $("user-avatar").alt = user.name || "user";
      $("user-avatar").hidden = false;
    } else {
      $("user-avatar").hidden = true;
    }
    $("auth-gate").hidden = true;
    $("profile-content").hidden = false;
  }

  // ------------------------------------------------------------------
  // Tokens
  // ------------------------------------------------------------------

  function revokeButton(token) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-neutral btn-sm";
    button.textContent = token.revoked ? "Deleted" : "Revoke";
    button.disabled = !!token.revoked;
    button.addEventListener("click", () => revokeToken(token.id));
    return button;
  }

  async function loadTokens() {
    const body = $("tokens-table-body");
    let tokens = [];
    try {
      const resp = await fetch("/profile/tokens");
      if (resp.ok) tokens = await resp.json();
      else if (resp.status === 401) return showSignInRequired();
    } catch (_err) {
      // fall through to the empty-state row below
    }
    clearChildren(body);
    if (tokens.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.textContent = "No tokens yet. Create one to start tracking usage.";
      td.colSpan = 7;
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
      row.appendChild(tdCell(token.revoked ? "revoked" : "active"));
      row.appendChild(
        tdCell(
          token.endpoint_scope && token.endpoint_scope.length
            ? token.endpoint_scope.join(", ")
            : "Full access"
        )
      );
      const actions = document.createElement("td");
      actions.appendChild(revokeButton(token));
      row.appendChild(actions);
      body.appendChild(row);
    }
  }

  function tdCell(text) {
    const td = document.createElement("td");
    td.textContent = text;
    return td;
  }

  async function revokeToken(tokenId) {
    const resp = await fetch("/profile/tokens/" + tokenId, { method: "DELETE" });
    if (!resp.ok) {
      showToast("Failed to revoke token.", "error");
      return;
    }
    showToast("Token revoked.", "success");
    loadTokens();
  }

  async function loadEndpoints() {
    const select = $("token-endpoints");
    clearChildren(select);
    try {
      const resp = await fetch("/profile/endpoints");
      if (resp.ok) {
        const endpoints = await resp.json();
        if (endpoints.length === 0) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No endpoints available to pin";
          option.disabled = true;
          select.appendChild(option);
          return;
        }
        for (const endpoint of endpoints) {
          const option = document.createElement("option");
          option.value = endpoint.node_id;
          option.textContent = endpoint.model
            ? endpoint.node_id + " (" + endpoint.model + ")"
            : endpoint.node_id;
          select.appendChild(option);
        }
      }
    } catch (_err) {
      // leave the empty option in place; scoping is optional
    }
  }

  function wireTokenForm() {
    const toggle = $("token-toggle");
    const form = $("token-form");
    toggle.addEventListener("click", () => {
      form.hidden = !form.hidden;
      if (!form.hidden) {
        loadEndpoints();
        $("token-name").focus();
      }
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const name = $("token-name").value.trim();
      if (!name) return;
      const selected = Array.from($("token-endpoints").selectedOptions).map(
        (option) => option.value
      );
      const body = { name: name };
      if (selected.length > 0) body.endpoints = selected;
      const button = $("token-create-btn");
      button.disabled = true;
      try {
        const resp = await fetch("/profile/tokens", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (resp.status === 401) return showSignInRequired();
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || "token creation failed");
        }
        const created = await resp.json();
        $("token-name").value = "";
        $("token-secret-value").textContent = created.token;
        $("token-secret").hidden = false;
        pendingToken = created.token;
        showToast("Token created.", "success");
        loadTokens();
      } catch (err) {
        showToast("Failed to create token: " + err.message, "error");
      } finally {
        button.disabled = false;
      }
    });
    $("token-copy-btn").addEventListener("click", () => {
      if (!pendingToken) return;
      const done = () => {
        showToast("Token copied to clipboard.", "success");
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(pendingToken).then(done, () => fallbackCopy(done));
      } else {
        fallbackCopy(done);
      }
    });
  }

  function fallbackCopy(done) {
    const area = document.createElement("textarea");
    area.value = pendingToken;
    area.setAttribute("readonly", "");
    area.style.position = "absolute";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.select();
    try {
      document.execCommand("copy");
    } catch (_err) {
      /* clipboard unavailable */
    }
    area.remove();
    done();
  }

  // ------------------------------------------------------------------
  // Usage
  // ------------------------------------------------------------------

  async function loadUsage() {
    let data = null;
    try {
      const resp = await fetch("/profile/usage");
      if (resp.ok) data = await resp.json();
      else if (resp.status === 401) return showSignInRequired();
    } catch (_err) {
      return;
    }
    if (!data) return;
    $("usage-totals").textContent =
      data.totals.request_count +
      " requests \u00b7 " +
      data.totals.prompt_tokens +
      " prompt tokens \u00b7 " +
      data.totals.completion_tokens +
      " completion tokens \u00b7 " +
      data.totals.total_tokens +
      " total tokens";
    const body = $("usage-table-body");
    clearChildren(body);
    if (data.rows.length === 0) {
      const row = document.createElement("tr");
      const td = document.createElement("td");
      td.textContent = "No usage recorded yet.";
      td.colSpan = 7;
      td.className = "muted-status";
      row.appendChild(td);
      body.appendChild(row);
      return;
    }
    for (const row of data.rows) {
      body.appendChild(
        tr([
          row.token_name || "deleted token",
          row.model || "-",
          row.endpoint || "-",
          String(row.request_count),
          String(row.prompt_tokens),
          String(row.completion_tokens),
          String(row.total_tokens),
        ]),
      );
    }
  }

  // ------------------------------------------------------------------
  // Sign-in / sign-out
  // ------------------------------------------------------------------

  function showSignInRequired() {
    showToast("Your session expired \u2014 sign in again.", "warning");
    setGate(
      "Sign in with your Google account to create API tokens for the inference API.",
      true,
    );
  }

  function wireLogout() {
    $("logout-btn").addEventListener("click", async () => {
      await fetch("/auth/logout", { method: "POST" });
      window.location.reload();
    });
  }

  async function init() {
    parseErrorParam();
    wireTokenForm();
    wireLogout();

    let resp;
    try {
      resp = await fetch("/auth/me");
    } catch (_err) {
      setGate("Could not check your sign-in state. Please reload and try again.", false);
      return;
    }
    if (resp.status === 200) {
      renderUser(await resp.json());
      loadTokens();
      loadUsage();
      return;
    }
    if (resp.status === 503) {
      setGate("Google sign-in is not configured on this gateway.", false);
      return;
    }
    if (resp.status === 401) {
      setGate(
        "Sign in with your Google account to create API tokens for the inference API.",
        true,
      );
      return;
    }
    setGate("Could not check your sign-in state. Please reload and try again.", false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
