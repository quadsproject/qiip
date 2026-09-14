// Admin page: hidden inference servers and admin-role users.
// ponytail: vanilla fetch + DOM, no framework needed

function showAdminToast(message, type) {
  const container = document.getElementById("toast-container");
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

function parseServerUrl(rawUrl) {
  const url = new URL(rawUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Only http(s) server URLs are supported");
  }
  if (!url.hostname) throw new Error("Server URL must include a hostname");
  return {
    hostname: url.hostname,
    port: url.port ? Number(url.port) : null,
  };
}

async function removeHiddenNode(nodeId) {
  const ok = await confirmDialog({
    title: "Remove hidden server",
    message: `Remove ${nodeId} from the hidden server list? The existing server will keep running.`,
    confirmLabel: "Remove",
    danger: false,
  });
  if (!ok) return;
  const resp = await fetch(
    "/admin/nodes/" + encodeURIComponent(nodeId) + "/pool",
    { method: "DELETE" }
  );
  const data = await resp.json().catch(() => ({}));
  if (resp.ok) {
    showAdminToast(`${nodeId} removed`, "success");
    refreshAdminPage();
  } else {
    showAdminToast(data.detail || `HTTP ${resp.status}`, "error");
  }
}

function renderHiddenNodes(nodes) {
  const tbody = document.getElementById("hidden-node-body");
  tbody.textContent = "";
  if (nodes.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No hidden inference servers configured";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const node of nodes) {
    const tr = document.createElement("tr");
    const tdId = document.createElement("td");
    const idSpan = document.createElement("span");
    idSpan.className = "node-id-hidden";
    idSpan.textContent = node.name || node.node_id;
    idSpan.title = node.node_id;
    tdId.appendChild(idSpan);
    tr.appendChild(tdId);

    const tdEndpoint = document.createElement("td");
    tdEndpoint.textContent = node.endpoint;
    tr.appendChild(tdEndpoint);

    const tdModel = document.createElement("td");
    tdModel.textContent = node.model;
    tr.appendChild(tdModel);

    const tdState = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `badge badge-${node.state}`;
    badge.textContent = node.state;
    tdState.appendChild(badge);
    tr.appendChild(tdState);

    const tdActions = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-secondary";
    btn.textContent = "Remove";
    btn.addEventListener("click", () => removeHiddenNode(node.node_id));
    tdActions.appendChild(btn);
    tr.appendChild(tdActions);

    tbody.appendChild(tr);
  }
}

function renderAdminUsers(users) {
  const tbody = document.getElementById("admin-user-body");
  tbody.textContent = "";
  if (users.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "No users have signed in yet";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const user of users) {
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    tdName.textContent = user.name || "—";
    tr.appendChild(tdName);

    const tdEmail = document.createElement("td");
    tdEmail.textContent = user.email;
    tr.appendChild(tdEmail);

    const tdAdmin = document.createElement("td");
    tdAdmin.textContent = user.is_admin ? "yes" : "no";
    tr.appendChild(tdAdmin);

    const tdActions = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "btn btn-sm " + (user.is_admin ? "btn-danger" : "btn-primary");
    btn.textContent = user.is_admin ? "Revoke Admin" : "Grant Admin";
    btn.disabled = false;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const method = user.is_admin ? "DELETE" : "POST";
        const resp = await fetch(
          "/admin/users/" + user.id + "/admin",
          {
            method: method,
            headers: { "Content-Type": "application/json" },
          }
        );
        if (resp.ok) {
          showAdminToast(
            user.is_admin
              ? `Admin revoked for ${user.email}`
              : `Admin granted to ${user.email}`,
            "success"
          );
          refreshAdminPage();
        } else {
          const data = await resp.json().catch(() => ({}));
          showAdminToast(data.detail || `HTTP ${resp.status}`, "error");
          btn.disabled = false;
        }
      } catch (err) {
        showAdminToast(`Role change failed: ${err.message}`, "error");
        btn.disabled = false;
      }
    });
    tdActions.appendChild(btn);
    tr.appendChild(tdActions);

    tbody.appendChild(tr);
  }
}

async function refreshAdminPage() {
  const statusEl = document.getElementById("admin-status");
  try {
    const [nodesResp, usersResp] = await Promise.all([
      fetch("/admin/nodes"),
      fetch("/admin/users"),
    ]);
    if (!nodesResp.ok || !usersResp.ok) {
      throw new Error(`HTTP ${nodesResp.status}/${usersResp.status}`);
    }
    const nodes = await nodesResp.json();
    const users = await usersResp.json();
    renderHiddenNodes(nodes.filter((node) => node.hidden));
    renderAdminUsers(users);
    statusEl.textContent = `Admin data loaded: ${nodes.filter((node) => node.hidden).length} hidden servers, ${users.length} users`;
    const lastUpdated = document.getElementById("last-updated");
    if (lastUpdated) {
      lastUpdated.textContent = "Updated " + new Date().toLocaleTimeString();
    }
  } catch (err) {
    statusEl.textContent = `Failed to load admin data: ${err.message}`;
  }
}

document.addEventListener("DOMContentLoaded", function () {
  refreshAdminPage();
  setInterval(refreshAdminPage, POLL_INTERVAL_MS);

  document
    .getElementById("hidden-server-form")
    .addEventListener("submit", async function (e) {
      e.preventDefault();
      const input = document.getElementById("hidden-server-url");
      const btn = document.getElementById("hidden-server-btn");
      const rawUrl = input.value.trim();
      if (!rawUrl) return;
      let parsed;
      try {
        parsed = parseServerUrl(rawUrl);
      } catch (err) {
        showAdminToast(err.message, "error");
        return;
      }
      btn.disabled = true;
      try {
        const body = {
          hostname: parsed.hostname,
          self_setup: true,
          hidden: true,
        };
        if (parsed.port !== null) body.port = parsed.port;
        const resp = await fetch("/admin/nodes/pool", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) {
          showAdminToast(`Hidden server ${parsed.hostname} registered`, "success");
          input.value = "";
          refreshAdminPage();
        } else {
          showAdminToast(data.detail || `HTTP ${resp.status}`, "error");
        }
      } catch (err) {
        showAdminToast(`Registration failed: ${err.message}`, "error");
      } finally {
        btn.disabled = false;
      }
    });
});
