// Admin page: admin-only inference servers and admin-user role management.
// ponytail: vanilla fetch + DOM, no framework needed

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

async function setUserAdmin(user, grant, btn) {
  btn.disabled = true;
  try {
    const resp = await fetch("/admin/users/" + user.id + "/admin", {
      method: grant ? "POST" : "DELETE",
      headers: { "Content-Type": "application/json" },
    });
    if (resp.ok) {
      if (!grant && resp.headers.get("x-qiip-self-revoked") === "true") {
        // Self-revocation: the OAuth session stays valid, so land on the
        // dashboard's trimmed fleet view directly — never the sign-in page,
        // never a native Basic pop-up.
        window.location.assign("/dashboard");
        return;
      }
      showToast(
        grant ? `Admin granted to ${user.email}` : `Admin revoked for ${user.email}`,
        "success",
      );
      refreshAdminPage();
    } else {
      const data = await resp.json().catch(() => ({}));
      showToast(data.detail || `HTTP ${resp.status}`, "error");
      btn.disabled = false;
    }
  } catch (err) {
    showToast(`Role change failed: ${err.message}`, "error");
    btn.disabled = false;
  }
}

function renderAdminUsers(users) {
  const tbody = document.getElementById("admin-user-body");
  if (!tbody || !Array.isArray(users)) return;
  clearChildren(tbody);
  if (users.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "No users yet.";
    td.className = "muted-status";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const user of users) {
    const tr = document.createElement("tr");

    const tdUser = document.createElement("td");
    const link = document.createElement("a");
    link.href = "/dashboard/users/" + encodeURIComponent(user.id);
    link.textContent = user.name || user.email;
    tdUser.appendChild(link);
    tr.appendChild(tdUser);

    const tdEmail = document.createElement("td");
    tdEmail.textContent = user.email;
    tr.appendChild(tdEmail);

    const tdAdmin = document.createElement("td");
    tdAdmin.textContent = user.is_admin ? "yes" : "no";
    tr.appendChild(tdAdmin);

    const tdActions = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm " + (user.is_admin ? "btn-danger" : "btn-primary");
    btn.textContent = user.is_admin ? "Revoke Admin" : "Grant Admin";
    btn.addEventListener("click", () => setUserAdmin(user, !user.is_admin, btn));
    tdActions.appendChild(btn);
    tr.appendChild(tdActions);

    tbody.appendChild(tr);
  }
}

async function removeAdminOnlyNode(nodeId) {
  const ok = await confirmDialog({
    title: "Remove admin-only server",
    message: `Remove ${nodeId} from the admin-only server list? The existing server will keep running.`,
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
    showToast(`${nodeId} removed`, "success");
    refreshAdminPage();
  } else {
    showToast(data.detail || `HTTP ${resp.status}`, "error");
  }
}

function renderAdminOnlyNodes(nodes) {
  const tbody = document.getElementById("admin-only-node-body");
  tbody.textContent = "";
  if (nodes.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No admin-only inference servers configured";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const node of nodes) {
    const tr = document.createElement("tr");
    const tdId = document.createElement("td");
    const idSpan = document.createElement("span");
    idSpan.className = "node-id-admin-only";
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
    btn.addEventListener("click", () => removeAdminOnlyNode(node.node_id));
    tdActions.appendChild(btn);
    tr.appendChild(tdActions);

    tbody.appendChild(tr);
  }
}

async function refreshAdminPage() {
  const statusEl = document.getElementById("admin-status");
  try {
    const nodesResp = await fetch("/admin/nodes");
    if (!nodesResp.ok) {
      throw new Error(`HTTP ${nodesResp.status}`);
    }
    const nodes = await nodesResp.json();
    const adminOnly = nodes.filter((node) => node.admin_only);
    renderAdminOnlyNodes(adminOnly);
    statusEl.textContent = `Admin data loaded: ${adminOnly.length} admin-only servers`;
    const lastUpdated = document.getElementById("last-updated");
    if (lastUpdated) {
      lastUpdated.textContent = "Updated " + new Date().toLocaleTimeString();
    }
  } catch (err) {
    statusEl.textContent = `Failed to load admin data: ${err.message}`;
  }
  // Users table is a separate concern: a failure to list users must not
  // blank out the servers table (or vice versa).
  try {
    const usersResp = await fetch("/admin/users");
    if (usersResp.ok) {
      renderAdminUsers(await usersResp.json());
    }
  } catch (_err) {
    // Keep whatever was rendered last; the next poll retries.
  }
}

document.addEventListener("DOMContentLoaded", function () {
  refreshAdminPage();
  setInterval(refreshAdminPage, POLL_INTERVAL_MS);

  document
    .getElementById("admin-only-server-form")
    .addEventListener("submit", async function (e) {
      e.preventDefault();
      const input = document.getElementById("admin-only-server-url");
      const portInput = document.getElementById("admin-only-server-port");
      const nameInput = document.getElementById("admin-only-server-name");
      const btn = document.getElementById("admin-only-server-btn");
      const hostname = input.value.trim();
      if (!hostname) return;
      btn.disabled = true;
      try {
        const body = {
          hostname: hostname,
          self_setup: true,
          admin_only: true,
        };
        const port = portInput.value.trim();
        if (port) body.port = Number(port);
        const name = nameInput.value.trim();
        if (name) body.name = name;
        const resp = await fetch("/admin/nodes/pool", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) {
          showToast(`admin-only server ${hostname} registered`, "success");
          input.value = "";
          portInput.value = "";
          nameInput.value = "";
          refreshAdminPage();
        } else {
          showToast(data.detail || `HTTP ${resp.status}`, "error");
        }
      } catch (err) {
        showToast(`Registration failed: ${err.message}`, "error");
      } finally {
        btn.disabled = false;
      }
    });
});
