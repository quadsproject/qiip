// Admin page: admin-only inference servers (user role management lives on
// the token dashboard).
// ponytail: vanilla fetch + DOM, no framework needed

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

async function removeAdminOnlyNode(nodeId) {
  const ok = await confirmDialog({
    title: "Remove admin_only server",
    message: `Remove ${nodeId} from the admin_only server list? The existing server will keep running.`,
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
    td.textContent = "No admin_only inference servers configured";
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
    statusEl.textContent = `Admin data loaded: ${adminOnly.length} admin_only servers`;
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
    .getElementById("admin-only-server-form")
    .addEventListener("submit", async function (e) {
      e.preventDefault();
      const input = document.getElementById("admin-only-server-url");
      const nameInput = document.getElementById("admin-only-server-name");
      const btn = document.getElementById("admin-only-server-btn");
      const rawUrl = input.value.trim();
      if (!rawUrl) return;
      let parsed;
      try {
        parsed = parseServerUrl(rawUrl);
      } catch (err) {
        showToast(err.message, "error");
        return;
      }
      btn.disabled = true;
      try {
        const body = {
          hostname: parsed.hostname,
          self_setup: true,
          admin_only: true,
        };
        if (parsed.port !== null) body.port = parsed.port;
        const name = nameInput.value.trim();
        if (name) body.name = name;
        const resp = await fetch("/admin/nodes/pool", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) {
          showToast(`admin_only server ${parsed.hostname} registered`, "success");
          input.value = "";
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
