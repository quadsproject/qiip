// ponytail: vanilla fetch + DOM, no framework needed

function showToast(message, type) {
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

// ponytail: data-driven action dispatch replaces per-action functions
const ACTION_CONFIG = {
  setup: {
    method: "POST",
    url: () => "/admin/nodes/setup",
    body: (nodeId, node) => ({ hostname: nodeId, managed: node ? node.managed !== false : true }),
    confirm: false,
    confirmMsg: null,
    label: "Setup Node",
    css: "btn-setup",
    successMsg: (nodeId) => `Setup started for ${nodeId}`,
  },
  teardown: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) =>
      `Teardown node ${nodeId}? This will drain connections and stop the container.`,
    label: "Teardown",
    css: "btn-teardown",
    successMsg: (nodeId) => `Teardown started for ${nodeId}`,
  },
  retry: {
    method: "POST",
    url: () => "/admin/nodes/setup",
    body: (nodeId, node) => ({ hostname: nodeId, managed: node ? node.managed !== false : true }),
    confirm: false,
    confirmMsg: null,
    label: "Retry",
    css: "btn-retry",
    successMsg: (nodeId) => `Retry started for ${nodeId}`,
  },
  cancel: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) => `Cancel provisioning for ${nodeId}?`,
    label: "Cancel",
    css: "btn-cancel",
    successMsg: (nodeId) => `Cancelled provisioning for ${nodeId}`,
  },
  force_teardown: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}?force=true`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) =>
      `Force teardown ${nodeId}? This will immediately stop the container without draining.`,
    label: "Force Teardown",
    css: "btn-force-teardown",
    successMsg: (nodeId) => `Teardown started for ${nodeId}`,
  },
};

async function handleAction(action, nodeId, node) {
  const config = ACTION_CONFIG[action];
  if (!config) return;
  if (config.confirm && !window.confirm(config.confirmMsg(nodeId))) return;
  const options = { method: config.method };
  if (config.body) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(config.body(nodeId, node));
  }
  try {
    const resp = await fetch(config.url(nodeId), options);
    if (resp.ok) {
      showToast(config.successMsg(nodeId), "success");
    } else {
      const data = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      showToast(data.detail || `HTTP ${resp.status}`, "error");
    }
  } catch (err) {
    showToast(`${config.label} failed: ${err.message}`, "error");
  }
}

function relativeTime(isoString) {
  if (!isoString) return "";
  const diffMs = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h`;
}

function renderQuadsStatus(data) {
  const el = document.getElementById("quads-status");
  el.textContent = "";
  const badge = document.createElement("span");
  badge.className = "badge";
  if (data.status === "connected") {
    badge.classList.add("badge-healthy");
    badge.textContent = `QUADS: connected — ${relativeTime(data.last_sync)} ago`;
  } else if (data.status === "stale") {
    badge.classList.add("badge-draining");
    badge.textContent = `QUADS: stale — last sync ${relativeTime(data.last_sync)} ago`;
  } else {
    badge.classList.add("badge-unhealthy");
    badge.textContent = "QUADS: unavailable";
  }
  el.appendChild(badge);
}

function createActionButton(action, nodeId, node) {
  const config = ACTION_CONFIG[action];
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = config.css;
  btn.textContent = config.label;
  btn.addEventListener("click", async function () {
    btn.disabled = true;
    try {
      await handleAction(action, nodeId, node);
    } finally {
      btn.disabled = false;
    }
  });
  return btn;
}

async function refreshDashboard() {
  const tbody = document.getElementById("node-table-body");
  const countEl = document.getElementById("node-count");
  const lastUpdatedEl = document.getElementById("last-updated");
  const warningEl = document.getElementById("poll-warning");
  try {
    const [nodesResp, metricsResp, quadsResp] = await Promise.all([
      fetch("/admin/nodes"),
      fetch("/admin/metrics"),
      fetch("/admin/quads/status"),
    ]);
    if (!nodesResp.ok) throw new Error(`HTTP ${nodesResp.status}`);
    if (!metricsResp.ok) throw new Error(`HTTP ${metricsResp.status}`);
    const nodes = await nodesResp.json();
    const metrics = await metricsResp.json();
    const perNode = metrics.per_node || {};

    // ponytail: graceful degradation if QUADS endpoint unavailable
    if (quadsResp.ok) {
      renderQuadsStatus(await quadsResp.json());
    }

    renderTaskDataWarning(nodesResp, warningEl);

    if (nodes.length === 0) {
      countEl.textContent = "0 nodes";
      tbody.textContent = "";
      const emptyRow = document.createElement("tr");
      const emptyCell = document.createElement("td");
      emptyCell.colSpan = 7;
      emptyCell.textContent = "No nodes found";
      emptyRow.appendChild(emptyCell);
      tbody.appendChild(emptyRow);
    } else {
      countEl.textContent = `${nodes.length} nodes`;
      tbody.textContent = "";

      for (const node of nodes) {
        const tr = document.createElement("tr");

        const tdId = document.createElement("td");
        const idLink = document.createElement("a");
        idLink.href = "/dashboard/nodes/" + encodeURIComponent(node.node_id);
        idLink.textContent = node.node_id.split(".")[0];
        idLink.title = node.node_id;
        tdId.appendChild(idLink);
        if (node.managed === false) {
          const tag = document.createElement("span");
          tag.className = "badge badge-standalone";
          tag.textContent = "standalone";
          tdId.appendChild(document.createTextNode(" "));
          tdId.appendChild(tag);
        }
        tr.appendChild(tdId);

        const tdGpuVendor = document.createElement("td");
        tdGpuVendor.textContent = node.gpu_vendor || "—";
        tr.appendChild(tdGpuVendor);

        const tdGpuModel = document.createElement("td");
        tdGpuModel.textContent = node.gpu_model || "—";
        tr.appendChild(tdGpuModel);

        const tdModel = document.createElement("td");
        tdModel.textContent = node.state === "available" ? "—" : node.model;
        tr.appendChild(tdModel);

        const tdState = document.createElement("td");
        const stateBadge = document.createElement("span");
        stateBadge.className = `badge badge-${node.state}`;
        stateBadge.textContent = node.state;
        tdState.appendChild(stateBadge);
        tr.appendChild(tdState);

        const tdReqs = document.createElement("td");
        tdReqs.textContent = node.state === "available" ? "—" : (perNode[node.node_id] || 0);
        tr.appendChild(tdReqs);

        const tdActions = document.createElement("td");
        const actions = node.actions || [];
        if (actions.length === 1) {
          tdActions.appendChild(createActionButton(actions[0], node.node_id, node));
        } else if (actions.length > 1) {
          const group = document.createElement("div");
          group.className = "action-group";
          group.appendChild(createActionButton(actions[0], node.node_id, node));

          const caret = document.createElement("button");
          caret.type = "button";
          caret.className = ACTION_CONFIG[actions[0]].css + " action-caret";
          caret.textContent = "▾";
          const menu = document.createElement("div");
          menu.className = "action-menu";
          for (let i = 1; i < actions.length; i++) {
            const menuBtn = createActionButton(actions[i], node.node_id, node);
            menu.appendChild(menuBtn);
          }
          caret.addEventListener("click", function (e) {
            e.stopPropagation();
            var wasOpen = menu.classList.contains("open");
            document.querySelectorAll(".action-menu.open").forEach(function (m) { m.classList.remove("open"); });
            if (!wasOpen) {
              menu.classList.add("open");
              var rect = caret.getBoundingClientRect();
              menu.style.top = (rect.top - menu.offsetHeight) + "px";
              menu.style.left = (rect.right - menu.offsetWidth) + "px";
            }
          });
          group.appendChild(caret);
          group.appendChild(menu);
          tdActions.appendChild(group);
        }
        tr.appendChild(tdActions);

        tbody.appendChild(tr);

        // ponytail: expandable error sub-row for failed nodes (D-05 through D-08)
        if (node.state === "failed" && (node.failed_step || node.error)) {
          const subRow = document.createElement("tr");
          subRow.className = "error-subrow";
          subRow.style.display = "none";

          const subTd = document.createElement("td");
          subTd.colSpan = 7;

          const detail = document.createElement("div");
          detail.className = "error-detail";

          if (node.failed_step) {
            const stepBadge = document.createElement("span");
            stepBadge.className = "badge badge-failed";
            stepBadge.textContent = "failed at " + node.failed_step;
            detail.appendChild(stepBadge);
          }

          if (node.error) {
            const errPre = document.createElement("pre");
            errPre.className = "error-message";
            errPre.textContent = node.error;
            detail.appendChild(errPre);
          }

          subTd.appendChild(detail);
          subRow.appendChild(subTd);
          tbody.appendChild(subRow);

          // Make the state badge clickable to toggle the sub-row
          stateBadge.style.cursor = "pointer";
          stateBadge.setAttribute("role", "button");
          stateBadge.setAttribute("tabindex", "0");
          stateBadge.setAttribute("aria-expanded", "false");

          function toggleSubRow() {
            const visible = subRow.style.display !== "none";
            subRow.style.display = visible ? "none" : "table-row";
            stateBadge.setAttribute("aria-expanded", String(!visible));
          }

          stateBadge.addEventListener("click", toggleSubRow);
          stateBadge.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              toggleSubRow();
            }
          });
        }
      }
    }

    lastUpdatedEl.textContent =
      "Updated " + new Date().toLocaleTimeString();
    lastUpdatedEl.className = "last-updated";
  } catch (err) {
    warningEl.textContent = "Update failed — retrying...";
    warningEl.className = "poll-warning";
  }
}

function renderTaskDataWarning(nodesResponse, warningEl) {
  const degraded = nodesResponse.headers.get("X-Inference-Proxy-Data-Degraded");
  if (degraded === "provisioning-tasks") {
    warningEl.textContent =
      "Provisioning task details are unavailable; failed step and error data may be incomplete.";
    warningEl.className = "poll-warning";
    return;
  }
  warningEl.textContent = "";
  warningEl.className = "";
}

document.addEventListener("DOMContentLoaded", function () {
  refreshDashboard();
  setInterval(refreshDashboard, POLL_INTERVAL_MS);

  // Dropdown dismissal on outside click
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".action-group")) {
      document.querySelectorAll(".action-menu.open").forEach(function (m) {
        m.classList.remove("open");
      });
    }
  });

  // Manual setup toggle (D-05)
  const toggle = document.getElementById("manual-setup-toggle");
  const setupRow = document.getElementById("manual-setup-row");
  toggle.addEventListener("click", function (e) {
    e.preventDefault();
    if (setupRow.style.display === "none") {
      setupRow.style.display = "flex";
      toggle.textContent = "- Manual setup";
    } else {
      setupRow.style.display = "none";
      toggle.textContent = "+ Manual setup";
    }
  });

  // Setup form handler
  const form = document.getElementById("setup-form");
  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const input = document.getElementById("setup-hostname");
    const btn = document.getElementById("setup-btn");
    const hostname = input.value.trim();
    if (!hostname) return;
    const standalone = document.getElementById("setup-standalone").checked;
    btn.disabled = true;
    try {
      const resp = await fetch("/admin/nodes/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hostname, managed: !standalone }),
      });
      if (resp.ok) {
        showToast(`Setup started for ${hostname}`, "success");
        input.value = "";
        setTimeout(function () { btn.disabled = false; }, 2000);
      } else {
        const data = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
        showToast(data.detail || `Error: HTTP ${resp.status}`, "error");
        btn.disabled = false;
      }
    } catch (err) {
      showToast(`Error: ${err.message}`, "error");
      btn.disabled = false;
    }
  });
});
