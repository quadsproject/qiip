// ponytail: vanilla fetch + DOM, no framework needed

const setupSelection = createSetupSelectionController();

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
    body: (nodeId, node) => {
      const base = { hostname: nodeId, managed: node ? node.managed !== false : true };
      return node && node.state !== "available"
        ? base
        : setupSelection.buildBody(base);
    },
    confirm: false,
    confirmMsg: null,
    danger: false,
    label: "Setup Node",
    pendingLabel: "Starting…",
    css: "btn-primary",
    successMsg: (nodeId) => `Setup started for ${nodeId}`,
  },
  teardown: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) =>
      `Teardown node ${nodeId}? This will drain connections and stop the container.`,
    danger: true,
    label: "Teardown",
    pendingLabel: "Tearing down…",
    css: "btn-danger",
    successMsg: (nodeId) => `Teardown started for ${nodeId}`,
  },
  retry: {
    method: "POST",
    url: () => "/admin/nodes/setup",
    body: (nodeId, node) => ({ hostname: nodeId, managed: node ? node.managed !== false : true }),
    confirm: false,
    confirmMsg: null,
    danger: false,
    label: "Retry",
    pendingLabel: "Retrying…",
    css: "btn-warning",
    successMsg: (nodeId) => `Retry started for ${nodeId}`,
  },
  cancel: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) => `Cancel provisioning for ${nodeId}?`,
    danger: true,
    label: "Cancel",
    pendingLabel: "Cancelling…",
    css: "btn-danger",
    successMsg: (nodeId) => `Cancelled provisioning for ${nodeId}`,
  },
  force_teardown: {
    method: "DELETE",
    url: (nodeId) => `/admin/nodes/${nodeId}?force=true`,
    body: null,
    confirm: true,
    confirmMsg: (nodeId) =>
      `Force teardown ${nodeId}? This will immediately stop the container without draining.`,
    danger: true,
    label: "Force Teardown",
    pendingLabel: "Forcing…",
    css: "btn-danger",
    successMsg: (nodeId) => `Teardown started for ${nodeId}`,
  },
};

const inFlightNodes = new Set();
const expandedErrorNodes = new Set();
let openActionMenuNode = null;
let openConfigMenuNode = null;
let dashboardPollInFlight = false;
let dashboardRequestSequence = 0;
let dashboardLastRenderedSequence = 0;

async function handleAction(action, nodeId, node, onStart) {
  const config = ACTION_CONFIG[action];
  if (!config) return;
  if (inFlightNodes.has(nodeId)) return;
  if (config.confirm) {
    const ok = await confirmDialog({
      title: config.label,
      message: config.confirmMsg(nodeId),
      confirmLabel: config.label,
      danger: config.danger,
    });
    if (!ok) return;
  }
  if (onStart) onStart();
  inFlightNodes.add(nodeId);
  const options = { method: config.method };
  if (config.body) {
    const body = config.body(nodeId, node);
    if (!body) {
      showToast(setupSelection.errorMessage(), "error");
      inFlightNodes.delete(nodeId);
      return;
    }
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
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
  } finally {
    inFlightNodes.delete(nodeId);
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
    badge.textContent = `QUADS: connected (${relativeTime(data.last_sync)} ago)`;
  } else if (data.status === "stale") {
    badge.classList.add("badge-draining");
    badge.textContent = `QUADS: stale (last sync ${relativeTime(data.last_sync)} ago)`;
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
  btn.className = "btn btn-sm " + config.css;
  btn.textContent = config.label;
  btn.disabled = inFlightNodes.has(nodeId);
  btn.addEventListener("click", async function () {
    if (inFlightNodes.has(nodeId)) return;
    btn.disabled = true;
    try {
      await handleAction(action, nodeId, node, function () {
        btn.textContent = config.pendingLabel;
        btn.setAttribute("aria-busy", "true");
      });
    } finally {
      btn.disabled = inFlightNodes.has(nodeId);
      btn.textContent = config.label;
      btn.removeAttribute("aria-busy");
    }
  });
  return btn;
}

function positionActionMenu(caret, menu) {
  const rect = caret.getBoundingClientRect();
  menu.style.top = (rect.top - menu.offsetHeight) + "px";
  menu.style.left = (rect.right - menu.offsetWidth) + "px";
}

async function refreshDashboard() {
  if (dashboardPollInFlight) return false;
  dashboardPollInFlight = true;
  const requestSequence = ++dashboardRequestSequence;
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

    if (requestSequence < dashboardLastRenderedSequence) return false;
    dashboardLastRenderedSequence = requestSequence;

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
      emptyCell.colSpan = 9;
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

        const tdEngine = document.createElement("td");
        tdEngine.textContent = formatInferenceEngine(node.engine);
        tr.appendChild(tdEngine);

        const tdModel = document.createElement("td");
        tdModel.textContent = node.state === "available" ? "—" : node.model;
        tr.appendChild(tdModel);

        const tdConfig = document.createElement("td");
        if (node.state === "healthy" && node.model) {
          const cfgDropdown = createConfigDropdown(
            window.location.origin, node.model, positionActionMenu,
            function (menuOpen) {
              openActionMenuNode = null;
              openConfigMenuNode = menuOpen ? node.node_id : null;
            }
          );
          if (openConfigMenuNode === node.node_id) {
            const cfgMenu = cfgDropdown.querySelector(".action-menu");
            const cfgTrigger = cfgDropdown.querySelector("button");
            cfgMenu.classList.add("open");
            requestAnimationFrame(function () { positionActionMenu(cfgTrigger, cfgMenu); });
          }
          tdConfig.appendChild(cfgDropdown);
        } else {
          tdConfig.textContent = "—";
        }
        tr.appendChild(tdConfig);

        const tdState = document.createElement("td");
        const stateBadge = document.createElement("span");
        stateBadge.className = `badge badge-${node.state}`;
        stateBadge.textContent = node.state;
        tdState.appendChild(stateBadge);
        tr.appendChild(tdState);

        const tdReqs = document.createElement("td");
        tdReqs.className = "num";
        tdReqs.textContent = node.state === "available" ? "—" : (perNode[node.node_id] || 0);
        tr.appendChild(tdReqs);

        const tdActions = document.createElement("td");
        const actions = node.actions || [];
        if (actions.length === 1) {
          tdActions.appendChild(createActionButton(actions[0], node.node_id, node));
        } else if (actions.length > 1) {
          const group = document.createElement("div");
          group.className = "action-group action-group--split";
          group.appendChild(createActionButton(actions[0], node.node_id, node));

          const caret = document.createElement("button");
          caret.type = "button";
          caret.className = "btn btn-sm " + ACTION_CONFIG[actions[0]].css + " action-caret";
          caret.textContent = "▾";
          const menu = document.createElement("div");
          menu.className = "action-menu";
          if (openActionMenuNode === node.node_id) {
            menu.classList.add("open");
            requestAnimationFrame(function () { positionActionMenu(caret, menu); });
          }
          for (let i = 1; i < actions.length; i++) {
            const menuBtn = createActionButton(actions[i], node.node_id, node);
            menu.appendChild(menuBtn);
          }
          caret.addEventListener("click", function (e) {
            e.stopPropagation();
            var wasOpen = menu.classList.contains("open");
            document.querySelectorAll(".action-menu.open").forEach(function (m) { m.classList.remove("open"); });
            openConfigMenuNode = null;
            if (!wasOpen) {
              openActionMenuNode = node.node_id;
              menu.classList.add("open");
              positionActionMenu(caret, menu);
            } else {
              openActionMenuNode = null;
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
          const expanded = expandedErrorNodes.has(node.node_id);
          subRow.style.display = expanded ? "table-row" : "none";

          const subTd = document.createElement("td");
          subTd.colSpan = 9;

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
          stateBadge.setAttribute("aria-expanded", String(expanded));

          function toggleSubRow() {
            const visible = subRow.style.display !== "none";
            subRow.style.display = visible ? "none" : "table-row";
            stateBadge.setAttribute("aria-expanded", String(!visible));
            if (visible) expandedErrorNodes.delete(node.node_id);
            else expandedErrorNodes.add(node.node_id);
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
    if (requestSequence >= dashboardLastRenderedSequence) {
      warningEl.textContent = "Update failed. Retrying…";
      warningEl.className = "poll-warning";
    }
  } finally {
    dashboardPollInFlight = false;
  }
  return true;
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

async function loadSetupCatalog() {
  try {
    const response = await fetch("/admin/models/catalog");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setupSelection.setCatalog(await response.json());
  } catch (_) {
    setupSelection.setCatalogUnavailable();
  }
}

document.addEventListener("DOMContentLoaded", function () {
  loadSetupCatalog();
  refreshDashboard();
  setInterval(refreshDashboard, POLL_INTERVAL_MS);

  // Dropdown dismissal on outside click
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".action-group")) {
      document.querySelectorAll(".action-menu.open").forEach(function (m) {
        m.classList.remove("open");
      });
      openActionMenuNode = null;
      openConfigMenuNode = null;
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
      const body = setupSelection.buildBody({ hostname, managed: !standalone });
      if (!body) {
        showToast(setupSelection.errorMessage(), "error");
        btn.disabled = false;
        return;
      }
      const resp = await fetch("/admin/nodes/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
