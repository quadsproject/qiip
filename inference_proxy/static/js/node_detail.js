// ponytail: vanilla fetch + DOM, same pattern as dashboard.js

function showToast(message, type) {
  var container = document.getElementById("toast-container");
  var toast = document.createElement("div");
  toast.className = "toast toast-" + (type || "info");
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(function () { toast.classList.add("toast-visible"); });
  setTimeout(function () {
    toast.classList.remove("toast-visible");
    setTimeout(function () { toast.remove(); }, 300);
  }, 4000);
}

var ACTION_CONFIG = {
  setup: {
    method: "POST", url: function () { return "/admin/nodes/setup"; },
    body: function (id) { var b = { hostname: id }; var m = getSelectedModel(); if (m) b.model = m; return b; }, confirm: false,
    label: "Setup Node", css: "btn-setup",
    successMsg: function (id) { return "Setup started for " + id; },
  },
  teardown: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id; },
    body: null, confirm: true,
    confirmMsg: function (id) { return "Teardown " + id + "? This will drain connections and stop the container."; },
    label: "Teardown", css: "btn-teardown",
    successMsg: function (id) { return "Teardown started for " + id; },
  },
  retry: {
    method: "POST", url: function () { return "/admin/nodes/setup"; },
    body: function (id, node) { var b = { hostname: id, managed: node ? node.managed !== false : true }; var m = getSelectedModel(); if (m) b.model = m; return b; }, confirm: false,
    label: "Retry", css: "btn-retry",
    successMsg: function (id) { return "Retry started for " + id; },
  },
  cancel: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id; },
    body: null, confirm: true,
    confirmMsg: function (id) { return "Cancel provisioning for " + id + "?"; },
    label: "Cancel", css: "btn-cancel",
    successMsg: function (id) { return "Cancelled provisioning for " + id; },
  },
  force_teardown: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id + "?force=true"; },
    body: null, confirm: true,
    confirmMsg: function (id) { return "Force teardown " + id + "? This will immediately stop the container without draining."; },
    label: "Force Teardown", css: "btn-force-teardown",
    successMsg: function (id) { return "Teardown started for " + id; },
  },
};

async function handleAction(action, nodeId, node) {
  var config = ACTION_CONFIG[action];
  if (!config) return;
  if (config.confirm && !window.confirm(config.confirmMsg(nodeId))) return;
  var options = { method: config.method };
  if (config.body) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(config.body(nodeId, node));
  }
  try {
    var resp = await fetch(config.url(nodeId), options);
    if (resp.ok) {
      showToast(config.successMsg(nodeId), "success");
      logReceivedAny = false; logStreamDone = false;
      if (logSource) { logSource.close(); logSource = null; }
    } else {
      var data = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      showToast(data.detail || "HTTP " + resp.status, "error");
    }
  } catch (err) {
    showToast(config.label + " failed: " + err.message, "error");
  }
}

var ALL_ACTIONS = ["setup", "teardown", "retry", "cancel", "force_teardown"];

function createActionsDropdown(nodeId, enabledActions, node) {
  var group = document.createElement("div");
  group.className = "action-group";

  var trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "btn-action-trigger";
  trigger.textContent = "Actions ▾";
  if (enabledActions.length === 0) trigger.disabled = true;

  var menu = document.createElement("div");
  menu.className = "action-menu";

  for (var i = 0; i < ALL_ACTIONS.length; i++) {
    (function (action) {
      var config = ACTION_CONFIG[action];
      var btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = config.label;
      btn.className = config.css;
      var enabled = enabledActions.indexOf(action) !== -1;
      btn.disabled = !enabled;
      if (enabled) {
        btn.addEventListener("click", async function () {
          btn.disabled = true;
          menu.classList.remove("open");
          try { await handleAction(action, nodeId, node); } finally { btn.disabled = false; }
        });
      }
      menu.appendChild(btn);
    })(ALL_ACTIONS[i]);
  }

  trigger.addEventListener("click", function (e) {
    e.stopPropagation();
    var wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".action-menu.open").forEach(function (m) { m.classList.remove("open"); });
    if (!wasOpen) {
      menu.classList.add("open");
      var rect = trigger.getBoundingClientRect();
      menu.style.top = (rect.top - menu.offsetHeight) + "px";
      menu.style.left = (rect.right - menu.offsetWidth) + "px";
    }
  });

  group.appendChild(trigger);
  group.appendChild(menu);
  return group;
}

function stepBadgeClass(step) {
  if (step === "complete" || step === "teardown_complete") return "badge-complete";
  if (step === "failed") return "badge-failed";
  return "badge-in-progress";
}

async function refreshDetail() {
  var stateEl = document.getElementById("node-state");
  var infoBody = document.getElementById("node-info-body");
  var tasksBody = document.getElementById("tasks-table-body");
  var lastUpdatedEl = document.getElementById("last-updated");

  try {
    var [nodesResp, metricsResp, tasksResp] = await Promise.all([
      fetch("/admin/nodes"),
      fetch("/admin/metrics"),
      fetch("/admin/provisioning/tasks"),
    ]);
    if (!nodesResp.ok) throw new Error("HTTP " + nodesResp.status);
    var nodes = await nodesResp.json();
    var metrics = metricsResp.ok ? await metricsResp.json() : {};
    var allTasks = tasksResp.ok ? await tasksResp.json() : [];
    var perNode = metrics.per_node || {};

    var node = nodes.find(function (n) { return n.node_id === NODE_ID; });
    if (!node) {
      stateEl.textContent = "Node not found";
      infoBody.innerHTML = '<tr><td colspan="9">Node not found in registry</td></tr>';
    } else {
      stateEl.textContent = node.state;

      infoBody.textContent = "";
      var tr = document.createElement("tr");

      var tdGV = document.createElement("td"); tdGV.textContent = node.gpu_vendor || "—"; tr.appendChild(tdGV);
      var tdGM = document.createElement("td"); tdGM.textContent = node.gpu_model || "—"; tr.appendChild(tdGM);
      var tdEp = document.createElement("td"); tdEp.textContent = node.state === "available" ? "—" : node.endpoint; tr.appendChild(tdEp);
      var tdMo = document.createElement("td"); tdMo.textContent = node.state === "available" ? "—" : node.model; tr.appendChild(tdMo);

      var tdSt = document.createElement("td");
      var sb = document.createElement("span"); sb.className = "badge badge-" + node.state; sb.textContent = node.state;
      tdSt.appendChild(sb); tr.appendChild(tdSt);

      var tdCo = document.createElement("td"); tdCo.textContent = node.state === "available" ? "—" : node.active_connections; tr.appendChild(tdCo);

      var tdCb = document.createElement("td");
      if (node.state === "available") { tdCb.textContent = "—"; }
      else { var cb = document.createElement("span"); cb.className = "badge badge-" + node.circuit_breaker_state; cb.textContent = node.circuit_breaker_state; tdCb.appendChild(cb); }
      tr.appendChild(tdCb);

      var tdRq = document.createElement("td"); tdRq.textContent = node.state === "available" ? "—" : (perNode[node.node_id] || 0); tr.appendChild(tdRq);

      var tdAc = document.createElement("td");
      var enabledActions = node.actions || [];
      if (catalogModels.length === 0) {
        enabledActions = enabledActions.filter(function (a) { return a !== "setup"; });
      }
      tdAc.appendChild(createActionsDropdown(node.node_id, enabledActions, node));
      tr.appendChild(tdAc);

      infoBody.appendChild(tr);
    }

    // ponytail: filter tasks by hostname — matching against node_id (which is the hostname)
    var tasks = allTasks.filter(function (t) { return t.hostname === NODE_ID; });
    if (tasks.length > 0) connectLogStream();
    if (tasks.length === 0) {
      tasksBody.innerHTML = '<tr><td colspan="5">No provisioning tasks for this node</td></tr>';
    } else {
      tasksBody.textContent = "";
      for (var j = 0; j < tasks.length; j++) {
        var task = tasks[j];
        var ttr = document.createElement("tr");

        var tdStep = document.createElement("td");
        var badge = document.createElement("span"); badge.className = "badge " + stepBadgeClass(task.current_step); badge.textContent = task.current_step;
        tdStep.appendChild(badge); ttr.appendChild(tdStep);

        var tdStatus = document.createElement("td");
        if (task.failed_step) {
          var fb = document.createElement("span"); fb.className = "badge badge-failed"; fb.textContent = "failed at " + task.failed_step; tdStatus.appendChild(fb);
        } else if (task.current_step === "complete" || task.current_step === "teardown_complete") {
          var db = document.createElement("span"); db.className = "badge badge-complete"; db.textContent = task.current_step; tdStatus.appendChild(db);
        } else {
          var pb = document.createElement("span"); pb.className = "badge badge-in-progress"; pb.textContent = "in progress"; tdStatus.appendChild(pb);
        }
        ttr.appendChild(tdStatus);

        var tdErr = document.createElement("td");
        if (task.error) { tdErr.className = "error-text"; tdErr.textContent = task.error; }
        else { tdErr.textContent = "—"; }
        ttr.appendChild(tdErr);

        var tdStart = document.createElement("td"); tdStart.textContent = new Date(task.started_at).toLocaleString(); ttr.appendChild(tdStart);
        var tdUpd = document.createElement("td"); tdUpd.textContent = new Date(task.updated_at).toLocaleString(); ttr.appendChild(tdUpd);

        tasksBody.appendChild(ttr);
      }
    }

    var degraded = nodesResp.headers.get("X-Inference-Proxy-Data-Degraded");
    if (degraded === "provisioning-tasks") {
      lastUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString() +
        " — task/error details unavailable";
      lastUpdatedEl.className = "poll-warning";
    } else {
      lastUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString();
      lastUpdatedEl.className = "last-updated";
    }
  } catch (err) {
    lastUpdatedEl.textContent = "Update failed — retrying...";
  }
}

// ponytail: SSE live log viewer — poll loop triggers connection when tasks exist
var logSource = null;
var logReceivedAny = false;
var logStreamDone = false;

function connectLogStream() {
  if (logSource || logStreamDone) return;

  var output = document.getElementById("logs-output");
  var status = document.getElementById("logs-status");

  if (!logReceivedAny) output.textContent = "";
  status.textContent = "connecting";
  status.className = "badge badge-in-progress";

  var es = new EventSource("/admin/provisioning/" + encodeURIComponent(NODE_ID) + "/logs");
  logSource = es;

  es.addEventListener("open", function () {
    status.textContent = "streaming";
  });

  es.addEventListener("message", function (ev) {
    logReceivedAny = true;
    try {
      var entry = JSON.parse(ev.data);
      var line = document.createElement("div");
      line.className = "log-line";
      if (entry.level) line.dataset.level = entry.level;
      if (entry.stream) line.dataset.stream = entry.stream;

      var ts = document.createElement("span");
      ts.className = "log-ts";
      ts.textContent = new Date(entry.ts).toLocaleTimeString();
      line.appendChild(ts);

      var msg = document.createElement("span");
      msg.className = "log-msg";
      msg.textContent = entry.msg;
      line.appendChild(msg);

      output.appendChild(line);
      var nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 60;
      if (nearBottom) output.scrollTop = output.scrollHeight;
    } catch (_) {}
  });

  es.addEventListener("error", function () {
    es.close();
    logSource = null;
    if (logReceivedAny) {
      logStreamDone = true;
      status.textContent = "ended";
      status.className = "badge badge-complete";
    } else {
      // ponytail: 404 or premature close — show placeholder, retry on next poll
      status.textContent = "waiting";
      status.className = "badge";
      output.innerHTML = '<span class="log-placeholder">Logs will appear here when provisioning starts.</span>';
    }
  });
}

// ponytail: module-level state for download polling and catalog cache
var catalogSetCache = new Set();
var downloadPollTimer = null;

function renderDownloadCell(td, modelName, catalogSet, downloadMap) {
  td.textContent = "";
  td.dataset.repoId = modelName;

  var dl = downloadMap[modelName];

  if (dl && dl.status === "downloading") {
    var badge = document.createElement("span");
    badge.className = "badge badge-in-progress";
    badge.textContent = "Downloading…";
    td.appendChild(badge);
  } else if ((dl && dl.status === "complete") || catalogSet.has(modelName)) {
    var badge = document.createElement("span");
    badge.className = "badge badge-complete";
    badge.textContent = "Downloaded";
    td.appendChild(badge);
  } else if (dl && dl.status === "failed") {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "badge badge-failed";
    btn.style.cursor = "pointer";
    btn.textContent = "Failed — Retry";
    btn.addEventListener("click", function () { triggerDownload(modelName, td); });
    td.appendChild(btn);
  } else {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-setup";
    btn.textContent = "Download";
    btn.addEventListener("click", function () { triggerDownload(modelName, td); });
    td.appendChild(btn);
  }
}

async function triggerDownload(repoId, td) {
  // ponytail: optimistic UI — show Downloading immediately, don't wait for poll (Pitfall 2)
  td.textContent = "";
  var badge = document.createElement("span");
  badge.className = "badge badge-in-progress";
  badge.textContent = "Downloading…";
  td.appendChild(badge);

  try {
    var resp = await fetch("/admin/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_id: repoId }),
    });
    if (resp.ok) {
      showToast("Download started for " + repoId, "success");
      startDownloadPolling();
    } else {
      var err = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      showToast(err.detail || "Download failed", "error");
      renderDownloadCell(td, repoId, catalogSetCache, { [repoId]: { status: "failed" } });
    }
  } catch (e) {
    showToast("Network error starting download", "error");
    renderDownloadCell(td, repoId, catalogSetCache, {});
  }
}

function startDownloadPolling() {
  if (downloadPollTimer) return; // ponytail: single timer guard (T-32-02, Pitfall 4)
  pollDownloadStatuses();
  downloadPollTimer = setInterval(pollDownloadStatuses, 4000);
}

async function pollDownloadStatuses() {
  try {
    var resp = await fetch("/admin/models/downloads");
    if (!resp.ok) return;
    var downloads = await resp.json();

    var downloadMap = {};
    for (var i = 0; i < downloads.length; i++) downloadMap[downloads[i].repo_id] = downloads[i];

    var cells = document.querySelectorAll("td[data-repo-id]");
    for (var j = 0; j < cells.length; j++) {
      var cell = cells[j];
      var repoId = cell.dataset.repoId;
      if (downloadMap[repoId]) {
        if (downloadMap[repoId].status === "complete") catalogSetCache.add(repoId);
        renderDownloadCell(cell, repoId, catalogSetCache, downloadMap);
      }
    }

    var anyActive = downloads.some(function (d) { return d.status === "downloading"; });
    if (!anyActive) {
      clearInterval(downloadPollTimer);
      downloadPollTimer = null;
    }
  } catch (_) { /* poll failure is silent — next tick retries */ }
}

// ponytail: on-demand fetch, not polled — each call triggers SSH+llmfit on remote host
async function loadRecommendations() {
  var btn = document.getElementById("load-recs-btn");
  var content = document.getElementById("recs-content");
  var hwSummary = document.getElementById("recs-hw-summary");

  btn.disabled = true;
  btn.textContent = "Loading...";
  content.innerHTML = '<span style="color:var(--muted);font-size:0.875rem">Fetching recommendations...</span>';

  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/recommendations");
    if (!resp.ok) {
      var err = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      var msgs = {
        timeout: "llmfit timed out on " + NODE_ID + ". The node may be under heavy load.",
        parse_error: "Failed to parse llmfit output. The tool may need updating.",
        connection_error: "Cannot reach " + NODE_ID + " via SSH. Check connectivity.",
        ssh_error: "llmfit command failed on " + NODE_ID + ".",
      };
      showToast(msgs[err.error_type] || err.detail || "Failed to load recommendations", "error");
      content.textContent = "";
      var errSpan = document.createElement("span");
      errSpan.className = "error-text";
      errSpan.textContent = err.detail || "Failed to load";
      content.appendChild(errSpan);
      btn.textContent = "Retry";
      btn.disabled = false;
      return;
    }

    var data = await resp.json();

    // ponytail: fetch catalog + downloads in parallel, graceful degradation per D-01/Pitfall 1
    var catalogSet = new Set();
    var downloadMap = {};
    try {
      var catalogResp = await fetch("/admin/models/catalog");
      if (catalogResp.ok) {
        var catalogData = await catalogResp.json();
        catalogSet = new Set(catalogData.models.map(function (m) { return m.repo_id; }));
      }
    } catch (_) { /* catalog unavailable — all models show Download button */ }
    try {
      var dlResp = await fetch("/admin/models/downloads");
      if (dlResp.ok) {
        var dlArray = await dlResp.json();
        for (var d = 0; d < dlArray.length; d++) downloadMap[dlArray[d].repo_id] = dlArray[d];
      }
    } catch (_) { /* downloads unavailable — no status overlay */ }
    catalogSetCache = catalogSet;

    // Hardware summary
    var sys = data.system;
    hwSummary.style.display = "";
    hwSummary.textContent = "";
    var hwParts = [
      ["GPU: ", sys.gpu_name || "Unknown"],
      [" · VRAM: ", sys.gpu_vram_gb.toFixed(1) + " GB"],
      [" · Backend: ", sys.backend || "Unknown"],
    ];
    for (var hp = 0; hp < hwParts.length; hp++) {
      var label = document.createElement("strong");
      label.textContent = hwParts[hp][0];
      hwSummary.appendChild(label);
      hwSummary.appendChild(document.createTextNode(hwParts[hp][1]));
    }

    // Model table
    if (data.models.length === 0) {
      content.innerHTML = '<span style="color:var(--muted);font-size:0.875rem">No model recommendations available for this hardware.</span>';
    } else {
      var FIT_BADGE = { perfect: "badge-complete", good: "badge-in-progress", marginal: "badge-failed" };
      var wrap = document.createElement("div");
      wrap.className = "table-wrap";
      var tbl = document.createElement("table");
      var thead = document.createElement("thead");
      var headRow = document.createElement("tr");
      var headers = ["Model", "Category", "Score", "Fit", "Est. tok/s", "Memory", "Download"];
      for (var h = 0; h < headers.length; h++) {
        var th = document.createElement("th"); th.textContent = headers[h]; headRow.appendChild(th);
      }
      thead.appendChild(headRow);
      tbl.appendChild(thead);

      var tbody = document.createElement("tbody");
      for (var i = 0; i < data.models.length; i++) {
        var m = data.models[i];
        var badgeCls = FIT_BADGE[m.fit_level] || "badge-in-progress";
        var row = document.createElement("tr");

        var tdName = document.createElement("td"); tdName.textContent = m.name; row.appendChild(tdName);
        var tdCat = document.createElement("td"); tdCat.textContent = m.category || "—"; row.appendChild(tdCat);
        var tdScore = document.createElement("td"); tdScore.textContent = m.score.toFixed(1) + "%"; row.appendChild(tdScore);
        var tdFit = document.createElement("td");
        var fitBadge = document.createElement("span"); fitBadge.className = "badge " + badgeCls; fitBadge.textContent = m.fit_level;
        tdFit.appendChild(fitBadge); row.appendChild(tdFit);
        var tdTps = document.createElement("td"); tdTps.textContent = m.estimated_tps.toFixed(1); row.appendChild(tdTps);
        var tdMem = document.createElement("td"); tdMem.textContent = m.memory_required_gb.toFixed(1) + " GB"; row.appendChild(tdMem);

        var tdDl = document.createElement("td");
        renderDownloadCell(tdDl, m.name, catalogSet, downloadMap);
        row.appendChild(tdDl);

        tbody.appendChild(row);
      }
      tbl.appendChild(tbody);
      wrap.appendChild(tbl);
      content.textContent = "";
      content.appendChild(wrap);

      // ponytail: if any downloads are active, start polling (handles Reload case)
      var anyActive = Object.keys(downloadMap).some(function (k) { return downloadMap[k].status === "downloading"; });
      if (anyActive) startDownloadPolling();
    }

    btn.textContent = "Reload";
    btn.disabled = false;
  } catch (e) {
    showToast("Network error loading recommendations", "error");
    content.textContent = "";
    var netErr = document.createElement("span");
    netErr.className = "error-text";
    netErr.textContent = "Network error";
    content.appendChild(netErr);
    btn.textContent = "Retry";
    btn.disabled = false;
  }
}

document.getElementById("load-recs-btn").addEventListener("click", loadRecommendations);

document.addEventListener("click", function () {
  var open = document.querySelectorAll(".action-menu.open");
  for (var i = 0; i < open.length; i++) open[i].classList.remove("open");
});

var POWER_BADGE = {
  On:  { cls: "badge-complete", text: "Power: On" },
  Off: { cls: "badge-failed",   text: "Power: Off" },
};
var POWER_UNKNOWN = { cls: "badge-unknown", text: "Power: Unknown" };

var currentPowerState = null;

var POWER_ACTIONS = [
  { action: "On", label: "Power On", css: "btn-setup", confirm: false },
  { action: "ForceOff", label: "Force Off", css: "btn-teardown", confirm: true,
    confirmMsg: "Force off " + NODE_ID + "? This will immediately cut power." },
  { action: "GracefulRestart", label: "Graceful Restart", css: "btn-cancel", confirm: false },
  { action: "ForceRestart", label: "Force Restart", css: "btn-cancel", confirm: true,
    confirmMsg: "Force restart " + NODE_ID + "? This will immediately reset the node." },
];

function renderPowerButtons() {
  var container = document.getElementById("power-actions");
  container.textContent = "";
  var visible;
  if (currentPowerState === "On") {
    visible = ["ForceOff", "GracefulRestart", "ForceRestart"];
  } else if (currentPowerState === "Off") {
    visible = ["On"];
  } else {
    visible = ["On", "ForceOff", "GracefulRestart", "ForceRestart"];
  }
  for (var i = 0; i < POWER_ACTIONS.length; i++) {
    (function (pa) {
      if (visible.indexOf(pa.action) === -1) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = pa.css;
      btn.textContent = pa.label;
      btn.addEventListener("click", function () { handlePowerAction(pa.action); });
      container.appendChild(btn);
    })(POWER_ACTIONS[i]);
  }
}

async function handlePowerAction(action) {
  var config;
  for (var i = 0; i < POWER_ACTIONS.length; i++) {
    if (POWER_ACTIONS[i].action === action) { config = POWER_ACTIONS[i]; break; }
  }
  if (!config) return;
  if (config.confirm && !window.confirm(config.confirmMsg)) return;

  var el = document.querySelector("#power-state span");
  var prevClass = el.className;
  var prevText = el.textContent;

  var btns = document.querySelectorAll("#power-actions button");
  for (var j = 0; j < btns.length; j++) btns[j].disabled = true;

  el.className = "badge badge-unknown";
  el.textContent = "Power: ...";

  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action }),
    });
    if (resp.ok) {
      showToast("Power action sent: " + config.label, "success");
      refreshPowerState();
    } else {
      var data = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      showToast(data.detail || "HTTP " + resp.status, "error");
      el.className = prevClass;
      el.textContent = prevText;
      for (var k = 0; k < btns.length; k++) btns[k].disabled = false;
    }
  } catch (err) {
    showToast(config.label + " failed: " + err.message, "error");
    el.className = prevClass;
    el.textContent = prevText;
    for (var k = 0; k < btns.length; k++) btns[k].disabled = false;
  }
}

async function refreshPowerState() {
  var el = document.querySelector("#power-state span");
  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    currentPowerState = data.power_state;
    var info = POWER_BADGE[data.power_state] || POWER_UNKNOWN;
  } catch (_) {
    currentPowerState = null;
    var info = POWER_UNKNOWN;
  }
  el.className = "badge " + info.cls;
  el.textContent = info.text;
  renderPowerButtons();
}

// ponytail: model catalog state for setup config card
var catalogModels = [];

function getSelectedModel() {
  var el = document.getElementById("model-select");
  return el ? el.value : "";
}

async function fetchCatalog() {
  var sel = document.getElementById("model-select");
  var status = document.getElementById("model-status");
  try {
    var resp = await fetch("/admin/models/catalog");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    catalogModels = data.models || [];
  } catch (_) {
    catalogModels = [];
  }
  if (catalogModels.length === 0) {
    sel.style.display = "none";
    status.style.display = "inline";
    status.textContent = "No models downloaded";
  } else {
    sel.style.display = "";
    status.style.display = "none";
    sel.textContent = "";
    for (var i = 0; i < catalogModels.length; i++) {
      var opt = document.createElement("option");
      opt.value = catalogModels[i].repo_id;
      opt.textContent = catalogModels[i].repo_id;
      sel.appendChild(opt);
    }
  }
}

document.addEventListener("DOMContentLoaded", function () {
  fetchCatalog();
  refreshDetail();
  refreshPowerState();
  setInterval(refreshDetail, POLL_INTERVAL_MS);
});
