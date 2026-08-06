// ponytail: vanilla fetch + DOM, same pattern as dashboard.js

var setupSelection = createSetupSelectionController();

function setupActionBody(id, node) {
  var base = {
    hostname: id,
    managed: node ? node.managed !== false : true,
  };
  return node && node.state !== "available"
    ? base
    : setupSelection.buildBody(base);
}

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

function renderTableMessage(tbody, colSpan, message) {
  tbody.textContent = "";
  var row = document.createElement("tr");
  var cell = document.createElement("td");
  cell.colSpan = colSpan;
  cell.textContent = message;
  row.appendChild(cell);
  tbody.appendChild(row);
}

function renderStatusMessage(container, message, className) {
  container.textContent = "";
  var span = document.createElement("span");
  if (className) span.className = className;
  span.textContent = message;
  container.appendChild(span);
}

var ACTION_CONFIG = {
  setup: {
    method: "POST", url: function () { return "/admin/nodes/setup"; },
    body: setupActionBody, confirm: false, danger: false,
    label: "Setup Node", pendingLabel: "Starting…", css: "btn-setup",
    successMsg: function (id) { return "Setup started for " + id; },
  },
  teardown: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id; },
    body: null, confirm: true, danger: true,
    confirmMsg: function (id) { return "Teardown " + id + "? This will drain connections and stop the container."; },
    label: "Teardown", pendingLabel: "Tearing down…", css: "btn-teardown",
    successMsg: function (id) { return "Teardown started for " + id; },
  },
  retry: {
    method: "POST", url: function () { return "/admin/nodes/setup"; },
    body: function (id, node) {
      return { hostname: id, managed: node ? node.managed !== false : true };
    }, confirm: false, danger: false,
    label: "Retry", pendingLabel: "Retrying…", css: "btn-retry",
    successMsg: function (id) { return "Retry started for " + id; },
  },
  cancel: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id; },
    body: null, confirm: true, danger: true,
    confirmMsg: function (id) { return "Cancel provisioning for " + id + "?"; },
    label: "Cancel", pendingLabel: "Cancelling…", css: "btn-cancel",
    successMsg: function (id) { return "Cancelled provisioning for " + id; },
  },
  force_teardown: {
    method: "DELETE", url: function (id) { return "/admin/nodes/" + id + "?force=true"; },
    body: null, confirm: true, danger: true,
    confirmMsg: function (id) { return "Force teardown " + id + "? This will immediately stop the container without draining."; },
    label: "Force Teardown", pendingLabel: "Forcing…", css: "btn-force-teardown",
    successMsg: function (id) { return "Teardown started for " + id; },
  },
};

async function handleAction(action, nodeId, node, onStart) {
  var config = ACTION_CONFIG[action];
  if (!config) return;
  if (config.confirm) {
    var ok = await confirmDialog({
      title: config.label,
      message: config.confirmMsg(nodeId),
      confirmLabel: config.label,
      danger: config.danger,
    });
    if (!ok) return;
  }
  if (onStart) onStart();
  var options = { method: config.method };
  if (config.body) {
    var body = config.body(nodeId, node);
    if (!body) {
      showToast(setupSelection.errorMessage(), "error");
      return;
    }
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  try {
    var resp = await fetch(config.url(nodeId), options);
    if (resp.ok) {
      showToast(config.successMsg(nodeId), "success");
      resetLogStreamState();
    } else {
      var data = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      showToast(data.detail || "HTTP " + resp.status, "error");
    }
  } catch (err) {
    showToast(config.label + " failed: " + err.message, "error");
  }
}

var ALL_ACTIONS = ["setup", "teardown", "retry", "cancel", "force_teardown"];

// ponytail: shared dropdown chrome (trigger + menu) reused by node actions and power controls
function buildActionMenu(items) {
  var menu = document.createElement("div");
  menu.className = "action-menu";
  items.forEach(function (item) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm " + item.variant;
    btn.textContent = item.label;
    btn.disabled = item.disabled;
    if (!item.disabled) {
      btn.addEventListener("click", function () {
        menu.classList.remove("open");
        item.onClick(btn);
      });
    }
    menu.appendChild(btn);
  });
  return menu;
}

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
          try {
            await handleAction(action, nodeId, node, function () {
              btn.textContent = config.pendingLabel;
              btn.setAttribute("aria-busy", "true");
            });
          } finally {
            btn.disabled = false;
            btn.textContent = config.label;
            btn.removeAttribute("aria-busy");
          }
        });
      }
      menu.appendChild(btn);
    })(ALL_ACTIONS[i]);
  }

function attachDropdownToggle(trigger, menu) {
  trigger.addEventListener("click", function (e) {
    e.stopPropagation();
    var wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".action-menu.open").forEach(function (m) { m.classList.remove("open"); });
    if (!wasOpen) {
      menu.classList.add("open");
      var rect = trigger.getBoundingClientRect();
      menu.style.top = rect.bottom + "px";
      menu.style.left = (rect.right - menu.offsetWidth) + "px";
    }
  });
}

function buildDropdownGroup(trigger, menu) {
  var group = document.createElement("div");
  group.className = "action-group";
  group.appendChild(trigger);
  group.appendChild(menu);
  return group;
}

function createActionsDropdown(nodeId, enabledActions, node) {
  var items = ALL_ACTIONS.map(function (action) {
    var config = ACTION_CONFIG[action];
    return {
      label: config.label,
      variant: config.css,
      disabled: enabledActions.indexOf(action) === -1,
      onClick: function (btn) {
        btn.disabled = true;
        handleAction(action, nodeId, node, function () {
          btn.textContent = config.pendingLabel;
          btn.setAttribute("aria-busy", "true");
        }).finally(function () {
          btn.disabled = false;
          btn.textContent = config.label;
          btn.removeAttribute("aria-busy");
        });
      },
    };
  });
  var menu = buildActionMenu(items);
  var trigger = buildDropdownTrigger("Actions", enabledActions.length === 0, null);
  attachDropdownToggle(trigger, menu);
  return buildDropdownGroup(trigger, menu);
}

function stepBadgeClass(step) {
  if (step === "complete" || step === "teardown_complete") return "badge-complete";
  if (step === "failed") return "badge-failed";
  return "badge-in-progress";
}

function formatRuntimeCount(value) {
  return Number(value).toLocaleString();
}

function formatRuntimeTimestamp(value) {
  var timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return "—";
  return timestamp.toLocaleString([], { timeZoneName: "short" });
}

function renderLlamaCppRuntime(node) {
  var panel = document.getElementById("llamacpp-runtime-panel");
  var status = document.getElementById("llamacpp-runtime-status");
  var values = document.getElementById("llamacpp-runtime-values");

  if (!node || node.engine !== "llama_cpp") {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  var runtime = node.llamacpp_runtime;
  if (!runtime) {
    status.hidden = false;
    status.textContent = "Runtime configuration is unavailable until the next successful managed llama.cpp setup.";
    values.hidden = true;
    return;
  }

  var requested = runtime.requested;
  var effective = runtime.effective;
  var sizingLabel = requested.sizing === "auto" ? "Automatic" : requested.sizing;
  var minimumFree = Math.min.apply(null, runtime.gpus.map(function (gpu) { return gpu.free_mib; }));
  var minimumHeadroom = minimumFree - requested.fit_target_mib;
  status.hidden = true;
  status.textContent = "";
  values.hidden = false;
  document.getElementById("llamacpp-runtime-min-free").textContent = formatRuntimeCount(minimumFree) + " MiB";
  document.getElementById("llamacpp-runtime-min-headroom").textContent = formatRuntimeCount(minimumHeadroom) + " MiB above target";
  document.getElementById("llamacpp-runtime-sizing").textContent = sizingLabel;
  document.getElementById("llamacpp-runtime-context").textContent = formatRuntimeCount(effective.context_per_slot) + " tokens";
  document.getElementById("llamacpp-runtime-train-context").textContent = formatRuntimeCount(effective.train_context) + " tokens";
  document.getElementById("llamacpp-runtime-slot-limit").textContent = formatRuntimeCount(effective.slot_context_limit) + " tokens";
  document.getElementById("llamacpp-runtime-slots").textContent = formatRuntimeCount(effective.slots);
  document.getElementById("llamacpp-runtime-aggregate").textContent = formatRuntimeCount(effective.aggregate_context) + " tokens";
  document.getElementById("llamacpp-runtime-kv").textContent = effective.cache_type_k.toUpperCase() + " / " + effective.cache_type_v.toUpperCase();
  document.getElementById("llamacpp-runtime-flash").textContent = effective.flash_attn === "on" ? "On" : "Auto";
  document.getElementById("llamacpp-runtime-offload").textContent = effective.gpu_layers + " / " + effective.total_layers + " layers";
  document.getElementById("llamacpp-runtime-reserve").textContent = formatRuntimeCount(requested.fit_target_mib) + " MiB per GPU";
  document.getElementById("llamacpp-runtime-observed").textContent = "Post-load snapshot at " + formatRuntimeTimestamp(runtime.observed_at);

  var gpuList = document.getElementById("llamacpp-runtime-gpus");
  gpuList.textContent = "";
  runtime.gpus.forEach(function (gpu) {
    var item = document.createElement("li");
    var headroom = gpu.free_mib - requested.fit_target_mib;
    item.textContent = "GPU " + gpu.index + ": " +
      formatRuntimeCount(gpu.used_mib) + " MiB used, " +
      formatRuntimeCount(gpu.free_mib) + " MiB free of " +
      formatRuntimeCount(gpu.total_mib) + " MiB (" +
      formatRuntimeCount(headroom) + " MiB above target)";
    gpuList.appendChild(item);
  });
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
    var taskDataAvailable = tasksResp.ok;
    var allTasks = taskDataAvailable ? await tasksResp.json() : [];
    var perNode = metrics.per_node || {};

    var node = nodes.find(function (n) { return n.node_id === NODE_ID; });
    if (!node) {
      stateEl.textContent = "Node not found";
      renderTableMessage(infoBody, 9, "Node not found in registry");
      document.getElementById("node-actions").textContent = "";
      document.getElementById("config-download-panel").style.display = "none";
      renderLlamaCppRuntime(null);
    } else {
      stateEl.textContent = node.state;
      setupSelection.setPreferredNode(node);
      renderLlamaCppRuntime(node);

      infoBody.textContent = "";
      var tr = document.createElement("tr");

      var tdGV = document.createElement("td"); tdGV.textContent = node.gpu_vendor || "—"; tr.appendChild(tdGV);
      var tdGM = document.createElement("td"); tdGM.textContent = node.gpu_model || "—"; tr.appendChild(tdGM);
      var tdEp = document.createElement("td"); tdEp.textContent = node.state === "available" ? "—" : node.endpoint; tr.appendChild(tdEp);
      var tdMo = document.createElement("td"); tdMo.textContent = node.state === "available" ? "—" : node.model; tr.appendChild(tdMo);
      var tdEn = document.createElement("td"); tdEn.textContent = formatInferenceEngine(node.engine); tr.appendChild(tdEn);

      var tdSt = document.createElement("td");
      var sb = document.createElement("span"); sb.className = "badge badge-" + node.state; sb.textContent = node.state;
      tdSt.appendChild(sb); tr.appendChild(tdSt);

      var tdCo = document.createElement("td"); tdCo.className = "num"; tdCo.textContent = node.state === "available" ? "—" : node.active_connections; tr.appendChild(tdCo);

      var tdCb = document.createElement("td");
      if (node.state === "available") { tdCb.textContent = "—"; }
      else { var cb = document.createElement("span"); cb.className = "badge badge-" + node.circuit_breaker_state; cb.textContent = node.circuit_breaker_state; tdCb.appendChild(cb); }
      tr.appendChild(tdCb);

      var tdRq = document.createElement("td"); tdRq.className = "num"; tdRq.textContent = node.state === "available" ? "—" : (perNode[node.node_id] || 0); tr.appendChild(tdRq);

      var enabledActions = node.actions || [];
      if (!setupSelection.isValid()) {
        enabledActions = enabledActions.filter(function (a) { return a !== "setup"; });
      }
      var nodeActionsContainer = document.getElementById("node-actions");
      nodeActionsContainer.textContent = "";
      nodeActionsContainer.appendChild(createActionsDropdown(node.node_id, enabledActions, node));

      infoBody.appendChild(tr);

      var cfgPanel = document.getElementById("config-download-panel");
      var cfgHint = document.getElementById("config-download-hint");
      var cfgButtons = document.getElementById("config-download-buttons");
      if (node.state === "healthy" && node.model) {
        cfgPanel.style.display = "";
        cfgHint.textContent = "Download agent configuration pointing directly at this node (" + node.endpoint + ").";
        cfgButtons.textContent = "";
        cfgButtons.appendChild(createConfigDropdown(node.endpoint, node.model));
      } else {
        cfgPanel.style.display = "none";
      }
    }

    // ponytail: filter tasks by hostname — matching against node_id (which is the hostname)
    var tasks = allTasks.filter(function (t) { return t.hostname === NODE_ID; });
    if (taskDataAvailable) updateLogTaskState(tasks);
    else markLogTaskStateUnavailable();
    if (tasks.length > 0 && (!logTaskTerminal || !logStreamStarted)) connectLogStream();
    if (tasks.length === 0) {
      renderTableMessage(tasksBody, 5, "No provisioning tasks for this node");
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
        " (task/error details unavailable)";
      lastUpdatedEl.className = "poll-warning";
    } else {
      lastUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString();
      lastUpdatedEl.className = "last-updated";
    }
  } catch (err) {
    markLogTaskStateUnavailable();
    lastUpdatedEl.textContent = "Update failed. Retrying…";
  }
}

// ponytail: SSE live log viewer — poll loop triggers connection when tasks exist
var logSource = null;
var logReceivedAny = false;
var logStreamDone = false;
var logTaskTerminal = false;
var logTaskStateKnown = false;
var logTaskObserved = false;
var logReconnectTimer = null;
var logReconnectAttempts = 0;
var logReconnectStartedAt = null;
var logSeenEntries = new Set();
var logStreamStarted = false;
var LOG_RECONNECT_BASE_MS = 1000;
var LOG_RECONNECT_MAX_DELAY_MS = 30000;
var LOG_RECONNECT_MAX_ELAPSED_MS = 5 * 60 * 1000;

function isTerminalTask(task) {
  return ["complete", "failed", "teardown_complete"].indexOf(task.current_step) !== -1;
}

function updateLogTaskState(tasks) {
  logTaskStateKnown = true;
  if (tasks.length > 0) logTaskObserved = true;
  logTaskTerminal = logTaskObserved && (tasks.length === 0 || tasks.every(isTerminalTask));
  if (logTaskTerminal) {
    // Let an active stream deliver its final buffered entries. If it already
    // disconnected, terminal task state cancels the pending reconnect.
    if (logStreamStarted && logSource === null) {
      finishLogStream("ended", "badge badge-complete");
    }
    return;
  }

  // A successful task poll proves the operation is still observable. A later
  // task-API outage gets a fresh, bounded reconnect window.
  logReconnectStartedAt = null;
  logReconnectAttempts = 0;
}

function markLogTaskStateUnavailable() {
  logTaskStateKnown = false;
}

function finishLogStream(message, className) {
  if (logReconnectTimer !== null) {
    clearTimeout(logReconnectTimer);
    logReconnectTimer = null;
  }
  if (logSource) {
    logSource.close();
    logSource = null;
  }
  logStreamDone = true;
  var status = document.getElementById("logs-status");
  status.textContent = message;
  status.className = className;
}

function resetLogStreamState() {
  if (logReconnectTimer !== null) clearTimeout(logReconnectTimer);
  if (logSource) logSource.close();
  logSource = null;
  logReconnectTimer = null;
  logReceivedAny = false;
  logStreamDone = false;
  logTaskTerminal = false;
  logTaskStateKnown = false;
  logTaskObserved = false;
  logReconnectAttempts = 0;
  logReconnectStartedAt = null;
  logSeenEntries = new Set();
  logStreamStarted = false;
}

function scheduleLogReconnect() {
  if (logStreamDone || logReconnectTimer !== null) return;

  var now = Date.now();
  if (logReconnectStartedAt === null) logReconnectStartedAt = now;
  var elapsed = now - logReconnectStartedAt;
  if (elapsed >= LOG_RECONNECT_MAX_ELAPSED_MS) {
    finishLogStream("status unavailable, reload to retry", "badge badge-failed");
    return;
  }

  logReconnectAttempts += 1;
  var delay = Math.min(
    LOG_RECONNECT_BASE_MS * Math.pow(2, Math.min(logReconnectAttempts - 1, 15)),
    LOG_RECONNECT_MAX_DELAY_MS,
    LOG_RECONNECT_MAX_ELAPSED_MS - elapsed,
  );
  var status = document.getElementById("logs-status");
  status.textContent = logTaskStateKnown ? "reconnecting" : "status unavailable, retrying";
  status.className = "badge badge-in-progress";
  logReconnectTimer = setTimeout(function () {
    logReconnectTimer = null;
    connectLogStream();
  }, delay);
}

function connectLogStream() {
  if (logSource || logStreamDone) return;

  var output = document.getElementById("logs-output");
  var status = document.getElementById("logs-status");

  if (!logReceivedAny) output.textContent = "";
  status.textContent = "connecting";
  status.className = "badge badge-in-progress";

  var es = new EventSource("/admin/provisioning/" + encodeURIComponent(NODE_ID) + "/logs");
  logSource = es;
  logStreamStarted = true;

  es.addEventListener("open", function () {
    status.textContent = "streaming";
  });

  es.addEventListener("message", function (ev) {
    try {
      var entry = JSON.parse(ev.data);
      var entryKey = JSON.stringify(entry);
      if (logSeenEntries.has(entryKey)) return;
      logSeenEntries.add(entryKey);
      logReceivedAny = true;
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
    if (logTaskTerminal) {
      finishLogStream("ended", "badge badge-complete");
      return;
    }
    if (!logReceivedAny) {
      renderStatusMessage(
        output,
        "Logs will appear here when provisioning starts.",
        "log-placeholder",
      );
    }
    scheduleLogReconnect();
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
    btn.textContent = "Failed, retry";
    btn.addEventListener("click", function () { triggerDownload(modelName, td); });
    td.appendChild(btn);
  } else {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-sm btn-primary";
    btn.textContent = "Download";
    btn.addEventListener("click", function () { triggerDownload(modelName, td); });
    td.appendChild(btn);
  }
}

function renderAvailabilityText(td, message, className) {
  td.textContent = "";
  var badge = document.createElement("span");
  badge.className = "badge " + (className || "badge-unknown");
  badge.textContent = message;
  td.appendChild(badge);
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
    var catalogChanged = false;
    for (var i = 0; i < downloads.length; i++) {
      var download = downloads[i];
      var isVllm = !download.engine || download.engine === "vllm";
      if (isVllm) downloadMap[download.repo_id] = download;
      if (download.status === "complete") {
        if (isVllm && !catalogSetCache.has(download.repo_id)) {
          catalogSetCache.add(download.repo_id);
          catalogChanged = true;
        } else if (!isVllm) {
          catalogChanged = true;
        }
      }
    }

    var cells = document.querySelectorAll("td[data-repo-id]");
    for (var j = 0; j < cells.length; j++) {
      var cell = cells[j];
      var repoId = cell.dataset.repoId;
      if (downloadMap[repoId]) {
        renderDownloadCell(cell, repoId, catalogSetCache, downloadMap);
      }
    }

    if (catalogChanged) {
      await fetchCatalog();
      await refreshDetail();
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
  renderStatusMessage(content, "Fetching recommendations...", "muted-status");

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
    var catalogArtifacts = [];
    var catalogAvailable = false;
    var downloadMap = {};
    try {
      var catalogResp = await fetch("/admin/models/catalog");
      if (catalogResp.ok) {
        var catalogData = await catalogResp.json();
        catalogSet = new Set(catalogData.models.map(function (m) { return m.repo_id; }));
        catalogArtifacts = catalogData.gguf_artifacts || [];
        catalogAvailable = true;
        setupSelection.setCatalog(catalogData);
      }
    } catch (_) { /* availability cells distinguish an unavailable catalog */ }
    try {
      var dlResp = await fetch("/admin/models/downloads");
      if (dlResp.ok) {
        var dlArray = await dlResp.json();
        for (var d = 0; d < dlArray.length; d++) {
          if (!dlArray[d].engine || dlArray[d].engine === "vllm") {
            downloadMap[dlArray[d].repo_id] = dlArray[d];
          }
        }
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
      renderStatusMessage(
        content,
        "No model recommendations available for this hardware.",
        "muted-status",
      );
    } else {
      var FIT_BADGE = { perfect: "badge-complete", good: "badge-in-progress", marginal: "badge-failed" };
      var wrap = document.createElement("div");
      wrap.className = "table-wrap";
      var tbl = document.createElement("table");
      var thead = document.createElement("thead");
      var headRow = document.createElement("tr");
      var headers = ["Model", "Engine", "Source", "Category", "Score", "Fit", "Est. tok/s", "Memory", "Availability"];
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
        var tdEngine = document.createElement("td"); tdEngine.textContent = formatRecommendationRuntime(m.runtime); row.appendChild(tdEngine);
        var sources = m.gguf_sources || [];
        var tdSource = document.createElement("td");
        tdSource.textContent = sources.length ? sources.map(function (source) {
          return (source.provider ? source.provider + " - " : "") + source.repo;
        }).join(", ") : "—";
        row.appendChild(tdSource);
        var tdCat = document.createElement("td"); tdCat.textContent = m.category || "—"; row.appendChild(tdCat);
        var tdScore = document.createElement("td"); tdScore.textContent = m.score.toFixed(1) + "%"; row.appendChild(tdScore);
        var tdFit = document.createElement("td");
        var fitBadge = document.createElement("span"); fitBadge.className = "badge " + badgeCls; fitBadge.textContent = m.fit_level;
        tdFit.appendChild(fitBadge); row.appendChild(tdFit);
        var tdTps = document.createElement("td"); tdTps.textContent = m.estimated_tps.toFixed(1); row.appendChild(tdTps);
        var tdMem = document.createElement("td"); tdMem.textContent = m.memory_required_gb.toFixed(1) + " GB"; row.appendChild(tdMem);

        var tdAvailability = document.createElement("td");
        if (m.runtime === "vllm") {
          renderDownloadCell(tdAvailability, m.name, catalogSet, downloadMap);
        } else if (m.runtime === "llama_cpp") {
          if (sources.length === 0) {
            renderAvailabilityText(tdAvailability, "No GGUF source", "badge-unknown");
          } else if (!catalogAvailable) {
            renderAvailabilityText(tdAvailability, "Catalog unavailable", "badge-failed");
          } else {
            var sourceRepos = new Set(sources.map(function (source) { return source.repo; }));
            var matchingArtifacts = catalogArtifacts.filter(function (artifact) {
              return sourceRepos.has(artifact.repo_id);
            });
            if (matchingArtifacts.length === 0) {
              renderAvailabilityText(tdAvailability, "Not downloaded", "badge-unknown");
            } else {
              renderAvailabilityText(
                tdAvailability,
                "Available (" + matchingArtifacts.length +
                  (matchingArtifacts.length === 1 ? " generation)" : " generations)"),
                "badge-complete"
              );
            }
          }
        } else if (m.runtime === "unknown") {
          renderAvailabilityText(tdAvailability, "Unknown runtime", "badge-unknown");
        } else {
          renderAvailabilityText(tdAvailability, "Unsupported", "badge-unknown");
        }
        row.appendChild(tdAvailability);

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
var POWER_UNCONFIGURED = { cls: "badge-unknown", text: "Power: unavailable" };

var currentPowerState = null;
var powerControlsState = "unknown";

var POWER_ACTIONS = [
  { action: "On", label: "Power On", pendingLabel: "Powering on…", css: "btn-setup", confirm: false, danger: false },
  { action: "ForceOff", label: "Force Off", pendingLabel: "Forcing off…", css: "btn-teardown", confirm: true, danger: true,
    confirmMsg: "Force off " + NODE_ID + "? This will immediately cut power." },
  { action: "GracefulRestart", label: "Graceful Restart", pendingLabel: "Restarting…", css: "btn-cancel", confirm: false, danger: false },
  { action: "ForceRestart", label: "Force Restart", pendingLabel: "Restarting…", css: "btn-cancel", confirm: true, danger: true,
    confirmMsg: "Force restart " + NODE_ID + "? This will immediately reset the node." },
];

function renderPowerButtons() {
  var container = document.getElementById("power-actions");
  container.textContent = "";
  var unconfigured = powerControlsState === "unconfigured";
  container.hidden = unconfigured;
  if (unconfigured) return;

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
      if (powerControlsState === "unknown") {
        btn.disabled = true;
        btn.title = "Power state is temporarily unavailable; controls are disabled.";
      }
      btn.addEventListener("click", function () { handlePowerAction(pa.action, btn); });
      container.appendChild(btn);
    })(POWER_ACTIONS[i]);
  }
}

async function handlePowerAction(action, triggerBtn) {
  var config;
  for (var i = 0; i < POWER_ACTIONS.length; i++) {
    if (POWER_ACTIONS[i].action === action) { config = POWER_ACTIONS[i]; break; }
  }
  if (!config) return;
  if (config.confirm) {
    var ok = await confirmDialog({
      title: config.label,
      message: config.confirmMsg,
      confirmLabel: config.label,
      danger: config.danger,
    });
    if (!ok) return;
  }

  var el = document.querySelector("#power-state span");
  var prevClass = el.className;
  var prevText = el.textContent;

  var btns = document.querySelectorAll("#power-actions button");
  for (var j = 0; j < btns.length; j++) btns[j].disabled = true;
  if (triggerBtn) {
    triggerBtn.textContent = config.pendingLabel;
    triggerBtn.setAttribute("aria-busy", "true");
  }

  el.className = "badge badge-unknown";
  el.textContent = "Power: …";

  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action }),
    });
    if (resp.ok) {
      showToast("Power action sent: " + config.label, "success");
      refreshPowerState(); // rebuilds #power-actions from scratch, so no manual label/disabled restore needed here
    } else {
      var data = await resp.json().catch(function () { return { detail: "HTTP " + resp.status }; });
      showToast(data.detail || "HTTP " + resp.status, "error");
      el.className = prevClass;
      el.textContent = prevText;
      for (var k = 0; k < btns.length; k++) btns[k].disabled = false;
      if (triggerBtn) { triggerBtn.textContent = config.label; triggerBtn.removeAttribute("aria-busy"); }
    }
  } catch (err) {
    showToast(config.label + " failed: " + err.message, "error");
    el.className = prevClass;
    el.textContent = prevText;
    for (var k = 0; k < btns.length; k++) btns[k].disabled = false;
    if (triggerBtn) { triggerBtn.textContent = config.label; triggerBtn.removeAttribute("aria-busy"); }
  }
}

async function refreshPowerState() {
  var el = document.querySelector("#power-state span");
  try {
    var resp = await fetch("/admin/nodes/" + encodeURIComponent(NODE_ID) + "/power");
    if (resp.status === 503) {
      currentPowerState = null;
      powerControlsState = "unconfigured";
      el.className = "badge " + POWER_UNCONFIGURED.cls;
      el.textContent = POWER_UNCONFIGURED.text;
      renderPowerButtons();
      return;
    }
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    currentPowerState = data.power_state;
    var info = POWER_BADGE[data.power_state] || POWER_UNKNOWN;
    powerControlsState = POWER_BADGE[data.power_state] ? "configured" : "unknown";
  } catch (_) {
    currentPowerState = null;
    powerControlsState = "unknown";
    var info = POWER_UNKNOWN;
  }
  el.className = "badge " + info.cls;
  el.textContent = info.text;
  renderPowerButtons();
}

async function fetchCatalog() {
  try {
    var resp = await fetch("/admin/models/catalog");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    setupSelection.setCatalog(data);
  } catch (_) {
    setupSelection.setCatalogUnavailable();
  }
}

document.addEventListener("DOMContentLoaded", function () {
  fetchCatalog().then(refreshDetail);
  refreshPowerState();
  setInterval(refreshDetail, POLL_INTERVAL_MS);
});
