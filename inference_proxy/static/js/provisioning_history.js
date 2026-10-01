"use strict";

// Separate retained history from the latest operation's live stream.
document.addEventListener("DOMContentLoaded", function () {
  var panel = document.getElementById("attempt-history-panel");
  if (!panel) return;
  var select = document.getElementById("attempt-select");
  var summary = document.getElementById("attempt-summary");
  var failureDetails = document.getElementById("attempt-failure-details");
  var diagnostics = document.getElementById("attempt-diagnostics");
  var diagnosticSources = document.getElementById("attempt-diagnostic-sources");
  var issues = document.getElementById("attempt-issues");
  var output = document.getElementById("attempt-output");
  var query = document.getElementById("attempt-query");
  var source = document.getElementById("attempt-source");
  var download = document.getElementById("attempt-download");
  var more = document.getElementById("attempt-more");
  var older = document.getElementById("attempt-older");
  var collect = document.getElementById("attempt-collect");
  var base = "/admin/provisioning/" + encodeURIComponent(NODE_ID) + "/attempts";
  var offset = 0;
  var historyOffset = 0;
  var generation = 0;

  async function request(url, options) {
    var response = await fetch(url, options);
    if (!response.ok) throw new Error("Log request failed (" + response.status + ")");
    return response.json();
  }

  function attemptUrl() { return base + "/" + encodeURIComponent(select.value); }
  function showManifest(attempt) {
    summary.textContent = (attempt.failure_summary ? attempt.failure_summary + "\n" : "") +
      attempt.status + " · " + attempt.engine + " · " +
      (attempt.model || "model not selected") + " · " + attempt.stage + " · " +
      attempt.bundle_version;
    var failure = attempt.failure;
    failureDetails.textContent = failure ? "Failed at " + failure.failed_at + " after " +
      Number(failure.duration_seconds).toFixed(1) + "s" +
      (failure.exit_code != null ? " · exit " + failure.exit_code : "") +
      (failure.signal != null ? " · signal " + failure.signal : "") +
      (failure.command ? " · command " + failure.command.stage + " (" +
        (failure.command.phase_id || failure.command.sha256) + ")" : "") : "";
    diagnostics.hidden = !attempt.diagnostics;
    diagnosticSources.textContent = "";
    Object.entries((attempt.diagnostics || {}).sources || {}).forEach(function (entry) {
      var name = entry[0], detail = entry[1];
      var item = document.createElement("li");
      item.textContent = name + ": " + detail.status.replace(/_/g, " ") +
        (detail.collected_at ? " · " + detail.collected_at : "") +
        (detail.reason ? " · " + detail.reason : "") +
        (detail.deferred ? " · retry available" : "");
      diagnosticSources.appendChild(item);
    });
    var notices = (attempt.issues || []).slice();
    if (attempt.dropped_records) notices.push(attempt.dropped_records + " gateway records evicted by retention.");
    issues.textContent = notices.join("\n");
    download.href = attemptUrl() + "/bundle";
    download.hidden = false;
    var selectedSource = source.value;
    source.textContent = "";
    var all = document.createElement("option");
    all.value = ""; all.textContent = "All sources"; source.appendChild(all);
    Object.keys(Object.assign({}, attempt.sources, attempt.remote_sources)).sort().forEach(function (name) {
      var option = document.createElement("option");
      option.value = name; option.textContent = name; source.appendChild(option);
    });
    source.value = selectedSource;
  }

  async function logs(append) {
    var current = ++generation;
    if (!select.value) return;
    if (!append) { offset = 0; output.textContent = ""; }
    more.hidden = true;
    try {
      var page = await request(attemptUrl() + "/logs?after=" + offset +
        "&q=" + encodeURIComponent(query.value) + "&source=" + encodeURIComponent(source.value));
      if (current !== generation) return;
      showManifest(page.attempt);
      page.records.forEach(function (entry) {
        var line = document.createElement("div");
        line.className = "log-line";
        line.dataset.level = entry.level;
        line.textContent = entry.ts + " [" + entry.seq + " · " + entry.stage + " · " + entry.source + "] " + entry.msg;
        output.appendChild(line);
      });
      if (!append && !page.records.length) output.textContent = "No retained records match this search.";
      offset = page.next_offset;
      more.hidden = !page.has_more;
    } catch (error) {
      if (current === generation) issues.textContent = error.message;
    }
  }

  async function history(append) {
    var previous = select.value;
    try {
      var data = await request(base + "?offset=" + (append ? historyOffset : 0));
      if (!append) { select.textContent = ""; historyOffset = 0; }
      data.attempts.forEach(function (attempt) {
        var option = document.createElement("option");
        option.value = attempt.attempt_id;
        option.textContent = new Date(attempt.started_at).toLocaleString() + " · " +
          attempt.operation + " · " + attempt.status + " · " + attempt.attempt_id.slice(0, 8);
        select.appendChild(option);
      });
      historyOffset += data.attempts.length;
      older.hidden = historyOffset >= data.total;
      if (previous && Array.from(select.options).some(function (option) { return option.value === previous; })) select.value = previous;
      collect.disabled = !select.value;
      if (!select.value) {
        summary.textContent = "No retained attempts for this node.";
        failureDetails.textContent = "";
        diagnostics.hidden = true;
        download.hidden = true;
      } else if (!append) await logs(false);
      if (data.evicted_attempts) issues.textContent += "\n" + data.evicted_attempts + " older attempt manifests evicted across this gateway.";
    } catch (error) { issues.textContent = error.message; }
  }

  select.addEventListener("change", function () { query.value = ""; source.value = ""; logs(false); });
  document.getElementById("attempt-search").addEventListener("submit", function (event) { event.preventDefault(); logs(false); });
  document.getElementById("attempt-refresh").addEventListener("click", function () { history(false); });
  older.addEventListener("click", function () { history(true); });
  more.addEventListener("click", function () { logs(true); });
  collect.addEventListener("click", async function () {
    if (!select.value) return;
    var selected = select.value;
    collect.disabled = true;
    issues.textContent = "Retrieving logs and deferred diagnostics from the node…";
    try {
      await request(attemptUrl() + "/collect", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      if (selected === select.value) await logs(false);
    } catch (error) { issues.textContent = error.message; }
    finally { collect.disabled = false; }
  });
  history(false);
});
