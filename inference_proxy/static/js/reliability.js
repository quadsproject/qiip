"use strict";

document.addEventListener("DOMContentLoaded", function () {
  var get = function (id) { return document.getElementById("reliability-" + id); };
  var generation = 0;
  function element(tag, text) {
    var node = document.createElement(tag);
    if (text != null) node.textContent = text;
    return node;
  }
  function attemptDetails(attempt) {
    var details = element("details");
    details.appendChild(element("summary", attempt.hostname + " · " + attempt.outcome + " · " +
      attempt.started_at + " · " + attempt.attempt_id));
    details.appendChild(element("p", [attempt.engine, attempt.model || "model not selected", attempt.stage,
      attempt.signature, "runtime " + attempt.runtime_version, "GPU " + attempt.gpu_family,
      attempt.os, attempt.kernel, attempt.bundle_version].join(" · ")));
    details.appendChild(element("p", "Attempt " + (attempt.attempt_number || "unknown") +
      " · origin " + (attempt.series_origin_known ? "known" : "unknown") +
      " · " + (attempt.incomplete ? "incomplete evidence" : "no recorded evidence gaps") +
      (attempt.unknown_dimensions.length ? " · unknown: " + attempt.unknown_dimensions.join(", ") : "")));
    if (attempt.original_error) details.appendChild(element("pre", attempt.original_error));
    var links = element("div"); links.className = "reliability-links";
    [["Node history", attempt.node_url], ["Attempt logs (JSON)", attempt.logs_url],
      ["Diagnostic bundle", attempt.bundle_url]].forEach(function (item) {
      var link = element("a", item[0]); link.href = item[1]; links.appendChild(link);
    });
    details.appendChild(links);
    return details;
  }
  function render(report) {
    get("outcomes").textContent = report.attempt_count + " attempts · " + Object.entries(report.outcomes)
      .map(function (entry) { return entry[1] + " " + entry[0]; }).join(" · ");
    var labels = {first_attempt_success: "First-attempt success", retry_recovery: "Retry recovery",
      cancellation: "Cancellation", unsupported: "Unsupported nodes", time_to_usable_inference: "Time to usable inference"};
    get("metrics").textContent = "";
    Object.entries(report.metrics).forEach(function (entry) {
      var key = entry[0], metric = entry[1], value;
      if (key === "time_to_usable_inference") {
        value = (metric.median_seconds == null ? "Not measured" : metric.median_seconds.toFixed(1) + "s median") +
          " · " + metric.samples + "/" + metric.eligible_successes + " successes timed";
      } else {
        value = (metric.percent == null ? "Not measured" : metric.percent.toFixed(1) + "%") +
          " · " + metric.numerator + "/" + metric.denominator;
      }
      var row = element("tr");
      [labels[key], value, metric.definition].forEach(function (text) { row.appendChild(element("td", text)); });
      get("metrics").appendChild(row);
    });
    var evidence = report.evidence;
    get("evidence").textContent = evidence.incomplete_attempts + " incomplete attempts · " +
      evidence.unknown_environment_attempts + " with unknown environment fields · " +
      evidence.unknown_origin_attempts + " with unknown series origin · " +
      evidence.evicted_attempts + " manifests evicted across this gateway · " +
      evidence.invalid_start_times + " invalid start times · " + evidence.excluded_operations + " other operations excluded";
    get("warnings").textContent = "";
    report.warnings.forEach(function (warning) { get("warnings").appendChild(element("li", warning)); });
    var attempts = new Map(report.attempts.map(function (attempt) { return [attempt.attempt_id, attempt]; }));
    get("groups").textContent = report.groups.length ? "" : "No failed or unsupported attempts in this selection.";
    report.groups.forEach(function (group) {
      var details = element("details");
      details.appendChild(element("summary", group.count + " attempts · " + Object.entries(group.dimensions)
        .map(function (entry) { return entry[0] + ": " + entry[1]; }).join(" · ") +
        " · " + group.incomplete_count + " incomplete"));
      group.attempt_ids.forEach(function (id) { details.appendChild(attemptDetails(attempts.get(id))); });
      get("groups").appendChild(details);
    });
    get("attempts").textContent = report.attempts.length ? "" : "No retained provisioning attempts match these filters.";
    report.attempts.forEach(function (attempt) { get("attempts").appendChild(attemptDetails(attempt)); });
    get("report").hidden = false;
  }
  async function refresh() {
    var current = ++generation, params = new URLSearchParams();
    get("download").hidden = true;
    get("report").hidden = true;
    get("status").textContent = "Loading report...";
    try {
      ["since", "until"].forEach(function (key) {
        if (get(key).value) params.set(key, new Date(get(key).value).toISOString());
      });
      get("hostnames").value.split(",").map(function (host) { return host.trim(); }).filter(Boolean)
        .forEach(function (host) { params.append("hostname", host); });
      params.set("group_by", get("group").value);
      var url = "/admin/provisioning/reliability?" + params.toString();
      var response = await fetch(url);
      if (!response.ok) {
        var message = "Report unavailable (" + response.status + "). Please retry.";
        try {
          var detail = (await response.json()).detail;
          if (Array.isArray(detail)) detail = detail.map(function (item) { return item && item.msg; }).filter(Boolean).join("; ");
          if (typeof detail === "string" && detail) message = detail;
        } catch (_) { /* A proxy may return a non-JSON error body. */ }
        throw new Error(message);
      }
      var report = await response.json();
      if (current !== generation) return;
      render(report);
      get("status").textContent = "Snapshot: " + report.generated_at + ". Filters use your local time; report and export timestamps are UTC. Records are selected by attempt start.";
      get("download").href = url + "&download=true";
      get("download").hidden = false;
    } catch (error) { if (current === generation) get("status").textContent = error.message; }
  }
  get("filters").addEventListener("submit", function (event) { event.preventDefault(); refresh(); });
  refresh();
});
