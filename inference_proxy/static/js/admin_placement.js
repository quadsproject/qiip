/* Automatic model placement card on the admin page.
 *
 * Shows placement status and lets admins resume suspended hosts. Everything is written
 * with textContent, so host names and error text from the API are never
 * interpreted as markup.
 */
"use strict";

function placementSummary(data) {
  if (!data.available) return data.error || "Automatic placement is unavailable.";
  const parts = [data.enabled ? "Enabled" : "Disabled"];
  if (data.last_run_at) parts.push("last pass " + data.last_run_at);
  if (!data.enabled) parts.push("no new placements are made; existing claims are shown");
  if (data.error) parts.push("problem: " + data.error);
  return parts.join(" · ");
}

function placementProfileRows(data) {
  const usage = new Map((data.usage || []).map((item) => [item.model, item]));
  return (data.profiles || []).map((profile) => {
    const demand = usage.get(profile.model);
    return [
      profile.display_name,
      String(profile.weight),
      profile.serving + " serving, " + profile.pending + " pending, " +
        profile.failed + " failed / target " + profile.target,
      profile.artifacts_present ? "present" : "missing",
      profile.qualified_gpus.length
        ? profile.qualified_gpus.map((gpu) => gpu.toUpperCase()).join(", ")
        : "none yet",
      demand ? String(demand.recorded_requests) : "0",
      demand ? String(demand.total_tokens) : "0",
    ];
  });
}

function placementIssueRows(data) {
  const rows = [];
  for (const item of data.missing_artifacts || []) {
    rows.push(["missing file", item.profile_id,
      item.repo_id + " @ " + item.revision.slice(0, 12) + " / " + item.filename]);
  }
  for (const claim of data.claims || []) {
    if (claim.state === "active") continue;
    const detail = claim.state + ", attempt " + claim.attempts +
      (claim.last_error ? ": " + claim.last_error : "");
    rows.push([claim.state === "provisioning" ? "in progress" : "failed placement",
      claim.hostname, claim.profile_id + " (" + detail + ")"]);
  }
  for (const host of data.unreadable_claims || []) {
    rows.push(["unreadable claim", host, "the host stays blocked until it is fixed"]);
  }
  for (const item of data.skipped_hosts || []) {
    rows.push(["skipped host", item.hostname, item.reason]);
  }
  return rows;
}

function fillPlacementTable(body, rows, columns, emptyText) {
  body.replaceChildren();
  const source = rows.length ? rows : [[emptyText]];
  for (const row of source) {
    const tr = document.createElement("tr");
    row.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (!rows.length) td.colSpan = columns;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

function fillSuspendedHosts(hosts) {
  const section = document.getElementById("placement-suspensions");
  const body = document.getElementById("placement-suspension-body");
  section.hidden = hosts.length === 0;
  body.replaceChildren();
  for (const hostname of hosts) {
    const tr = document.createElement("tr");
    const host = document.createElement("td");
    host.textContent = hostname;
    const action = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-neutral";
    button.textContent = "Resume automatic placement";
    button.setAttribute("aria-label", "Resume automatic placement for " + hostname);
    button.addEventListener("click", async () => {
      const confirmed = await confirmDialog({
        title: "Resume automatic placement",
        message: hostname + " will be eligible on the next placement pass, subject to availability and configuration.",
        confirmLabel: "Resume",
        danger: false,
      });
      if (!confirmed) return;
      button.disabled = true;
      const status = document.getElementById("placement-resume-status");
      status.textContent = "Resuming " + hostname + "…";
      try {
        const response = await fetch("/admin/placement/suspensions/" + encodeURIComponent(hostname), {
          method: "DELETE",
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "HTTP " + response.status);
        }
        status.textContent = "Automatic placement resumed for " + hostname + ".";
        await refreshPlacement();
      } catch (error) {
        status.textContent = "Could not resume: " + error.message;
        button.disabled = false;
      }
    });
    action.appendChild(button);
    tr.appendChild(host);
    tr.appendChild(action);
    body.appendChild(tr);
  }
}

async function refreshPlacement() {
  const status = document.getElementById("placement-status");
  if (!status) return;
  try {
    const resp = await fetch("/admin/placement");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    status.textContent = placementSummary(data);
    fillSuspendedHosts(data.suspended_hosts || []);
    fillPlacementTable(document.getElementById("placement-profile-body"),
      placementProfileRows(data), 7, "No catalog profiles to show.");
    fillPlacementTable(document.getElementById("placement-issue-body"),
      placementIssueRows(data), 3, "Nothing needs attention.");
  } catch (err) {
    status.textContent = "Could not load placement status: " + err.message;
  }
}

if (typeof document !== "undefined") {
  refreshPlacement();
  if (typeof POLL_INTERVAL_MS !== "undefined") {
    setInterval(refreshPlacement, POLL_INTERVAL_MS);
  }
}
