// ponytail: vanilla fetch + DOM, same pattern as dashboard.js

// Poll guards, modeled on dashboard.js: never let overlapping catalog polls
// accumulate (the catalog request scans the NFS cache and can outlive the
// polling interval), and never let an older response overwrite a newer one.
let modelsPollInFlight = false;
let modelsRequestSequence = 0;
let modelsLastRenderedSequence = 0;

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

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function tr(cells) {
  const row = document.createElement("tr");
  for (const cell of cells) {
    const td = document.createElement("td");
    if (typeof cell === "string") {
      td.textContent = cell;
    } else {
      td.appendChild(cell);
    }
    row.appendChild(td);
  }
  return row;
}

async function fetchCatalog(requestSequence) {
  try {
    const resp = await fetch("/admin/models/catalog");
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();
    if (requestSequence >= modelsLastRenderedSequence) {
      renderCatalog(data);
    }
  } catch (err) {
    if (requestSequence >= modelsLastRenderedSequence) {
      document.getElementById("model-count").textContent = "Failed to load catalog";
    }
  }
}

function renderCatalog(data) {
  const tbody = document.getElementById("catalog-table-body");
  const models = data.models || [];
  const gguf = data.gguf_artifacts || [];
  const total = models.length + gguf.length;

  document.getElementById("model-count").textContent =
    total + " model" + (total !== 1 ? "s" : "") + " cached";

  clearChildren(tbody);

  if (total === 0) {
    const empty = tr(["No models in cache"]);
    empty.firstChild.className = "empty-state";
    tbody.appendChild(empty);
    return;
  }

  for (const m of models) {
    tbody.appendChild(tr([m.repo_id]));
  }
  for (const a of gguf) {
    tbody.appendChild(tr([a.repo_id + " (" + a.entrypoint + ")"]));
  }
}

async function fetchDownloads(requestSequence) {
  try {
    const resp = await fetch("/admin/models/downloads");
    if (!resp.ok) return;
    const downloads = await resp.json();
    if (requestSequence >= modelsLastRenderedSequence) {
      renderDownloads(downloads);
    }
  } catch (_) {
    // silent — downloads section is supplementary
  }
}

// Out-of-band refresh after starting a download. It is given a newer request
// sequence so an older in-flight periodic poll cannot pass the render guard
// and overwrite the freshly rendered download list.
async function refreshDownloads() {
  const requestSequence = ++modelsRequestSequence;
  await fetchDownloads(requestSequence);
  if (requestSequence >= modelsLastRenderedSequence) {
    modelsLastRenderedSequence = requestSequence;
  }
}

const STATUS_BADGE = {
  complete: "badge-healthy",
  failed: "badge-failed",
  downloading: "badge-provisioning",
};

function renderDownloads(downloads) {
  const section = document.getElementById("downloads-section");
  const tbody = document.getElementById("downloads-table-body");
  if (!downloads.length) {
    section.style.display = "none";
    return;
  }
  section.style.display = "";
  clearChildren(tbody);
  for (const d of downloads) {
    const badge = document.createElement("span");
    badge.className = "badge " + (STATUS_BADGE[d.status] || "");
    badge.textContent = d.status;
    tbody.appendChild(
      tr([
        d.repo_id,
        d.resolved_revision || d.requested_revision || "-",
        badge,
        formatTime(d.started_at),
      ]),
    );
  }
}

function formatTime(iso) {
  if (!iso) return "-";
  try {
    return new Date(iso).toLocaleString();
  } catch (_) {
    return iso;
  }
}

// Download form
const toggle = document.getElementById("download-toggle");
const row = document.getElementById("download-row");
toggle.addEventListener("click", () => {
  const open = row.style.display !== "none";
  row.style.display = open ? "none" : "";
  toggle.textContent = open ? "+ New download" : "- Cancel";
});

document.getElementById("download-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const repoId = document.getElementById("download-repo").value.trim();
  const revision = document.getElementById("download-revision").value.trim() || null;
  if (!repoId) return;

  const btn = document.getElementById("download-btn");
  btn.disabled = true;
  btn.textContent = "Downloading...";

  try {
    const body = { repo_id: repoId };
    if (revision) body.revision = revision;
    const resp = await fetch("/admin/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || resp.statusText);
    }
    showToast("Download started for " + repoId, "success");
    document.getElementById("download-repo").value = "";
    document.getElementById("download-revision").value = "";
    row.style.display = "none";
    toggle.textContent = "+ New download";
    await refreshDownloads();
  } catch (err) {
    showToast("Download failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Download";
  }
});

function updateTimestamp() {
  const el = document.getElementById("last-updated");
  el.textContent = "Updated " + new Date().toLocaleTimeString();
}

async function poll() {
  if (modelsPollInFlight) return false;
  modelsPollInFlight = true;
  const requestSequence = ++modelsRequestSequence;
  try {
    await Promise.all([fetchCatalog(requestSequence), fetchDownloads(requestSequence)]);
    if (requestSequence >= modelsLastRenderedSequence) {
      modelsLastRenderedSequence = requestSequence;
      updateTimestamp();
    }
  } finally {
    modelsPollInFlight = false;
  }
  return true;
}

document.addEventListener("DOMContentLoaded", function () {
  poll();
  setInterval(poll, POLL_INTERVAL_MS);
});
