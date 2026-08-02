# Feature Landscape: HuggingFace Hub Integration (v1.7)

**Domain:** Model download + NFS catalog for LLM inference gateway
**Researched:** 2026-07-28

## Existing Infrastructure (Already Built)

| Component | What It Does | New Features Build On |
|-----------|-------------|----------------------|
| NFS mount at `/srv/hf-cache` | Shared HF cache across all vLLM nodes (NFS server: `storage.example.com:/exports/huggingface`) | Gateway downloads to same NFS share; catalog scans it |
| `start-vllm.sh` symlink | Links `/root/.cache/huggingface` -> NFS mount. vLLM resolves HF repo_ids (e.g. `Qwen/Qwen2.5-7B-Instruct`) through this symlink to the HF cache structure | Downloaded models immediately available to vLLM -- no config change on nodes |
| `setup.sh` NFS step | Mounts NFS on target servers via `mount -t nfs -o vers=3,soft,timeo=100,retrans=2` | No change needed; mount already exists |
| llmfit recommendations | Ranked model suggestions per server hardware. The `name` field uses HuggingFace repo_id format (e.g. `meta-llama/Llama-3.1-8B-Instruct`) -- confirmed via llmfit source (model DB scraped from HF API) | Download button per recommendation; `name` maps directly to `snapshot_download(repo_id=)` |
| Node detail page | Per-node dashboard with recommendations panel, hardware summary, action buttons | Download status integrated into recommendations table |
| Admin API (`/admin/`) | Operational endpoints with FastAPI DI, Pydantic response models, structlog | New download/catalog routes follow same pattern |
| `ModelRecommendation` Pydantic model | Parsed llmfit output: `name`, `provider`, `score`, `fit_level`, `estimated_tps`, `memory_required_gb`, `category` | `name` field is the HuggingFace repo_id -- the download key |
| In-memory state tracking | `ProvisioningState` / `DownloadStatus` pattern with frozen Pydantic models and StrEnums | Download task tracking reuses same patterns |
| `asyncio.to_thread()` wrapping | etcd3gw sync calls wrapped for FastAPI async handlers | Same pattern for `snapshot_download()` (sync with internal thread pool) |

## Table Stakes

Features operators expect. Missing = the integration feels incomplete.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Download a model from HuggingFace to NFS | Core value -- operators want to pre-stage models before deploying to vLLM nodes | Med | Settings, huggingface-hub | `POST /admin/models/download` with `repo_id`. Background `snapshot_download(cache_dir=)` to NFS. Uses HF cache layout so vLLM finds models via the existing symlink. |
| HF API token support for gated models | Llama 4 (Meta), Mistral, Gemma (Google) are gated. Without a valid token with gated-repo read permission, downloads fail with `GatedRepoError`. Most llmfit-recommended models are gated. | Low | Settings (SecretStr) | `HuggingFaceSettings.token: SecretStr`. Passed as `token=` param. Two-step process for operators: (1) request access on HF model page, (2) create read token with "public gated repos" permission enabled. |
| NFS model catalog | "What's already downloaded?" -- operators need to see available models before downloading or provisioning | Med | Settings (models_dir path) | `GET /admin/models/catalog`. Uses `scan_cache_dir(cache_dir=)` which returns `HFCacheInfo` with per-repo `CachedRepoInfo` (repo_id, size_on_disk, nb_files, revisions). |
| Download status tracking | "Is it downloading? Did it finish? Did it fail?" -- operators need feedback on multi-GB downloads that take minutes to hours | Med | Download service | In-memory dict: `{repo_id: DownloadTask}`. Status enum: pending/downloading/complete/failed. `GET /admin/models/downloads`. Completed downloads also visible via catalog scan. |
| "Already downloaded" indicator on recommendations | When viewing llmfit results, operator needs to know which models exist on NFS without checking separately | Low | Catalog API, JS | Dashboard JS cross-references catalog response against recommendation `name` fields. Badge = "downloaded" / "not downloaded". |
| Download button on recommendation rows | One-click download from the recommendations panel -- the primary workflow entry point | Low | Download API, JS | Button in recommendations table calls `POST /admin/models/download`. Disabled + "downloading" text when in progress. |
| Feature disabled when NFS not configured | Gateway may run without NFS (dev, test, CI). Download features must degrade gracefully, not crash. | Low | Settings | `HuggingFaceSettings.models_dir: Path | None`. When None, endpoints return 503. Same pattern as QUADS (`base_url: str | None`) and Redfish (`bmc_username: str | None`). |

## Differentiators

Features that add operational polish. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Download size estimate before starting | Operator knows "this will download 140GB" before clicking. Prevents surprise disk fills. | Low | `snapshot_download(repo_id, dry_run=True)` returns `list[DryRunFileInfo]` with file sizes and cached status. Sums to total bytes-to-download. Simpler than `model_info(files_metadata=True)` -- already filters cached files. Show in a confirmation dialog or as text next to button. |
| Pre-download auth validation | Catch "you need to accept the license agreement" before starting a multi-hour download that will fail at the end | Low | `HfApi.auth_check(repo_id, token=)` raises `GatedRepoError` or `RepositoryNotFoundError` immediately. Call before queuing download. Fast (single HTTP HEAD). |
| Retry failed downloads | Downloads fail (network, disk, HF outage). Retry without re-downloading completed blobs. | Low | `snapshot_download()` is resumable by design -- HF cache uses content-addressed blobs with integrity checks. Re-trigger same repo_id, it picks up where it left off. Clear failed status on retry. |
| Multiple concurrent downloads | Download several models at once for initial cluster setup | Low | Each download runs in its own `asyncio.to_thread()`. Natural concurrency. Track all in the tasks dict. Consider limiting to 2-3 concurrent to avoid NFS throughput saturation. |
| Catalog disk usage summary | "NFS has 1.2TB of models across 8 repos" at a glance | Low | `scan_cache_dir()` returns `size_on_disk` per repo and total. Sum and display in dashboard header or catalog panel. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Download progress percentage streaming | `snapshot_download()` uses internal tqdm with per-file progress bars. Intercepting progress requires custom `tqdm_class` subclass that pipes updates to shared state, SSE endpoint for streaming, JS polling or EventSource -- significant complexity for marginal UX gain. Scope says "simple status." | Three states: downloading / complete / failed. Poll on dashboard refresh interval (10s default). The download either finishes or it does not. |
| Model deletion from dashboard | Deleting models from shared NFS is destructive and affects all vLLM nodes currently serving that model. One wrong click = production outage. | Omit delete button. Document `huggingface-cli delete-cache` or manual `rm` for cleanup. Require SSH access for destructive ops. |
| Auto-download on provisioning | Automatically downloading missing models during node setup ties provisioning success to HuggingFace availability and network speed. Models are 5-140GB. Setup would block for hours. | Download models separately via dashboard. Provisioning picks from what is available on NFS. Operator controls when downloads happen. |
| Custom HuggingFace mirror/endpoint support | Internal mirror/proxy for HF Hub adds config complexity. Internal network has direct internet access. | Use standard HuggingFace endpoint. If mirror needed later, `HF_ENDPOINT` env var is handled by `huggingface_hub` library natively -- no code changes needed. |
| Model format conversion (GGUF, AWQ, GPTQ) | Conversion is compute-intensive and requires model loading. Gateway is a CPU-only FastAPI process, no GPU. | Download models in their native safetensors format. vLLM handles quantization at serve time via `--quantization` flag. |
| Model version/revision management | Tracking multiple revisions of the same model adds catalog complexity, UI complexity, and NFS space consumption. | Download latest (main branch). HF cache deduplicates unchanged blobs between revisions. If specific revision needed later, add optional `revision` param to download endpoint. |
| WebSocket/SSE for real-time download updates | Adds transport complexity for marginal benefit. Existing dashboard uses polling for everything (node status, metrics, provisioning). | Stick with existing polling pattern. 10-second refresh is responsive enough for downloads that take 5-60 minutes. |

## Feature Dependencies

```
HuggingFaceSettings (config)
        |
        +---> NFSModelCatalog (scan NFS)
        |         |
        +---> ModelDownloadService (download models)
        |         |  (uses catalog for "already downloaded" check)
        v         v
    Admin API endpoints (catalog, download, status)
              |
              v
    Dashboard JS (download button + status in recommendations table)
```

Orthogonal (no ordering dependency on each other, but all depend on settings):
- Catalog scan: independent of download service
- Download service: uses catalog for "already downloaded" check (optional optimization)
- Dashboard: consumes both catalog and download APIs

Specific dependency chains:
- Download button requires download API + catalog API (to show "already downloaded")
- "Already downloaded" indicator requires catalog API only
- Download status display requires download API only
- All API endpoints require settings + Pydantic models

## MVP Recommendation

Prioritize in this order:

1. **Settings + Pydantic models** -- `HuggingFaceSettings`, download status/catalog data contracts. Foundation everything else depends on.
2. **NFS catalog scanner** -- `scan_cache_dir()` wrapper with `has_model()` fast-path. Independent, immediately testable.
3. **Dependency wiring** -- DI providers, lifespan init, `disable_progress_bars()` at startup.
4. **Catalog API** -- `GET /admin/models/catalog`. Operators can see what is downloaded before any download features exist.
5. **Download service + API** -- `POST /admin/models/download`, `GET /admin/models/downloads`. Core download flow with error mapping (`GatedRepoError` -> 403, `RepositoryNotFoundError` -> 404).
6. **Dashboard integration** -- Download column in recommendations table. "Already downloaded" badges. Download button per row.

Defer:
- **Download size estimate (`dry_run=True`)**: Nice-to-have. Add when operators ask. Single function call to integrate.
- **Pre-download auth validation (`auth_check`)**: Nice-to-have. Saves time but not blocking.
- **Catalog disk usage summary**: Trivial to add once catalog exists (sum `size_on_disk` fields). Not blocking.
- **Concurrent download limit**: Start unlimited, add semaphore when NFS throughput becomes an issue.

## Sources

- huggingface-hub download API: verified via Context7 `/huggingface/huggingface_hub` -- `snapshot_download` parameters, `cache_dir` vs `local_dir`, `token`, `dry_run`, `allow_patterns` (HIGH)
- huggingface-hub cache management: verified via Context7 `/huggingface/huggingface_hub` -- `scan_cache_dir`, `HFCacheInfo`, `CachedRepoInfo` (HIGH)
- huggingface-hub error types: verified via Context7 -- `GatedRepoError` derives from `RepositoryNotFoundError`, `HfHubHTTPError`, `EntryNotFoundError` (HIGH)
- huggingface-hub HfApi: verified via Context7 -- `model_info`, `auth_check`, `repo_exists` (HIGH)
- HuggingFace gated models docs: https://huggingface.co/docs/hub/en/models-gated -- access request process, token permissions (HIGH)
- HuggingFace download guide: https://huggingface.co/docs/huggingface_hub/en/guides/download (HIGH)
- HuggingFace cache docs: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache (HIGH)
- vLLM local model loading: https://github.com/vllm-project/vllm/issues/10721 -- requires HF model format with config.json (HIGH)
- vLLM NFS config: https://docs.vllm.ai/en/latest/api/vllm/config/load/ -- NFS auto-detection, prefetch mode (MEDIUM)
- llmfit model naming: https://github.com/AlexsJones/llmfit -- model DB sourced from HF API, `name` field uses HF repo_id format (MEDIUM)
- Existing codebase: `start-vllm.sh` (NFS symlink to `/root/.cache/huggingface`), `setup.sh` (NFS mount), `node_detail.js` (recommendations table), `settings.py` (SecretStr pattern) (HIGH)
- PROJECT.md milestone v1.7 target features (HIGH)
