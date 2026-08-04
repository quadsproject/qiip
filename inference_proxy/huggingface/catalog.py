"""HuggingFace model catalog service.

Scans a local HuggingFace cache directory (typically NFS-mounted) and
returns the list of cached model repositories.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, NamedTuple

import structlog
from huggingface_hub import CachedRepoInfo, scan_cache_dir, snapshot_download
from huggingface_hub.errors import (
    CorruptedCacheException,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
)
from pydantic import BaseModel, Field

from inference_proxy.huggingface.artifacts import GGUFArtifact, GGUFArtifactStore

logger = structlog.get_logger()


class CatalogEntry(BaseModel):
    """A single cached model repository."""

    repo_id: str


class ModelCatalogResponse(BaseModel):
    """Response payload for the model catalog endpoint."""

    models: list[CatalogEntry]
    gguf_artifacts: list[GGUFArtifact] = Field(default_factory=list)
    incomplete_count: int = 0
    unverifiable_count: int = 0
    invalid_artifact_count: int = 0
    cache_warning_count: int = 0


_SnapshotState = Literal["complete", "incomplete", "unverifiable"]


class _SnapshotCheck(NamedTuple):
    state: _SnapshotState
    resolved_revision: str | None = None


class ModelCatalogService:
    """Lists model repos found in a HuggingFace cache directory.

    Wraps ``scan_cache_dir`` in ``asyncio.to_thread`` so the blocking
    filesystem scan does not stall the event loop.
    """

    def __init__(
        self,
        cache_dir: str,
        artifact_store: GGUFArtifactStore | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._artifact_store = artifact_store or GGUFArtifactStore(cache_dir)

    async def list_models(self) -> ModelCatalogResponse:
        """Return complete models and counts for hidden cache entries."""
        return await asyncio.to_thread(self._scan_catalog)

    def _scan_catalog(self) -> ModelCatalogResponse:
        artifact_scan = self._artifact_store.scan()
        artifacts = list(artifact_scan.artifacts)
        artifact_revisions = {
            (artifact.repo_id, artifact.resolved_revision) for artifact in artifacts
        }
        cache_info = scan_cache_dir(self._cache_dir)
        models: list[CatalogEntry] = []
        incomplete_count = 0
        unverifiable_count = 0
        cache_warning_count = sum(
            not self._is_artifact_root_warning(warning)
            for warning in cache_info.warnings
        )
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue
            check = self._snapshot_state(repo)
            if check.state == "complete":
                models.append(CatalogEntry(repo_id=repo.repo_id))
            elif check.state == "incomplete":
                intentionally_partial = (
                    repo.repo_id,
                    check.resolved_revision,
                ) in artifact_revisions
                if not intentionally_partial:
                    incomplete_count += 1
            else:
                unverifiable_count += 1

        if (
            incomplete_count
            or unverifiable_count
            or artifact_scan.invalid_count
            or cache_warning_count
        ):
            logger.warning(
                "catalog scan skipped models",
                incomplete_count=incomplete_count,
                unverifiable_count=unverifiable_count,
                invalid_artifact_count=artifact_scan.invalid_count,
                cache_warning_count=cache_warning_count,
            )
        return ModelCatalogResponse(
            models=models,
            gguf_artifacts=artifacts,
            incomplete_count=incomplete_count,
            unverifiable_count=unverifiable_count,
            invalid_artifact_count=artifact_scan.invalid_count,
            cache_warning_count=cache_warning_count,
        )

    def _is_artifact_root_warning(self, warning: Exception) -> bool:
        """Ignore only the cache scanner warning caused by our ``gguf`` root."""
        expected = (
            "Repo path is not a valid HuggingFace cache directory: "
            f"{self._artifact_store.root}"
        )
        return isinstance(warning, CorruptedCacheException) and warning.args == (
            expected,
        )

    def _snapshot_state(self, repo: CachedRepoInfo) -> _SnapshotCheck:
        try:
            # The local-only path checks the cached tree manifest and rejects
            # snapshots with missing files. This also catches failed-download
            # remnants left before a gateway restart.
            snapshot_path = snapshot_download(
                repo.repo_id,
                repo_type="model",
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
        except IncompleteSnapshotError as exc:
            return _SnapshotCheck("incomplete", Path(exc.snapshot_path).name)
        except LocalEntryNotFoundError:
            return _SnapshotCheck("unverifiable")

        if not isinstance(snapshot_path, str):
            raise TypeError("snapshot_download returned a dry-run result unexpectedly")

        # huggingface_hub accepts legacy snapshots without a tree manifest
        # because it cannot prove they are incomplete. The catalog takes the
        # safer position and exposes these separately for operator migration.
        commit = Path(snapshot_path).name
        manifest = repo.repo_path / "trees" / f"{commit}.json"
        if manifest.is_file():
            return _SnapshotCheck("complete", commit)
        return _SnapshotCheck("unverifiable", commit)
