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
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
)
from pydantic import BaseModel, Field

from inference_proxy.huggingface.artifacts import GGUFArtifact, GGUFArtifactIndex

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
        artifact_index: GGUFArtifactIndex | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._artifact_index = artifact_index or GGUFArtifactIndex(cache_dir)

    async def list_models(self) -> ModelCatalogResponse:
        """Return complete models and counts for hidden cache entries."""
        return await asyncio.to_thread(self._scan_catalog)

    def _scan_catalog(self) -> ModelCatalogResponse:
        cache_info = scan_cache_dir(self._cache_dir)
        artifact_scan = self._artifact_index.scan(cache_info)
        artifacts = list(artifact_scan.artifacts)
        models: list[CatalogEntry] = []
        incomplete_count = 0
        unverifiable_count = 0
        cache_warning_count = len(cache_info.warnings)
        for repo in sorted(cache_info.repos, key=lambda item: item.repo_id):
            if repo.repo_type != "model":
                continue
            check = self._snapshot_state(repo)
            if check.state == "complete":
                models.append(CatalogEntry(repo_id=repo.repo_id))
            elif check.state == "incomplete":
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
