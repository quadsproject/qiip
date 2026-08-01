"""HuggingFace model catalog service.

Scans a local HuggingFace cache directory (typically NFS-mounted) and
returns the list of cached model repositories.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import structlog
from huggingface_hub import CachedRepoInfo, scan_cache_dir, snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError
from pydantic import BaseModel

logger = structlog.get_logger()


class CatalogEntry(BaseModel):
    """A single cached model repository."""

    repo_id: str


class ModelCatalogResponse(BaseModel):
    """Response payload for the model catalog endpoint."""

    models: list[CatalogEntry]
    incomplete_count: int = 0
    unverifiable_count: int = 0


_SnapshotState = Literal["complete", "incomplete", "unverifiable"]


class ModelCatalogService:
    """Lists model repos found in a HuggingFace cache directory.

    Wraps ``scan_cache_dir`` in ``asyncio.to_thread`` so the blocking
    filesystem scan does not stall the event loop.
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir

    async def list_models(self) -> ModelCatalogResponse:
        """Return complete models and counts for hidden cache entries."""
        return await asyncio.to_thread(self._scan_catalog)

    def _scan_catalog(self) -> ModelCatalogResponse:
        cache_info = scan_cache_dir(self._cache_dir)
        models: list[CatalogEntry] = []
        incomplete_count = 0
        unverifiable_count = 0
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue
            state = self._snapshot_state(repo)
            if state == "complete":
                models.append(CatalogEntry(repo_id=repo.repo_id))
            elif state == "incomplete":
                incomplete_count += 1
            else:
                unverifiable_count += 1

        if incomplete_count or unverifiable_count:
            logger.warning(
                "catalog scan skipped models",
                incomplete_count=incomplete_count,
                unverifiable_count=unverifiable_count,
            )
        return ModelCatalogResponse(
            models=models,
            incomplete_count=incomplete_count,
            unverifiable_count=unverifiable_count,
        )

    def _snapshot_state(self, repo: CachedRepoInfo) -> _SnapshotState:
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
        except IncompleteSnapshotError:
            return "incomplete"
        except LocalEntryNotFoundError:
            return "unverifiable"

        if not isinstance(snapshot_path, str):
            raise TypeError("snapshot_download returned a dry-run result unexpectedly")

        # huggingface_hub accepts legacy snapshots without a tree manifest
        # because it cannot prove they are incomplete. The catalog takes the
        # safer position and exposes these separately for operator migration.
        commit = Path(snapshot_path).name
        manifest = repo.repo_path / "trees" / f"{commit}.json"
        return "complete" if manifest.is_file() else "unverifiable"
