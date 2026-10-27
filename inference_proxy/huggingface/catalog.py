"""HuggingFace model catalog service.

Scans a local HuggingFace cache directory (typically NFS-mounted) and
returns the list of cached model repositories.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import structlog
from huggingface_hub import (
    CachedRepoInfo,
    HFCacheInfo,
    scan_cache_dir,
    snapshot_download,
)
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


class ModelNotCachedError(LookupError):
    """Raised when a requested repository is not deletable from the cache."""

    def __init__(
        self, repo_id: str, reason: str = "is not present in the cache"
    ) -> None:
        self.repo_id = repo_id
        super().__init__(f"Model '{repo_id}' {reason}")


@dataclass(frozen=True, slots=True)
class DeletedModel:
    """Summary of an explicit model-cache deletion."""

    repo_id: str
    revision_count: int
    freed_bytes: int


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
    """Lists and deletes model repos in a HuggingFace cache directory.

    Wraps ``scan_cache_dir`` and explicit deletions in ``asyncio.to_thread`` so
    the blocking filesystem work does not stall the event loop.
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

    async def delete_model(self, repo_id: str) -> DeletedModel:
        """Delete every cached revision of one model repository.

        Only Hugging Face ``model`` repositories are eligible. The returned
        summary reports the exact revisions removed and the cache's expected
        freed size. Callers must coordinate with active downloads and reject a
        deletion while a registered node still serves the repository.
        """
        return await asyncio.to_thread(self._delete_model, repo_id)

    def _delete_model(self, repo_id: str) -> DeletedModel:
        cache_info = scan_cache_dir(self._cache_dir)
        repo = next(
            (
                item
                for item in cache_info.repos
                if item.repo_id == repo_id and item.repo_type == "model"
            ),
            None,
        )
        if repo is None:
            raise ModelNotCachedError(repo_id)
        revisions = sorted(repo.revisions, key=lambda item: item.commit_hash)
        if not revisions:
            raise ModelNotCachedError(repo_id, "has no cached revisions")

        # Hugging Face resolves revisions by commit hash across the whole
        # cache, and assumes a hash is unique to one repository. A mirror can
        # legitimately have the same commit as its source, so restrict the
        # deletion view to the repository the operator named.
        target_cache = HFCacheInfo(
            size_on_disk=repo.size_on_disk,
            repos=frozenset({repo}),
            incomplete_files=cache_info.incomplete_files,
            warnings=cache_info.warnings,
        )
        strategy = target_cache.delete_revisions(
            *(revision.commit_hash for revision in revisions)
        )
        freed_bytes = strategy.expected_freed_size
        strategy.execute()
        logger.info(
            "model cache deleted",
            repo_id=repo_id,
            revision_count=len(revisions),
            freed_bytes=freed_bytes,
        )
        return DeletedModel(
            repo_id=repo_id,
            revision_count=len(revisions),
            freed_bytes=freed_bytes,
        )

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

        return _SnapshotCheck("complete", Path(snapshot_path).name)
