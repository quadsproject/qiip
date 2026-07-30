"""Background HuggingFace model download service.

Downloads models via ``snapshot_download`` in background threads,
tracking status in a thread-safe dict. Concurrent downloads are
capped by an asyncio semaphore (D-09).
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

import structlog
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from inference_proxy.models.admin import DownloadState, DownloadStatusResponse

logger = structlog.get_logger()


class DownloadService:
    """Manages background model downloads with thread-safe status tracking.

    Args:
        cache_dir: Path to the HuggingFace cache directory.
        token: Optional HuggingFace API token for gated models.
    """

    def __init__(self, cache_dir: str, token: str | None) -> None:
        self._cache_dir = cache_dir
        self._token = token
        self._statuses: dict[str, DownloadStatusResponse] = {}
        self._lock = threading.Lock()
        # ponytail: lazy semaphore -- must be created inside a running event loop
        self._semaphore: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def _ensure_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(2)
        return self._semaphore

    def get_status(self, repo_id: str) -> DownloadStatusResponse | None:
        """Return the download status for *repo_id*, or ``None``."""
        with self._lock:
            return self._statuses.get(repo_id)

    def get_all_statuses(self) -> list[DownloadStatusResponse]:
        """Return all tracked download statuses."""
        with self._lock:
            return list(self._statuses.values())

    async def trigger_download(self, repo_id: str) -> DownloadStatusResponse:
        """Start a background download for *repo_id*.

        Returns existing status if a download is already in progress (D-10).
        Allows re-download if previous attempt completed or failed (D-11).
        """
        with self._lock:
            existing = self._statuses.get(repo_id)
            if existing is not None and existing.status == DownloadState.DOWNLOADING:
                return existing

            status = DownloadStatusResponse(
                repo_id=repo_id,
                status=DownloadState.DOWNLOADING,
                started_at=datetime.now(UTC),
            )
            self._statuses[repo_id] = status

        task = asyncio.create_task(self._run_download(repo_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return status

    async def _run_download(self, repo_id: str) -> None:
        """Execute the download, gated by the concurrency semaphore."""
        async with self._ensure_semaphore():
            log = logger.bind(repo_id=repo_id)
            log.info("download started")
            try:
                await asyncio.to_thread(
                    snapshot_download,
                    repo_id,
                    cache_dir=self._cache_dir,
                    token=self._token,
                )
            except GatedRepoError:
                self._set_failed(
                    repo_id, f"Repository '{repo_id}' requires access approval"
                )
                return
            except RepositoryNotFoundError:
                self._set_failed(repo_id, f"Repository '{repo_id}' not found")
                return
            except RevisionNotFoundError:
                self._set_failed(repo_id, f"Revision not found for '{repo_id}'")
                return
            except Exception as exc:
                self._set_failed(repo_id, str(exc))
                return

            now = datetime.now(UTC)
            with self._lock:
                prev = self._statuses[repo_id]
                self._statuses[repo_id] = DownloadStatusResponse(
                    repo_id=repo_id,
                    status=DownloadState.COMPLETE,
                    started_at=prev.started_at,
                    completed_at=now,
                )
            log.info("download complete")

    def _set_failed(self, repo_id: str, error: str) -> None:
        now = datetime.now(UTC)
        with self._lock:
            prev = self._statuses[repo_id]
            self._statuses[repo_id] = DownloadStatusResponse(
                repo_id=repo_id,
                status=DownloadState.FAILED,
                started_at=prev.started_at,
                completed_at=now,
                error=error,
            )
        logger.error("download failed", repo_id=repo_id, error=error)
