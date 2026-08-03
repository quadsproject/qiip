"""Background HuggingFace model download service.

Downloads models via ``snapshot_download`` in background threads,
tracking status in a thread-safe dict. Concurrent downloads are
capped by an asyncio semaphore (D-09).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

import structlog
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from inference_proxy.models.admin import DownloadState, DownloadStatusResponse

logger = structlog.get_logger()


async def _run_in_daemon_thread[T](function: Callable[[], T], *, name: str) -> T:
    """Run blocking work without registering a process-blocking executor thread."""
    finished = threading.Event()
    values: list[T] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            values.append(function())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    threading.Thread(target=_worker, name=name, daemon=True).start()
    while not finished.is_set():
        await asyncio.sleep(0.01)
    if errors:
        raise errors[0]
    return values[0]


@dataclass(frozen=True, slots=True)
class DownloadTriggerResult:
    """Atomic result of requesting a model download."""

    status: DownloadStatusResponse
    started: bool


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
        self._closed = False

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

    @staticmethod
    def _dedup_key(repo_id: str, allow_patterns: list[str] | None) -> str:
        if not allow_patterns:
            return repo_id
        return f"{repo_id}::{','.join(sorted(allow_patterns))}"

    async def trigger_download(
        self,
        repo_id: str,
        allow_patterns: list[str] | None = None,
    ) -> DownloadTriggerResult:
        """Start a background download for *repo_id*.

        The result identifies atomically whether this call started the download
        or found one already in progress (D-10). Allows re-download if a
        previous attempt completed or failed (D-11).

        Pass *allow_patterns* to filter files (e.g. ``["*q4_k_m*"]`` for
        a specific GGUF quantization).
        """
        key = self._dedup_key(repo_id, allow_patterns)
        with self._lock:
            if self._closed:
                raise RuntimeError("download service is shutting down")
            existing = self._statuses.get(key)
            if existing is not None and existing.status == DownloadState.DOWNLOADING:
                return DownloadTriggerResult(status=existing, started=False)

            status = DownloadStatusResponse(
                repo_id=repo_id,
                status=DownloadState.DOWNLOADING,
                started_at=datetime.now(UTC),
            )
            self._statuses[key] = status

        task = asyncio.create_task(
            self._run_download(key, repo_id, allow_patterns=allow_patterns)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return DownloadTriggerResult(status=status, started=True)

    async def _run_download(
        self,
        key: str,
        repo_id: str,
        allow_patterns: list[str] | None = None,
    ) -> None:
        """Execute the download, gated by the concurrency semaphore."""
        async with self._ensure_semaphore():
            log = logger.bind(repo_id=repo_id)
            log.info("download started")
            download_kwargs: dict[str, object] = {
                "cache_dir": self._cache_dir,
                "token": self._token,
            }
            if allow_patterns:
                download_kwargs["allow_patterns"] = allow_patterns
            try:
                await _run_in_daemon_thread(
                    partial(snapshot_download, repo_id, **download_kwargs),
                    name=f"huggingface-download:{repo_id}",
                )
            except GatedRepoError:
                self._set_failed(
                    key, f"Repository '{repo_id}' requires access approval"
                )
                return
            except RepositoryNotFoundError:
                self._set_failed(key, f"Repository '{repo_id}' not found")
                return
            except RevisionNotFoundError:
                self._set_failed(key, f"Revision not found for '{repo_id}'")
                return
            except Exception as exc:
                self._set_failed(key, str(exc))
                return

            now = datetime.now(UTC)
            with self._lock:
                prev = self._statuses[key]
                self._statuses[key] = DownloadStatusResponse(
                    repo_id=repo_id,
                    status=DownloadState.COMPLETE,
                    started_at=prev.started_at,
                    completed_at=now,
                )
            log.info("download complete")

    async def shutdown(self) -> None:
        """Cancel download wrappers without joining their daemon workers.

        A blocking ``snapshot_download`` cannot be interrupted safely. Its raw
        worker thread is therefore allowed to end with the process, while the
        asyncio tasks are cancelled and awaited so loop shutdown stays bounded.
        """
        with self._lock:
            self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _set_failed(self, key: str, error: str) -> None:
        now = datetime.now(UTC)
        with self._lock:
            prev = self._statuses[key]
            self._statuses[key] = DownloadStatusResponse(
                repo_id=prev.repo_id,
                status=DownloadState.FAILED,
                started_at=prev.started_at,
                completed_at=now,
                error=error,
            )
        logger.error("download failed", key=key, error=error)
