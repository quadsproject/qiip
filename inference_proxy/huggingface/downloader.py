"""Background HuggingFace model download service.

Downloads models via ``snapshot_download`` in background threads,
tracking status in a thread-safe dict. Concurrent downloads are
capped by an asyncio semaphore (D-09).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import structlog
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from inference_proxy.huggingface.artifacts import (
    GGUFArtifact,
    GGUFArtifactIndex,
    GGUFDownloadSpec,
)
from inference_proxy.models.admin import DownloadState, DownloadStatusResponse
from inference_proxy.models.node import InferenceEngine

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

    def __init__(
        self,
        cache_dir: str,
        token: str | None,
        artifact_index: GGUFArtifactIndex | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._token = token
        self._artifact_index = artifact_index or GGUFArtifactIndex(cache_dir)
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

    def get_all_statuses(self) -> list[DownloadStatusResponse]:
        """Return all tracked download statuses."""
        with self._lock:
            return list(self._statuses.values())

    @staticmethod
    def _download_id(
        repo_id: str,
        *,
        revision: str | None,
        engine: InferenceEngine,
        gguf: GGUFDownloadSpec | None,
    ) -> str:
        payload = json.dumps(
            {
                "repo_id": repo_id,
                "requested_revision": revision,
                "engine": engine.value,
                "gguf": gguf.model_dump(mode="json") if gguf else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def trigger_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        engine: InferenceEngine = InferenceEngine.VLLM,
        gguf: GGUFDownloadSpec | None = None,
    ) -> DownloadTriggerResult:
        """Start a background download for *repo_id*.

        The result identifies atomically whether this call started the download
        or found one already in progress (D-10). Allows re-download if a
        previous attempt completed or failed (D-11).

        GGUF downloads require an exact file set and entrypoint. Full vLLM
        snapshots do not accept a partial-file specification.
        """
        if engine == InferenceEngine.LLAMA_CPP and gguf is None:
            raise ValueError("llama_cpp downloads require an exact GGUF specification")
        if engine == InferenceEngine.VLLM and gguf is not None:
            raise ValueError("GGUF specifications are only valid for llama_cpp")
        key = self._download_id(repo_id, revision=revision, engine=engine, gguf=gguf)
        with self._lock:
            if self._closed:
                raise RuntimeError("download service is shutting down")
            existing = self._statuses.get(key)
            if existing is not None and existing.status == DownloadState.DOWNLOADING:
                return DownloadTriggerResult(status=existing, started=False)

            status = DownloadStatusResponse(
                download_id=key,
                repo_id=repo_id,
                requested_revision=revision,
                engine=engine,
                gguf=gguf,
                status=DownloadState.DOWNLOADING,
                started_at=datetime.now(UTC),
            )
            self._statuses[key] = status

        task = asyncio.create_task(self._run_download(key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return DownloadTriggerResult(status=status, started=True)

    async def _run_download(
        self,
        key: str,
    ) -> None:
        """Execute the download, gated by the concurrency semaphore."""
        async with self._ensure_semaphore():
            with self._lock:
                requested = self._statuses[key]
            repo_id = requested.repo_id
            log = logger.bind(repo_id=repo_id)
            log.info("download started")
            try:
                resolved_revision, artifacts = await _run_in_daemon_thread(
                    partial(
                        self._download_and_index,
                        repo_id,
                        requested.requested_revision,
                        requested.gguf,
                    ),
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
                    download_id=prev.download_id,
                    repo_id=repo_id,
                    requested_revision=prev.requested_revision,
                    resolved_revision=resolved_revision,
                    engine=prev.engine,
                    gguf=prev.gguf,
                    artifacts=artifacts,
                    status=DownloadState.COMPLETE,
                    started_at=prev.started_at,
                    completed_at=now,
                )
            log.info("download complete")

    def _download_and_index(
        self,
        repo_id: str,
        revision: str | None,
        gguf: GGUFDownloadSpec | None,
    ) -> tuple[str, tuple[GGUFArtifact, ...]]:
        if revision is not None and gguf is not None:
            snapshot_result = snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=list(gguf.files),
                cache_dir=self._cache_dir,
                token=self._token,
            )
        elif revision is not None:
            snapshot_result = snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=self._cache_dir,
                token=self._token,
            )
        elif gguf is not None:
            snapshot_result = snapshot_download(
                repo_id,
                allow_patterns=list(gguf.files),
                cache_dir=self._cache_dir,
                token=self._token,
            )
        else:
            snapshot_result = snapshot_download(
                repo_id, cache_dir=self._cache_dir, token=self._token
            )
        if not isinstance(snapshot_result, str):
            raise TypeError("snapshot_download returned a dry-run result unexpectedly")
        snapshot_path = Path(snapshot_result)
        resolved_revision = snapshot_path.name
        if re.fullmatch(r"[0-9a-f]{40,64}", resolved_revision) is None:
            raise ValueError(
                "HuggingFace snapshot path does not end in an immutable commit SHA: "
                f"{snapshot_result}"
            )
        if gguf is None:
            return resolved_revision, ()
        artifact = self._artifact_index.artifact_from_download(
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            snapshot_path=snapshot_path,
            spec=gguf,
        )
        return resolved_revision, (artifact,)

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
                download_id=prev.download_id,
                repo_id=prev.repo_id,
                requested_revision=prev.requested_revision,
                resolved_revision=prev.resolved_revision,
                engine=prev.engine,
                gguf=prev.gguf,
                artifacts=prev.artifacts,
                status=DownloadState.FAILED,
                started_at=prev.started_at,
                completed_at=now,
                error=error,
            )
        logger.error("download failed", key=key, error=error)
