"""Unit tests for DownloadService."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.models.admin import DownloadState


async def _wait_for_thread_event(event: threading.Event, timeout: float) -> bool:
    """Poll a thread event without creating a process-blocking default worker."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.001)
    return event.is_set()


class TestTriggerDownload:
    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_sets_status_downloading(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        result = await svc.trigger_download("org/model")

        assert result.started is True
        assert result.status.status == DownloadState.DOWNLOADING
        assert result.status.repo_id == "org/model"
        assert result.status.started_at is not None

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_completes_on_success(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model")
        # Let the background task finish
        await asyncio.sleep(0.1)

        status = svc.get_status("org/model")
        assert status is not None
        assert status.status == DownloadState.COMPLETE
        assert status.completed_at is not None

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_fails_on_exception(self, mock_sd: MagicMock) -> None:
        mock_sd.side_effect = RuntimeError("disk full")
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        status = svc.get_status("org/model")
        assert status is not None
        assert status.status == DownloadState.FAILED
        assert status.error == "disk full"

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_fails_on_gated_repo(self, mock_sd: MagicMock) -> None:
        resp = httpx.Response(403, request=httpx.Request("GET", "https://hf.co"))
        mock_sd.side_effect = GatedRepoError("gated", response=resp)
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        status = svc.get_status("org/model")
        assert status is not None
        assert status.status == DownloadState.FAILED
        assert "access approval" in status.error  # type: ignore[operator]

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_duplicate_returns_existing(self, mock_sd: MagicMock) -> None:
        started = threading.Event()
        release = threading.Event()

        def block_download(*_args: object, **_kwargs: object) -> None:
            started.set()
            if not release.wait(timeout=2):
                raise AssertionError("test did not release the download worker")

        mock_sd.side_effect = block_download

        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        first = await svc.trigger_download("org/model")
        assert await _wait_for_thread_event(started, 1)
        duplicate = await svc.trigger_download("org/model")

        tasks = tuple(svc._tasks)
        try:
            assert first.started is True
            assert duplicate.started is False
            assert duplicate.status is first.status
            assert mock_sd.call_count == 1
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        assert svc.get_status("org/model") is not None
        assert svc.get_status("org/model").status == DownloadState.COMPLETE  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_allows_redownload_after_complete(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        assert svc.get_status("org/model") is not None
        assert svc.get_status("org/model").status == DownloadState.COMPLETE  # type: ignore[union-attr]

        # Trigger again -- should start a new download
        result = await svc.trigger_download("org/model")
        assert result.started is True
        assert result.status.status == DownloadState.DOWNLOADING

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_concurrent_downloads_are_capped_at_two(
        self, mock_sd: MagicMock
    ) -> None:
        release = threading.Event()
        two_started = threading.Event()
        third_started = threading.Event()
        state_lock = threading.Lock()
        active = 0

        def block_download(*_args: object, **_kwargs: object) -> None:
            nonlocal active
            with state_lock:
                active += 1
                if active == 2:
                    two_started.set()
                elif active == 3:
                    third_started.set()
            if not release.wait(timeout=2):
                raise AssertionError("test did not release the download workers")

        mock_sd.side_effect = block_download
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")

        await svc.trigger_download("org/model-a")
        await svc.trigger_download("org/model-b")
        await svc.trigger_download("org/model-c")

        tasks = tuple(svc._tasks)
        try:
            assert await _wait_for_thread_event(two_started, 1)
            assert not await _wait_for_thread_event(third_started, 0.1)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        assert mock_sd.call_count == 3
        assert {status.status for status in svc.get_all_statuses()} == {
            DownloadState.COMPLETE
        }


class TestDownloadShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_wrappers_without_waiting_for_worker(
        self,
    ) -> None:
        started = asyncio.Event()

        async def block_download(*_args: object, **_kwargs: object) -> None:
            started.set()
            await asyncio.Event().wait()

        service = DownloadService(cache_dir="/tmp/test", token=None)
        with patch(
            "inference_proxy.huggingface.downloader._run_in_daemon_thread",
            side_effect=block_download,
        ):
            await service.trigger_download("org/model")
            await asyncio.wait_for(started.wait(), timeout=1)
            await asyncio.wait_for(service.shutdown(), timeout=0.5)

        assert service._tasks == set()
        with pytest.raises(RuntimeError, match="shutting down"):
            await service.trigger_download("org/other")

    def test_process_exits_with_blocked_snapshot_download(
        self,
        tmp_path: Path,
    ) -> None:
        script = textwrap.dedent(
            """
            import asyncio
            import threading
            from unittest.mock import patch

            from inference_proxy.huggingface.downloader import DownloadService

            started = threading.Event()
            blocked_forever = threading.Event()

            def block_download(*_args, **_kwargs):
                started.set()
                blocked_forever.wait()

            async def main():
                with patch(
                    "inference_proxy.huggingface.downloader.snapshot_download",
                    side_effect=block_download,
                ):
                    service = DownloadService(cache_dir="/tmp/test", token=None)
                    await service.trigger_download("org/model")
                    deadline = asyncio.get_running_loop().time() + 1
                    while not started.is_set():
                        if asyncio.get_running_loop().time() >= deadline:
                            raise RuntimeError("download worker did not start")
                        await asyncio.sleep(0.001)
                    await service.shutdown()

            asyncio.run(main())
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )

        assert result.returncode == 0, result.stderr


class TestGetAllStatuses:
    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_returns_all(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model-a")
        await svc.trigger_download("org/model-b")
        await asyncio.sleep(0.1)

        all_statuses = svc.get_all_statuses()
        assert len(all_statuses) == 2
        repo_ids = {s.repo_id for s in all_statuses}
        assert repo_ids == {"org/model-a", "org/model-b"}


class TestTokenPassing:
    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_passes_token_to_snapshot_download(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="hf_secret")
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        mock_sd.assert_called_once_with(
            "org/model", cache_dir="/tmp/test", token="hf_secret"
        )

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_passes_none_token(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token=None)
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        mock_sd.assert_called_once_with("org/model", cache_dir="/tmp/test", token=None)
