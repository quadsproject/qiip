"""Unit tests for DownloadService."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.models.admin import DownloadState


class TestTriggerDownload:
    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_sets_status_downloading(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        status = await svc.trigger_download("org/model")

        assert status.status == DownloadState.DOWNLOADING
        assert status.repo_id == "org/model"
        assert status.started_at is not None

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
        # Make download hang so it stays in DOWNLOADING state
        event = asyncio.Event()
        mock_sd.side_effect = lambda *a, **kw: event.wait()

        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        status1 = await svc.trigger_download("org/model")
        status2 = await svc.trigger_download("org/model")

        assert status1 is status2
        # snapshot_download should only be invoked once (one background task)
        # Give a tick for the task to start
        await asyncio.sleep(0)
        assert mock_sd.call_count <= 1

        # Clean up: unblock the hanging call
        event.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.downloader.snapshot_download")
    async def test_allows_redownload_after_complete(self, mock_sd: MagicMock) -> None:
        svc = DownloadService(cache_dir="/tmp/test", token="test-token")
        await svc.trigger_download("org/model")
        await asyncio.sleep(0.1)

        assert svc.get_status("org/model") is not None
        assert svc.get_status("org/model").status == DownloadState.COMPLETE  # type: ignore[union-attr]

        # Trigger again -- should start a new download
        status = await svc.trigger_download("org/model")
        assert status.status == DownloadState.DOWNLOADING


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
