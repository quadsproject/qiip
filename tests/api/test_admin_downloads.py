"""Integration tests for the download admin endpoints.

Tests cover:
- POST /admin/models/download triggers background download (DL-01)
- GET /admin/models/downloads lists all download statuses (DL-03)
- Duplicate POST returns existing status (D-10)
- Missing repo_id returns 422
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.huggingface.downloader import DownloadTriggerResult
from inference_proxy.models.admin import DownloadState, DownloadStatusResponse


class TestTriggerDownload:
    """POST /admin/models/download triggers a background download."""

    def test_returns_202(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        mock_svc: MagicMock = app.state.download_service
        mock_svc.trigger_download.return_value = DownloadTriggerResult(
            status=DownloadStatusResponse(
                repo_id="meta-llama/Llama-3.1-8B",
                status=DownloadState.DOWNLOADING,
                started_at=datetime.now(UTC),
            ),
            started=True,
        )

        response = client.post(
            "/admin/models/download",
            json={"repo_id": "meta-llama/Llama-3.1-8B"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["repo_id"] == "meta-llama/Llama-3.1-8B"
        assert data["status"] == "downloading"

    def test_requires_repo_id(self, client: TestClient) -> None:
        response = client.post("/admin/models/download", json={})
        assert response.status_code == 422

    def test_duplicate_returns_200(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        mock_svc: MagicMock = app.state.download_service
        mock_svc.trigger_download.return_value = DownloadTriggerResult(
            status=DownloadStatusResponse(
                repo_id="org/model",
                status=DownloadState.DOWNLOADING,
                started_at=datetime.now(UTC),
            ),
            started=False,
        )

        response = client.post("/admin/models/download", json={"repo_id": "org/model"})

        assert response.status_code == 200
        assert response.json()["status"] == "downloading"


class TestListDownloads:
    """GET /admin/models/downloads returns download statuses."""

    def test_returns_empty_list(self, client: TestClient) -> None:
        response = client.get("/admin/models/downloads")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_statuses(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        mock_svc: MagicMock = app.state.download_service
        now = datetime.now(UTC)
        mock_svc.get_all_statuses.return_value = [
            DownloadStatusResponse(
                repo_id="org/model-a",
                status=DownloadState.DOWNLOADING,
                started_at=now,
            ),
            DownloadStatusResponse(
                repo_id="org/model-b",
                status=DownloadState.COMPLETE,
                started_at=now,
                completed_at=now,
            ),
        ]

        response = client.get("/admin/models/downloads")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        repo_ids = {d["repo_id"] for d in data}
        assert repo_ids == {"org/model-a", "org/model-b"}
