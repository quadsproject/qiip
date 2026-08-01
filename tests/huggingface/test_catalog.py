"""Unit tests for ModelCatalogService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from structlog.testing import capture_logs

from inference_proxy.huggingface.catalog import CatalogEntry, ModelCatalogService


def _mock_repo(
    repo_id: str,
    repo_type: str = "model",
    *,
    repo_path: Path | None = None,
) -> MagicMock:
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.repo_type = repo_type
    if repo_path is not None:
        repo.repo_path = repo_path
    return repo


def _mock_cache_info(repos: list[MagicMock]) -> MagicMock:
    info = MagicMock()
    info.repos = frozenset(repos)
    return info


def _write_cached_snapshot(
    cache_dir: Path,
    repo_id: str,
    *,
    present_files: set[str],
    write_manifest: bool = True,
) -> Path:
    commit = "a" * 40
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = repo_dir / "snapshots" / commit
    snapshot_dir.mkdir(parents=True)
    for filename in present_files:
        path = snapshot_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")

    refs_dir = repo_dir / "refs"
    refs_dir.mkdir()
    (refs_dir / "main").write_text(commit, encoding="utf-8")

    if write_manifest:
        trees_dir = repo_dir / "trees"
        trees_dir.mkdir()
        (trees_dir / f"{commit}.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "files": {
                        "config.json": {"size": 1, "blob_id": "config"},
                        "model.safetensors": {"size": 1, "blob_id": "weights"},
                    },
                }
            ),
            encoding="utf-8",
        )
    return repo_dir


class TestListModels:
    @pytest.mark.asyncio
    @patch.object(ModelCatalogService, "_snapshot_state", return_value="complete")
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_returns_model_repos(
        self, mock_scan: MagicMock, mock_state: MagicMock
    ) -> None:
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo("meta-llama/Llama-3-8B"),
                _mock_repo("mistralai/Mistral-7B"),
            ]
        )
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert len(result.models) == 2
        ids = {e.repo_id for e in result.models}
        assert ids == {"meta-llama/Llama-3-8B", "mistralai/Mistral-7B"}
        assert all(isinstance(e, CatalogEntry) for e in result.models)
        assert result.incomplete_count == 0
        assert result.unverifiable_count == 0
        mock_scan.assert_called_once_with("/data/hf")
        assert mock_state.call_count == 2

    @pytest.mark.asyncio
    @patch.object(ModelCatalogService, "_snapshot_state", return_value="complete")
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_filters_non_model_repos(
        self, mock_scan: MagicMock, mock_state: MagicMock
    ) -> None:
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo("meta-llama/Llama-3-8B", "model"),
                _mock_repo("some-org/some-dataset", "dataset"),
                _mock_repo("some-org/some-space", "space"),
            ]
        )
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert len(result.models) == 1
        assert result.models[0].repo_id == "meta-llama/Llama-3-8B"
        mock_state.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(ModelCatalogService, "_snapshot_state")
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_empty_cache_returns_empty_list(
        self, mock_scan: MagicMock, mock_state: MagicMock
    ) -> None:
        mock_scan.return_value = _mock_cache_info([])
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert result.models == []
        assert result.incomplete_count == 0
        assert result.unverifiable_count == 0
        mock_state.assert_not_called()

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_incomplete_and_unverifiable_snapshots_are_not_listed(
        self, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        complete_path = _write_cached_snapshot(
            tmp_path,
            "org/complete",
            present_files={"config.json", "model.safetensors"},
        )
        partial_path = _write_cached_snapshot(
            tmp_path,
            "org/partial",
            present_files={"config.json"},
        )
        legacy_path = _write_cached_snapshot(
            tmp_path,
            "org/legacy",
            present_files={"config.json", "model.safetensors"},
            write_manifest=False,
        )
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo("org/complete", repo_path=complete_path),
                _mock_repo("org/partial", repo_path=partial_path),
                _mock_repo("org/legacy", repo_path=legacy_path),
            ]
        )
        svc = ModelCatalogService(cache_dir=str(tmp_path))

        with capture_logs() as logs:
            result = await svc.list_models()

        assert [entry.repo_id for entry in result.models] == ["org/complete"]
        assert result.incomplete_count == 1
        assert result.unverifiable_count == 1
        assert logs == [
            {
                "event": "catalog scan skipped models",
                "incomplete_count": 1,
                "log_level": "warning",
                "unverifiable_count": 1,
            }
        ]
