"""Unit tests for ModelCatalogService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inference_proxy.huggingface.catalog import CatalogEntry, ModelCatalogService


def _mock_repo(repo_id: str, repo_type: str = "model") -> MagicMock:
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.repo_type = repo_type
    return repo


def _mock_cache_info(repos: list[MagicMock]) -> MagicMock:
    info = MagicMock()
    info.repos = frozenset(repos)
    return info


class TestListModels:
    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_returns_model_repos(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo("meta-llama/Llama-3-8B"),
                _mock_repo("mistralai/Mistral-7B"),
            ]
        )
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert len(result) == 2
        ids = {e.repo_id for e in result}
        assert ids == {"meta-llama/Llama-3-8B", "mistralai/Mistral-7B"}
        assert all(isinstance(e, CatalogEntry) for e in result)
        mock_scan.assert_called_once_with("/data/hf")

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_filters_non_model_repos(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo("meta-llama/Llama-3-8B", "model"),
                _mock_repo("some-org/some-dataset", "dataset"),
                _mock_repo("some-org/some-space", "space"),
            ]
        )
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert len(result) == 1
        assert result[0].repo_id == "meta-llama/Llama-3-8B"

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_empty_cache_returns_empty_list(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = _mock_cache_info([])
        svc = ModelCatalogService(cache_dir="/data/hf")
        result = await svc.list_models()

        assert result == []
