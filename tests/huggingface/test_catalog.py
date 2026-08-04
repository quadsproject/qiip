"""Unit tests for ModelCatalogService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import CorruptedCacheException
from structlog.testing import capture_logs

from inference_proxy.huggingface.artifacts import GGUFArtifactStore, GGUFDownloadSpec
from inference_proxy.huggingface.catalog import (
    CatalogEntry,
    ModelCatalogService,
    _SnapshotCheck,
)


def _mock_repo(
    repo_id: str,
    repo_type: str = "model",
    *,
    repo_path: Path | None = None,
    revisions: tuple[str, ...] = (),
) -> MagicMock:
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.repo_type = repo_type
    if repo_path is not None:
        repo.repo_path = repo_path
    repo.revisions = frozenset(
        MagicMock(commit_hash=commit_hash) for commit_hash in revisions
    )
    return repo


def _mock_cache_info(
    repos: list[MagicMock], *, warnings: tuple[Exception, ...] = ()
) -> MagicMock:
    info = MagicMock()
    info.repos = frozenset(repos)
    info.warnings = warnings
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
    @patch.object(
        ModelCatalogService, "_snapshot_state", return_value=_SnapshotCheck("complete")
    )
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
        assert result.gguf_artifacts == []
        mock_scan.assert_called_once_with("/data/hf")
        assert mock_state.call_count == 2

    @pytest.mark.asyncio
    @patch.object(
        ModelCatalogService, "_snapshot_state", return_value=_SnapshotCheck("complete")
    )
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
                "invalid_artifact_count": 0,
                "log_level": "warning",
                "cache_warning_count": 0,
                "unverifiable_count": 1,
            }
        ]

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_valid_partial_gguf_is_cataloged_without_degradation(
        self, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        repo_id = "org/model-GGUF"
        revision = "c" * 40
        snapshot = tmp_path / "models--org--model-GGUF" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.gguf").write_bytes(b"weights")
        artifact = GGUFArtifactStore(tmp_path).publish(
            repo_id=repo_id,
            resolved_revision=revision,
            snapshot_path=snapshot,
            spec=GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf"),
        )
        repo = _mock_repo(
            repo_id,
            repo_path=snapshot.parents[1],
            revisions=(revision,),
        )
        mock_scan.return_value = _mock_cache_info([repo])
        service = ModelCatalogService(str(tmp_path))

        with patch.object(
            service,
            "_snapshot_state",
            return_value=_SnapshotCheck("incomplete", revision),
        ):
            result = await service.list_models()

        assert result.models == []
        assert result.gguf_artifacts == [artifact]
        assert result.incomplete_count == 0
        assert result.invalid_artifact_count == 0
        assert result.cache_warning_count == 0

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_artifact_does_not_hide_a_different_incomplete_revision(
        self, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        repo_id = "org/model-GGUF"
        artifact_revision = "c" * 40
        incomplete_revision = "d" * 40
        snapshot = (
            tmp_path / "models--org--model-GGUF" / "snapshots" / artifact_revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.gguf").write_bytes(b"weights")
        artifact = GGUFArtifactStore(tmp_path).publish(
            repo_id=repo_id,
            resolved_revision=artifact_revision,
            snapshot_path=snapshot,
            spec=GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf"),
        )
        mock_scan.return_value = _mock_cache_info(
            [
                _mock_repo(
                    repo_id,
                    repo_path=snapshot.parents[1],
                    revisions=(artifact_revision, incomplete_revision),
                )
            ]
        )
        service = ModelCatalogService(str(tmp_path))

        with patch.object(
            service,
            "_snapshot_state",
            return_value=_SnapshotCheck("incomplete", incomplete_revision),
        ):
            result = await service.list_models()

        assert result.gguf_artifacts == [artifact]
        assert result.incomplete_count == 1

    @pytest.mark.asyncio
    @patch("inference_proxy.huggingface.catalog.scan_cache_dir")
    async def test_only_exact_artifact_root_cache_warning_is_tolerated(
        self, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        artifact_root = tmp_path / "gguf"
        artifact_root.mkdir()
        (artifact_root / "invalid-artifact").mkdir()
        artifact_root_warning = CorruptedCacheException(
            f"Repo path is not a valid HuggingFace cache directory: {artifact_root}"
        )
        real_corruption = CorruptedCacheException(
            "Repo path is not a valid HuggingFace cache directory: "
            f"{tmp_path / 'broken'}"
        )
        mock_scan.return_value = _mock_cache_info(
            [], warnings=(artifact_root_warning, real_corruption)
        )

        result = await ModelCatalogService(str(tmp_path)).list_models()

        assert result.cache_warning_count == 1
        assert result.invalid_artifact_count == 1
