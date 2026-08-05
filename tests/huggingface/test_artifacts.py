"""Tests for deterministic GGUF discovery in the native Hugging Face cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import scan_cache_dir

from inference_proxy.huggingface.artifacts import (
    GGUFArtifact,
    GGUFArtifactError,
    GGUFArtifactIndex,
    GGUFDownloadSpec,
)


def _cached_snapshot(
    cache: Path,
    *,
    repo_id: str,
    revision: str,
    files: dict[str, bytes],
) -> Path:
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    blobs = repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        blob = blobs / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        link = snapshot / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(os.path.relpath(blob, link.parent)))
    refs = repo / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    return snapshot


def _index_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    shared_root = tmp_path / "shared export"
    cache_dir = shared_root / "hub"
    node_mount = tmp_path / "node mount"
    cache_dir.mkdir(parents=True)
    node_mount.mkdir()
    return shared_root, cache_dir, node_mount


def test_single_file_discovery_matches_shipped_r2_identity_vector(
    tmp_path: Path,
) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    repo_id = "LiquidAI/LFM2.5-8B-A1B-GGUF"
    revision = "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0"
    entrypoint = "LFM2.5-8B-A1B-Q4_K_M.gguf"
    _cached_snapshot(
        cache_dir,
        repo_id=repo_id,
        revision=revision,
        files={entrypoint: b"staging q4 weights"},
    )

    result = GGUFArtifactIndex(cache_dir, shared_root=shared_root).scan()

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.artifact_id == (
        "7fcf10e5a78a81f27d43e9d37a13adcdac993649d56bae0cd3ef8ab498b1f19d"
    )
    assert artifact.files == (entrypoint,)
    assert not (cache_dir / "gguf").exists()


def test_split_discovery_matches_shipped_r2_identity_vector(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    files = {
        "big-00002-of-00002.gguf": b"second",
        "big-00001-of-00002.gguf": b"first",
    }
    _cached_snapshot(
        cache_dir,
        repo_id="org/Big-GGUF",
        revision="c" * 40,
        files=files,
    )

    artifact = GGUFArtifactIndex(cache_dir, shared_root=shared_root).scan().artifacts[0]

    assert artifact.artifact_id == (
        "5dd18d722798ef5cdd526c3f32eab862f6d18e586cd3517d7647bde32bff6059"
    )
    assert artifact.files == (
        "big-00001-of-00002.gguf",
        "big-00002-of-00002.gguf",
    )
    assert artifact.entrypoint == "big-00001-of-00002.gguf"


def test_get_maps_cache_snapshot_beneath_distinct_shared_and_node_roots(
    tmp_path: Path,
) -> None:
    shared_root, cache_dir, node_mount = _index_layout(tmp_path)
    repo_id = "org/model--with---separators-GGUF"
    revision = "a" * 40
    entrypoint = "quant/model -- Q4_K_M.gguf"
    _cached_snapshot(
        cache_dir,
        repo_id=repo_id,
        revision=revision,
        files={entrypoint: b"weights"},
    )
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)
    public = index.scan().artifacts[0]

    resolved = index.get(public.artifact_id)

    assert resolved is not None
    assert resolved.artifact == public
    assert resolved.node_relative_entrypoint == (
        "hub/models--org--model--with---separators-GGUF/snapshots/"
        f"{revision}/{entrypoint}"
    )
    node_candidate = node_mount / resolved.node_relative_entrypoint
    assert node_candidate.relative_to(node_mount).parts[0] == "hub"


def test_get_validates_only_the_matching_artifact(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    _cached_snapshot(
        cache_dir,
        repo_id="org/first-GGUF",
        revision="a" * 40,
        files={"q4.gguf": b"first"},
    )
    _cached_snapshot(
        cache_dir,
        repo_id="org/target-GGUF",
        revision="b" * 40,
        files={"q8.gguf": b"target"},
    )
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)
    target = next(
        artifact
        for artifact in index.scan().artifacts
        if artifact.repo_id == "org/target-GGUF"
    )

    with patch.object(
        index,
        "_build_indexed_artifact",
        wraps=index._build_indexed_artifact,
    ) as build:
        resolved = index.get(target.artifact_id)

    assert resolved is not None
    assert resolved.artifact == target
    assert build.call_count == 1
    assert build.call_args.kwargs["repo_id"] == "org/target-GGUF"


def test_discovery_is_deterministic_uncapped_and_writes_nothing(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    _cached_snapshot(
        cache_dir,
        repo_id="org/z-GGUF",
        revision="b" * 40,
        files={f"quant-{index:02d}.gguf": str(index).encode() for index in range(30)},
    )
    _cached_snapshot(
        cache_dir,
        repo_id="org/a-GGUF",
        revision="a" * 40,
        files={"nested/q8.gguf": b"q8", "nested/q4.gguf": b"q4"},
    )
    before = sorted(str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*"))
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)

    first = index.scan()
    second = GGUFArtifactIndex(cache_dir, shared_root=shared_root).scan(
        scan_cache_dir(cache_dir)
    )
    after = sorted(str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*"))

    expected_order = sorted(
        (item.repo_id, item.resolved_revision, item.entrypoint)
        for item in first.artifacts
    )
    assert len(first.artifacts) == 32
    assert [
        (item.repo_id, item.resolved_revision, item.entrypoint)
        for item in first.artifacts
    ] == expected_order
    assert second == first
    assert after == before


@pytest.mark.parametrize(
    "files",
    [
        {"model-00001-of-00002.gguf": b"first"},
        {
            "model-00001-of-00002.gguf": b"first",
            "model-00002-of-00003.gguf": b"second",
        },
        {"model-00000-of-00002.gguf": b"invalid"},
    ],
)
def test_incomplete_or_malformed_split_families_are_hidden(
    tmp_path: Path, files: dict[str, bytes]
) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    _cached_snapshot(
        cache_dir,
        repo_id="org/broken-GGUF",
        revision="a" * 40,
        files=files,
    )

    result = GGUFArtifactIndex(cache_dir, shared_root=shared_root).scan()

    assert result.artifacts == ()
    assert result.invalid_count >= 1


def test_broken_or_escaping_cache_entries_fail_closed(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    snapshot = _cached_snapshot(
        cache_dir,
        repo_id="org/model-GGUF",
        revision="a" * 40,
        files={"model.gguf": b"inside"},
    )
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)
    artifact_id = index.scan().artifacts[0].artifact_id
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"outside")
    entrypoint = snapshot / "model.gguf"
    entrypoint.unlink()
    entrypoint.symlink_to(outside)

    result = index.scan()

    assert result.artifacts == ()
    assert result.invalid_count == 1
    with pytest.raises(GGUFArtifactError, match="outside the configured cache"):
        index.get(artifact_id)


def test_download_reconstruction_returns_only_the_requested_artifact(
    tmp_path: Path,
) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    snapshot = _cached_snapshot(
        cache_dir,
        repo_id="org/model-GGUF",
        revision="a" * 40,
        files={"q4.gguf": b"q4", "q8.gguf": b"q8"},
    )
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)

    artifact = index.artifact_from_download(
        repo_id="org/model-GGUF",
        resolved_revision="a" * 40,
        snapshot_path=snapshot,
        spec=GGUFDownloadSpec(files=("q4.gguf",), entrypoint="q4.gguf"),
    )

    assert artifact.files == ("q4.gguf",)
    assert artifact.entrypoint == "q4.gguf"
    assert len(index.scan().artifacts) == 2


def test_download_reconstruction_names_missing_files(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    snapshot = _cached_snapshot(
        cache_dir,
        repo_id="org/model-GGUF",
        revision="a" * 40,
        files={"other.gguf": b"other"},
    )

    with pytest.raises(GGUFArtifactError, match="missing/model.gguf"):
        GGUFArtifactIndex(cache_dir, shared_root=shared_root).artifact_from_download(
            repo_id="org/model-GGUF",
            resolved_revision="a" * 40,
            snapshot_path=snapshot,
            spec=GGUFDownloadSpec(
                files=("missing/model.gguf",), entrypoint="missing/model.gguf"
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        "/absolute.gguf",
        "../escape.gguf",
        "nested//model.gguf",
        "dir\\escape.gguf",
        "model.bin",
        "model.GGUF",
    ],
)
def test_download_spec_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        GGUFDownloadSpec(files=(value,), entrypoint=value)


def test_download_spec_normalizes_complete_split_and_rejects_loose_groups() -> None:
    spec = GGUFDownloadSpec(
        files=("model-00002-of-00002.gguf", "model-00001-of-00002.gguf"),
        entrypoint="model-00001-of-00002.gguf",
    )
    assert spec.files == (
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    )

    with pytest.raises(ValueError, match="one file or one complete split family"):
        GGUFDownloadSpec(
            files=("q4.gguf", "q8.gguf"),
            entrypoint="q4.gguf",
        )
    with pytest.raises(ValueError, match="every declared shard"):
        GGUFDownloadSpec(
            files=("model-00001-of-00002.gguf",),
            entrypoint="model-00001-of-00002.gguf",
        )


@pytest.mark.parametrize(
    ("files", "entrypoint", "message"),
    [
        (
            ("model-00000-of-00001.gguf", "model-00001-of-00001.gguf"),
            "model-00001-of-00001.gguf",
            "more than one shard",
        ),
        (
            ("a-00001-of-00002.gguf", "b-00002-of-00002.gguf"),
            "a-00001-of-00002.gguf",
            "one filename family",
        ),
        (
            ("model-00001-of-00003.gguf", "model-00002-of-00003.gguf"),
            "model-00001-of-00003.gguf",
            "every declared shard",
        ),
        (
            ("model-00001-of-00002.gguf", "model-00003-of-00002.gguf"),
            "model-00001-of-00002.gguf",
            "shards 1 through N",
        ),
        (
            ("model-00001-of-00002.gguf", "model-00002-of-00002.gguf"),
            "model-00002-of-00002.gguf",
            "entrypoint must be shard 1",
        ),
    ],
)
def test_download_spec_rejects_noncanonical_split_contracts(
    files: tuple[str, ...], entrypoint: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GGUFDownloadSpec(files=files, entrypoint=entrypoint)


def test_shared_root_is_optional_for_browsing_but_required_for_resolution(
    tmp_path: Path,
) -> None:
    _shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    _cached_snapshot(
        cache_dir,
        repo_id="org/model-GGUF",
        revision="a" * 40,
        files={"model.gguf": b"model"},
    )
    index = GGUFArtifactIndex(cache_dir)
    artifact = index.scan().artifacts[0]

    with pytest.raises(GGUFArtifactError, match="shared_root is not configured"):
        index.get(artifact.artifact_id)


def test_shared_root_must_contain_cache_dir(tmp_path: Path) -> None:
    _shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    with pytest.raises(GGUFArtifactError, match="contained within shared_root"):
        GGUFArtifactIndex(cache_dir, shared_root=unrelated).validate_shared_root()


def test_shared_root_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)
    assert index.cache_dir == cache_dir.resolve()

    missing = tmp_path / "missing"
    with pytest.raises(GGUFArtifactError, match="configuration is unavailable"):
        GGUFArtifactIndex(cache_dir, shared_root=missing).validate_shared_root()

    file_root = tmp_path / "shared-root-file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(GGUFArtifactError, match="is not a directory"):
        GGUFArtifactIndex(cache_dir, shared_root=file_root).validate_shared_root()


def test_get_rejects_invalid_or_unknown_artifact_ids(tmp_path: Path) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    index = GGUFArtifactIndex(cache_dir, shared_root=shared_root)

    assert index.get("not-an-artifact-id") is None
    assert index.get("a" * 64) is None


@pytest.mark.parametrize(
    ("revision", "snapshot_kind", "message"),
    [
        ("main", "valid", "not a commit SHA"),
        ("a" * 40, "missing", "snapshot is unavailable"),
        ("a" * 40, "outside", "outside the configured cache"),
        ("a" * 40, "wrong-name", "does not match its commit SHA"),
    ],
)
def test_download_reconstruction_rejects_invalid_snapshot_identity(
    tmp_path: Path,
    revision: str,
    snapshot_kind: str,
    message: str,
) -> None:
    shared_root, cache_dir, _node_mount = _index_layout(tmp_path)
    if snapshot_kind == "valid":
        snapshot = _cached_snapshot(
            cache_dir,
            repo_id="org/model-GGUF",
            revision="a" * 40,
            files={"model.gguf": b"weights"},
        )
    elif snapshot_kind == "missing":
        snapshot = cache_dir / "models--org--model-GGUF" / "snapshots" / revision
    elif snapshot_kind == "outside":
        snapshot = tmp_path / revision
        snapshot.mkdir()
        (snapshot / "model.gguf").write_bytes(b"weights")
    else:
        snapshot = cache_dir / "models--org--model-GGUF" / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "model.gguf").write_bytes(b"weights")

    with pytest.raises(GGUFArtifactError, match=message):
        GGUFArtifactIndex(cache_dir, shared_root=shared_root).artifact_from_download(
            repo_id="org/model-GGUF",
            resolved_revision=revision,
            snapshot_path=snapshot,
            spec=GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf"),
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"files": ("model.gguf", "model.gguf")}, "unique"),
        ({"entrypoint": "other.gguf"}, "included in files"),
        ({"file_sizes": {"other.gguf": 1}}, "exact file set"),
        ({"model_alias": ""}, "must not be empty"),
    ],
)
def test_artifact_schema_rejects_inconsistent_contract(
    update: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "artifact_id": "a" * 64,
        "repo_id": "org/model",
        "resolved_revision": "b" * 40,
        "files": ("model.gguf",),
        "entrypoint": "model.gguf",
        "model_alias": "org/model",
        "file_sizes": {"model.gguf": 1},
    }
    values.update(update)

    with pytest.raises(ValueError, match=message):
        GGUFArtifact.model_validate(values)
