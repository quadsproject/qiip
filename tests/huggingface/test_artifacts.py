"""Tests for immutable GGUF artifact publication."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from inference_proxy.huggingface.artifacts import (
    GGUFArtifact,
    GGUFArtifactError,
    GGUFArtifactStore,
    GGUFDownloadSpec,
)


def _snapshot(
    cache: Path,
    *,
    repo_id: str = "org/model--with---separators-GGUF",
    revision: str = "a" * 40,
    files: tuple[str, ...] = ("weights/model Q4_K_M.gguf",),
) -> Path:
    snapshot = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    for relative in files:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    return snapshot


def test_publish_uses_resolved_revision_and_relative_links(tmp_path: Path) -> None:
    repo_id = "org/model--with---separators-GGUF"
    revision = "a" * 40
    files = (
        "quant/model -- Q4-00001-of-00002.gguf",
        "quant/model -- Q4-00002-of-00002.gguf",
    )
    snapshot = _snapshot(tmp_path, repo_id=repo_id, revision=revision, files=files)
    store = GGUFArtifactStore(tmp_path)

    artifact = store.publish(
        repo_id=repo_id,
        resolved_revision=revision,
        snapshot_path=snapshot,
        spec=GGUFDownloadSpec(files=files, entrypoint=files[0]),
    )

    assert artifact.repo_id == repo_id
    assert artifact.resolved_revision == revision
    assert artifact.model_alias == repo_id
    assert artifact.cache_relative_entrypoint.endswith(
        f"/{artifact.artifact_id}/files/{files[0]}"
    )
    for relative in files:
        link = tmp_path / "gguf" / artifact.artifact_id / "files" / relative
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
        assert link.resolve() == (snapshot / relative).resolve()
    assert store.get(artifact.artifact_id) == artifact


def test_resolved_revision_creates_distinct_generations(tmp_path: Path) -> None:
    spec = GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf")
    store = GGUFArtifactStore(tmp_path)
    first = store.publish(
        repo_id="org/model",
        resolved_revision="a" * 40,
        snapshot_path=_snapshot(
            tmp_path, repo_id="org/model", revision="a" * 40, files=spec.files
        ),
        spec=spec,
    )
    second = store.publish(
        repo_id="org/model",
        resolved_revision="b" * 40,
        snapshot_path=_snapshot(
            tmp_path, repo_id="org/model", revision="b" * 40, files=spec.files
        ),
        spec=spec,
    )

    assert first.artifact_id != second.artifact_id
    assert {item.artifact_id for item in store.scan().artifacts} == {
        first.artifact_id,
        second.artifact_id,
    }


def test_missing_exact_file_names_the_file(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, files=("other.gguf",))
    store = GGUFArtifactStore(tmp_path)

    with pytest.raises(GGUFArtifactError, match="missing/model.gguf"):
        store.publish(
            repo_id="org/model--with---separators-GGUF",
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
        "dir\\escape.gguf",
        "model.bin",
        "model.GGUF",
    ],
)
def test_download_spec_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        GGUFDownloadSpec(files=(value,), entrypoint=value)


def test_download_spec_rejects_noncanonical_duplicate_and_missing_entrypoint() -> None:
    with pytest.raises(ValueError, match="canonical"):
        GGUFDownloadSpec(files=("dir//model.gguf",), entrypoint="dir//model.gguf")
    with pytest.raises(ValueError, match="unique"):
        GGUFDownloadSpec(files=("model.gguf", "model.gguf"), entrypoint="model.gguf")
    with pytest.raises(ValueError, match="included in files"):
        GGUFDownloadSpec(files=("model.gguf",), entrypoint="other.gguf")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"files": ("model.gguf", "model.gguf")}, "unique"),
        ({"entrypoint": "other.gguf"}, "included in files"),
        ({"file_sizes": {"other.gguf": 1}}, "exact file set"),
        ({"model_alias": ""}, "must not be empty"),
    ],
)
def test_artifact_manifest_rejects_inconsistent_contract(
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


def test_publish_rejects_source_symlink_outside_cache(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, files=("placeholder.gguf",))
    outside = tmp_path.parent / f"{tmp_path.name}-outside.gguf"
    outside.write_bytes(b"outside")
    escaped = snapshot / "escape.gguf"
    escaped.symlink_to(outside)

    with pytest.raises(GGUFArtifactError, match="outside the configured cache"):
        GGUFArtifactStore(tmp_path).publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="a" * 40,
            snapshot_path=snapshot,
            spec=GGUFDownloadSpec(files=("escape.gguf",), entrypoint="escape.gguf"),
        )


def test_publish_rejects_invalid_revision_and_snapshot_location(tmp_path: Path) -> None:
    store = GGUFArtifactStore(tmp_path)
    spec = GGUFDownloadSpec(files=("model.gguf",), entrypoint="model.gguf")
    snapshot = _snapshot(tmp_path, files=spec.files)

    with pytest.raises(GGUFArtifactError, match="not a commit SHA"):
        store.publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="main",
            snapshot_path=snapshot,
            spec=spec,
        )

    outside_cache = tmp_path.parent / f"{tmp_path.name}-outside-cache"
    outside_snapshot = _snapshot(outside_cache, files=spec.files)
    with pytest.raises(GGUFArtifactError, match="outside the configured cache"):
        store.publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="a" * 40,
            snapshot_path=outside_snapshot,
            spec=spec,
        )

    with pytest.raises(GGUFArtifactError, match="does not match its commit SHA"):
        store.publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="b" * 40,
            snapshot_path=snapshot,
            spec=spec,
        )


def test_republishing_identical_artifact_is_idempotent(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    store = GGUFArtifactStore(tmp_path)
    spec = GGUFDownloadSpec(
        files=("weights/model Q4_K_M.gguf",),
        entrypoint="weights/model Q4_K_M.gguf",
    )

    first = store.publish(
        repo_id="org/model--with---separators-GGUF",
        resolved_revision="a" * 40,
        snapshot_path=snapshot,
        spec=spec,
    )
    second = store.publish(
        repo_id="org/model--with---separators-GGUF",
        resolved_revision="a" * 40,
        snapshot_path=snapshot,
        spec=spec,
    )

    assert second == first


def test_interrupted_staging_directory_is_not_cataloged(tmp_path: Path) -> None:
    root = tmp_path / "gguf"
    (root / ".staging-dead").mkdir(parents=True)
    (root / "not-an-artifact").mkdir()

    result = GGUFArtifactStore(tmp_path).scan()

    assert result.artifacts == ()
    assert result.invalid_count == 1


def test_get_rejects_invalid_or_missing_artifact_id(tmp_path: Path) -> None:
    store = GGUFArtifactStore(tmp_path)

    assert store.get("not-an-id") is None
    assert store.get("a" * 64) is None


def test_scan_counts_non_directory_entries(tmp_path: Path) -> None:
    root = tmp_path / "gguf"
    root.mkdir()
    (root / "unexpected-file").write_text("not an artifact", encoding="utf-8")

    result = GGUFArtifactStore(tmp_path).scan()

    assert result.artifacts == ()
    assert result.invalid_count == 1


@pytest.mark.parametrize("root_kind", ["symlink", "file"])
def test_artifact_root_cannot_redirect_publication(
    tmp_path: Path, root_kind: str
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-artifact-root"
    outside.mkdir()
    root = tmp_path / "gguf"
    if root_kind == "symlink":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.write_text("not a directory", encoding="utf-8")
    snapshot = _snapshot(tmp_path)
    store = GGUFArtifactStore(tmp_path)
    spec = GGUFDownloadSpec(
        files=("weights/model Q4_K_M.gguf",),
        entrypoint="weights/model Q4_K_M.gguf",
    )

    with pytest.raises(GGUFArtifactError, match="not a real directory"):
        store.publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="a" * 40,
            snapshot_path=snapshot,
            spec=spec,
        )

    assert store.scan().invalid_count == 1
    assert not any(outside.iterdir())


def _published_artifact(tmp_path: Path) -> tuple[GGUFArtifactStore, GGUFArtifact, Path]:
    store = GGUFArtifactStore(tmp_path)
    snapshot = _snapshot(tmp_path)
    artifact = store.publish(
        repo_id="org/model--with---separators-GGUF",
        resolved_revision="a" * 40,
        snapshot_path=snapshot,
        spec=GGUFDownloadSpec(
            files=("weights/model Q4_K_M.gguf",),
            entrypoint="weights/model Q4_K_M.gguf",
        ),
    )
    return store, artifact, snapshot


def test_corrupt_manifest_identity_is_hidden_from_scan(tmp_path: Path) -> None:
    store, artifact, _snapshot_path = _published_artifact(tmp_path)
    directory = store.root / artifact.artifact_id
    manifest = directory / "artifact.json"
    values = json.loads(manifest.read_text(encoding="utf-8"))
    values["repo_id"] = "org/different-model"
    manifest.write_text(json.dumps(values), encoding="utf-8")

    result = store.scan()

    assert result.artifacts == ()
    assert result.invalid_count == 1
    with pytest.raises(GGUFArtifactError, match="identity is invalid"):
        store.get(artifact.artifact_id)


def test_manifest_id_must_match_directory_name(tmp_path: Path) -> None:
    store, artifact, _snapshot_path = _published_artifact(tmp_path)
    original = store.root / artifact.artifact_id
    wrong = store.root / ("b" * 64)
    original.rename(wrong)

    with pytest.raises(GGUFArtifactError, match="directory does not match"):
        store.get("b" * 64)


@pytest.mark.parametrize("failure", ["absolute", "broken", "wrong-source", "size"])
def test_artifact_files_fail_closed_after_publication(
    tmp_path: Path, failure: str
) -> None:
    store, artifact, snapshot = _published_artifact(tmp_path)
    link = store.root / artifact.artifact_id / "files" / artifact.entrypoint
    source = snapshot / artifact.entrypoint

    if failure == "absolute":
        link.unlink()
        link.symlink_to(source.resolve())
        message = "not a relative symlink"
    elif failure == "broken":
        source.unlink()
        message = "is unavailable"
    elif failure == "wrong-source":
        other = snapshot / "other.gguf"
        other.write_bytes(b"other")
        link.unlink()
        link.symlink_to(os.path.relpath(other, link.parent))
        message = "does not match its snapshot source"
    else:
        source.write_bytes(b"changed-size")
        message = "size does not match its manifest"

    with pytest.raises(GGUFArtifactError, match=message):
        store.get(artifact.artifact_id)


def test_concurrent_identical_publications_accept_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    store = GGUFArtifactStore(tmp_path)
    spec = GGUFDownloadSpec(
        files=("weights/model Q4_K_M.gguf",),
        entrypoint="weights/model Q4_K_M.gguf",
    )
    barrier = threading.Barrier(2)
    original_rename = Path.rename

    def synchronized_rename(path: Path, target: Path) -> Path:
        if path.name.startswith(".staging-"):
            barrier.wait(timeout=2)
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", synchronized_rename)

    def publish() -> str:
        return store.publish(
            repo_id="org/model--with---separators-GGUF",
            resolved_revision="a" * 40,
            snapshot_path=snapshot,
            spec=spec,
        ).artifact_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: publish(), range(2)))

    assert results[0] == results[1]
    assert len(store.scan().artifacts) == 1
    assert not list(store.root.glob(".staging-*"))
