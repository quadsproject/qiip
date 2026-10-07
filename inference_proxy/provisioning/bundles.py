"""Snapshot the exact engine/recorder files sent to a node."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def bundle_files(engine_dir: Path | None, common_dir: Path) -> dict[str, bytes]:
    files = {}
    for directory in (engine_dir, common_dir):
        if directory is None:
            continue
        for path in sorted(directory.glob("*")):
            if path.is_file():
                files[f"{directory.name}/{path.name}"] = path.read_bytes()
    for name in ("log_store.py", "diagnostics.py"):
        files[f"common/{name}"] = Path(__file__).with_name(name).read_bytes()
    return files


def bundle_manifest(files: dict[str, bytes]) -> tuple[str, dict[str, object]]:
    manifest: dict[str, object] = {
        "schema": 1,
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(files.items())
        },
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return identity, manifest


def write_bundle(root: Path, files: dict[str, bytes]) -> str:
    identity, manifest = bundle_manifest(files)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "BUNDLE.json").write_text(json.dumps(manifest, sort_keys=True))
    return identity
