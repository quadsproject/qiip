"""Immutable GGUF artifact publication and validation.

GGUF downloads are partial Hugging Face snapshots.  This module publishes an
explicit, versioned manifest plus relative links under ``<cache>/gguf`` so the
gateway and provisioned nodes share one exact serving contract.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GGUFArtifactError(ValueError):
    """Raised when a GGUF artifact cannot be published or validated."""


def _validated_relative_gguf_path(value: str) -> str:
    """Return one canonical repository-relative GGUF path."""
    if not value or "\\" in value:
        raise ValueError("GGUF paths must be non-empty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("GGUF paths must be canonical relative paths")
    if str(path) != value:
        raise ValueError("GGUF paths must be canonical relative paths")
    if path.suffix != ".gguf":
        raise ValueError("GGUF paths must end in .gguf")
    return value


class GGUFDownloadSpec(BaseModel):
    """Exact repository files forming one loadable GGUF artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[str, ...] = Field(min_length=1)
    entrypoint: str

    @field_validator("files")
    @classmethod
    def validate_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validated_relative_gguf_path(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("GGUF file paths must be unique")
        return normalized

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        return _validated_relative_gguf_path(value)

    @model_validator(mode="after")
    def entrypoint_is_in_files(self) -> GGUFDownloadSpec:
        if self.entrypoint not in self.files:
            raise ValueError("GGUF entrypoint must be included in files")
        return self


class GGUFArtifact(BaseModel):
    """Versioned manifest for one immutable, loadable GGUF generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo_id: str
    resolved_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    files: tuple[str, ...] = Field(min_length=1)
    entrypoint: str
    model_alias: str
    file_sizes: dict[str, int]

    @field_validator("files")
    @classmethod
    def validate_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validated_relative_gguf_path(value) for value in values)

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        return _validated_relative_gguf_path(value)

    @model_validator(mode="after")
    def validate_file_contract(self) -> GGUFArtifact:
        if len(set(self.files)) != len(self.files):
            raise ValueError("GGUF artifact files must be unique")
        if self.entrypoint not in self.files:
            raise ValueError("GGUF artifact entrypoint must be included in files")
        if set(self.file_sizes) != set(self.files):
            raise ValueError("GGUF artifact sizes must cover the exact file set")
        if not self.model_alias:
            raise ValueError("GGUF artifact model alias must not be empty")
        return self

    @property
    def cache_relative_entrypoint(self) -> str:
        """Return the exact path understood by the node launcher."""
        return str(PurePosixPath("gguf", self.artifact_id, "files", self.entrypoint))


class ArtifactScanResult(BaseModel):
    """Validated artifacts plus the number of hidden invalid manifests."""

    model_config = ConfigDict(frozen=True)

    artifacts: tuple[GGUFArtifact, ...]
    invalid_count: int = 0


def _artifact_identity(
    *,
    repo_id: str,
    resolved_revision: str,
    files: tuple[str, ...],
    entrypoint: str,
    model_alias: str,
) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "repo_id": repo_id,
            "resolved_revision": resolved_revision,
            "files": files,
            "entrypoint": entrypoint,
            "model_alias": model_alias,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class GGUFArtifactStore:
    """Publish and resolve immutable GGUF manifests under one HF cache."""

    _MANIFEST_NAME = "artifact.json"

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir).resolve()
        self._root = self._cache_dir / "gguf"

    @property
    def root(self) -> Path:
        return self._root

    def _prepare_root(self, *, create: bool) -> bool:
        """Reject an artifact root that could redirect publication elsewhere."""
        if self._root.is_symlink() or (
            os.path.lexists(self._root) and not self._root.is_dir()
        ):
            raise GGUFArtifactError(
                f"GGUF artifact root is not a real directory: {self._root}"
            )
        if self._root.is_dir():
            return True
        if not create:
            return False
        try:
            self._root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Another identical publication may create the shared root between
            # the checks above and mkdir. Accept only the intended directory.
            if self._root.is_symlink() or not self._root.is_dir():
                raise GGUFArtifactError(
                    f"GGUF artifact root is not a real directory: {self._root}"
                ) from None
        return True

    def publish(
        self,
        *,
        repo_id: str,
        resolved_revision: str,
        snapshot_path: str | Path,
        spec: GGUFDownloadSpec,
    ) -> GGUFArtifact:
        """Atomically publish *spec* from one resolved HF snapshot."""
        if not re.fullmatch(r"[0-9a-f]{40,64}", resolved_revision):
            raise GGUFArtifactError(
                f"Resolved HuggingFace revision is not a commit SHA: {resolved_revision!r}"
            )

        snapshot = Path(snapshot_path).resolve(strict=True)
        if not _is_relative_to(snapshot, self._cache_dir):
            raise GGUFArtifactError("Resolved snapshot is outside the configured cache")
        if snapshot.name != resolved_revision:
            raise GGUFArtifactError(
                "Resolved snapshot path does not match its commit SHA"
            )

        sources: dict[str, Path] = {}
        missing: list[str] = []
        escaped: list[str] = []
        for relative in spec.files:
            source = snapshot.joinpath(*PurePosixPath(relative).parts)
            if not source.is_file():
                missing.append(relative)
                continue
            resolved_source = source.resolve(strict=True)
            if not _is_relative_to(resolved_source, self._cache_dir):
                escaped.append(relative)
                continue
            sources[relative] = source
        if missing:
            raise GGUFArtifactError(
                "Resolved snapshot "
                f"{resolved_revision} does not contain requested GGUF file(s): "
                + ", ".join(missing)
            )
        if escaped:
            raise GGUFArtifactError(
                "Requested GGUF file(s) resolve outside the configured cache: "
                + ", ".join(escaped)
            )

        files = tuple(spec.files)
        artifact_id = _artifact_identity(
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            files=files,
            entrypoint=spec.entrypoint,
            model_alias=repo_id,
        )
        artifact = GGUFArtifact(
            artifact_id=artifact_id,
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            files=files,
            entrypoint=spec.entrypoint,
            model_alias=repo_id,
            file_sizes={
                name: source.stat().st_size for name, source in sources.items()
            },
        )

        self._prepare_root(create=True)
        final_dir = self._root / artifact_id
        if final_dir.exists():
            return self._require_matching(final_dir, artifact)

        staging = Path(
            tempfile.mkdtemp(prefix=f".staging-{artifact_id[:12]}-", dir=self._root)
        )
        try:
            for relative, source in sources.items():
                destination = staging / "files" / Path(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(os.path.relpath(source, destination.parent))
                if destination.resolve(strict=True) != source.resolve(strict=True):
                    raise GGUFArtifactError(
                        f"Published GGUF link does not resolve to {relative!r}"
                    )

            manifest_tmp = staging / f".{self._MANIFEST_NAME}.tmp"
            manifest_tmp.write_text(
                artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            os.replace(manifest_tmp, staging / self._MANIFEST_NAME)
            try:
                staging.rename(final_dir)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                return self._require_matching(final_dir, artifact)
            return self._load_directory(final_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get(self, artifact_id: str) -> GGUFArtifact | None:
        """Return one fully validated artifact, or ``None`` if absent."""
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            return None
        if not self._prepare_root(create=False):
            return None
        directory = self._root / artifact_id
        if not directory.is_dir():
            return None
        return self._load_directory(directory)

    def scan(self) -> ArtifactScanResult:
        """Return all valid published artifacts and count invalid entries."""
        try:
            root_exists = self._prepare_root(create=False)
        except GGUFArtifactError:
            return ArtifactScanResult(artifacts=(), invalid_count=1)
        if not root_exists:
            return ArtifactScanResult(artifacts=())
        artifacts: list[GGUFArtifact] = []
        invalid_count = 0
        for directory in sorted(self._root.iterdir()):
            if directory.name.startswith("."):
                continue
            if not directory.is_dir():
                invalid_count += 1
                continue
            try:
                artifacts.append(self._load_directory(directory))
            except (GGUFArtifactError, OSError, ValueError):
                invalid_count += 1
        return ArtifactScanResult(
            artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
            invalid_count=invalid_count,
        )

    def _require_matching(
        self, directory: Path, expected: GGUFArtifact
    ) -> GGUFArtifact:
        actual = self._load_directory(directory)
        if actual != expected:
            raise GGUFArtifactError(
                f"Existing artifact {expected.artifact_id} does not match its identity"
            )
        return actual

    def _load_directory(self, directory: Path) -> GGUFArtifact:
        manifest = directory / self._MANIFEST_NAME
        try:
            artifact = GGUFArtifact.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise GGUFArtifactError(
                f"Invalid GGUF artifact manifest in {directory.name!r}: {exc}"
            ) from exc

        if artifact.artifact_id != directory.name:
            raise GGUFArtifactError("Artifact directory does not match manifest ID")
        expected_id = _artifact_identity(
            repo_id=artifact.repo_id,
            resolved_revision=artifact.resolved_revision,
            files=artifact.files,
            entrypoint=artifact.entrypoint,
            model_alias=artifact.model_alias,
        )
        if artifact.artifact_id != expected_id:
            raise GGUFArtifactError("Artifact manifest identity is invalid")

        snapshot = (
            self._cache_dir
            / f"models--{artifact.repo_id.replace('/', '--')}"
            / "snapshots"
            / artifact.resolved_revision
        )
        for relative in artifact.files:
            link = directory / "files" / Path(*PurePosixPath(relative).parts)
            if not link.is_symlink() or os.path.isabs(os.readlink(link)):
                raise GGUFArtifactError(
                    f"Artifact file {relative!r} is not a relative symlink"
                )
            expected_source = snapshot.joinpath(*PurePosixPath(relative).parts)
            try:
                resolved_link = link.resolve(strict=True)
                resolved_source = expected_source.resolve(strict=True)
            except OSError as exc:
                raise GGUFArtifactError(
                    f"Artifact file {relative!r} is unavailable: {exc}"
                ) from exc
            if resolved_link != resolved_source or not _is_relative_to(
                resolved_link, self._cache_dir
            ):
                raise GGUFArtifactError(
                    f"Artifact file {relative!r} does not match its snapshot source"
                )
            if resolved_link.stat().st_size != artifact.file_sizes[relative]:
                raise GGUFArtifactError(
                    f"Artifact file {relative!r} size does not match its manifest"
                )
        return artifact
