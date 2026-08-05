"""Deterministic GGUF artifact discovery over the native Hugging Face cache."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from huggingface_hub import CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GGUFArtifactError(ValueError):
    """Raised when a GGUF artifact cannot be discovered or resolved safely."""


_SPLIT_FILE_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$"
)


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


@dataclass(frozen=True, slots=True)
class _SplitPart:
    path: str
    parent: str
    prefix: str
    index: int
    total: int


def _split_part(value: str) -> _SplitPart | None:
    path = PurePosixPath(value)
    match = _SPLIT_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return _SplitPart(
        path=value,
        parent=path.parent.as_posix(),
        prefix=match.group("prefix"),
        index=int(match.group("index")),
        total=int(match.group("total")),
    )


def _normalize_gguf_family(
    files: tuple[str, ...], entrypoint: str
) -> tuple[tuple[str, ...], str]:
    """Return the one canonical single-file or split-family contract."""
    normalized = tuple(_validated_relative_gguf_path(value) for value in files)
    normalized_entrypoint = _validated_relative_gguf_path(entrypoint)
    if len(set(normalized)) != len(normalized):
        raise ValueError("GGUF file paths must be unique")
    if normalized_entrypoint not in normalized:
        raise ValueError("GGUF entrypoint must be included in files")

    if len(normalized) == 1:
        part = _split_part(normalized[0])
        if part is not None and part.total > 1:
            raise ValueError("GGUF split families must include every declared shard")
        return normalized, normalized_entrypoint

    parts = tuple(_split_part(value) for value in normalized)
    if any(part is None for part in parts):
        raise ValueError(
            "GGUF artifacts must contain one file or one complete split family"
        )
    split_parts = tuple(part for part in parts if part is not None)
    first = split_parts[0]
    if first.total <= 1:
        raise ValueError("GGUF split families must declare more than one shard")
    if any(
        (part.parent, part.prefix, part.total)
        != (first.parent, first.prefix, first.total)
        for part in split_parts
    ):
        raise ValueError("GGUF split shards must belong to one filename family")
    if len(split_parts) != first.total:
        raise ValueError("GGUF split families must include every declared shard")
    by_index = {part.index: part.path for part in split_parts}
    if set(by_index) != set(range(1, first.total + 1)):
        raise ValueError("GGUF split families must contain shards 1 through N exactly")
    ordered = tuple(by_index[index] for index in range(1, first.total + 1))
    if normalized_entrypoint != ordered[0]:
        raise ValueError("GGUF split entrypoint must be shard 1")
    return ordered, ordered[0]


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


class GGUFDownloadSpec(BaseModel):
    """Exact repository files forming one loadable GGUF artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[str, ...] = Field(min_length=1)
    entrypoint: str

    @model_validator(mode="after")
    def normalize_file_family(self) -> GGUFDownloadSpec:
        files, entrypoint = _normalize_gguf_family(self.files, self.entrypoint)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "entrypoint", entrypoint)
        return self


class GGUFArtifact(BaseModel):
    """Public identity and metadata for one immutable GGUF generation."""

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
        files, entrypoint = _normalize_gguf_family(self.files, self.entrypoint)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "entrypoint", entrypoint)
        if set(self.file_sizes) != set(files):
            raise ValueError("GGUF artifact sizes must cover the exact file set")
        if not self.model_alias:
            raise ValueError("GGUF artifact model alias must not be empty")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedGGUFArtifact:
    """Artifact identity plus its exact path relative to the shared export."""

    artifact: GGUFArtifact
    node_relative_entrypoint: str

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def model_alias(self) -> str:
        return self.artifact.model_alias


class ArtifactScanResult(BaseModel):
    """Discovered artifacts plus the number of hidden invalid candidates."""

    model_config = ConfigDict(frozen=True)

    artifacts: tuple[GGUFArtifact, ...]
    invalid_count: int = 0


@dataclass(frozen=True, slots=True)
class _IndexedArtifact:
    artifact: GGUFArtifact
    entrypoint_path: Path


@dataclass(frozen=True, slots=True)
class _IndexScan:
    artifacts: tuple[_IndexedArtifact, ...]
    invalid_count: int


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class GGUFArtifactIndex:
    """Discover and resolve GGUF artifacts without writing cache metadata."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        shared_root: str | Path | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir).resolve()
        self._shared_root = Path(shared_root) if shared_root is not None else None

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def validate_shared_root(self) -> Path:
        """Return the resolved export root or reject llama.cpp provisioning."""
        if self._shared_root is None:
            raise GGUFArtifactError("HuggingFace shared_root is not configured")
        try:
            shared_root = self._shared_root.resolve(strict=True)
            cache_dir = self._cache_dir.resolve(strict=True)
        except OSError as exc:
            raise GGUFArtifactError(
                f"HuggingFace shared-root configuration is unavailable: {exc}"
            ) from exc
        if not shared_root.is_dir():
            raise GGUFArtifactError(
                f"HuggingFace shared_root is not a directory: {shared_root}"
            )
        if not _is_relative_to(cache_dir, shared_root):
            raise GGUFArtifactError(
                "HuggingFace cache_dir must be contained within shared_root"
            )
        return shared_root

    def scan(self, cache_info: HFCacheInfo | None = None) -> ArtifactScanResult:
        """Return native-cache artifacts in deterministic display order."""
        info = (
            cache_info
            if cache_info is not None
            else scan_cache_dir(str(self._cache_dir))
        )
        result = self._discover(info)
        return ArtifactScanResult(
            artifacts=tuple(item.artifact for item in result.artifacts),
            invalid_count=result.invalid_count,
        )

    def get(self, artifact_id: str) -> ResolvedGGUFArtifact | None:
        """Resolve one artifact and its node-facing path from a fresh cache scan."""
        if re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None:
            return None
        cache_info = scan_cache_dir(str(self._cache_dir))
        for repo in sorted(cache_info.repos, key=lambda item: item.repo_id):
            if repo.repo_type != "model":
                continue
            for revision in sorted(repo.revisions, key=lambda item: item.commit_hash):
                specs, _invalid_specs = self._revision_specs(revision)
                for spec in specs:
                    candidate_id = _artifact_identity(
                        repo_id=repo.repo_id,
                        resolved_revision=revision.commit_hash,
                        files=spec.files,
                        entrypoint=spec.entrypoint,
                        model_alias=repo.repo_id,
                    )
                    if candidate_id != artifact_id:
                        continue
                    try:
                        indexed = self._build_indexed_artifact(
                            repo_id=repo.repo_id,
                            resolved_revision=revision.commit_hash,
                            snapshot_path=revision.snapshot_path,
                            spec=spec,
                        )
                    except (GGUFArtifactError, OSError, ValueError) as exc:
                        raise GGUFArtifactError(str(exc)) from exc
                    return self._resolve_for_node(indexed)
        return None

    def artifact_from_download(
        self,
        *,
        repo_id: str,
        resolved_revision: str,
        snapshot_path: str | Path,
        spec: GGUFDownloadSpec,
    ) -> GGUFArtifact:
        """Validate and reconstruct exactly the artifact requested for download."""
        return self._build_indexed_artifact(
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            snapshot_path=Path(snapshot_path),
            spec=spec,
        ).artifact

    def _discover(self, cache_info: HFCacheInfo) -> _IndexScan:
        artifacts: list[_IndexedArtifact] = []
        invalid_count = 0
        for repo in sorted(cache_info.repos, key=lambda item: item.repo_id):
            if repo.repo_type != "model":
                continue
            for revision in sorted(repo.revisions, key=lambda item: item.commit_hash):
                specs, invalid_specs = self._revision_specs(revision)
                invalid_count += invalid_specs
                for spec in specs:
                    try:
                        indexed = self._build_indexed_artifact(
                            repo_id=repo.repo_id,
                            resolved_revision=revision.commit_hash,
                            snapshot_path=revision.snapshot_path,
                            spec=spec,
                        )
                    except (GGUFArtifactError, OSError, ValueError):
                        invalid_count += 1
                    else:
                        artifacts.append(indexed)
        artifacts.sort(
            key=lambda item: (
                item.artifact.repo_id,
                item.artifact.resolved_revision,
                item.artifact.entrypoint,
            )
        )
        return _IndexScan(tuple(artifacts), invalid_count)

    def _revision_specs(
        self, revision: CachedRevisionInfo
    ) -> tuple[tuple[GGUFDownloadSpec, ...], int]:
        standalone: list[str] = []
        split_groups: dict[tuple[str, str, int], dict[int, str]] = defaultdict(dict)
        invalid_count = 0

        for file in sorted(revision.files, key=lambda item: str(item.file_path)):
            if file.file_path.suffix != ".gguf":
                continue
            try:
                relative = file.file_path.relative_to(revision.snapshot_path).as_posix()
                relative = _validated_relative_gguf_path(relative)
            except ValueError:
                invalid_count += 1
                continue
            part = _split_part(relative)
            if part is None or (part.total == 1 and part.index == 1):
                standalone.append(relative)
                continue
            if part.total <= 1 or not 1 <= part.index <= part.total:
                invalid_count += 1
                continue
            key = (part.parent, part.prefix, part.total)
            if part.index in split_groups[key]:
                invalid_count += 1
                continue
            split_groups[key][part.index] = relative

        specs = [
            GGUFDownloadSpec(files=(relative,), entrypoint=relative)
            for relative in standalone
        ]
        for (_parent, _prefix, total), by_index in split_groups.items():
            if set(by_index) != set(range(1, total + 1)):
                invalid_count += 1
                continue
            files = tuple(by_index[index] for index in range(1, total + 1))
            specs.append(GGUFDownloadSpec(files=files, entrypoint=files[0]))
        return tuple(sorted(specs, key=lambda item: item.entrypoint)), invalid_count

    def _build_indexed_artifact(
        self,
        *,
        repo_id: str,
        resolved_revision: str,
        snapshot_path: Path,
        spec: GGUFDownloadSpec,
    ) -> _IndexedArtifact:
        if re.fullmatch(r"[0-9a-f]{40,64}", resolved_revision) is None:
            raise GGUFArtifactError(
                f"Resolved HuggingFace revision is not a commit SHA: {resolved_revision!r}"
            )
        try:
            snapshot = snapshot_path.resolve(strict=True)
        except OSError as exc:
            raise GGUFArtifactError(f"Resolved snapshot is unavailable: {exc}") from exc
        if not _is_relative_to(snapshot, self._cache_dir):
            raise GGUFArtifactError("Resolved snapshot is outside the configured cache")
        if snapshot.name != resolved_revision:
            raise GGUFArtifactError(
                "Resolved snapshot path does not match its commit SHA"
            )

        sources: dict[str, Path] = {}
        sizes: dict[str, int] = {}
        missing: list[str] = []
        escaped: list[str] = []
        for relative in spec.files:
            source = snapshot.joinpath(*PurePosixPath(relative).parts)
            if not source.is_file():
                missing.append(relative)
                continue
            try:
                resolved_source = source.resolve(strict=True)
            except OSError:
                missing.append(relative)
                continue
            if not _is_relative_to(resolved_source, self._cache_dir):
                escaped.append(relative)
                continue
            sources[relative] = source
            sizes[relative] = resolved_source.stat().st_size
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

        artifact_id = _artifact_identity(
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            files=spec.files,
            entrypoint=spec.entrypoint,
            model_alias=repo_id,
        )
        artifact = GGUFArtifact(
            artifact_id=artifact_id,
            repo_id=repo_id,
            resolved_revision=resolved_revision,
            files=spec.files,
            entrypoint=spec.entrypoint,
            model_alias=repo_id,
            file_sizes=sizes,
        )
        return _IndexedArtifact(artifact, sources[spec.entrypoint])

    def _resolve_for_node(self, indexed: _IndexedArtifact) -> ResolvedGGUFArtifact:
        shared_root = self.validate_shared_root()
        try:
            candidate = indexed.entrypoint_path
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise GGUFArtifactError(
                f"GGUF artifact {indexed.artifact.artifact_id!r} is unavailable: {exc}"
            ) from exc
        if not _is_relative_to(candidate, shared_root) or not _is_relative_to(
            resolved, shared_root
        ):
            raise GGUFArtifactError(
                "GGUF artifact entrypoint resolves outside HuggingFace shared_root"
            )
        relative = candidate.relative_to(shared_root).as_posix()
        return ResolvedGGUFArtifact(indexed.artifact, relative)
