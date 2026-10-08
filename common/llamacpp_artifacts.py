#!/usr/bin/env python3
"""Verified llama.cpp artifact publication/selection and build sizing (EL9 stdlib)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    from common import generations
except ImportError:  # Executed from a node's immutable script bundle.
    import generations  # type: ignore[no-redef,import-not-found]

TOOLS = ["llama-server", "llama-fit-params", "llama-quantize", "cuda-probe"]
IDENTITY_KEYS = {
    "publication_schema",
    "version",
    "source_sha256",
    "build_profile",
    "fit_cli_patch_sha256",
    "compute_capabilities",
    "compiled_cuda_architectures",
    "cmake_cuda_architectures",
    "cuda_toolkit",
    "profile",
    "compiler",
    "cpu_flags",
    "os_id",
    "os_major",
    "arch",
    "glibc",
}


def metadata(identity: str) -> dict[str, str]:
    if not isinstance(identity, str):
        raise ValueError("Invalid llama.cpp artifact identity")
    rows = identity.splitlines()
    result = dict(row.split("=", 1) for row in rows)
    if len(result) != len(rows) or set(result) != IDENTITY_KEYS:
        raise ValueError("Incomplete llama.cpp artifact identity")
    if result["publication_schema"] != "4":
        raise ValueError("Unsupported llama.cpp artifact schema")
    for key in ("source_sha256", "fit_cli_patch_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", result[key]):
            raise ValueError("Invalid source/transformation digest")
    if not result["compiler"] or not re.fullmatch(
        r"\d+\.\d+\.\d+", result["cuda_toolkit"]
    ):
        raise ValueError("Missing compiler/CUDA identity")
    version_tuple(result["glibc"])
    feature_set(result["cpu_flags"])
    if result["cmake_cuda_architectures"] == "native":
        raise ValueError("Artifact must record resolved CUDA compile targets")
    if not re.fullmatch(
        r"[1-9][0-9]*(?:,[1-9][0-9]*)*", result["compiled_cuda_architectures"]
    ):
        raise ValueError("Invalid compiled CUDA architectures")
    resolved = targets(
        result["cmake_cuda_architectures"], result["compute_capabilities"]
    )
    if resolved["compiled_cuda_architectures"] != result["compiled_cuda_architectures"]:
        raise ValueError("Compiled CUDA architecture identity mismatch")
    return result


def version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)+", value):
        raise ValueError("Invalid ABI version")
    return tuple(int(part) for part in value.split("."))


def feature_set(value: str) -> set[str]:
    if not re.fullmatch(r"[a-z0-9_]+(?:,[a-z0-9_]+)*", value):
        raise ValueError("Invalid CPU feature identity")
    flags = set(value.split(","))
    if ",".join(sorted(flags)) != value:
        raise ValueError("CPU feature identity must be sorted and unique")
    return flags


def cpu_features(cpuinfo: Path) -> dict[str, str]:
    """Require artifact features on every consumer CPU, including mixed CPUs."""
    cpus = []
    for block in re.split(r"\n\s*\n", cpuinfo.read_text()):
        if not re.search(r"^processor\s*:", block, re.M):
            continue
        lines = re.findall(r"^flags\s*:\s*([^\n]+)", block, re.M)
        if len(lines) != 1:
            raise ValueError("Cannot determine CPU features for every processor")
        flags = ",".join(sorted(set(lines[0].split())))
        cpus.append(feature_set(flags))
    if not cpus:
        raise ValueError("Cannot determine CPU features from cpuinfo")
    return {
        "cpu_flags": ",".join(sorted(set.intersection(*cpus))),
        "build_cpu_flags": ",".join(sorted(set.union(*cpus))),
    }


def compatible(identity: str, host: dict[str, str]) -> bool:
    info = metadata(identity)
    exact = (
        "version",
        "source_sha256",
        "build_profile",
        "fit_cli_patch_sha256",
        "profile",
        "os_id",
        "os_major",
        "arch",
    )
    if any(info[key] != host[key] for key in exact):
        return False
    if not info["cuda_toolkit"].startswith(host["cuda_toolkit"] + "."):
        return False
    if version_tuple(info["glibc"]) > version_tuple(host["glibc"]):
        return False
    if not feature_set(info["cpu_flags"]).issubset(feature_set(host["cpu_flags"])):
        return False
    # Match the actual engine/probe targets, never the producer's GPU inventory.
    # Require exact targets even for PTX; do not guess forward compatibility.
    required = {
        capability.replace(".", "")
        for capability in host["compute_capabilities"].split(",")
    }
    if not required.issubset(info["compiled_cuda_architectures"].split(",")):
        return False
    return host["cmake_cuda_architectures"] == "native" or (
        info["cmake_cuda_architectures"] == host["cmake_cuda_architectures"]
    )


def targets(architectures: str, capabilities: str) -> dict[str, str]:
    """Resolve native to explicit CMake targets shared by the engine and probe."""
    if not re.fullmatch(r"[1-9][0-9]*\.[0-9](?:,[1-9][0-9]*\.[0-9])*", capabilities):
        raise ValueError("Invalid measured CUDA capabilities")
    if architectures == "native":
        architectures = ";".join(
            capability.replace(".", "") for capability in capabilities.split(",")
        )
    if not re.fullmatch(
        r"[1-9][0-9]*(?:-(?:real|virtual))?(?:;[1-9][0-9]*(?:-(?:real|virtual))?)*",
        architectures,
    ):
        raise ValueError(
            "CUDA architectures must be native or explicit numeric CMake targets"
        )
    compiled = sorted(
        {item.split("-", 1)[0] for item in architectures.split(";")}, key=int
    )
    return {
        "cmake_cuda_architectures": architectures,
        "compiled_cuda_architectures": ",".join(compiled),
    }


def verify_package(root: Path, identity: str | None = None) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("Artifact root must not be a symlink")
    manifest = json.loads((root / "RUNTIME.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid artifact manifest")
    metadata(manifest["identity"])
    expected = {f"bin/{name}" for name in TOOLS}
    if (
        set(manifest["files"]) != expected
        or not isinstance(manifest.get("external_links", {}), dict)
        or manifest.get("external_links")
    ):
        raise ValueError("Artifact must contain all three tools and its CUDA probe")
    data = manifest.get("data_files", {})
    if not isinstance(data, dict) or not all(isinstance(name, str) for name in data):
        raise ValueError("Invalid artifact data manifest")
    allowed = expected | set(data) | {"RUNTIME.json"}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Artifact contains a link or special file")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != allowed or any(
        name != "BUILD-INFO"
        and (
            len(name.split("/")) != 2
            or not name.startswith("lib/")
            or name.split("/")[1] in {"", ".", ".."}
        )
        for name in data
    ):
        raise ValueError("Artifact file inventory mismatch")
    generations.verify_runtime(root, identity)
    if (
        "BUILD-INFO" not in data
        or (root / "BUILD-INFO").read_text().rstrip("\n") != manifest["identity"]
    ):
        raise ValueError("Artifact build identity mismatch")
    return manifest


def safe_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Invalid artifact URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(
            "Artifact URL must be HTTP(S), without credentials or fragments"
        )
    return value


def select(catalog: Path, host: dict[str, str]) -> dict[str, Any] | None:
    document = json.loads(catalog.read_text())
    if (
        not isinstance(document, dict)
        or document.get("schema") != 1
        or not isinstance(document.get("artifacts"), list)
    ):
        raise ValueError("Unsupported artifact catalog")
    for entry in document["artifacts"]:
        try:
            if not isinstance(entry, dict) or not compatible(entry["identity"], host):
                continue
            safe_url(entry["url"])
            if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("Invalid artifact digest")
            for key in ("archive_bytes", "unpacked_bytes"):
                if type(entry[key]) is not int or not 0 < entry[key] <= 32 * 1024**3:
                    raise ValueError("Invalid artifact size")
            return entry
        except (ValueError, KeyError, TypeError):
            continue
    return None


def unpack(archive: Path, destination: Path, entry: dict[str, Any]) -> None:
    if (
        archive.stat().st_size != entry["archive_bytes"]
        or generations.file_digest(archive) != entry["sha256"]
    ):
        raise ValueError("Artifact SHA-256/size verification failed")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = set()
        total = 0
        for member in members:
            name = member.name
            parts = name.split("/")
            if (
                name in names
                or name.startswith("/")
                or any(p in {"", ".", ".."} for p in parts)
                or not member.isfile()
                or not (
                    name in {"BUILD-INFO", "RUNTIME.json"}
                    or (len(parts) == 2 and parts[0] in {"bin", "lib"})
                )
            ):
                raise ValueError("Unsafe artifact archive member")
            names.add(name)
            total += member.size
        if total != entry["unpacked_bytes"]:
            raise ValueError("Artifact unpacked size mismatch")
        destination.mkdir()
        # Manual extraction supports system Python 3.9 and never follows links.
        for member in members:
            path = destination / member.name
            path.parent.mkdir(exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("Unreadable artifact archive member")
            with source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            path.chmod(0o755 if member.name.startswith(("bin/", "lib/")) else 0o644)
    verify_package(destination, entry["identity"])


def bundle_libraries(binaries: Path, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    pending = list(binaries.iterdir())
    seen = set()
    # Driver libraries and the platform libc/loader remain node-owned. Bundle
    # CUDA, libstdc++, libgcc and other compiler dependencies with their SONAMEs.
    platform = re.compile(
        r"^(?:lib(?:c|m|pthread|dl|rt|resolv)\.so|ld-linux|libcuda\.so|libnvidia-)"
    )
    while pending:
        binary = pending.pop()
        result = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        if "not found" in output:
            raise ValueError(f"Unresolved dependency for {binary}: {output}")
        if result.returncode and not any(
            x in output for x in ("not a dynamic executable", "statically linked")
        ):
            raise ValueError(f"Could not inspect dependencies for {binary}: {output}")
        for name, filename in re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)", output, re.M):
            if platform.match(name) or name in seen:
                continue
            if Path(name).name != name:
                raise ValueError("Invalid library SONAME")
            seen.add(name)
            target = destination / name
            shutil.copyfile(filename, target)
            target.chmod(0o755)
            pending.append(target)


def seal(root: Path, identity: str) -> None:
    data_files = [
        "BUILD-INFO",
        *(p.relative_to(root).as_posix() for p in sorted((root / "lib").glob("*"))),
    ]
    generations.seal_runtime(root, identity, TOOLS, data_files)
    verify_package(root, identity)


def install(staged: Path, final: Path) -> None:
    manifest = verify_package(staged)
    final.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".install-", dir=final.parent))
    try:
        shutil.copytree(staged, pending, dirs_exist_ok=True)
        generations.publish_directory(
            pending,
            final,
            lambda path: verify_package(path, manifest["identity"]),
            final.parent / ".runtime-publication.lock",
        )
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def publish(runtime: Path, output: Path, base_url: str) -> dict[str, Any]:
    # Archive/catalog publication is a different transaction from directory
    # publication: the archive is durable before the catalog references it, and
    # both operations share one lock to preserve entries from concurrent writers.
    manifest = verify_package(runtime)
    info = metadata(manifest["identity"])
    # Publication is a hardware action on the producer, not an assertion from
    # a hand-written manifest. Execute the shipped CLI and CUDA verification.
    setup = Path(__file__).resolve().parent.parent / "auto-llamacpp/setup.sh"
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; LLAMACPP_VERSION="$2"; verify_llamacpp_binaries "$3"; verify_llamacpp_cuda "$3"',
            "publish",
            str(setup),
            info["version"],
            str(runtime.resolve()),
        ],
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".artifact-", delete=False
        ) as temporary:
            staged = Path(temporary.name)
        try:
            files = sorted(p for p in runtime.rglob("*") if p.is_file())
            with tarfile.open(staged, "w:gz") as archive:
                for path in files:
                    archive.add(
                        path, arcname=path.relative_to(runtime), recursive=False
                    )
            checksum = generations.file_digest(staged)
            filename = checksum + ".tar.gz"
            entry = {
                "identity": manifest["identity"],
                "sha256": checksum,
                "url": safe_url(base_url.rstrip("/") + "/" + filename),
                "archive_bytes": staged.stat().st_size,
                "unpacked_bytes": sum(p.stat().st_size for p in files),
            }
            with staged.open("rb") as stream:
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            os.replace(staged, output / filename)
            generations.sync_directory(output)
            catalog = output / "catalog.json"
            document = (
                json.loads(catalog.read_text())
                if catalog.exists()
                else {"schema": 1, "artifacts": []}
            )
            document["artifacts"] = [
                e for e in document["artifacts"] if e["identity"] != entry["identity"]
            ] + [entry]
            generations.write_json(catalog, document)
            return {
                "catalog_sha256": generations.file_digest(catalog),
                "artifact": entry,
            }
        finally:
            staged.unlink(missing_ok=True)


def existing_ancestor(path: Path) -> Path:
    while not path.exists():
        if path.parent == path:
            raise ValueError(f"No filesystem for {path}")
        path = path.parent
    return path


def capacity(requirements: list[tuple[Path, int]]) -> None:
    devices: dict[int, tuple[Path, int]] = {}
    for path, required in requirements:
        path = existing_ancestor(path)
        device = path.stat().st_dev
        devices[device] = (path, devices.get(device, (path, 0))[1] + required)
    for path, required in devices.values():
        free = shutil.disk_usage(path).free
        if free < required:
            raise ValueError(
                f"{path} has {free} bytes free; {required} bytes required for build/install"
            )


def resources(temporary: Path, install: Path) -> dict[str, Any]:
    meminfo = Path(os.environ.get("AUTOLLAMACPP_MEMINFO", "/proc/meminfo")).read_text()
    matched = re.search(r"^MemAvailable:\s+(\d+) kB$", meminfo, re.M)
    if matched is None:
        raise ValueError("Cannot determine available build memory")
    available_mib = int(matched[1]) // 1024

    def number(name: str, default: int, *, automatic: bool = False) -> int:
        value = os.environ.get(name, str(default))
        if automatic and value == "0":
            return default
        if not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    reserve = number("AUTOLLAMACPP_BUILD_RESERVE_MIB", 2048)
    per_job = number("AUTOLLAMACPP_BUILD_JOB_MIB", 4096)
    memory_jobs = (available_mib - reserve) // per_job
    if memory_jobs < 1:
        raise ValueError(
            f"Insufficient available build memory: {available_mib} MiB; need {reserve + per_job} MiB for one job"
        )
    cpu = int(subprocess.check_output(["nproc"], text=True, timeout=5))
    jobs = min(cpu, memory_jobs, number("AUTOLLAMACPP_BUILD_JOBS", cpu, automatic=True))
    build_bytes = number("AUTOLLAMACPP_BUILD_FREE_MIB", 12288) * 1024**2
    install_bytes = number("AUTOLLAMACPP_INSTALL_FREE_MIB", 4096) * 1024**2
    candidates = [temporary]
    if "AUTOLLAMACPP_BUILD_TMP_DIR" in os.environ:
        candidates = [Path(os.environ["AUTOLLAMACPP_BUILD_TMP_DIR"])]
    else:
        candidates.extend([Path("/var/tmp"), Path("/tmp")])
    errors = []
    for candidate in candidates:
        if not candidate.is_dir() or not os.access(candidate, os.W_OK):
            errors.append(f"{candidate}: not a writable build directory")
            continue
        try:
            capacity([(candidate, build_bytes), (install, install_bytes)])
            return {
                "jobs": jobs,
                "temporary": str(candidate),
                "available_mib": available_mib,
                "reserve_mib": reserve,
                "job_mib": per_job,
                "cpu": cpu,
            }
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "select",
            "local",
            "unpack",
            "seal",
            "install",
            "libraries",
            "publish",
            "resources",
            "capacity",
            "targets",
            "verify",
            "cpu",
        ],
    )
    parser.add_argument("arguments", nargs="+")
    args = parser.parse_args()
    values = args.arguments
    if args.command == "cpu":
        print(json.dumps(cpu_features(Path(values[0]))))
    elif args.command == "targets":
        print(json.dumps(targets(values[0], values[1])))
    elif args.command == "select":
        entry = select(Path(values[0]), json.loads(values[1]))
        if entry is None:
            return 4
        print(json.dumps(entry))
    elif args.command == "local":
        host = json.loads(values[1])
        for path in sorted(Path(values[0]).glob("*/RUNTIME.json")):
            try:
                identity = json.loads(path.read_text())["identity"]
                info = metadata(identity)
                expected_directory = (
                    info["version"]
                    + "-"
                    + hashlib.sha256(identity.encode()).hexdigest()[:16]
                )
                # A copied manifest can precede the publication commit. Never
                # select an interrupted .install-* directory or pointer alias.
                if path.parent.name != expected_directory or path.parent.is_symlink():
                    continue
                if compatible(identity, host) and (
                    host["profile"] != "unselected"
                    or not host.get("installed_cuda_toolkit")
                    or info["cuda_toolkit"] == host["installed_cuda_toolkit"]
                ):
                    verify_package(path.parent)
                    print(path.parent.resolve())
                    return 0
            except (OSError, ValueError, KeyError):
                continue
        return 4
    elif args.command == "unpack":
        unpack(Path(values[0]), Path(values[1]), json.loads(values[2]))
    elif args.command == "seal":
        seal(Path(values[0]), values[1])
    elif args.command == "verify":
        verify_package(Path(values[0]), values[1])
    elif args.command == "install":
        install(Path(values[0]), Path(values[1]))
    elif args.command == "libraries":
        bundle_libraries(Path(values[0]), Path(values[1]))
    elif args.command == "publish":
        print(json.dumps(publish(Path(values[0]), Path(values[1]), values[2])))
    elif args.command == "resources":
        print(json.dumps(resources(Path(values[0]), Path(values[1]))))
    elif args.command == "capacity":
        capacity(
            [(Path(values[i]), int(values[i + 1])) for i in range(0, len(values), 2)]
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        sys.exit(1)
