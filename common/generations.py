#!/usr/bin/env python3
"""Durable, content-addressed node bundles and engine generations (stdlib only)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    # EL9's system Python is 3.9; hashlib.file_digest arrived in 3.11.
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def bundle_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Bundle contains a symlink: {path}")
        if path.is_file() and path.name != "BUNDLE.json":
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return {"schema": 1, "files": files}


def verify_bundle(root: Path, expected: str | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "BUNDLE.json").read_text())
    if manifest != bundle_manifest(root) or (expected and digest(manifest) != expected):
        raise ValueError("Incomplete or corrupt setup bundle")
    for name in manifest["files"]:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(root / name)], check=True)
        elif name.endswith(".py"):
            compile((root / name).read_bytes(), name, "exec")
    return manifest


def publish_bundle(staged: Path, final: Path, expected: str) -> None:
    verify_bundle(staged, expected)
    # Flush uploads before the rename. An incomplete directory never becomes a
    # candidate, including when an upload runs out of space or loses its SSH.
    for path in staged.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(staged.rglob("*"), reverse=True):
        if path.is_dir():
            sync_directory(path)
    sync_directory(staged)
    with (final.parent / ".publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            try:
                verify_bundle(final, expected)
            except (OSError, ValueError, subprocess.CalledProcessError):
                # Keep damaged content for inspection, then publish the already
                # verified upload. A crash between renames is repaired by retry.
                quarantine = Path(
                    tempfile.mkdtemp(
                        prefix=".corrupt-" + expected + "-", dir=final.parent
                    )
                )
                os.rename(final, quarantine)
                sync_directory(final.parent)
            else:
                shutil.rmtree(staged)
        if staged.exists():
            os.rename(staged, final)
        sync_directory(final.parent)


def setup_bundle(root: Path, engine: Path) -> Path:
    """Snapshot a standalone checkout; managed uploads are already sealed."""
    if (engine.parent / "BUNDLE.json").is_file():
        verify_bundle(engine.parent)
        return engine.parent.resolve()
    bundles = root / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=bundles))
    for source in (engine, engine.parent / "common"):
        destination = staged / source.name
        destination.mkdir()
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    manifest = bundle_manifest(staged)
    write_json(staged / "BUNDLE.json", manifest)
    final = bundles / digest(manifest)
    publish_bundle(staged, final, final.name)
    return final.resolve()


def seal_runtime(
    root: Path,
    identity: str,
    binaries: list[str],
    data_files: list[str] | None = None,
) -> None:
    files = {}
    external_links = {}
    for name in binaries:
        path = root / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Required executable is missing: {path}")
        if path.is_symlink() and root.resolve() not in path.resolve().parents:
            # uv uses the RPM-managed interpreter. Validate its link/existence,
            # without treating an ordinary system update as runtime corruption.
            external_links[f"bin/{name}"] = os.readlink(path)
        else:
            files[f"bin/{name}"] = file_digest(path)
    # Flush installed package data as well as entry points before publishing
    # the completion marker. A durable marker must not outrun its dependencies.
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            sync_directory(path)
    sync_directory(root)
    write_json(
        root / "RUNTIME.json",
        {
            "identity": identity,
            "files": files,
            "external_links": external_links,
            "data_files": {
                name: file_digest(root / name) for name in (data_files or [])
            },
        },
    )


def verify_runtime(root: Path, identity: str | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "RUNTIME.json").read_text())
    if identity is not None and manifest["identity"] != identity:
        raise ValueError("Runtime dependency/profile identity mismatch")
    for name, expected in manifest["files"].items():
        path = root / name
        if not os.access(path, os.X_OK) or file_digest(path) != expected:
            raise ValueError(f"Required executable is corrupt: {path}")
    for name, target in manifest.get("external_links", {}).items():
        path = root / name
        if (
            not path.is_symlink()
            or os.readlink(path) != target
            or not os.access(path, os.X_OK)
        ):
            raise ValueError(f"Required external executable link is corrupt: {path}")
    for name, expected in manifest.get("data_files", {}).items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Required runtime data is corrupt: {root / name}")
    return manifest


def atomic_link(target: Path, link: Path) -> None:
    temporary = link.with_name(link.name + ".pending")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)
    sync_directory(link.parent)


def evidence(generation: Path, *, runtime_integrity: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((generation / "GENERATION.json").read_text())
    if generation.name != digest(value):
        raise ValueError("Generation manifest identity mismatch")
    for name, expected in (
        (value["engine_dir"], Path(value["bundle_path"]) / value["engine_dir"]),
        ("common", Path(value["bundle_path"]) / "common"),
        ("runtime", Path(value["runtime_path"])),
    ):
        if (generation / name).resolve(
            strict=runtime_integrity or name != "runtime"
        ) != expected:
            raise ValueError(f"Generation link is corrupt: {name}")
    verify_bundle(Path(value["bundle_path"]), value["bundle_id"])
    if runtime_integrity:
        runtime = verify_runtime(Path(value["runtime_path"]), value["runtime_id"])
        if digest(runtime) != value["runtime_manifest_id"]:
            raise ValueError("Runtime manifest identity mismatch")
    for name, expected in value["files"].items():
        if file_digest(generation / name) != expected:
            raise ValueError(f"Generation configuration is corrupt: {name}")
    return {"generation_id": generation.name, **value}


def service_unit(root: Path, bundle: Path, engine_dir: str) -> str:
    def quote(path: Path) -> str:
        text = (
            str(path)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("%", "%%")
            .replace("$", "$$")
        )
        return '"' + text + '"'

    # The unit always dispatches through current, even if systemd still has
    # the prior unit loaded after an interrupted reload.
    helper = bundle / "common/generations.py"
    command = f"/usr/bin/python3 {quote(helper)} exec-service {quote(root.resolve())} {quote(bundle / engine_dir)}"
    return (
        "[Unit]\nDescription=vLLM inference server\n"
        "After=network-online.target nvidia-fabricmanager.service\n"
        "Wants=network-online.target\n\n[Service]\nType=exec\n"
        f"ExecStartPre={command} wait-fabric.sh\n"
        f"ExecStartPre={command} preflight.sh --check-only\n"
        f"ExecStart={command} start-vllm.sh\n"
        "Restart=on-failure\nRestartSec=10\n\n[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def finish_rollback(root: Path) -> dict[str, Any]:
    """Complete a journaled rollback while holding the activation lock."""
    journal = root / "ROLLBACK.json"
    change = json.loads(journal.read_text())
    target = Path(change["target"])
    selected = evidence(target)
    previous = Path(change["previous"])
    atomic_link(previous, root / "previous")
    atomic_link(target, root / "current")
    journal.unlink()
    sync_directory(root)
    return selected


def rollback(root: Path) -> dict[str, Any]:
    with (root / "activation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / "ROLLBACK.json").exists():
            target = (root / "previous").resolve(strict=True)
            evidence(target)
            write_json(
                root / "ROLLBACK.json",
                {
                    "target": str(target),
                    "previous": str((root / "current").resolve(strict=True)),
                },
            )
        return finish_rollback(root)


def activate(
    root: Path, bundle: Path, engine_dir: str, runtime: Path, config: dict[str, str]
) -> dict[str, Any]:
    bundle = bundle.resolve()
    runtime = runtime.resolve()
    manifest = verify_bundle(bundle)
    runtime_manifest = verify_runtime(runtime)
    unit = (
        service_unit(root, bundle, engine_dir)
        if config["QIIP_ENGINE"] == "vllm"
        else None
    )
    value = dict(
        schema=1,
        bundle_id=digest(manifest),
        bundle_path=str(bundle),
        engine_dir=engine_dir,
        runtime_id=runtime_manifest["identity"],
        runtime_manifest_id=digest(runtime_manifest),
        runtime_path=str(runtime),
        config=config,
        files={"vllm.service": hashlib.sha256(unit.encode()).hexdigest()}
        if unit
        else {},
    )
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    with (root / "activation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "ROLLBACK.json").exists():
            finish_rollback(root)
        generation = generations / digest(value)
        if not generation.exists():
            staged = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=generations))
            (staged / engine_dir).symlink_to(bundle / engine_dir)
            (staged / "common").symlink_to(bundle / "common")
            (staged / "runtime").symlink_to(runtime)
            if unit:
                with (staged / "vllm.service").open("w") as stream:
                    stream.write(unit)
                    stream.flush()
                    os.fsync(stream.fileno())
            write_json(staged / "GENERATION.json", value)
            os.rename(staged, generation)
            sync_directory(generations)
        selected = evidence(generation)
        if unit:
            # Setup must reset restart settings even when reusing the same
            # generation; retained *other* generations keep their own settings.
            (generation / "vllm.env").unlink(missing_ok=True)
            sync_directory(generation)
        current = root / "current"
        try:
            if current.exists() and current.resolve() != generation.resolve():
                # Write the rollback pointer first. If interrupted before the
                # commit, current is still valid; retries complete that commit.
                atomic_link(current.resolve(), root / "previous")
            atomic_link(generation.resolve(), current)
        except OSError:
            # A failed acknowledgement/fsync can follow a successful rename.
            # Record what is actually selected before reporting that failure.
            if current.is_symlink():
                emit(evidence(current.resolve(strict=True)))
            raise
        return selected


def emit(value: dict[str, Any]) -> None:
    print("[GENERATION:" + json.dumps(value, sort_keys=True) + "]", flush=True)


def launch(
    root: Path, fallback: Path, script: str, args: list[str], *, service: bool = False
) -> None:
    current = root / "current"
    environment = dict(os.environ)
    if current.is_symlink():
        generation = current.resolve(strict=True)
        value = evidence(
            generation,
            runtime_integrity=script not in {"stop-vllm.sh", "stop-llamacpp.sh"},
        )
        directory = Path(value["bundle_path"]) / value["engine_dir"]
        runtime = Path(value["runtime_path"])
        environment = {**environment, **value["config"]}
        if value["config"]["QIIP_ENGINE"] == "vllm":
            saved = generation / "vllm.env"
            if service and saved.is_file():
                launch_config = dict(
                    line.split("=", 1)
                    for line in saved.read_text().splitlines()
                    if "=" in line
                )
                environment = {**launch_config, **environment}
            environment["AUTOVLLM_BIN"] = str(runtime / "bin/vllm")
            environment["AUTOVLLM_PYTHON"] = str(runtime / "bin/python")
            environment["AUTOVLLM_RUNTIME_ROOT"] = str(runtime.parent)
            environment["AUTOVLLM_ENV_FILE"] = str(generation / "vllm.env")
        else:
            environment["AUTOLLAMACPP_BIN"] = str(runtime / "bin/llama-server")
            environment["AUTOLLAMACPP_FIT_BIN"] = str(runtime / "bin/llama-fit-params")
            environment["AUTOLLAMACPP_INSTALL_ROOT"] = str(runtime.parent)
        emit(value)
    else:
        # Pre-generation nodes remain stoppable/relaunchable. Setup always
        # creates a generation before the first managed launch.
        directory = fallback
    environment.pop("AUTOVLLM_SCRIPT_DIR", None)
    environment.pop("AUTOLLAMACPP_SCRIPT_DIR", None)
    environment["QIIP_GENERATION_PINNED"] = "1"
    os.execvpe("bash", ["bash", str(directory / script), *args], environment)


def main() -> None:
    action, *args = sys.argv[1:]
    if action == "publish-bundle":
        publish_bundle(Path(args[0]), Path(args[1]), args[2])
    elif action == "setup-bundle":
        print(setup_bundle(Path(args[0]), Path(args[1])))
    elif action == "seal-runtime":
        seal_runtime(Path(args[0]), args[1], args[2:])
    elif action == "verify-runtime":
        verify_runtime(Path(args[0]), args[1])
    elif action == "selected":
        current = Path(args[0]) / "current"
        if current.is_symlink():
            generation = current.resolve(strict=True)
            value = json.loads((generation / "GENERATION.json").read_text())
            if generation.name != digest(value):
                raise ValueError("Generation manifest identity mismatch")
            emit({"generation_id": generation.name, **value})
    elif action == "activate":
        emit(
            activate(
                Path(args[0]),
                Path(args[1]),
                args[2],
                Path(args[3]),
                json.loads(args[4]),
            )
        )
    elif action in {"exec", "exec-service"}:
        launch(
            Path(args[0]),
            Path(args[1]),
            args[2],
            args[3:],
            service=action == "exec-service",
        )
    elif action == "rollback":
        emit(rollback(Path(args[0])))
    else:
        raise ValueError(f"Unknown generation action: {action}")


if __name__ == "__main__":
    main()
