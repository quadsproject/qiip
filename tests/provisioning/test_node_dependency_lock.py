"""Structural and tool-compatibility guards for the node uv environment."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJECT = REPO_ROOT / "auto-vllm"
EXPECTED_ENVIRONMENT = (
    "sys_platform == 'linux' and platform_machine == 'x86_64' "
    "and implementation_name == 'cpython'"
)


def _load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text())


def test_node_project_targets_cpython312_linux_x86_64() -> None:
    project = _load_toml(NODE_PROJECT / "pyproject.toml")
    metadata = project["project"]
    uv_settings = project["tool"]["uv"]  # type: ignore[index]

    assert metadata["requires-python"] == "==3.12.*"  # type: ignore[index]
    assert metadata["dependencies"] == [  # type: ignore[index]
        "flashinfer-cubin==0.6.14",
        "flashinfer-python==0.6.14",
        "vllm==0.26.0",
    ]
    assert uv_settings["environments"] == [EXPECTED_ENVIRONMENT]
    assert uv_settings["required-environments"] == [EXPECTED_ENVIRONMENT]
    assert uv_settings["package"] is False

    indexes = uv_settings["index"]
    assert indexes == [
        {
            "name": "flashinfer",
            "url": "https://flashinfer.ai/whl",
            "explicit": True,
        }
    ]
    assert uv_settings["sources"] == {"flashinfer-cubin": {"index": "flashinfer"}}


def test_node_lock_contains_only_verified_registry_dependencies() -> None:
    lock = _load_toml(NODE_PROJECT / "uv.lock")
    packages = lock["package"]
    virtual_packages: list[str] = []

    for package in packages:  # type: ignore[union-attr]
        source = package["source"]
        if source == {"virtual": "."}:
            virtual_packages.append(package["name"])
            continue

        assert set(source) == {"registry"}, package["name"]
        assert source["registry"] in {
            "https://pypi.org/simple",
            "https://flashinfer.ai/whl",
        }
        wheels = package.get("wheels", [])
        assert wheels, f"{package['name']} has no locked wheel"
        for wheel in wheels:
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", wheel["hash"])
        if "sdist" in package:
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", package["sdist"]["hash"])

    assert virtual_packages == ["auto-vllm-node"]


def test_provisioning_scripts_never_invoke_pip() -> None:
    pip_command = re.compile(
        r"(?:^|[\s/'\"])(?:python\S*\s+-m\s+)?"
        r"pip(?:3(?:\.\d+)?)?(?=$|[\s\"'])"
    )
    offenders: list[str] = []
    for script in sorted(NODE_PROJECT.glob("*.sh")):
        for line_number, line in enumerate(script.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pip_command.search(line):
                offenders.append(f"{script.name}:{line_number}:{line.strip()}")
    assert offenders == []


def test_uv_pin_records_official_checksum_provenance() -> None:
    version = (NODE_PROJECT / ".uv-version").read_text().strip()
    checksum_line = (
        (NODE_PROJECT / "uv-x86_64-unknown-linux-gnu.tar.gz.sha256").read_text().strip()
    )
    documentation = (NODE_PROJECT / "README.md").read_text()

    assert version == "0.12.1"
    assert checksum_line == (
        "90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb  "
        "uv-x86_64-unknown-linux-gnu.tar.gz"
    )
    assert (
        f"https://github.com/astral-sh/uv/releases/download/{version}/"
        "uv-x86_64-unknown-linux-gnu.tar.gz.sha256"
    ) in documentation
    assert "uv lock --project auto-vllm --python 3.12 --upgrade" in documentation


def test_pinned_uv_reads_and_resolves_node_lock(tmp_path: Path) -> None:
    uv_binary = os.environ.get("AUTOVLLM_PINNED_UV_BIN", "uv")
    version = subprocess.run(
        [uv_binary, "--version"],
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert version.stdout.startswith("uv 0.12.1 ")

    subprocess.run(
        [uv_binary, "lock", "--check", "--project", str(NODE_PROJECT)],
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "cache")},
    )
    sync = subprocess.run(
        [
            uv_binary,
            "sync",
            "--project",
            str(NODE_PROJECT),
            "--frozen",
            "--no-dev",
            "--no-install-project",
            "--no-build",
            "--python",
            "3.12",
            "--python-platform",
            "x86_64-manylinux_2_34",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
        env={
            **os.environ,
            "UV_CACHE_DIR": str(tmp_path / "cache"),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "node-venv"),
        },
    )
    assert "Would install 196 packages" in sync.stderr
