"""Guards for runtime dependency support claims."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_etcd3gw_floor_matches_exercised_locked_version() -> None:
    """Keep the known-bad etcd3gw floor at the version CI actually exercises.

    Unlike the project's other unexercised dependency floors, the previous
    etcd3gw floor was demonstrably false when declared: its ``types`` module
    was unavailable below 2.6. Later raw-watch, lease, and transaction work
    expanded the API surface exercised only at 2.7. The gateway deploys from
    this lock, so upgrading etcd3gw must deliberately raise the floor too.
    """
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependency = next(
        item
        for item in project["project"]["dependencies"]
        if item.startswith("etcd3gw")
    )

    assert dependency == f"etcd3gw>={version('etcd3gw')}"
