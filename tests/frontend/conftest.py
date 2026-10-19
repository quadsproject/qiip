"""Shared settings for the frontend JavaScript harnesses."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _pinned_node_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Node's Intl output deterministic regardless of the runner locale."""
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
