"""Unit tests for BasePlugin metadata and lifecycle defaults."""

from __future__ import annotations

import pytest

from inference_proxy.config.settings import Settings
from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.manager import PluginManager


class StubPlugin(BasePlugin):
    """Concrete plugin for lifecycle tests."""

    name = "stub"
    version = "2.0.0"
    description = "stub plugin"
    author = "tests"


def test_plugin_metadata_defaults() -> None:
    plugin = StubPlugin()

    assert plugin.name == "stub"
    assert plugin.version == "2.0.0"
    assert plugin.description == "stub plugin"
    assert plugin.author == "tests"
    assert plugin.enabled is True
    assert plugin.logger.name == "qiip.plugins.stub"


def test_plugin_config_enabled_flag() -> None:
    plugin = StubPlugin({"enabled": False})

    assert plugin.enabled is False


def test_initialize_default_accepts_and_attaches_manager(
    test_settings: Settings,
) -> None:
    manager = PluginManager(test_settings)
    plugin = StubPlugin()

    assert plugin.initialize(manager) is True
    assert plugin.manager is manager


def test_manager_property_raises_when_unattached() -> None:
    plugin = StubPlugin()

    with pytest.raises(RuntimeError, match="not attached to a manager"):
        _ = plugin.manager
