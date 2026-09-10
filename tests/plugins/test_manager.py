"""Unit tests for PluginManager lifecycle."""

from __future__ import annotations

import pytest

from inference_proxy.config.settings import PluginSettings, Settings
from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.manager import PluginManager


class StubPlugin(BasePlugin):
    """Healthy plugin."""

    name = "stub"
    description = "healthy stub"
    author = "tests"


class RefusingPlugin(BasePlugin):
    """Plugin whose initialize refuses to load."""

    name = "refusing"

    def initialize(self, plugin_manager: object = None) -> bool:
        return False


class RaisingPlugin(BasePlugin):
    """Plugin whose initialize raises."""

    name = "raising"

    def initialize(self, plugin_manager: object = None) -> bool:
        raise RuntimeError("boom")


class ConfigPlugin(BasePlugin):
    """Plugin that records the config dict it was constructed with."""

    name = "config"
    recorded: dict[str, object]

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self.recorded = self.config


class OtherPlugin(BasePlugin):
    """Plugin of a different category type."""

    name = "other"


class FakeDiscovery:
    """Scripted discovery returning a fixed plugin registry."""

    def __init__(
        self,
        plugins: dict[str, type[BasePlugin]],
        external_keys: set[str] | None = None,
    ) -> None:
        self._plugins = plugins
        self.external_keys = external_keys or set()

    def discover_plugins(self) -> dict[str, type[BasePlugin]]:
        return dict(self._plugins)


def _manager(
    test_settings: Settings,
    plugins: dict[str, type[BasePlugin]],
    *,
    disabled: list[str] | None = None,
    config: dict[str, dict[str, object]] | None = None,
    external_keys: set[str] | None = None,
) -> PluginManager:
    settings = test_settings.model_copy(
        deep=True,
        update={
            "plugins": PluginSettings(
                disabled=disabled or [],
                external_dir=None,
                config=config or {},
            )
        },
    )
    return PluginManager(settings, discovery=FakeDiscovery(plugins, external_keys))


def test_initialize_loads_and_skips(test_settings: Settings) -> None:
    manager = _manager(
        test_settings,
        {
            "stub": StubPlugin,
            "refusing": RefusingPlugin,
            "raising": RaisingPlugin,
            "other": OtherPlugin,
        },
        external_keys={"raising"},
    )

    manager.initialize()

    assert set(manager.loaded_plugins) == {"stub", "other"}
    assert isinstance(manager.get_plugin("stub"), StubPlugin)
    assert manager.get_plugin("raising") is None
    assert manager.get_plugin("refusing") is None


def test_disabled_plugins_not_loaded(test_settings: Settings) -> None:
    manager = _manager(
        test_settings,
        {"stub": StubPlugin, "other": OtherPlugin},
        disabled=["other"],
    )

    manager.initialize()

    assert set(manager.loaded_plugins) == {"stub"}


def test_builtin_initialize_error_propagates(test_settings: Settings) -> None:
    manager = _manager(test_settings, {"raising": RaisingPlugin})

    with pytest.raises(RuntimeError, match="boom"):
        manager.initialize()


def test_external_initialize_error_is_skipped(test_settings: Settings) -> None:
    manager = _manager(
        test_settings,
        {"raising": RaisingPlugin},
        external_keys={"raising"},
    )

    manager.initialize()

    assert manager.loaded_plugins == {}


def test_plugin_receives_config_dict(test_settings: Settings) -> None:
    manager = _manager(
        test_settings,
        {"config": ConfigPlugin},
        config={"config": {"enabled": True, "api_key": "abc"}},
    )
    manager.initialize()

    plugin = manager.get_plugin("config")
    assert isinstance(plugin, ConfigPlugin)
    assert plugin.recorded == {"enabled": True, "api_key": "abc"}
    assert plugin.enabled is True


def test_plugin_config_enabled_false_not_loaded(test_settings: Settings) -> None:
    manager = _manager(
        test_settings,
        {"config": ConfigPlugin},
        config={"config": {"enabled": False, "api_key": "abc"}},
    )
    manager.initialize()

    assert manager.get_plugin("config") is None
    assert manager.loaded_plugins == {}


def test_get_plugin(test_settings: Settings) -> None:
    manager = _manager(test_settings, {"stub": StubPlugin, "other": OtherPlugin})
    manager.initialize()

    assert manager.get_plugin("stub") is manager.loaded_plugins["stub"]
    assert manager.get_plugin("other") is manager.loaded_plugins["other"]
    assert manager.get_plugin("missing") is None


def test_get_plugins_by_type(test_settings: Settings) -> None:
    manager = _manager(test_settings, {"stub": StubPlugin, "other": OtherPlugin})
    manager.initialize()

    assert manager.get_plugins_by_type(StubPlugin) == [manager.loaded_plugins["stub"]]
    assert manager.get_plugins_by_type(OtherPlugin) == [manager.loaded_plugins["other"]]


def test_load_plugin_returns_none_on_refusal(test_settings: Settings) -> None:
    manager = _manager(test_settings, {"stub": StubPlugin})

    assert manager.load_plugin("refusing", RefusingPlugin) is None
    assert manager.load_plugin("stub", StubPlugin) is not None
