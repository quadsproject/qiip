"""Plugin lifecycle: discovery, loading, and lookup."""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from inference_proxy.config.settings import Settings
from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.discovery import PluginDiscovery

logger = structlog.get_logger()


class DiscoveryProtocol(Protocol):
    """Minimal discovery contract used by PluginManager (testable)."""

    external_keys: set[str]

    def discover_plugins(self) -> dict[str, type[BasePlugin]]: ...


class PluginManager:
    """Discover and load enabled plugins for the application.

    The manager owns plugin lifecycle: discovery, per-plugin configuration
    (``settings.plugins.config``), disabled handling, construction, and
    initialization. A plugin whose ``initialize`` returns ``False`` is
    skipped. Exceptions from built-in plugins propagate so a broken built-in
    fails startup loudly; exceptions from external (downstream) plugins are
    logged and skipped so third-party code cannot take the gateway down.

    Components access plugins through ``get_plugin`` /
    ``get_plugins_by_type``; they never construct plugin classes themselves.
    """

    def __init__(
        self,
        settings: Settings,
        discovery: DiscoveryProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.discovery = discovery or PluginDiscovery(
            settings.plugins.external_dir,
            disabled=frozenset(settings.plugins.disabled),
        )
        self.available_plugins: dict[str, type[BasePlugin]] = {}
        self.loaded_plugins: dict[str, BasePlugin] = {}

    def initialize(self) -> None:
        """Discover plugins and load every enabled one."""
        self.available_plugins = self.discovery.discover_plugins()
        for name, plugin_class in self.available_plugins.items():
            if name in self.settings.plugins.disabled:
                logger.info("plugin disabled by configuration", plugin=name)
                continue
            self.load_plugin(
                name,
                plugin_class,
                config=self.settings.plugins.config.get(name, {}),
                raise_on_error=name not in self.discovery.external_keys,
            )

    def load_plugin(
        self,
        name: str,
        plugin_class: type[BasePlugin],
        *,
        config: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> BasePlugin | None:
        """Instantiate and initialize one plugin.

        Returns the loaded instance, or ``None`` when initialization fails
        or returns ``False``. When *raise_on_error* is True (built-in
        plugins) exceptions propagate; otherwise they are logged and the
        plugin is skipped.
        """
        try:
            plugin = plugin_class(config)
            if not plugin.enabled:
                logger.info("plugin disabled by configuration", plugin=name)
                return None
            if plugin.initialize(self):
                self.loaded_plugins[name] = plugin
                logger.info("plugin loaded", plugin=name, version=plugin.version)
                return plugin
            logger.info("plugin refused to initialize", plugin=name)
        except Exception:
            if raise_on_error:
                raise
            logger.error("plugin failed to initialize", plugin=name, exc_info=True)
        return None

    def get_plugin(self, name: str) -> BasePlugin | None:
        """Return the loaded plugin *name*, or None."""
        return self.loaded_plugins.get(name)

    def get_plugins_by_type(self, plugin_type: type[BasePlugin]) -> list[BasePlugin]:
        """Return every loaded plugin of the given type, in load order."""
        return [
            plugin
            for plugin in self.loaded_plugins.values()
            if isinstance(plugin, plugin_type)
        ]
