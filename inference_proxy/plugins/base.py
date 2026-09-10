"""Base class for all QIIP plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inference_proxy.plugins.manager import PluginManager


class BasePlugin:
    """Base class for all QIIP plugins.

    Subclasses declare metadata class attributes (``name`` must match the
    module filename for built-in discovery) and implement the category
    interface they belong to. Plugins are only created through
    :class:`PluginManager`.
    """

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(f"qiip.plugins.{self.name}")
        self._enabled = bool(self.config.get("enabled", True))
        self._manager: PluginManager | None = None

    @property
    def enabled(self) -> bool:
        """Return whether this plugin is enabled by configuration."""
        return self._enabled

    @property
    def manager(self) -> PluginManager:
        """Return the manager that loaded this plugin.

        Raises ``RuntimeError`` when the plugin was constructed outside the
        manager (it has no access to application settings).
        """
        if self._manager is None:
            raise RuntimeError(f"plugin {self.name!r} is not attached to a manager")
        return self._manager

    def initialize(self, plugin_manager: PluginManager | None = None) -> bool:
        """Initialize the plugin after construction.

        Override to validate configuration; return ``False`` to refuse
        loading. The default accepts any plugin.
        """
        self._manager = plugin_manager
        return True
