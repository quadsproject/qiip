"""Plugin architecture for QIIP.

Mirrors the QUADS plugin model (``quads.plugins``) at a smaller scale:
``BasePlugin`` subclasses implement a category interface, are discovered
from built-in packages or an external directory, and are loaded by
``PluginManager`` according to configuration.
"""

from __future__ import annotations

from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.interfaces.auth import (
    AuthCallbackError,
    AuthIdentity,
    AuthPlugin,
)
from inference_proxy.plugins.manager import PluginManager

__all__ = [
    "AuthCallbackError",
    "AuthIdentity",
    "AuthPlugin",
    "BasePlugin",
    "PluginManager",
]
