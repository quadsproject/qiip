"""Plugin discovery from built-in packages and an external directory."""

from __future__ import annotations

import importlib
import importlib.util
import os
import stat
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType

import structlog

from inference_proxy.plugins.base import BasePlugin

logger = structlog.get_logger()

_BUILTIN_ROOT = "inference_proxy.plugins.builtin"


def _as_plugin_class(value: object, module_stem: str) -> type[BasePlugin] | None:
    """Return *value* when it is a plugin class whose name matches its module."""
    if not isinstance(value, type) or not issubclass(value, BasePlugin):
        return None
    if value is BasePlugin or value.name != module_stem:
        return None
    return value


class PluginDiscovery:
    """Discover plugin classes from built-in packages and an external dir.

    Built-in discovery scans ``inference_proxy.plugins.builtin`` one level
    deep: each category subpackage (e.g. ``auth``) contributes modules, and
    a module's plugin class must declare ``name`` equal to the module stem.
    Discovered keys are ``"<category>.<module>"``.

    External discovery mirrors QUADS ``/opt/quads/plugins``: the directory
    itself or per-category subdirectories may hold plugin modules, and the
    plugin ``name`` must match the module filename stem. External modules
    are executed at import time as the service user, so every scanned
    directory must be root/user-owned, not group/world-writable, and not a
    symlink; files must not be group/world-writable; an untrusted path is
    skipped with a warning. Plugins listed in *disabled* are never imported
    at all. External plugins can never shadow a built-in plugin of the same
    key (a warning is logged and the built-in wins); to replace a built-in,
    disable it and register the external plugin under a distinct name.
    """

    def __init__(
        self,
        external_dir: Path | None = None,
        disabled: frozenset[str] | None = None,
    ) -> None:
        self.external_dir = external_dir
        self.disabled = disabled or frozenset()
        self.external_keys: set[str] = set()

    def discover_plugins(self) -> dict[str, type[BasePlugin]]:
        """Return discovered plugins keyed by their registered name."""
        plugins = self._discover_builtin()
        if self.external_dir is None or not self.external_dir.is_dir():
            return plugins
        if not self._trusted_directory(self.external_dir):
            logger.warning(
                "external plugin directory is not root/user-owned, is "
                "group/world-writable, or is a symlink; skipping scan",
                path=str(self.external_dir),
            )
            return plugins
        try:
            external = self._discover_directory(self.external_dir)
        except OSError:
            logger.warning(
                "external plugin directory could not be scanned; skipping",
                path=str(self.external_dir),
                exc_info=True,
            )
            return plugins
        for key, plugin_class in external.items():
            if key in plugins:
                logger.warning(
                    "external plugin name collides with a built-in plugin; "
                    "keeping the built-in",
                    plugin=key,
                )
                continue
            plugins[key] = plugin_class
            self.external_keys.add(key)
        return plugins

    @staticmethod
    def _trusted_directory(directory: Path) -> bool:
        """Return True when *directory* is not a symlink, is root/user-owned,
        and is not group/world-writable."""
        if directory.is_symlink():
            return False
        try:
            metadata = directory.stat()
        except OSError:
            return False
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False
        return metadata.st_uid == 0 or metadata.st_uid == os.getuid()

    @staticmethod
    def _writable_file(path: Path) -> bool:
        """Return True when *path* is group/world-writable (or unstatable)."""
        try:
            return bool(path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        except OSError:
            return True

    def _discover_builtin(self) -> dict[str, type[BasePlugin]]:
        plugins: dict[str, type[BasePlugin]] = {}
        try:
            root = importlib.import_module(_BUILTIN_ROOT)
        except ImportError:
            logger.warning("built-in plugin root unavailable", exc_info=True)
            return plugins
        for _finder, category, is_package in iter_modules(root.__path__):
            if not is_package:
                continue
            package = importlib.import_module(f"{_BUILTIN_ROOT}.{category}")
            plugins.update(self._modules_in_package(package, prefix=f"{category}."))
        return plugins

    @staticmethod
    def _modules_in_package(
        package: ModuleType, *, prefix: str
    ) -> dict[str, type[BasePlugin]]:
        plugins: dict[str, type[BasePlugin]] = {}
        for _finder, modname, is_package in iter_modules(package.__path__):
            if is_package:
                continue
            module = importlib.import_module(f"{package.__name__}.{modname}")
            for attr in vars(module).values():
                plugin_class = _as_plugin_class(attr, modname)
                if plugin_class is not None:
                    plugins[f"{prefix}{modname}"] = plugin_class
        return plugins

    def _discover_directory(self, directory: Path) -> dict[str, type[BasePlugin]]:
        plugins: dict[str, type[BasePlugin]] = {}
        scopes: list[tuple[Path, str]] = [(directory, "")]
        for subdir in sorted(directory.iterdir()):
            if (
                not subdir.is_dir()
                or subdir.name.startswith("_")
                or subdir.is_symlink()
            ):
                continue
            if not self._trusted_directory(subdir):
                logger.warning(
                    "external plugin subdirectory is not trusted; skipping",
                    path=str(subdir),
                )
                continue
            scopes.append((subdir, f"{subdir.name}."))
        for scope_dir, prefix in scopes:
            plugins.update(self._modules_in_directory(scope_dir, prefix=prefix))
        return plugins

    def _modules_in_directory(
        self, directory: Path, *, prefix: str
    ) -> dict[str, type[BasePlugin]]:
        plugins: dict[str, type[BasePlugin]] = {}
        for file in sorted(directory.glob("*.py")):
            if file.name == "__init__.py" or file.is_symlink():
                continue
            if self._writable_file(file):
                logger.warning(
                    "external plugin file is group/world-writable; skipping",
                    path=str(file),
                )
                continue
            key = f"{prefix}{file.stem}"
            if key in self.disabled:
                logger.info("disabled external plugin not imported", plugin=key)
                continue
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                logger.info("loading external plugin module", path=str(file))
                spec.loader.exec_module(module)
            except Exception:
                logger.warning(
                    "external plugin load failed", path=str(file), exc_info=True
                )
                continue
            for attr in vars(module).values():
                if not isinstance(attr, type) or not issubclass(attr, BasePlugin):
                    continue
                if attr is BasePlugin:
                    continue
                if attr.name != file.stem:
                    logger.warning(
                        "external plugin name does not match module filename; skipping",
                        plugin=attr.name or file.stem,
                        path=str(file),
                    )
                    continue
                plugins[f"{prefix}{attr.name}"] = attr
        return plugins
