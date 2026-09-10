"""Unit tests for plugin discovery (built-in packages and external dirs)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.discovery import PluginDiscovery, _as_plugin_class


class _ExternalPlugin(BasePlugin):
    """Concrete plugin for external-directory discovery tests."""

    name = "external"
    version = "1.0.0"
    description = "external plugin"
    author = "tests"


def test_builtin_discovery_finds_google_auth() -> None:
    plugins = PluginDiscovery().discover_plugins()

    assert "auth.google" in plugins
    plugin_class = plugins["auth.google"]
    assert issubclass(plugin_class, BasePlugin)
    assert plugin_class.__module__ == ("inference_proxy.plugins.builtin.auth.google")


def test_external_dir_none_and_missing_are_ignored(tmp_path: Path) -> None:
    discovery = PluginDiscovery(None)
    assert "auth.google" in discovery.discover_plugins()

    discovery = PluginDiscovery(tmp_path / "nope")
    plugins = discovery.discover_plugins()

    assert "auth.google" in plugins
    assert discovery.external_keys == set()


def test_external_flat_plugin_discovered(tmp_path: Path) -> None:
    (tmp_path / "external.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class ExternalPlugin(BasePlugin):\n"
        "    name = 'external'\n",
        encoding="utf-8",
    )

    discovery = PluginDiscovery(tmp_path)
    plugins = discovery.discover_plugins()

    assert "external" in plugins
    assert discovery.external_keys == {"external"}


def test_external_category_plugin_discovered(tmp_path: Path) -> None:
    category = tmp_path / "auth"
    category.mkdir()
    (category / "okta.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class OktaPlugin(BasePlugin):\n"
        "    name = 'okta'\n",
        encoding="utf-8",
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "auth.okta" in plugins


def test_external_cannot_shadow_builtin(tmp_path: Path) -> None:
    category = tmp_path / "auth"
    category.mkdir()
    (category / "google.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class GooglePlugin(BasePlugin):\n"
        "    name = 'google'\n",
        encoding="utf-8",
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert plugins["auth.google"].__module__ == (
        "inference_proxy.plugins.builtin.auth.google"
    )


def test_untrusted_external_dir_is_skipped(tmp_path: Path) -> None:
    os.chmod(tmp_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    (tmp_path / "external.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class ExternalPlugin(BasePlugin):\n"
        "    name = 'external'\n",
        encoding="utf-8",
    )

    discovery = PluginDiscovery(tmp_path)
    plugins = discovery.discover_plugins()

    assert "external" not in plugins
    assert discovery.external_keys == set()


def test_untrusted_external_subdir_is_skipped(tmp_path: Path) -> None:
    category = tmp_path / "auth"
    category.mkdir()
    os.chmod(category, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    (category / "okta.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class OktaPlugin(BasePlugin):\n"
        "    name = 'okta'\n",
        encoding="utf-8",
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "auth.okta" not in plugins


def test_symlinked_external_subdir_is_skipped(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class EvilPlugin(BasePlugin):\n"
        "    name = 'evil'\n",
        encoding="utf-8",
    )
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "linked.evil" not in plugins


def test_world_writable_external_file_is_skipped(tmp_path: Path) -> None:
    plugin_file = tmp_path / "external.py"
    plugin_file.write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class ExternalPlugin(BasePlugin):\n"
        "    name = 'external'\n",
        encoding="utf-8",
    )
    os.chmod(
        plugin_file,
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IRGRP
        | stat.S_IWGRP
        | stat.S_IROTH
        | stat.S_IWOTH,
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "external" not in plugins


def test_disabled_external_module_is_not_imported(tmp_path: Path) -> None:
    marker = tmp_path / "imported.marker"
    (tmp_path / "external.py").write_text(
        "from pathlib import Path\n"
        "from inference_proxy.plugins.base import BasePlugin\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "class ExternalPlugin(BasePlugin):\n"
        "    name = 'external'\n",
        encoding="utf-8",
    )

    discovery = PluginDiscovery(tmp_path, disabled=frozenset({"external"}))
    discovery.discover_plugins()

    assert marker.exists() is False


def test_external_name_must_match_filename(tmp_path: Path) -> None:
    (tmp_path / "external.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class RenamedPlugin(BasePlugin):\n"
        "    name = 'renamed'\n",
        encoding="utf-8",
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "external" not in plugins
    assert "renamed" not in plugins


def test_symlinked_external_root_is_skipped(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class EvilPlugin(BasePlugin):\n"
        "    name = 'evil'\n",
        encoding="utf-8",
    )
    linked_root = tmp_path / "plugins"
    linked_root.symlink_to(outside, target_is_directory=True)

    discovery = PluginDiscovery(linked_root)
    plugins = discovery.discover_plugins()

    assert "evil" not in plugins
    assert discovery.external_keys == set()


def test_underscore_prefixed_subdir_is_skipped(tmp_path: Path) -> None:
    category = tmp_path / "_internal"
    category.mkdir()
    (category / "helper.py").write_text(
        "from inference_proxy.plugins.base import BasePlugin\n"
        "class HelperPlugin(BasePlugin):\n"
        "    name = 'helper'\n",
        encoding="utf-8",
    )

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "_internal.helper" not in plugins


def test_init_py_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "auth.google" in plugins


def test_module_without_plugin_class_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "nomatch.py").write_text("VALUE = 42\n", encoding="utf-8")

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "nomatch" not in plugins


def test_external_broken_module_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("this is not python !!!\n", encoding="utf-8")

    plugins = PluginDiscovery(tmp_path).discover_plugins()

    assert "broken" not in plugins
    assert "auth.google" in plugins


def test_as_plugin_class_matches_module_stem() -> None:
    assert _as_plugin_class(_ExternalPlugin, "external") is _ExternalPlugin
    assert _as_plugin_class(_ExternalPlugin, "other") is None
    assert _as_plugin_class(BasePlugin, "base") is None
    assert _as_plugin_class(str, "str") is None
