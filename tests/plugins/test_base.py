"""Unit tests for BasePlugin metadata and lifecycle defaults."""

from __future__ import annotations

import pytest

from inference_proxy.auth._constants import GOOGLE_ISSUER
from inference_proxy.config.settings import Settings
from inference_proxy.plugins.base import BasePlugin
from inference_proxy.plugins.interfaces.auth import AuthIdentity
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


def test_auth_identity_defaults_to_google_issuer() -> None:
    """External plugins built before the issuer field keep working
    (review #231): identities land under the Google issuer by default."""
    identity = AuthIdentity(sub="sub-1", email="alice@example.com", email_verified=True)

    assert identity.issuer == GOOGLE_ISSUER


def test_auth_identity_positional_issuer_slot_is_stable() -> None:
    identity = AuthIdentity("sub-1", "alice@example.com", True, "https://local.oidc")

    assert identity.issuer == "https://local.oidc"
    assert identity.name == ""
    assert identity.picture == ""
