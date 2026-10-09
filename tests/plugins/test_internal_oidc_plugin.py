"""Unit tests for the built-in local OIDC auth plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from authlib.integrations.starlette_client import OAuthError
from fastapi.responses import RedirectResponse
from pydantic import SecretStr

from inference_proxy.config.settings import OAuthSettings, Settings
from inference_proxy.plugins.builtin.auth.internal_oidc import InternalOidcPlugin
from inference_proxy.plugins.interfaces.auth import AuthCallbackError, AuthIdentity
from inference_proxy.plugins.manager import PluginManager

_LOCAL_ISSUER = "http://oidc.localdomain/oidc"

_VALID_USERINFO: dict[str, object] = {
    "iss": _LOCAL_ISSUER,
    "sub": "subject-123",
    "email": "alice@localdomain",
    "email_verified": True,
    "name": "Alice Example",
    "picture": "",
}


class _FakeOidc:
    """Stand-in for the authlib OIDC client with scripted responses."""

    def __init__(
        self,
        token: dict[str, object] | None = None,
        error: str | None = None,
        redirect_to: str = "http://oidc.localdomain/oidc/authorize?state=abc",
    ) -> None:
        self._token = token or {"userinfo": dict(_VALID_USERINFO)}
        self._error = error
        self.redirect_to = redirect_to
        self.authorize_redirect_calls = 0

    async def authorize_redirect(
        self, _request: object, redirect_uri: str | None = None
    ) -> RedirectResponse:
        self.authorize_redirect_calls += 1
        return RedirectResponse(self.redirect_to, status_code=302)

    async def authorize_access_token(self, _request: object) -> dict[str, object]:
        if self._error is not None:
            raise OAuthError(error=self._error, description="denied")
        return self._token


class _FakeOAuth:
    """Stand-in for the authlib OAuth registry."""

    def __init__(self, oidc: _FakeOidc | None = None) -> None:
        self.oidc = oidc or _FakeOidc()


def _request() -> MagicMock:
    return MagicMock()


def _configured_settings(
    test_settings: Settings, *, metadata_url: str = _LOCAL_ISSUER + "/.well-known/openid-configuration"
) -> Settings:
    oauth = OAuthSettings(
        client_id="my-qiip-client",
        client_secret=SecretStr("s3cret"),
        redirect_uri="https://proxy.example.com/auth/callback",
    )
    plugins = test_settings.plugins.model_copy(
        update={
            "config": {
                "auth.internal_oidc": {
                    "enabled": True,
                    "server_metadata_url": metadata_url,
                }
            }
        }
    )
    return test_settings.model_copy(deep=True, update={"oauth": oauth, "plugins": plugins})


def _plugin(settings: Settings) -> InternalOidcPlugin:
    manager = PluginManager(settings)
    manager.initialize()
    loaded = manager.get_plugin("auth.internal_oidc")
    assert isinstance(loaded, InternalOidcPlugin)
    return loaded


def _inject_fake(plugin: InternalOidcPlugin, fake: _FakeOAuth) -> None:
    plugin._client = fake


class TestInternalOidcPluginInitialize:
    def test_defaults_to_disabled(self, test_settings: Settings) -> None:
        plugin = InternalOidcPlugin()

        assert plugin.initialize(PluginManager(test_settings)) is False
        assert plugin.is_configured() is False

    def test_manager_skips_plugin_without_opt_in(self, test_settings: Settings) -> None:
        manager = PluginManager(test_settings)
        manager.initialize()

        assert manager.get_plugin("auth.internal_oidc") is None

    def test_refuses_when_metadata_url_missing(
        self, test_settings: Settings
    ) -> None:
        oauth = OAuthSettings(
            client_id="my-qiip-client",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
        )
        plugins = test_settings.plugins.model_copy(
            update={"config": {"auth.internal_oidc": {"enabled": True}}}
        )
        settings = test_settings.model_copy(
            deep=True, update={"oauth": oauth, "plugins": plugins}
        )
        plugin = InternalOidcPlugin({"enabled": True})

        assert plugin.initialize(PluginManager(settings)) is False

    def test_refuses_when_oauth_disabled(self, test_settings: Settings) -> None:
        plugin = InternalOidcPlugin(
            {
                "enabled": True,
                "server_metadata_url": _LOCAL_ISSUER + "/.well-known/openid-configuration",
            }
        )

        assert plugin.initialize(PluginManager(test_settings)) is False
        assert plugin.is_configured() is False

    def test_loads_when_opted_in(self, test_settings: Settings) -> None:
        plugin = _plugin(_configured_settings(test_settings))

        assert plugin.is_configured() is True
        assert plugin.name == "internal_oidc"
        assert plugin.label == "Local Auth"
        assert plugin.enabled is True


class TestInternalOidcPluginFlow:
    async def test_start_login_redirects_to_provider(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        fake = _FakeOidc()
        _inject_fake(plugin, _FakeOAuth(fake))

        result = await plugin.start_login(_request(), "https://proxy/callback")

        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert fake.authorize_redirect_calls == 1

    async def test_complete_login_returns_identity(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        _inject_fake(plugin, _FakeOAuth())

        identity = await plugin.complete_login(_request())

        assert isinstance(identity, AuthIdentity)
        assert identity.issuer == _LOCAL_ISSUER
        assert identity.sub == "subject-123"
        assert identity.email == "alice@localdomain"
        assert identity.email_verified is True
        assert identity.name == "Alice Example"

    async def test_complete_login_provider_error_maps_to_login_failed(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        _inject_fake(plugin, _FakeOAuth(_FakeOidc(error="access_denied")))

        with pytest.raises(AuthCallbackError) as exc_info:
            await plugin.complete_login(_request())

        assert exc_info.value.code == "login_failed"

    async def test_complete_login_missing_issuer_is_no_profile(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        userinfo = dict(_VALID_USERINFO)
        del userinfo["iss"]
        _inject_fake(plugin, _FakeOAuth(_FakeOidc(token={"userinfo": userinfo})))

        with pytest.raises(AuthCallbackError) as exc_info:
            await plugin.complete_login(_request())

        assert exc_info.value.code == "no_profile"
