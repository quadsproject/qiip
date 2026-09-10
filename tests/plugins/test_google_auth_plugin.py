"""Unit tests for the built-in Google auth plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from authlib.integrations.starlette_client import OAuthError
from fastapi.responses import RedirectResponse
from pydantic import SecretStr

from inference_proxy.config.settings import OAuthSettings, Settings
from inference_proxy.plugins.builtin.auth.google import GoogleAuthPlugin
from inference_proxy.plugins.interfaces.auth import AuthCallbackError, AuthIdentity
from inference_proxy.plugins.manager import PluginManager

_VALID_USERINFO: dict[str, object] = {
    "sub": "subject-123",
    "email": "alice@example.com",
    "email_verified": True,
    "name": "Alice Example",
    "picture": "https://example.com/avatar.png",
}


class _FakeGoogle:
    """Stand-in for the authlib Google client with scripted responses."""

    def __init__(
        self,
        token: dict[str, object] | None = None,
        error: str | None = None,
        redirect_to: str = "https://accounts.google.com/o/oauth2/auth?state=abc",
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

    def __init__(self, google: _FakeGoogle | None = None) -> None:
        self.google = google or _FakeGoogle()


def _request() -> MagicMock:
    return MagicMock()


def _configured_settings(test_settings: Settings) -> Settings:
    oauth = OAuthSettings(
        client_id="123.apps.googleusercontent.com",
        client_secret=SecretStr("s3cret"),
        redirect_uri="https://proxy.example.com/auth/callback",
    )
    return test_settings.model_copy(deep=True, update={"oauth": oauth})


def _plugin(settings: Settings) -> GoogleAuthPlugin:
    manager = PluginManager(settings)
    manager.initialize()
    loaded = manager.get_plugin("auth.google")
    assert isinstance(loaded, GoogleAuthPlugin)
    return loaded


def _inject_fake(plugin: GoogleAuthPlugin, fake: _FakeOAuth) -> None:
    plugin._client = fake


class TestGoogleAuthPluginInitialize:
    def test_refuses_when_oauth_disabled(self, test_settings: Settings) -> None:
        plugin = GoogleAuthPlugin()

        assert plugin.initialize(PluginManager(test_settings)) is False
        assert plugin.is_configured() is False

    def test_loads_when_oauth_configured(self, test_settings: Settings) -> None:
        plugin = _plugin(_configured_settings(test_settings))

        assert plugin.is_configured() is True
        assert plugin.name == "google"
        assert plugin.enabled is True

    def test_manager_skips_unconfigured_plugin(self, test_settings: Settings) -> None:
        manager = PluginManager(test_settings)
        manager.initialize()

        assert manager.get_plugin("auth.google") is None


class TestGoogleAuthPluginFlow:
    async def test_start_login_redirects_to_provider(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        fake = _FakeGoogle()
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
        assert identity.sub == "subject-123"
        assert identity.email == "alice@example.com"
        assert identity.email_verified is True
        assert identity.name == "Alice Example"
        assert identity.picture == "https://example.com/avatar.png"

    async def test_complete_login_provider_error_maps_to_login_failed(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        _inject_fake(plugin, _FakeOAuth(_FakeGoogle(error="access_denied")))

        with pytest.raises(AuthCallbackError) as exc_info:
            await plugin.complete_login(_request())

        assert exc_info.value.code == "login_failed"

    @pytest.mark.parametrize(
        "userinfo",
        [
            {"email": "alice@example.com", "email_verified": True},
            {"sub": "s1", "email": "", "email_verified": True},
            {"sub": "s1", "email": None, "email_verified": True},
        ],
    )
    async def test_complete_login_missing_claims_rejected(
        self, test_settings: Settings, userinfo: dict[str, object]
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        _inject_fake(plugin, _FakeOAuth(_FakeGoogle(token={"userinfo": userinfo})))

        with pytest.raises(AuthCallbackError) as exc_info:
            await plugin.complete_login(_request())

        assert exc_info.value.code == "no_profile"

    async def test_complete_login_treats_non_boolean_verified_as_false(
        self, test_settings: Settings
    ) -> None:
        plugin = _plugin(_configured_settings(test_settings))
        info: dict[str, object] = {
            "sub": "s1",
            "email": "alice@example.com",
            "email_verified": "yes",
            "name": 42,
            "picture": None,
        }
        _inject_fake(plugin, _FakeOAuth(_FakeGoogle(token={"userinfo": info})))

        identity = await plugin.complete_login(_request())

        assert identity.email_verified is False
        assert identity.name == ""
        assert identity.picture == ""


class TestAuthCallbackError:
    def test_invalid_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid auth callback error code"):
            AuthCallbackError("bad code!")

    def test_valid_code_accepted(self) -> None:
        error = AuthCallbackError("login_failed")

        assert error.code == "login_failed"
