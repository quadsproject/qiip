"""Unit tests for auth FastAPI dependencies (store/plugin resolution)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException

from inference_proxy.api.errors import ApiAuthError
from inference_proxy.auth.dependencies import (
    get_api_auth,
    get_auth_plugin,
    get_auth_store,
)
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.settings import Settings

from .conftest import FakeAllowlist, FakeAuthPlugin


def _app_with_state(
    **state_kwargs: object,
) -> tuple[FastAPI, MagicMock]:
    app = FastAPI()
    for key, value in state_kwargs.items():
        setattr(app.state, key, value)
    request = MagicMock()
    request.app = app
    return app, request


class TestGetAuthStore:
    def test_missing_store_raises_503(self) -> None:
        _app, request = _app_with_state()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_store(request)

        assert exc_info.value.status_code == 503

    def test_present_store_returned(self, tmp_path: Path) -> None:
        store = AuthStore(tmp_path / "auth.db")
        _app, request = _app_with_state(auth_store=store)

        assert get_auth_store(request) is store
        store.close()


class TestGetApiAuth:
    """Use-time enforcement of the SSO whitelist (fail closed)."""

    @staticmethod
    def _enforced_settings(
        test_settings: Settings, *, enforce: bool = True, api_enforce: bool = False
    ) -> Settings:
        auth = test_settings.auth.model_copy(
            update={
                "enforce_api_tokens": api_enforce,
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "enforce_sso_whitelist": enforce,
            }
        )
        return test_settings.model_copy(deep=True, update={"auth": auth})

    @staticmethod
    def _request(token: str | None) -> MagicMock:
        request = MagicMock()
        request.headers.get.return_value = (
            f"Bearer {token}" if token is not None else ""
        )
        return request

    def _user_token(self, auth_store: AuthStore) -> tuple[str, str]:
        user = auth_store.upsert_google_user(
            google_sub="sub-1", email="alice@example.com", name="", picture=""
        )
        return user.email, auth_store.create_token(user.id, "ci").token

    async def test_valid_token_allowed(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        email, token = self._user_token(auth_store)
        result = await get_api_auth(
            self._request(token),
            self._enforced_settings(test_settings),
            auth_store,
            FakeAllowlist(allowed=True),
        )

        assert result is not None
        assert result.user.email == email

    async def test_removed_user_rejected(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        _email, token = self._user_token(auth_store)

        with pytest.raises(ApiAuthError) as exc_info:
            await get_api_auth(
                self._request(token),
                self._enforced_settings(test_settings),
                auth_store,
                FakeAllowlist(allowed=False),
            )

        assert exc_info.value.message == "SSO whitelist denied"

    async def test_missing_allowlist_fails_closed(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        _email, token = self._user_token(auth_store)

        with pytest.raises(ApiAuthError) as exc_info:
            await get_api_auth(
                self._request(token),
                self._enforced_settings(test_settings),
                auth_store,
                None,
            )

        assert exc_info.value.message == "SSO whitelist unavailable"

    async def test_unavailable_allowlist_fails_closed(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        _email, token = self._user_token(auth_store)

        with pytest.raises(ApiAuthError) as exc_info:
            await get_api_auth(
                self._request(token),
                self._enforced_settings(test_settings),
                auth_store,
                FakeAllowlist(raises=True),
            )

        assert exc_info.value.message == "SSO whitelist unavailable"

    async def test_enforcement_off_skips_allowlist(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        email, token = self._user_token(auth_store)
        result = await get_api_auth(
            self._request(token),
            self._enforced_settings(test_settings, enforce=False),
            auth_store,
            None,
        )

        assert result is not None
        assert result.user.email == email

    async def test_api_and_whitelist_enforced_allowed(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        email, token = self._user_token(auth_store)
        result = await get_api_auth(
            self._request(token),
            self._enforced_settings(test_settings, api_enforce=True),
            auth_store,
            FakeAllowlist(allowed=True),
        )

        assert result is not None
        assert result.user.email == email

    async def test_api_enforced_invalid_token_skips_allowlist(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        with pytest.raises(ApiAuthError) as exc_info:
            await get_api_auth(
                self._request("qiip_invalid"),
                self._enforced_settings(test_settings, api_enforce=True),
                auth_store,
                None,
            )

        assert exc_info.value.message == "Invalid API token"

    async def test_api_enforced_missing_token_rejected(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        with pytest.raises(ApiAuthError) as exc_info:
            await get_api_auth(
                self._request(None),
                self._enforced_settings(test_settings, api_enforce=True),
                auth_store,
                None,
            )

        assert (
            exc_info.value.message == "Authentication required for inference requests"
        )

    async def test_invalid_token_is_anonymous_without_enforcement(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        result = await get_api_auth(
            self._request("qiip_invalid"),
            self._enforced_settings(test_settings, enforce=False),
            auth_store,
            None,
        )

        assert result is None


class TestGetAuthPlugin:
    def test_missing_plugin_raises_404(self) -> None:
        _app, request = _app_with_state()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request)

        assert exc_info.value.status_code == 404

    def test_unconfigured_present_plugin_raises_404(self) -> None:
        plugin = FakeAuthPlugin(error="access_denied")
        _app, request = _app_with_state(auth_plugin=plugin)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request)

        assert exc_info.value.status_code == 404

    def test_present_plugin_returned(self) -> None:
        plugin = FakeAuthPlugin()
        _app, request = _app_with_state(auth_plugin=plugin)

        assert get_auth_plugin(request) is plugin


class TestGetApiAuthAdminBypass:
    """Admin full-access list bypasses the use-time whitelist gate (RFE #107)."""

    @staticmethod
    def _admin_settings(test_settings: Settings) -> Settings:
        auth = test_settings.auth.model_copy(
            update={
                "enforce_sso_whitelist": True,
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "admin_only_tokens_full_access": ["ops@example.com"],
            }
        )
        return test_settings.model_copy(deep=True, update={"auth": auth})

    async def test_admin_token_bypasses_denied_whitelist(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        user = auth_store.upsert_google_user(
            google_sub="sub-ops",
            email="ops@example.com",
            name="Ops",
            picture="",
        )
        token = auth_store.create_token(user.id, "admin").token
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"

        result = await get_api_auth(
            request,
            self._admin_settings(test_settings),
            auth_store,
            FakeAllowlist(allowed=False),
        )

        assert result is not None
        assert result.user.email == "ops@example.com"

    async def test_non_admin_still_denied(
        self, test_settings: Settings, auth_store: AuthStore
    ) -> None:
        user = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )
        token = auth_store.create_token(user.id, "ci").token
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"

        with pytest.raises(ApiAuthError):
            await get_api_auth(
                request,
                self._admin_settings(test_settings),
                auth_store,
                FakeAllowlist(allowed=False),
            )


class _GoogleAuthPlugin(FakeAuthPlugin):
    """Named fake so provider selection can be exercised."""

    name = "google"


class _LocalOidcPlugin(FakeAuthPlugin):
    """Named fake so provider selection can be exercised."""

    name = "internal_oidc"


class _FakePluginManager:
    """Minimal PluginManager stand-in (get_plugins_by_type + get_plugin)."""

    def __init__(self, plugins: list[FakeAuthPlugin]) -> None:
        self.plugins = plugins

    def get_plugins_by_type(self, plugin_type: object) -> list[FakeAuthPlugin]:
        return self.plugins

    def get_plugin(self, name: str) -> FakeAuthPlugin | None:
        return next((p for p in self.plugins if p.name == name), None)


def _request_with_manager(plugins: list[FakeAuthPlugin]) -> MagicMock:
    app = FastAPI()
    app.state.plugin_manager = _FakePluginManager(plugins)
    request = MagicMock()
    request.app = app
    request.session = {}
    return request


class TestGetAuthPluginProviderSelection:
    def test_404_when_no_provider_configured(self) -> None:
        request = _request_with_manager([])

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request)
        assert exc_info.value.status_code == 404

    def test_unconfigured_plugin_is_skipped(self) -> None:
        disabled = _LocalOidcPlugin(error="unconfigured")
        request = _request_with_manager([disabled])

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request, provider="internal_oidc")
        assert exc_info.value.status_code == 404

    def test_selects_provider_by_name(self) -> None:
        google = _GoogleAuthPlugin()
        local = _LocalOidcPlugin()
        request = _request_with_manager([google, local])

        assert get_auth_plugin(request, provider="internal_oidc") is local
        assert get_auth_plugin(request, provider="google") is google

    def test_unknown_provider_is_404(self) -> None:
        request = _request_with_manager([_GoogleAuthPlugin()])

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request, provider="internal_oidc")
        assert exc_info.value.status_code == 404

    def test_session_marker_selects_provider(self) -> None:
        google = _GoogleAuthPlugin()
        local = _LocalOidcPlugin()
        request = _request_with_manager([google, local])
        request.session["auth_provider"] = "internal_oidc"

        assert get_auth_plugin(request) is local

    def test_stale_marker_falls_back_to_first(self) -> None:
        google = _GoogleAuthPlugin()
        request = _request_with_manager([google])
        request.session["auth_provider"] = "internal_oidc"

        assert get_auth_plugin(request) is google

    def test_without_marker_uses_first_configured(self) -> None:
        google = _GoogleAuthPlugin()
        local = _LocalOidcPlugin()
        request = _request_with_manager([google, local])

        assert get_auth_plugin(request) is google
