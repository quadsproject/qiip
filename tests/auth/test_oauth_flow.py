"""End-to-end tests for the OAuth login flow (via a fake auth plugin)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from inference_proxy.auth.dependencies import get_auth_plugin, get_sso_allowlist
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import OAuthSettings, Settings

from .conftest import FakeAllowlist, FakeAuthPlugin, FakeAuthPluginBuilder


class _RecordingPlugin(FakeAuthPlugin):
    """FakeAuthPlugin that records the redirect_uri passed to start_login."""

    def __init__(
        self,
        userinfo: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        super().__init__(userinfo, error)
        self.last_redirect_uri: str | None = None

    async def start_login(self, request: Request, redirect_uri: str):  # type: ignore[no-untyped-def]
        self.last_redirect_uri = redirect_uri
        return await super().start_login(request, redirect_uri)


def _client_with_auth(
    app: FastAPI,
    auth: object,
    settings: Settings,
    base_url: str = "http://testserver",
) -> TestClient:
    app.dependency_overrides[get_auth_plugin] = lambda: auth
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, base_url=base_url)


class TestOAuthLogin:
    def test_login_redirects_to_google(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        oauth = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
        )
        settings = test_settings.model_copy(update={"oauth": oauth})
        client = _client_with_auth(app, make_fake_auth_plugin(), settings)

        response = client.get("/auth/login", follow_redirects=False)

        assert response.status_code == 302
        assert response.headers["location"].startswith(
            "https://accounts.google.com/o/oauth2/auth"
        )

    def test_login_uses_request_host_when_allowlisted(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        """Multi-name deployments send the callback to the hostname the user
        started from, so the OAuth state nonce (host-scoped cookie) is still
        visible at the callback (regression: bare-name sign-in always failed
        with "Sign-in was not completed")."""
        oauth = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
            allowed_redirect_hosts=["inference-proxy.scalelab.example.com"],
        )
        settings = test_settings.model_copy(update={"oauth": oauth})
        plugin = _RecordingPlugin()
        client = _client_with_auth(
            app,
            plugin,
            settings,
            base_url="https://inference-proxy.scalelab.example.com",
        )

        response = client.get("/auth/login", follow_redirects=False)

        assert response.status_code == 302
        assert (
            plugin.last_redirect_uri
            == "https://inference-proxy.scalelab.example.com/auth/callback"
        )

    def test_login_preserves_port_and_configured_path_for_allowlisted_host(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        """Regression (sjug review): rebuilding the callback from
        ``.hostname`` alone dropped the request port and any configured path
        prefix, so ``http://localhost:5000/auth/callback`` became
        ``http://localhost/auth/callback`` and broke previously valid OAuth
        redirects. An allowlisted host must keep the request port and the
        configured callback path."""
        oauth = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="http://localhost:5000/prefix/auth/callback",
            allowed_redirect_hosts=["localhost"],
        )
        settings = test_settings.model_copy(update={"oauth": oauth})
        plugin = _RecordingPlugin()
        client = _client_with_auth(
            app,
            plugin,
            settings,
            base_url="http://localhost:5000",
        )

        response = client.get("/auth/login", follow_redirects=False)

        assert response.status_code == 302
        assert plugin.last_redirect_uri == "http://localhost:5000/prefix/auth/callback"

    def test_login_falls_back_to_configured_uri_for_unlisted_host(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        """A never-trusted Host header can never steer the redirect: any host
        outside the allowlist uses the configured redirect_uri, so single-name
        deployments are unchanged."""
        oauth = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
        )
        settings = test_settings.model_copy(update={"oauth": oauth})
        plugin = _RecordingPlugin()
        client = _client_with_auth(
            app,
            plugin,
            settings,
            base_url="https://evil.example.com",
        )

        response = client.get("/auth/login", follow_redirects=False)

        assert response.status_code == 302
        assert plugin.last_redirect_uri == "https://proxy.example.com/auth/callback"

    def test_login_returns_404_when_oauth_disabled(self, app: FastAPI) -> None:
        client = TestClient(app)

        response = client.get("/auth/login")

        assert response.status_code == 404


class TestOAuthCallback:
    def test_callback_empty_identity_rejected(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        from inference_proxy.plugins.interfaces.auth import AuthIdentity

        from .conftest import FakeAuthPlugin

        class EmptyIdentityPlugin(FakeAuthPlugin):
            async def complete_login(self, request: object) -> AuthIdentity:
                return AuthIdentity(sub="", email="", email_verified=True)

        app.dependency_overrides[get_auth_plugin] = lambda: EmptyIdentityPlugin()
        client = TestClient(app)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=no_profile" in response.headers["location"]

    def test_callback_signs_in_user_and_sets_session(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(app, make_fake_auth_plugin(), test_settings)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert response.status_code == 302
        assert response.headers["location"] == "/profile"

        me = client.get("/auth/me")
        assert me.status_code == 200
        assert me.json()["email"] == "alice@example.com"
        assert me.json()["name"] == "Alice Example"

    def test_callback_denied_redirects_with_error(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(
            app, make_fake_auth_plugin(error="access_denied"), test_settings
        )

        response = client.get(
            "/auth/callback?error=access_denied", follow_redirects=False
        )

        assert response.status_code == 302
        assert "error=login_failed" in response.headers["location"]

    def test_callback_unverified_email_rejected(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(
            app,
            make_fake_auth_plugin(
                {
                    "sub": "s1",
                    "email": "nobody@example.com",
                    "email_verified": False,
                }
            ),
            test_settings,
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=unverified_email" in response.headers["location"]

    def test_callback_missing_profile_rejected(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(app, make_fake_auth_plugin({}), test_settings)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=no_profile" in response.headers["location"]

    def test_callback_domain_allowlist_enforced(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        restricted = test_settings.model_copy(
            deep=True,
            update={
                "oauth": test_settings.oauth.model_copy(
                    update={"allowed_domains": ["allowed.example.com"]}
                )
            },
        )
        client = _client_with_auth(
            app,
            make_fake_auth_plugin(
                {
                    "sub": "s2",
                    "email": "alice@example.com",
                    "email_verified": True,
                }
            ),
            restricted,
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )
        assert "error=domain_not_allowed" in response.headers["location"]

        allowed_client = _client_with_auth(
            app,
            make_fake_auth_plugin(
                {
                    "sub": "s3",
                    "email": "bob@allowed.example.com",
                    "email_verified": True,
                }
            ),
            restricted,
        )
        allowed = allowed_client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )
        assert allowed.headers["location"] == "/profile"


class TestOAuthCallbackWhitelist:
    """Callback gate: whitelist enforced before user upsert and session."""

    @staticmethod
    def _enforced_settings(test_settings: Settings) -> Settings:
        auth = test_settings.auth.model_copy(
            update={
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "enforce_sso_whitelist": True,
            }
        )
        return test_settings.model_copy(deep=True, update={"auth": auth})

    def _client(
        self,
        app: FastAPI,
        settings: Settings,
        allowlist: object,
        auth: object,
    ) -> TestClient:
        app.dependency_overrides[get_auth_plugin] = lambda: auth
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_sso_allowlist] = lambda: allowlist
        return TestClient(app)

    def test_not_whitelisted_rejected_without_user_row(
        self,
        app: FastAPI,
        test_settings: Settings,
        auth_store: AuthStore,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._client(
            app,
            TestOAuthCallbackWhitelist._enforced_settings(test_settings),
            FakeAllowlist(allowed=False),
            make_fake_auth_plugin(),
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=not_whitelisted" in response.headers["location"]
        count = auth_store._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == 0

    def test_whitelisted_user_signs_in(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._client(
            app,
            TestOAuthCallbackWhitelist._enforced_settings(test_settings),
            FakeAllowlist(allowed=True),
            make_fake_auth_plugin(),
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert response.headers["location"] == "/profile"
        assert client.get("/auth/me").status_code == 200

    def test_unavailable_allowlist_fails_closed(
        self,
        app: FastAPI,
        test_settings: Settings,
        auth_store: AuthStore,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._client(
            app,
            TestOAuthCallbackWhitelist._enforced_settings(test_settings),
            FakeAllowlist(raises=True),
            make_fake_auth_plugin(),
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=allowlist_unavailable" in response.headers["location"]
        count = auth_store._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == 0

    def test_missing_allowlist_fails_closed(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._client(
            app,
            TestOAuthCallbackWhitelist._enforced_settings(test_settings),
            None,
            make_fake_auth_plugin(),
        )

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=allowlist_unavailable" in response.headers["location"]


class TestAuthMe:
    def test_me_anonymous_returns_401(self, app: FastAPI) -> None:
        client = TestClient(app)

        response = client.get("/auth/me")

        assert response.status_code == 401

    def test_me_without_sessions_returns_503(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        no_session = test_settings.model_copy(
            deep=True,
            update={
                "auth": test_settings.auth.model_copy(update={"session_secret": None})
            },
        )
        app.dependency_overrides[get_settings] = lambda: no_session
        client = TestClient(app)

        response = client.get("/auth/me")

        assert response.status_code == 503

    def test_me_roundtrip_after_callback_returns_user_identity(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(app, make_fake_auth_plugin(), test_settings)
        assert (
            client.get(
                "/auth/callback?code=code&state=state", follow_redirects=False
            ).status_code
            == 302
        )

        me = client.get("/auth/me")

        assert me.status_code == 200
        payload: dict[str, Any] = me.json()
        assert payload["id"] >= 1

    def test_me_reports_admin_role_after_grant(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
        auth_store: AuthStore,
    ) -> None:
        """/auth/me must surface the live admin role (regression)."""
        client = _client_with_auth(app, make_fake_auth_plugin(), test_settings)
        client.get("/auth/callback?code=code&state=state", follow_redirects=False)

        me = client.get("/auth/me").json()
        assert me["is_admin"] is False

        assert auth_store.set_user_admin(me["id"], True)
        me = client.get("/auth/me").json()
        assert me["is_admin"] is True


class TestOAuthLogout:
    def test_logout_without_sessions_redirects(self, test_settings: Settings) -> None:
        from inference_proxy.main import create_app

        no_session = test_settings.model_copy(
            deep=True,
            update={
                "auth": test_settings.auth.model_copy(update={"session_secret": None})
            },
        )
        client = TestClient(create_app(settings=no_session))

        response = client.post("/auth/logout", follow_redirects=False)

        assert response.status_code == 302
        assert response.headers["location"] == "/dashboard"

    def test_logout_clears_session(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = _client_with_auth(app, make_fake_auth_plugin(), test_settings)
        assert (
            client.get(
                "/auth/callback?code=code&state=state",
                follow_redirects=False,
            ).status_code
            == 302
        )
        assert client.get("/auth/me").status_code == 200

        response = client.post("/auth/logout", follow_redirects=False)

        assert response.status_code == 302
        assert client.get("/auth/me").status_code == 401


class TestBuildGoogleOAuth:
    def test_registers_google_client_with_credentials(self) -> None:
        from inference_proxy.plugins.builtin.auth.google import build_google_oauth

        settings = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
        )

        oauth = build_google_oauth(settings)

        assert oauth.google.client_id == "123.apps.googleusercontent.com"

    def test_disabled_settings_rejected(self) -> None:
        from inference_proxy.plugins.builtin.auth.google import build_google_oauth

        with pytest.raises(ValueError, match="disabled"):
            build_google_oauth(OAuthSettings())


class TestOAuthCallbackAdminBypass:
    """Admin full-access list bypasses the callback whitelist gate (RFE #107)."""

    @staticmethod
    def _admin_enforced_settings(test_settings: Settings) -> Settings:
        auth = test_settings.auth.model_copy(
            update={
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "enforce_sso_whitelist": True,
                "admin_only_tokens_full_access": ["ops@example.com"],
            }
        )
        return test_settings.model_copy(deep=True, update={"auth": auth})

    def test_admin_not_on_whitelist_still_signs_in(
        self,
        app: FastAPI,
        test_settings: Settings,
        auth_store: AuthStore,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        app.dependency_overrides[get_auth_plugin] = lambda: make_fake_auth_plugin(
            userinfo={
                "sub": "sub-ops",
                "email": "ops@example.com",
                "email_verified": True,
                "name": "Ops",
                "picture": "",
            }
        )
        app.dependency_overrides[get_settings] = lambda: (
            TestOAuthCallbackAdminBypass._admin_enforced_settings(test_settings)
        )
        app.dependency_overrides[get_sso_allowlist] = lambda: FakeAllowlist(
            allowed=False
        )
        client = TestClient(app)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert response.status_code == 302
        assert "error=" not in response.headers["location"]
        user = (
            auth_store.get_user_by_email("ops@example.com")
            if hasattr(auth_store, "get_user_by_email")
            else None
        )
        assert (
            user is not None
            or auth_store._conn.execute(
                "SELECT COUNT(*) FROM users WHERE email = 'ops@example.com'"
            ).fetchone()[0]
            == 1
        )

    def test_non_admin_still_redirected_error(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        app.dependency_overrides[get_auth_plugin] = lambda: make_fake_auth_plugin()
        app.dependency_overrides[get_settings] = lambda: (
            TestOAuthCallbackAdminBypass._admin_enforced_settings(test_settings)
        )
        app.dependency_overrides[get_sso_allowlist] = lambda: FakeAllowlist(
            allowed=False
        )
        client = TestClient(app)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=not_whitelisted" in response.headers["location"]
