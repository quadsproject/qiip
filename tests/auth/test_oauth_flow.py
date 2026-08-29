"""End-to-end tests for the Google OAuth login flow (via a fake client)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from inference_proxy.auth.dependencies import get_oauth_client
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import OAuthSettings, Settings

from .conftest import FakeOAuth, FakeOAuthBuilder


def _client_with_oauth(
    app: FastAPI,
    oauth: FakeOAuth,
    settings: Settings,
) -> TestClient:
    app.dependency_overrides[get_oauth_client] = lambda: oauth
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


class TestOAuthLogin:
    def test_login_redirects_to_google(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(app, make_fake_oauth(), test_settings)

        response = client.get("/auth/login", follow_redirects=False)

        assert response.status_code == 302
        assert response.headers["location"].startswith(
            "https://accounts.google.com/o/oauth2/auth"
        )

    def test_login_returns_404_when_oauth_disabled(self, app: FastAPI) -> None:
        client = TestClient(app)

        response = client.get("/auth/login")

        assert response.status_code == 404


class TestOAuthCallback:
    def test_callback_signs_in_user_and_sets_session(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(app, make_fake_oauth(), test_settings)

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
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(
            app, make_fake_oauth(error="access_denied"), test_settings
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
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(
            app,
            make_fake_oauth(
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
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(app, make_fake_oauth(userinfo={}), test_settings)

        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )

        assert "error=no_profile" in response.headers["location"]

    def test_callback_domain_allowlist_enforced(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        restricted = test_settings.model_copy(
            deep=True,
            update={
                "oauth": test_settings.oauth.model_copy(
                    update={"allowed_domains": ["allowed.example.com"]}
                )
            },
        )
        client = _client_with_oauth(
            app,
            make_fake_oauth(
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

        allowed_client = _client_with_oauth(
            app,
            make_fake_oauth(
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
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(app, make_fake_oauth(), test_settings)
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


class TestOAuthLogout:
    def test_logout_clears_session(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_oauth: FakeOAuthBuilder,
    ) -> None:
        client = _client_with_oauth(app, make_fake_oauth(), test_settings)
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
        from inference_proxy.auth.oauth import build_google_oauth

        settings = OAuthSettings(
            client_id="123.apps.googleusercontent.com",
            client_secret=SecretStr("s3cret"),
            redirect_uri="https://proxy.example.com/auth/callback",
        )

        oauth = build_google_oauth(settings)

        assert oauth.google.client_id == "123.apps.googleusercontent.com"

    def test_disabled_settings_rejected(self) -> None:
        from inference_proxy.auth.oauth import build_google_oauth

        with pytest.raises(ValueError, match="disabled"):
            build_google_oauth(OAuthSettings())
