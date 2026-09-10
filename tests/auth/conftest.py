"""Shared helpers for exercising the OAuth + profile surface in tests.

A fake auth plugin is injected through ``get_auth_plugin`` so the real
login/callback routes run end-to-end without network access to Google. The
session cookie is produced by the real session machinery, which lets the
same ``TestClient`` stay signed in across subsequent requests.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient

from inference_proxy.auth.allowlist import AllowlistUnavailableError
from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.plugins.interfaces.auth import (
    AuthCallbackError,
    AuthIdentity,
    AuthPlugin,
)

_VALID_USERINFO: dict[str, object] = {
    "sub": "subject-123",
    "email": "alice@example.com",
    "email_verified": True,
    "name": "Alice Example",
    "picture": "https://example.com/avatar.png",
}


class FakeAuthPlugin(AuthPlugin):
    """Scripted stand-in for an AuthPlugin (no network, no authlib)."""

    name = "fake-auth"
    version = "0.0.0"
    description = "test double"
    author = "tests"

    def __init__(
        self,
        userinfo: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        super().__init__()
        self.userinfo = userinfo if userinfo is not None else dict(_VALID_USERINFO)
        self.error = error

    def is_configured(self) -> bool:
        """Return True when no scripted provider error is active."""
        return self.error is None

    async def start_login(
        self, request: Request, redirect_uri: str
    ) -> RedirectResponse:
        return RedirectResponse(
            "https://accounts.google.com/o/oauth2/auth?state=abc",
            status_code=302,
        )

    async def complete_login(self, request: Request) -> AuthIdentity:
        if self.error is not None:
            raise AuthCallbackError("login_failed")
        info = self.userinfo
        sub = info.get("sub")
        email = info.get("email")
        if not isinstance(sub, str) or not sub:
            raise AuthCallbackError("no_profile")
        if not isinstance(email, str) or not email:
            raise AuthCallbackError("no_profile")
        name = info.get("name")
        picture = info.get("picture")
        return AuthIdentity(
            sub=sub,
            email=email,
            email_verified=info.get("email_verified") is True,
            name=name if isinstance(name, str) else "",
            picture=picture if isinstance(picture, str) else "",
        )


class FakeAllowlist:
    """Scripted SSOAllowlist: allow/deny everything, or raise unavailable."""

    def __init__(self, allowed: bool = True, raises: bool = False) -> None:
        self._allowed = allowed
        self._raises = raises

    async def is_allowed(self, _email: str) -> bool:
        if self._raises:
            raise AllowlistUnavailableError("scripted failure")
        return self._allowed


FakeAuthPluginBuilder = Callable[..., FakeAuthPlugin]


@pytest.fixture
def make_fake_auth_plugin() -> FakeAuthPluginBuilder:
    """Return a builder for a scripted fake auth plugin."""

    def _make(
        userinfo: dict[str, object] | None = None,
        error: str | None = None,
    ) -> FakeAuthPlugin:
        return FakeAuthPlugin(userinfo, error)

    return _make


@pytest.fixture
def profile_client(
    app: FastAPI,
    make_fake_auth_plugin: FakeAuthPluginBuilder,
) -> TestClient:
    """Return a TestClient already signed in through the real /auth/callback."""
    plugin = make_fake_auth_plugin()
    app.dependency_overrides[get_auth_plugin] = lambda: plugin
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/profile"
    return client
