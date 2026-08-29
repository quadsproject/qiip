"""Shared helpers for exercising the OAuth + profile surface in tests.

A fake authlib client is injected through ``get_oauth_client`` so the real
login/callback routes run end-to-end without network access to Google. The
session cookie is produced by the real session machinery, which lets the
same ``TestClient`` stay signed in across subsequent requests.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient

from inference_proxy.auth.dependencies import get_oauth_client

_VALID_USERINFO: dict[str, object] = {
    "sub": "subject-123",
    "email": "alice@example.com",
    "email_verified": True,
    "name": "Alice Example",
    "picture": "https://example.com/avatar.png",
}


class FakeGoogle:
    """Stand-in for ``oauth.google`` with scripted OAuth responses."""

    def __init__(
        self,
        userinfo: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        self.userinfo = userinfo
        self.error = error

    async def authorize_redirect(
        self,
        _request: object,
        redirect_uri: str | None = None,
    ) -> RedirectResponse:
        return RedirectResponse(
            "https://accounts.google.com/o/oauth2/auth?state=abc",
            status_code=302,
        )

    async def authorize_access_token(self, _request: object) -> dict[str, object]:
        if self.error is not None:
            from authlib.integrations.starlette_client import OAuthError

            raise OAuthError(error=self.error, description="denied")
        return {"userinfo": self.userinfo or {}}


class FakeOAuth:
    """Stand-in for the authlib OAuth registry."""

    def __init__(self, google: FakeGoogle | None = None) -> None:
        self.google = google or FakeGoogle(_VALID_USERINFO)


FakeOAuthBuilder = Callable[..., FakeOAuth]


@pytest.fixture
def make_fake_oauth() -> FakeOAuthBuilder:
    """Return a builder for a scripted fake OAuth client."""

    def _make(
        userinfo: dict[str, object] | None = None,
        error: str | None = None,
    ) -> FakeOAuth:
        if userinfo is None:
            userinfo = _VALID_USERINFO
        return FakeOAuth(FakeGoogle(userinfo, error))

    return _make


@pytest.fixture
def profile_client(
    app: FastAPI,
    make_fake_oauth: FakeOAuthBuilder,
) -> TestClient:
    """Return a TestClient already signed in through the real /auth/callback."""
    oauth = make_fake_oauth()
    app.dependency_overrides[get_oauth_client] = lambda: oauth
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/profile"
    return client
