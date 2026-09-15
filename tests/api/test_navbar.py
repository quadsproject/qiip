"""Shared navbar regression tests.

The top navigation bar is a single shared partial
(``templates/partials/navbar.html``) included by every page. These tests
guard against the navbar drifting out of sync across the site -- e.g. a
link (like the OAuth ``/profile`` link) appearing on some pages but not
others -- which is the exact failure the shared partial prevents.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.api.templating import templates
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "inference_proxy" / "templates"
_PARTIAL = _TEMPLATES_DIR / "partials" / "navbar.html"

_ALL_PAGES = (
    "/dashboard",
    "/dashboard/nodes/test-node",
    "/dashboard/tokens",
    "/dashboard/users/1",
    "/models",
    "/chat",
    "/profile",
)


class _TemplateRequest:
    def url_for(self, _name: str, **params: str) -> str:
        return f"/static/{params['path']}"

    headers: dict[str, str] = {}


class TestNavbarRenderedOnEveryPage:
    """Every page renders the shared top bar and the profile link."""

    @pytest.mark.parametrize("route", _ALL_PAGES)
    def test_every_page_renders_the_top_bar(
        self, client: TestClient, route: str
    ) -> None:
        response = client.get(route)
        assert response.status_code == 200
        assert '<nav class="top-bar" aria-label="Primary">' in response.text

    @pytest.mark.parametrize("route", _ALL_PAGES)
    def test_every_page_has_the_profile_link(
        self, client: TestClient, route: str
    ) -> None:
        """The OAuth /profile link appears on every page (incl. node detail)."""
        response = client.get(route)
        assert response.status_code == 200
        assert '<a href="/profile"' in response.text
        assert ">Profile</a>" in response.text


class TestNavbarActiveState:
    """The current page is marked active in the shared navbar."""

    def test_dashboard_marks_dashboard_active(self, client: TestClient) -> None:
        response = client.get("/dashboard")
        assert (
            '<a href="/dashboard" class="nav-link nav-link-active" aria-current="page">Dashboard</a>'
            in response.text
        )

    def test_node_detail_marks_dashboard_active(self, client: TestClient) -> None:
        """Node detail is part of the dashboard section, so Dashboard stays active."""
        response = client.get("/dashboard/nodes/test-node")
        assert (
            '<a href="/dashboard" class="nav-link nav-link-active" aria-current="page">Dashboard</a>'
            in response.text
        )

    def test_models_marks_models_active(self, client: TestClient) -> None:
        response = client.get("/models")
        assert (
            '<a href="/models" class="nav-link nav-link-active" aria-current="page">Models</a>'
            in response.text
        )

    def test_tokens_marks_tokens_active(self, client: TestClient) -> None:
        response = client.get("/dashboard/tokens")
        assert (
            '<a href="/dashboard/tokens" class="nav-link nav-link-active" aria-current="page">Tokens</a>'
            in response.text
        )

    def test_user_detail_marks_tokens_active(self, client: TestClient) -> None:
        response = client.get("/dashboard/users/1")
        assert (
            '<a href="/dashboard/tokens" class="nav-link nav-link-active" aria-current="page">Tokens</a>'
            in response.text
        )

    def test_chat_marks_chat_active(self, client: TestClient) -> None:
        response = client.get("/chat")
        assert (
            '<a href="/chat" class="nav-link nav-link-active" aria-current="page">Chat</a>'
            in response.text
        )

    def test_profile_marks_profile_active(self, client: TestClient) -> None:
        response = client.get("/profile")
        assert (
            '<a href="/profile" class="nav-link nav-link-active" aria-current="page">Profile</a>'
            in response.text
        )


class TestNavbarIsSingleSourceOfTruth:
    """The navbar only lives in the shared partial; pages include it."""

    def test_shared_partial_exists(self) -> None:
        assert _PARTIAL.is_file()

    def test_partial_contains_the_nav_markup(self) -> None:
        partial_text = _PARTIAL.read_text()
        assert '<nav class="top-bar" aria-label="Primary">' in partial_text
        assert '<a href="/profile"' in partial_text

    @pytest.mark.parametrize(
        "template",
        [
            "dashboard.html",
            "node_detail.html",
            "models.html",
            "chat.html",
            "profile.html",
            "tokens.html",
            "user_detail.html",
        ],
    )
    def test_every_page_includes_the_shared_partial(self, template: str) -> None:
        text = (_TEMPLATES_DIR / template).read_text()
        assert '{% include "partials/navbar.html" %}' in text

    @pytest.mark.parametrize(
        "template",
        [
            "dashboard.html",
            "node_detail.html",
            "models.html",
            "chat.html",
            "profile.html",
            "tokens.html",
            "user_detail.html",
        ],
    )
    def test_no_page_embeds_its_own_nav_markup(self, template: str) -> None:
        """Pages must not re-define the nav markup (this is what caused the drift)."""
        text = (_TEMPLATES_DIR / template).read_text()
        assert '<nav class="top-bar"' not in text
        assert 'class="nav-link' not in text


class TestNavbarLogout:
    """The Logout control appears for every signed-in viewer, next to the theme toggle."""

    def test_logout_visible_for_basic_admin(self, client: TestClient) -> None:
        response = client.get("/chat")

        assert response.status_code == 200
        assert ">Logout</button>" in response.text
        # Placed just left of the theme toggle.
        assert response.text.index(">Logout</button>") < response.text.index(
            'class="theme-toggle"'
        )

    def test_logout_hidden_for_anonymous(self, app: FastAPI) -> None:
        response = TestClient(app).get("/chat")

        assert response.status_code == 200
        assert ">Logout</button>" not in response.text


class TestNavbarRendersDirectly:
    """The partial renders in isolation (used by frontend security harness)."""

    def test_partial_renders_without_active_page(self) -> None:
        """Rendering with no active_page must still produce a usable navbar."""
        rendered = templates.get_template("partials/navbar.html").render(
            request=_TemplateRequest()
        )
        assert '<nav class="top-bar" aria-label="Primary">' in rendered
        assert '<a href="/profile"' in rendered


def test_navbar_resolves_app_settings_not_env(
    app: FastAPI,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (sjug review): the navbar helpers called get_settings()
    directly, bypassing create_app(settings=...). An injected app therefore
    showed the wrong admin links (or failed to render) when the environment
    held different or missing credentials.
    """
    monkeypatch.setenv("INFERENCE_PROXY_ADMIN__USERNAME", "other-admin")
    monkeypatch.setenv("INFERENCE_PROXY_ADMIN__PASSWORD", "other-pass")
    get_settings.cache_clear()

    token = base64.b64encode(b"test-admin:test-password").decode()
    client = TestClient(app, headers={"Authorization": "Basic " + token})
    response = client.get("/dashboard")

    assert response.status_code == 200
    # Admin-only navigation is visible because the navbar resolves the same
    # settings the app was created with (test_settings), not the env cache.
    assert '<a href="/dashboard/tokens"' in response.text
