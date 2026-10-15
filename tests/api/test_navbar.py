"""Shared navbar regression tests.

The top navigation bar is a single shared partial
(``templates/partials/navbar.html``) included by every page. These tests
guard against the navbar drifting out of sync across the site -- e.g. a
link (like the OAuth ``/profile`` link) appearing on some pages but not
others -- which is the exact failure the shared partial prevents.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from inference_proxy.api.templating import templates
from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import OAuthSettings, Settings
from tests.auth.conftest import FakeAuthPlugin

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "inference_proxy" / "templates"
_PARTIAL = _TEMPLATES_DIR / "partials" / "navbar.html"
_ACCOUNT_PARTIAL = _TEMPLATES_DIR / "partials" / "account_actions.html"
_ACCOUNT_INCLUDE = '{% include "partials/account_actions.html" %}'

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


@pytest.fixture
def user_session_client(app: FastAPI, test_settings: Settings) -> TestClient:
    """A signed-in normal user (not an admin), as in tests/api/test_onboarding.py."""
    test_settings.oauth = OAuthSettings(
        client_id="test-client",
        client_secret=SecretStr("test-secret"),
        redirect_uri="http://testserver/auth/callback",
    )
    app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state", follow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/start"
    return client


def _account_cluster(html: str) -> str:
    """Return the rendered account cluster (identity + logout + theme toggle)."""
    marker = '<div class="nav-actions">'
    start = html.index(marker)
    return html[start : html.index("</div>", start) + len("</div>")]


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
        # /profile renders the shared navbar for anonymous visitors; the chat
        # page sends them to the sign-in shell instead.
        response = TestClient(app).get("/profile")

        assert response.status_code == 200
        assert ">Logout</button>" not in response.text


class TestAccountClusterIsShared:
    """Identity, logout and theme render the same way on every signed-in surface.

    A signed-in normal user is the only viewer who reaches both the onboarding
    page and a navbar page, so those two surfaces are compared directly.
    """

    def test_normal_user_reaches_both_surfaces(
        self, user_session_client: TestClient
    ) -> None:
        for route in ("/start", "/leaderboard"):
            response = user_session_client.get(route, follow_redirects=False)
            assert response.status_code == 200, route

    def test_cluster_is_identical_on_start_and_leaderboard(
        self, user_session_client: TestClient
    ) -> None:
        onboarding = user_session_client.get("/start", follow_redirects=False).text
        leaderboard = user_session_client.get(
            "/leaderboard", follow_redirects=False
        ).text

        cluster = _account_cluster(onboarding)

        assert cluster == _account_cluster(leaderboard)
        assert "alice@example.com" in cluster

    def test_cluster_orders_identity_logout_then_theme(
        self, user_session_client: TestClient
    ) -> None:
        for route in ("/start", "/leaderboard"):
            cluster = _account_cluster(
                user_session_client.get(route, follow_redirects=False).text
            )
            assert (
                cluster.index('class="nav-identity"')
                < cluster.index("nav-logout-btn")
                < cluster.index('class="theme-toggle"')
            ), route

    def test_logout_label_is_its_own_accessible_name(
        self, user_session_client: TestClient
    ) -> None:
        """WCAG 2.2 Label in Name: no aria-label may contradict the visible text."""
        for route in ("/start", "/leaderboard"):
            cluster = _account_cluster(
                user_session_client.get(route, follow_redirects=False).text
            )
            button = re.search(
                r"<button[^>]*nav-logout-btn[^>]*>([^<]*)</button>", cluster
            )
            assert button is not None, route
            assert "aria-label" not in button.group(0), route
            assert button.group(1).strip() == "Logout", route

    def test_identity_is_rendered_before_the_page_loads(
        self, user_session_client: TestClient
    ) -> None:
        """The identity comes from the server, not from /onboarding/state."""
        onboarding = user_session_client.get("/start", follow_redirects=False).text

        assert "alice@example.com" in onboarding.split("<main", 1)[0]

    def test_leaderboard_link_is_not_inside_the_logout_form(
        self, user_session_client: TestClient
    ) -> None:
        onboarding = user_session_client.get("/start", follow_redirects=False).text
        form = re.search(
            r'<form method="post" action="/auth/logout".*?</form>', onboarding, re.S
        )

        assert form is not None
        assert "Leaderboard" not in form.group(0)


class TestAccountClusterIsSingleSource:
    """Neither top bar may define its own identity, logout or theme markup."""

    def test_partial_defines_the_cluster(self) -> None:
        text = _ACCOUNT_PARTIAL.read_text()

        assert '<div class="nav-actions">' in text
        assert 'action="/auth/logout"' in text
        assert 'class="theme-toggle"' in text

    @pytest.mark.parametrize("template", ["partials/navbar.html", "start.html"])
    def test_top_bars_include_the_account_cluster(self, template: str) -> None:
        assert _ACCOUNT_INCLUDE in (_TEMPLATES_DIR / template).read_text()

    @pytest.mark.parametrize("template", ["partials/navbar.html", "start.html"])
    def test_top_bars_do_not_embed_the_cluster(self, template: str) -> None:
        """A second copy is what let the two bars drift apart."""
        text = (_TEMPLATES_DIR / template).read_text()

        for marker in (
            'class="nav-identity"',
            'class="nav-logout-form"',
            'class="theme-toggle"',
        ):
            assert marker not in text, f"{template} embeds {marker}"


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
