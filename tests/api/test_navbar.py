"""Shared navbar regression tests.

The top navigation bar is a single shared partial
(``templates/partials/navbar.html``) included by every page. These tests
guard against the navbar drifting out of sync across the site -- e.g. a
link (like the OAuth ``/profile`` link) appearing on some pages but not
others -- which is the exact failure the shared partial prevents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_proxy.api.templating import templates

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "inference_proxy" / "templates"
_PARTIAL = _TEMPLATES_DIR / "partials" / "navbar.html"

_ALL_PAGES = (
    "/dashboard",
    "/dashboard/nodes/test-node",
    "/models",
    "/chat",
    "/profile",
)


class _TemplateRequest:
    def url_for(self, _name: str, **params: str) -> str:
        return f"/static/{params['path']}"


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
        ],
    )
    def test_no_page_embeds_its_own_nav_markup(self, template: str) -> None:
        """Pages must not re-define the nav markup (this is what caused the drift)."""
        text = (_TEMPLATES_DIR / template).read_text()
        assert '<nav class="top-bar"' not in text
        assert 'class="nav-link' not in text


class TestNavbarRendersDirectly:
    """The partial renders in isolation (used by frontend security harness)."""

    def test_partial_renders_without_active_page(self) -> None:
        """Rendering with no active_page must still produce a usable navbar."""
        rendered = templates.get_template("partials/navbar.html").render(
            request=_TemplateRequest()
        )
        assert '<nav class="top-bar" aria-label="Primary">' in rendered
        assert '<a href="/profile"' in rendered
