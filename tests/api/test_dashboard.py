"""Integration tests for the dashboard route and HTML content.

Tests cover:
- GET /dashboard returns 200 with text/html content type (DASH-01)
- Dashboard served by same app as API (DASH-03)
- HTML contains Google Fonts and dashboard assets (TMPL-01, TMPL-02)
- Table structure with 10 column headers including GPU Vendor, GPU Model, State, Actions (NODE-01)
- Badge CSS classes for status, circuit breaker, and provisioning states (NODE-02)
- Manual setup toggle and QUADS status element (D-04, D-05, D-09)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import DashboardSettings, Settings
from inference_proxy.main import create_app


class TestDashboardRoute:
    """GET /dashboard returns 200 HTML from the same app (DASH-01, DASH-03, TMPL-01)."""

    def test_dashboard_returns_200(self, client: TestClient) -> None:
        """GET /dashboard returns status code 200."""
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_dashboard_returns_html(self, client: TestClient) -> None:
        """Response content-type contains text/html."""
        response = client.get("/dashboard")
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_served_by_same_app(self, client: TestClient) -> None:
        """TestClient (wrapping create_app()) serves /dashboard -- proves DASH-03."""
        # The client fixture uses the same FastAPI app that serves /admin/nodes.
        # If this request succeeds, the dashboard shares the app.
        admin_response = client.get("/admin/nodes")
        dashboard_response = client.get("/dashboard")
        assert admin_response.status_code == 200
        assert dashboard_response.status_code == 200


class TestDashboardTemplate:
    """Dashboard HTML includes expected asset references (TMPL-01, TMPL-02)."""

    def test_contains_google_fonts_link(self, client: TestClient) -> None:
        """HTML contains Google Fonts link for Open Sans, Poppins, IBM Plex Mono."""
        response = client.get("/dashboard")
        assert "fonts.googleapis.com" in response.text

    def test_contains_dashboard_css_link(self, client: TestClient) -> None:
        """HTML contains link to dashboard.css."""
        response = client.get("/dashboard")
        assert "dashboard.css" in response.text

    def test_contains_dashboard_js_script(self, client: TestClient) -> None:
        """HTML contains script tag for dashboard.js."""
        response = client.get("/dashboard")
        assert "dashboard.js" in response.text

    def test_contains_config_download_js(self, client: TestClient) -> None:
        """HTML contains script tag for config_download.js."""
        response = client.get("/dashboard")
        assert "config_download.js" in response.text

    def test_config_download_js_loaded_before_dashboard_js(
        self, client: TestClient
    ) -> None:
        """config_download.js appears before dashboard.js in the HTML."""
        response = client.get("/dashboard")
        config_pos = response.text.index("config_download.js")
        dashboard_pos = response.text.index("dashboard.js")
        assert config_pos < dashboard_pos

    def test_setup_selection_js_loaded_before_dashboard_js(
        self, client: TestClient
    ) -> None:
        """The shared setup selector is available before dashboard.js runs."""
        response = client.get("/dashboard")
        setup_pos = response.text.index("setup_selection.js")
        dashboard_pos = response.text.index("dashboard.js")
        assert setup_pos < dashboard_pos

    def test_fonts_loaded_before_dashboard_css(self, client: TestClient) -> None:
        """Google Fonts link appears before dashboard.css link in the HTML."""
        response = client.get("/dashboard")
        fonts_pos = response.text.index("fonts.googleapis.com")
        dashboard_pos = response.text.index("dashboard.css")
        assert fonts_pos < dashboard_pos


class TestDashboardTableStructure:
    """Dashboard HTML contains the node fleet table structure (NODE-01, D-01, D-02)."""

    def test_contains_all_column_headers(self, client: TestClient) -> None:
        """HTML contains all th elements for the node table."""
        response = client.get("/dashboard")
        headers = [
            "Node ID",
            "GPU Vendor",
            "GPU Model",
            "Engine",
            "Model",
            "Config",
            "State",
            "Requests",
            "Actions",
        ]
        for header in headers:
            assert header in response.text, f"Missing column header: {header}"

    def test_contains_requests_column_header(self, client: TestClient) -> None:
        """HTML contains the Requests column header (METR-02)."""
        response = client.get("/dashboard")
        assert "Requests" in response.text

    def test_contains_table_body_id(self, client: TestClient) -> None:
        """HTML contains tbody with id="node-table-body" for JS population."""
        response = client.get("/dashboard")
        assert 'id="node-table-body"' in response.text

    def test_loading_row_colspan_matches_column_count(self, client: TestClient) -> None:
        """Loading placeholder colspan matches the number of column headers."""
        response = client.get("/dashboard")
        assert 'colspan="9"' in response.text


class TestDashboardPolling:
    """Dashboard HTML includes polling configuration (DASH-02)."""

    def test_contains_poll_interval_js_variable(self, client: TestClient) -> None:
        """HTML contains POLL_INTERVAL_MS JavaScript variable."""
        response = client.get("/dashboard")
        assert "POLL_INTERVAL_MS" in response.text

    def test_poll_interval_default_value(self, client: TestClient) -> None:
        """Default poll interval is 10s = 10000ms in the JS variable."""
        response = client.get("/dashboard")
        assert "10000" in response.text

    async def test_create_app_explicit_settings_reach_dashboard(
        self,
        test_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        admin_auth_headers: dict[str, str],
    ) -> None:
        """The factory's explicit settings own route dependency resolution."""
        monkeypatch.setenv("INFERENCE_PROXY_DASHBOARD__POLL_INTERVAL", "23")
        get_settings.cache_clear()
        explicit = test_settings.model_copy(
            update={"dashboard": DashboardSettings(poll_interval=17)}
        )
        application = create_app(settings=explicit)

        try:
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=admin_auth_headers,
            ) as client:
                response = await asyncio.wait_for(client.get("/dashboard"), timeout=2)
        finally:
            get_settings.cache_clear()

        assert response.status_code == 200
        assert "const POLL_INTERVAL_MS = 17000;" in response.text

    def test_contains_last_updated_element(self, client: TestClient) -> None:
        """HTML contains element with id='last-updated'."""
        response = client.get("/dashboard")
        assert 'id="last-updated"' in response.text

    def test_contains_poll_warning_element(self, client: TestClient) -> None:
        """HTML contains element with id='poll-warning'."""
        response = client.get("/dashboard")
        assert 'id="poll-warning"' in response.text

    def test_contains_quads_status_element(self, client: TestClient) -> None:
        """HTML contains span with id='quads-status' for QUADS indicator (D-09)."""
        response = client.get("/dashboard")
        assert 'id="quads-status"' in response.text


class TestDashboardBadgeCSS:
    """Badge CSS contains classes for all status and circuit breaker states (NODE-02)."""

    _css_path = (
        Path(__file__).resolve().parent.parent.parent
        / "inference_proxy"
        / "static"
        / "css"
        / "dashboard.css"
    )

    def test_badge_css_contains_all_status_classes(self) -> None:
        """dashboard.css contains .badge-healthy, .badge-unhealthy, .badge-draining."""
        css = self._css_path.read_text()
        for cls in (".badge-healthy", ".badge-unhealthy", ".badge-draining"):
            assert cls in css, f"Missing CSS class: {cls}"

    def test_badge_css_contains_all_cb_classes(self) -> None:
        """dashboard.css contains .badge-closed, .badge-open, .badge-half_open."""
        css = self._css_path.read_text()
        for cls in (".badge-closed", ".badge-open", ".badge-half_open"):
            assert cls in css, f"Missing CSS class: {cls}"

    def test_badge_css_contains_provisioning_classes(self) -> None:
        """dashboard.css contains .badge-complete, .badge-failed, .badge-in-progress."""
        css = self._css_path.read_text()
        for cls in (".badge-complete", ".badge-failed", ".badge-in-progress"):
            assert cls in css, f"Missing CSS class: {cls}"

    def test_badge_css_contains_available_class(self) -> None:
        """dashboard.css contains .badge-available for available state badge (DASH-01)."""
        css = self._css_path.read_text()
        assert ".badge-available" in css, "Missing CSS class: .badge-available"

    def test_badge_css_contains_action_button_classes(self) -> None:
        """dashboard.css contains .btn-setup and .btn-teardown action variants (D-06)."""
        css = self._css_path.read_text()
        assert ".btn-setup" in css, "Missing CSS class: .btn-setup"
        assert ".btn-teardown" in css, "Missing CSS class: .btn-teardown"

    def test_badge_css_contains_config_button_class(self) -> None:
        """dashboard.css contains .btn-config for config download buttons."""
        css = self._css_path.read_text()
        assert ".btn-config" in css, "Missing CSS class: .btn-config"

    def test_setup_engine_selector_has_visible_theme_styling(self) -> None:
        """The engine selector cannot collapse to an unstyled empty control."""
        css = self._css_path.read_text()
        assert ".setup-select {" in css
        assert ".setup-engine-select { min-width: 6.5rem; }" in css


class TestSetupForm:
    """Dashboard HTML contains the setup form elements (DASH-01, D-04, D-05)."""

    def test_contains_setup_form(self, client: TestClient) -> None:
        """HTML contains form with id='setup-form' (moved inside Node Fleet card)."""
        response = client.get("/dashboard")
        assert 'id="setup-form"' in response.text

    def test_standalone_provision_card_removed(self, client: TestClient) -> None:
        """Standalone 'Provision Node' card is removed (D-04)."""
        response = client.get("/dashboard")
        assert "Provision Node" not in response.text

    def test_contains_manual_setup_toggle(self, client: TestClient) -> None:
        """HTML contains manual setup toggle link (D-05)."""
        response = client.get("/dashboard")
        assert 'id="manual-setup-toggle"' in response.text
        assert "+ Manual setup" in response.text

    def test_contains_manual_setup_row(self, client: TestClient) -> None:
        """HTML contains hidden manual setup row container."""
        response = client.get("/dashboard")
        assert 'id="manual-setup-row"' in response.text

    def test_contains_hostname_input(self, client: TestClient) -> None:
        """HTML contains input with id='setup-hostname'."""
        response = client.get("/dashboard")
        assert 'id="setup-hostname"' in response.text

    def test_contains_setup_button(self, client: TestClient) -> None:
        """HTML contains button with id='setup-btn'."""
        response = client.get("/dashboard")
        assert 'id="setup-btn"' in response.text

    def test_contains_engine_specific_setup_selectors(self, client: TestClient) -> None:
        """Manual setup exposes mutually exclusive engine artifact selectors."""
        response = client.get("/dashboard")
        assert (
            'id="setup-engine-select" class="setup-select setup-engine-select"'
            in response.text
        )
        assert 'id="model-select"' in response.text
        assert 'id="artifact-select"' in response.text


class TestTasksPanel:
    """Provisioning tasks panel moved to per-node detail page."""

    def test_tasks_panel_not_on_main_dashboard(self, client: TestClient) -> None:
        """Tasks panel was removed from the main dashboard."""
        response = client.get("/dashboard")
        assert 'id="tasks-panel"' not in response.text

    def test_tasks_panel_on_node_detail(self, client: TestClient) -> None:
        """Tasks panel is present on the node detail page."""
        response = client.get("/dashboard/nodes/test-node")
        assert response.status_code == 200
        assert 'id="tasks-panel"' in response.text
        assert 'id="tasks-table-body"' in response.text


class TestNodeDetailPage:
    """Per-node detail page at /dashboard/nodes/{node_id}."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET /dashboard/nodes/{node_id} returns 200."""
        response = client.get("/dashboard/nodes/test-node")
        assert response.status_code == 200

    def test_returns_html(self, client: TestClient) -> None:
        """Response is HTML."""
        response = client.get("/dashboard/nodes/test-node")
        assert "text/html" in response.headers["content-type"]

    def test_contains_node_id(self, client: TestClient) -> None:
        """Page contains the node_id."""
        response = client.get("/dashboard/nodes/test-node")
        assert "test-node" in response.text

    def test_contains_back_link(self, client: TestClient) -> None:
        """Page contains a link back to the fleet dashboard."""
        response = client.get("/dashboard/nodes/test-node")
        assert 'href="/dashboard"' in response.text

    def test_contains_node_detail_js(self, client: TestClient) -> None:
        """Page loads node_detail.js."""
        response = client.get("/dashboard/nodes/test-node")
        assert "node_detail.js" in response.text

    def test_fqdn_node_id(self, client: TestClient) -> None:
        """Supports FQDN node IDs with dots."""
        response = client.get("/dashboard/nodes/host01.example.com")
        assert response.status_code == 200
        assert "host01.example.com" in response.text

    def test_contains_config_download_panel(self, client: TestClient) -> None:
        """Page contains the config download panel section."""
        response = client.get("/dashboard/nodes/test-node")
        assert 'id="config-download-panel"' in response.text

    def test_contains_config_download_js(self, client: TestClient) -> None:
        """Page loads config_download.js."""
        response = client.get("/dashboard/nodes/test-node")
        assert "config_download.js" in response.text

    def test_config_download_js_loaded_before_node_detail_js(
        self, client: TestClient
    ) -> None:
        """config_download.js appears before node_detail.js in the HTML."""
        response = client.get("/dashboard/nodes/test-node")
        config_pos = response.text.index("config_download.js")
        detail_pos = response.text.index("node_detail.js")
        assert config_pos < detail_pos

    def test_setup_selection_js_loaded_before_node_detail_js(
        self, client: TestClient
    ) -> None:
        """The shared setup selector is available before node_detail.js runs."""
        response = client.get("/dashboard/nodes/test-node")
        setup_pos = response.text.index("setup_selection.js")
        detail_pos = response.text.index("node_detail.js")
        assert setup_pos < detail_pos

    def test_node_detail_contains_engine_and_artifact_controls(
        self, client: TestClient
    ) -> None:
        """Node detail renders engine identity and catalog-backed setup controls."""
        response = client.get("/dashboard/nodes/test-node")
        assert '<th scope="col">Engine</th>' in response.text
        assert (
            'id="setup-engine-select" class="setup-select setup-engine-select"'
            in response.text
        )
        assert 'id="artifact-select"' in response.text
        assert 'colspan="10"' in response.text

    def test_node_detail_contains_read_only_llamacpp_runtime_card(
        self, client: TestClient
    ) -> None:
        response = client.get("/dashboard/nodes/test-node")

        assert 'id="llamacpp-runtime-panel" hidden' in response.text
        assert 'id="llamacpp-runtime-status"' in response.text
        assert 'aria-live="polite"' in response.text
        assert 'id="llamacpp-runtime-values" class="runtime-grid"' in response.text
        assert "<dl" in response.text
        assert 'id="llamacpp-runtime-min-free"' in response.text
        assert 'id="llamacpp-runtime-min-headroom"' in response.text
        assert "runtime-summary" not in response.text
        assert 'id="llamacpp-runtime-gpus"' in response.text
