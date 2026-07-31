"""Integration tests for the chat route and HTML content.

Tests cover:
- GET /chat returns 200 with text/html content type (CHAT-01)
- Chat page assets: chat.css, chat.js, locally vendored Markdown rendering (CHAT-02)
- Chat page elements: model selector, message area, textarea, send button (CHAT-01, CHAT-03)
- Accessibility: role="log", aria-live="polite"
- Navigation: Chat link on dashboard, Dashboard link on chat (D-03)
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestChatRoute:
    """GET /chat returns 200 HTML from the same app (CHAT-01)."""

    def test_chat_returns_200(self, client: TestClient) -> None:
        """GET /chat returns status code 200."""
        response = client.get("/chat")
        assert response.status_code == 200

    def test_chat_returns_html(self, client: TestClient) -> None:
        """Response content-type contains text/html."""
        response = client.get("/chat")
        assert "text/html" in response.headers["content-type"]

    def test_chat_served_by_same_app(self, client: TestClient) -> None:
        """TestClient serves both /chat and /v1/models -- proves wiring (CHAT-01)."""
        models_response = client.get("/v1/models")
        chat_response = client.get("/chat")
        assert models_response.status_code == 200
        assert chat_response.status_code == 200


class TestChatTemplate:
    """Chat HTML includes expected assets and elements (CHAT-01, CHAT-02, CHAT-03)."""

    def test_contains_chat_css_link(self, client: TestClient) -> None:
        """HTML contains link to chat.css."""
        response = client.get("/chat")
        assert "chat.css" in response.text

    def test_contains_chat_js_script(self, client: TestClient) -> None:
        """HTML contains script tag for chat.js."""
        response = client.get("/chat")
        assert "chat.js" in response.text

    def test_contains_local_markdown_scripts(self, client: TestClient) -> None:
        """HTML loads pinned local Marked and DOMPurify assets (CHAT-02)."""
        response = client.get("/chat")
        assert "/static/vendor/marked-18.0.7/marked.umd.js" in response.text
        assert "/static/vendor/dompurify-3.4.12/purify.min.js" in response.text
        assert "cdn.jsdelivr.net" not in response.text

    def test_contains_model_select(self, client: TestClient) -> None:
        """HTML contains select element with id='model-select' (CHAT-03)."""
        response = client.get("/chat")
        assert 'id="model-select"' in response.text

    def test_contains_message_area(self, client: TestClient) -> None:
        """HTML contains div with id='message-area'."""
        response = client.get("/chat")
        assert 'id="message-area"' in response.text

    def test_contains_chat_input(self, client: TestClient) -> None:
        """HTML contains textarea with id='chat-input' (CHAT-01)."""
        response = client.get("/chat")
        assert 'id="chat-input"' in response.text

    def test_contains_send_button(self, client: TestClient) -> None:
        """HTML contains button with id='send-btn' (CHAT-01)."""
        response = client.get("/chat")
        assert 'id="send-btn"' in response.text

    def test_contains_role_log(self, client: TestClient) -> None:
        """Message area has role='log' for accessibility."""
        response = client.get("/chat")
        assert 'role="log"' in response.text

    def test_contains_aria_live_polite(self, client: TestClient) -> None:
        """Message area has aria-live='polite' for screen readers."""
        response = client.get("/chat")
        assert 'aria-live="polite"' in response.text

    def test_contains_empty_state(self, client: TestClient) -> None:
        """HTML contains empty state heading text."""
        response = client.get("/chat")
        assert "Start a conversation" in response.text

    def test_contains_toast_container(self, client: TestClient) -> None:
        """HTML contains toast container div."""
        response = client.get("/chat")
        assert 'id="toast-container"' in response.text


class TestChatSystemPrompt:
    """System prompt HTML elements are present with correct attributes (CFG-01, CFG-02)."""

    def test_contains_system_prompt_toggle(self, client: TestClient) -> None:
        """HTML contains button with class='system-prompt-toggle' (CFG-01)."""
        response = client.get("/chat")
        assert 'class="system-prompt-toggle"' in response.text

    def test_toggle_has_aria_expanded(self, client: TestClient) -> None:
        """Toggle button has aria-expanded='false' by default (CFG-01)."""
        response = client.get("/chat")
        assert 'aria-expanded="false"' in response.text

    def test_toggle_has_aria_controls(self, client: TestClient) -> None:
        """Toggle button has aria-controls linking to panel (CFG-01)."""
        response = client.get("/chat")
        assert 'aria-controls="system-prompt-panel"' in response.text

    def test_contains_system_prompt_panel(self, client: TestClient) -> None:
        """HTML contains collapsible panel div (CFG-01)."""
        response = client.get("/chat")
        assert 'id="system-prompt-panel"' in response.text

    def test_contains_system_prompt_textarea(self, client: TestClient) -> None:
        """HTML contains system prompt textarea (CFG-01)."""
        response = client.get("/chat")
        assert 'id="system-prompt"' in response.text

    def test_textarea_has_aria_label(self, client: TestClient) -> None:
        """System prompt textarea has aria-label for accessibility (CFG-02)."""
        response = client.get("/chat")
        assert 'aria-label="System prompt"' in response.text

    def test_textarea_has_placeholder(self, client: TestClient) -> None:
        """System prompt textarea has placeholder text (CFG-01)."""
        response = client.get("/chat")
        assert 'placeholder="You are a helpful assistant..."' in response.text


class TestChatNavigation:
    """Both pages have cross-navigation links (D-03)."""

    def test_dashboard_has_chat_link(self, client: TestClient) -> None:
        """Dashboard nav bar contains href='/chat'."""
        response = client.get("/dashboard")
        assert 'href="/chat"' in response.text

    def test_chat_has_dashboard_link(self, client: TestClient) -> None:
        """Chat nav bar contains href='/dashboard'."""
        response = client.get("/chat")
        assert 'href="/dashboard"' in response.text
