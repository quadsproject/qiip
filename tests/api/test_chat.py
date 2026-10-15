"""Integration tests for the chat route and HTML content.

Tests cover:
- GET /chat returns 200 with text/html content type (CHAT-01)
- Chat page assets: chat.css, chat.js, locally vendored Markdown rendering (CHAT-02)
- Chat page elements: model selector, message area, textarea, send button (CHAT-01, CHAT-03)
- Accessibility: role="log", aria-live="polite"
- Navigation: Chat link on dashboard, Dashboard link on chat (D-03)
- Viewer access: only signed-in viewers get the playground, and a signed-in
  normal user chats with their own token so usage is attributed (CHAT-04)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from tests.auth.conftest import FakeAuthPlugin

_MODEL = "org/model-a"


def _node(node_id: str, model: str, *, admin_only: bool = False) -> Node:
    return Node(
        node_id=node_id,
        endpoint="10.0.1.100:8000",
        status=NodeStatus.HEALTHY,
        model=model,
        managed=False,
        self_setup=True,
        admin_only=admin_only,
    )


def _signed_in_user_client(app: FastAPI) -> TestClient:
    """Return a client signed in through the real callback as a normal user."""
    app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state", follow_redirects=False
    )
    assert response.headers["location"] == "/start"
    return client


def _session_secret(settings: Settings) -> str:
    """Return the session secret the test fixtures configured."""
    secret = settings.auth.session_secret
    assert secret is not None
    return secret.get_secret_value()


def _completion(total_tokens: int) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": total_tokens - 5,
            "completion_tokens": 5,
            "total_tokens": total_tokens,
        },
    }


def _playground_headers(token: str) -> dict[str, str]:
    """Exactly what the page sends: the token plus the strict-auth request."""
    return {
        "Authorization": f"Bearer {token}",
        "X-Qiip-Require-Token": "1",
    }


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


class TestChatViewerAccess:
    """The playground is for signed-in viewers only (CHAT-04)."""

    def test_anonymous_gets_the_signin_page(self, app: FastAPI) -> None:
        response = TestClient(app).get("/chat")

        assert response.status_code == 200
        assert "Local Admin" in response.text
        assert 'id="model-select"' not in response.text

    def test_signed_in_user_gets_the_playground(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client = _signed_in_user_client(app)

        response = client.get("/chat")

        assert response.status_code == 200
        assert 'id="model-select"' in response.text
        # The page must ask this viewer for their own token.
        assert "const ANONYMOUS_PLAYGROUND = false;" in response.text

    def test_admin_role_gets_the_anonymous_playground(
        self,
        client: TestClient,
    ) -> None:
        response = client.get("/chat")

        assert response.status_code == 200
        assert "const ANONYMOUS_PLAYGROUND = true;" in response.text

    def test_full_access_user_gets_the_anonymous_playground(
        self,
        app: FastAPI,
        test_settings: Settings,
        auth_store: AuthStore,
    ) -> None:
        """The trust list keeps a portfolio of tokens, so it is never asked to
        reveal a personal one either."""
        enabled = test_settings.model_copy(
            deep=True,
            update={
                "auth": test_settings.auth.model_copy(
                    update={"admin_only_tokens_full_access": ["alice@example.com"]}
                )
            },
        )
        app.dependency_overrides[get_settings] = lambda: enabled
        client = _signed_in_user_client(app)

        response = client.get("/chat")

        assert response.status_code == 200
        assert "const ANONYMOUS_PLAYGROUND = true;" in response.text


class TestPlaygroundToken:
    """A normal user's playground runs on their own token (CHAT-04).

    The credential is ``POST /onboarding/token/reveal``: its contract, the
    refusals the page turns into its notice, and the admin ``403`` that keeps
    admins anonymous are covered in test_onboarding.py. These tests pin what
    the playground does with the revealed token.
    """

    def test_revealed_token_attributes_usage_and_narrows_models(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        test_settings: Settings,
        httpx_mock: HTTPXMock,
    ) -> None:
        client = _signed_in_user_client(app)
        user = auth_store.get_user(auth_store.list_users_with_stats()[0].id)
        assert user is not None
        created = auth_store.create_personal_token(
            user.id,
            "chat tool",
            [_MODEL],
            _session_secret(test_settings),
        )
        test_registry.add(_node("gpu01", _MODEL))
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            json=_completion(10),
        )

        credential = client.post("/onboarding/token/reveal").json()

        assert credential["token"] == created.token
        headers = _playground_headers(credential["token"])
        assert [
            m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]
        ] == [_MODEL]

        turn = client.post(
            "/v1/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )

        assert turn.status_code == 200
        usage = auth_store.get_usage_summary(created.id)
        assert [(row.model, row.total_tokens) for row in usage] == [(_MODEL, 10)]

    def test_admin_only_model_is_not_reachable(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        test_settings: Settings,
        httpx_mock: HTTPXMock,
    ) -> None:
        """A user's playground token cannot reach an admin-only server, even
        when the token was scoped to that model name."""
        client = _signed_in_user_client(app)
        user = auth_store.get_user(auth_store.list_users_with_stats()[0].id)
        assert user is not None
        created = auth_store.create_personal_token(
            user.id,
            "chat tool",
            ["secret"],
            _session_secret(test_settings),
        )
        test_registry.add(_node("admin-only1", "secret", admin_only=True))
        test_registry.add(_node("pub1", _MODEL))

        raw = client.post("/onboarding/token/reveal").json()["token"]
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [_MODEL]

        response = client.post(
            "/v1/chat/completions",
            json={"model": "secret", "messages": [{"role": "user", "content": "hi"}]},
            headers=_playground_headers(raw),
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"
        assert len(httpx_mock.get_requests()) == 0
        assert raw == created.token

    def test_replacing_the_token_between_turns_is_followed(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        test_settings: Settings,
        httpx_mock: HTTPXMock,
    ) -> None:
        """The page reveals before every turn, so a token replaced in another
        /start tab is followed. The revoked value is refused instead of being
        served as an anonymous request (enforcement is off by default), which
        would silently lose the turn's usage."""
        client = _signed_in_user_client(app)
        user = auth_store.get_user(auth_store.list_users_with_stats()[0].id)
        assert user is not None
        first = auth_store.create_personal_token(
            user.id,
            "first",
            [_MODEL],
            _session_secret(test_settings),
        )
        test_registry.add(_node("gpu01", _MODEL))
        for total in (10, 20):
            httpx_mock.add_response(
                url="http://10.0.1.100:8000/v1/chat/completions",
                json=_completion(total),
            )

        turned = client.post(
            "/v1/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "one"}]},
            headers=_playground_headers(
                client.post("/onboarding/token/reveal").json()["token"]
            ),
        )
        replacement = auth_store.create_personal_token(
            user.id,
            "second",
            [_MODEL],
            _session_secret(test_settings),
        )
        revealed = client.post("/onboarding/token/reveal").json()["token"]
        again = client.post(
            "/v1/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "two"}]},
            headers=_playground_headers(revealed),
        )
        stale = client.post(
            "/v1/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "three"}]},
            headers=_playground_headers(first.token),
        )

        assert revealed == replacement.token
        assert replacement.token != first.token
        assert (turned.status_code, again.status_code) == (200, 200)
        assert stale.status_code == 401
        assert [
            row.total_tokens
            for row in auth_store.get_usage_summary(user.id)
            if row.token_id == first.id
        ] == [10]
        assert [
            row.total_tokens
            for row in auth_store.get_usage_summary(user.id)
            if row.token_id == replacement.id
        ] == [20]
        assert len(httpx_mock.get_requests()) == 2
