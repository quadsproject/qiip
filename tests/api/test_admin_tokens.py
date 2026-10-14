"""Tests for the admin token-management API surface (RFE #113)."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth.store import AuthStore


class _Seed(TypedDict):
    user_id: int
    token_id: int
    token: str


def _seed(store: AuthStore) -> _Seed:
    """Create one user with a token and recorded usage; return ids."""
    alice = store.upsert_google_user(
        google_sub="sub-alice",
        email="alice@example.com",
        name="Alice",
        picture="",
    )
    token = store.create_token(alice.id, "ci-job")
    store.record_usage(
        user_id=alice.id,
        token_id=token.id,
        model="llama-3",
        endpoint="/v1/chat/completions",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        total_tokens=1_500_000,
    )
    return {"user_id": alice.id, "token_id": token.id, "token": token.token}


class TestAdminAuth:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/admin/users"),
            ("GET", "/admin/tokens"),
            ("GET", "/admin/users/1"),
            ("DELETE", "/admin/tokens/1"),
            ("GET", "/admin/billing"),
        ],
    )
    def test_admin_routes_require_basic_auth(
        self, app: FastAPI, method: str, path: str
    ) -> None:
        response = TestClient(app).request(method, path)

        assert response.status_code == 401


class TestAdminUsers:
    def test_list_users_returns_stats_and_cost(
        self, client: TestClient, auth_store: AuthStore
    ) -> None:
        _seed(auth_store)

        response = client.get("/admin/users")

        assert response.status_code == 200
        users = response.json()
        assert len(users) == 1
        user = users[0]
        assert user["email"] == "alice@example.com"
        assert user["token_count"] == 1
        assert user["active_token_count"] == 1
        assert user["request_count"] == 1
        assert user["total_tokens"] == 1_500_000
        # 1M prompt * $2 + 500k completion * $10 = $7.00 (defaults).
        assert user["estimated_cost_usd"] == pytest.approx(7.0)

    def test_list_users_empty(self, client: TestClient) -> None:
        response = client.get("/admin/users")

        assert response.status_code == 200
        assert response.json() == []


class TestAdminTokens:
    def test_list_tokens_with_owner(
        self, client: TestClient, auth_store: AuthStore
    ) -> None:
        seeded = _seed(auth_store)

        response = client.get("/admin/tokens")

        assert response.status_code == 200
        tokens = response.json()
        assert len(tokens) == 1
        assert tokens[0]["user_id"] == seeded["user_id"]
        assert tokens[0]["user_email"] == "alice@example.com"
        assert tokens[0]["user_name"] == "Alice"
        assert tokens[0]["name"] == "ci-job"
        assert tokens[0]["request_count"] == 1
        assert tokens[0]["prompt_tokens"] == 1_000_000
        assert tokens[0]["completion_tokens"] == 500_000
        assert tokens[0]["total_tokens"] == 1_500_000
        # AUTH-02: the raw secret is exposed exactly once, at creation only.
        assert "token" not in tokens[0]

    def test_revoke_any_token(self, client: TestClient, auth_store: AuthStore) -> None:
        seeded = _seed(auth_store)

        revoked = client.delete(f"/admin/tokens/{seeded['token_id']}")

        assert revoked.status_code == 200
        assert revoked.json() == {"revoked": True}
        tokens = client.get("/admin/tokens").json()
        assert tokens[0]["revoked"] is True
        assert auth_store.resolve_token(seeded["token"]) is None

    def test_revoke_unknown_token_404(self, client: TestClient) -> None:
        response = client.delete("/admin/tokens/4242")

        assert response.status_code == 404


class TestAdminUserDetail:
    def test_user_detail_full_view(
        self, client: TestClient, auth_store: AuthStore
    ) -> None:
        seeded = _seed(auth_store)

        response = client.get(f"/admin/users/{seeded['user_id']}")

        assert response.status_code == 200
        detail = response.json()
        assert detail["user"]["email"] == "alice@example.com"
        assert detail["tokens"][0]["name"] == "ci-job"
        assert detail["usage"][0]["model"] == "llama-3"
        assert detail["usage"][0]["endpoint"] == "/v1/chat/completions"
        assert detail["totals"]["total_tokens"] == 1_500_000
        assert detail["timeline"][0]["total_tokens"] == 1_500_000
        assert detail["estimated_cost_usd"] == pytest.approx(7.0)
        assert detail["model_label"] == "claude-sonnet-5"

    def test_user_detail_unknown_404(self, client: TestClient) -> None:
        response = client.get("/admin/users/999")

        assert response.status_code == 404


class TestAdminBilling:
    def test_billing_summary_math(
        self, client: TestClient, auth_store: AuthStore
    ) -> None:
        _seed(auth_store)

        response = client.get("/admin/billing")

        assert response.status_code == 200
        billing: dict[str, Any] = response.json()
        assert billing["totals"]["total_tokens"] == 1_500_000
        assert billing["totals"]["request_count"] == 1
        assert billing["estimated_cost_usd"] == pytest.approx(7.0)
        assert billing["model_label"] == "claude-sonnet-5"

    def test_billing_empty(self, client: TestClient) -> None:
        response = client.get("/admin/billing")

        assert response.status_code == 200
        billing = response.json()
        assert billing["totals"]["total_tokens"] == 0
        assert billing["estimated_cost_usd"] == 0.0


class TestAdminTokenPages:
    def test_tokens_page_requires_admin(self, app: FastAPI) -> None:
        """Anonymous visitors get the sign-in page instead of the token view."""
        response = TestClient(app).get("/dashboard/tokens")

        assert response.status_code == 200
        assert "Local Admin" in response.text
        assert "QIIP - Token Management" not in response.text

    def test_user_detail_page_requires_admin(self, app: FastAPI) -> None:
        response = TestClient(app).get("/dashboard/users/1")

        assert response.status_code == 200
        assert "Local Admin" in response.text
        assert "QIIP - User Tokens" not in response.text

    def test_tokens_page_renders(self, client: TestClient) -> None:
        response = client.get("/dashboard/tokens")

        assert response.status_code == 200
        assert "QIIP - Token Management" in response.text
        assert "users-table-body" in response.text

    def test_user_detail_page_renders(self, client: TestClient) -> None:
        response = client.get("/dashboard/users/1")

        assert response.status_code == 200
        assert "QIIP - User Tokens" in response.text
        assert "timeline-table-body" in response.text


@pytest.mark.parametrize(
    ("route", "labels"),
    [
        ("/profile", ("API Tokens", "Usage")),
        ("/dashboard/tokens", ("Users", "All tokens")),
        ("/dashboard/users/1", ("Tokens", "Usage", "Daily timeline")),
    ],
)
def test_token_pages_declare_sortable_columns(
    client: TestClient, route: str, labels: tuple[str, ...]
) -> None:
    """Each token page loads the shared sorter and labels its sortable columns."""
    html = client.get(route).text

    assert "js/table_sort.js" in html
    kinds = re.findall(r'data-sort="([^"]*)"', html)
    assert kinds
    assert set(kinds) <= {"text", "number", "date"}
    headers = re.findall(r'<th[^>]*data-sort="[^"]*"[^>]*>(.*?)</th>', html)
    assert len(headers) == len(kinds)
    for header in headers:
        assert 'type="button"' in header
    for label in labels:
        assert f'aria-label="{label}"' in html
