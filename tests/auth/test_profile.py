"""Tests for the per-user profile surface: tokens CRUD and usage reporting."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth.store import AuthStore


class TestProfilePage:
    def test_profile_page_is_public_shell(self, app: FastAPI) -> None:
        client = TestClient(app)

        response = client.get("/profile")

        assert response.status_code == 200
        assert "QIIP - Profile" in response.text
        assert "API Tokens" in response.text


class TestProfileMe:
    def test_me_requires_sign_in(self, app: FastAPI) -> None:
        client = TestClient(app)

        response = client.get("/profile/me")

        assert response.status_code == 401

    def test_me_returns_identity(self, profile_client: TestClient) -> None:
        response = profile_client.get("/profile/me")

        assert response.status_code == 200
        assert response.json()["email"] == "alice@example.com"
        assert "id" in response.json()


class TestTokenManagement:
    def test_list_tokens_starts_empty(self, profile_client: TestClient) -> None:
        response = profile_client.get("/profile/tokens")

        assert response.status_code == 200
        assert response.json() == []

    def test_create_list_and_revoke_roundtrip(
        self,
        profile_client: TestClient,
    ) -> None:
        created_resp = profile_client.post("/profile/tokens", json={"name": "ci-job"})
        assert created_resp.status_code == 201
        created: dict[str, Any] = created_resp.json()
        assert created["name"] == "ci-job"
        assert created["token"].startswith("qiip_")
        assert created["revoked"] is False

        tokens = profile_client.get("/profile/tokens")
        assert tokens.status_code == 200
        listed = tokens.json()
        assert len(listed) == 1
        # The raw secret never appears in the list endpoint (AUTH-02).
        assert all("token" not in item for item in listed)
        assert listed[0]["prefix"] == created["prefix"]
        assert listed[0]["id"] == created["id"]

        revoked = profile_client.delete(f"/profile/tokens/{created['id']}")
        assert revoked.status_code == 200
        assert revoked.json() == {"revoked": True}

        revoked_list = profile_client.get("/profile/tokens")
        assert revoked_list.json()[0]["revoked"] is True

    def test_revoke_unknown_token_returns_404(
        self,
        profile_client: TestClient,
    ) -> None:
        response = profile_client.delete("/profile/tokens/4242")

        assert response.status_code == 404

    def test_create_blank_name_rejected(self, profile_client: TestClient) -> None:
        response = profile_client.post("/profile/tokens", json={"name": "  "})

        assert response.status_code == 422

    def test_tokens_require_sign_in(self, app: FastAPI) -> None:
        client = TestClient(app)

        assert client.get("/profile/tokens").status_code == 401
        assert client.post("/profile/tokens", json={"name": "x"}).status_code == 401

    def test_stale_session_user_returns_401(
        self,
        profile_client: TestClient,
        auth_store: AuthStore,
    ) -> None:
        # A signed-in session whose user row has been removed must not
        # resurrect the identity (session re-read from the store).
        auth_store._conn.execute("DELETE FROM users")
        auth_store._conn.commit()

        response = profile_client.get("/profile/tokens")

        assert response.status_code == 401


class TestUsageSummary:
    def test_usage_starts_empty(self, profile_client: TestClient) -> None:
        response = profile_client.get("/profile/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["rows"] == []
        assert data["totals"]["request_count"] == 0

    def test_usage_reflects_recorded_usage(
        self,
        profile_client: TestClient,
        auth_store: AuthStore,
    ) -> None:
        me = profile_client.get("/profile/me").json()
        created = profile_client.post("/profile/tokens", json={"name": "ci-job"}).json()
        auth_store.record_usage(
            user_id=me["id"],
            token_id=created["id"],
            model="llama-3",
            endpoint="/v1/chat/completions",
            prompt_tokens=4,
            completion_tokens=6,
            total_tokens=10,
        )

        data = profile_client.get("/profile/usage").json()

        assert data["rows"][0]["token_name"] == "ci-job"
        assert data["rows"][0]["total_tokens"] == 10
        assert data["totals"]["request_count"] == 1
        assert data["totals"]["total_tokens"] == 10

    def test_usage_requires_sign_in(self, app: FastAPI) -> None:
        client = TestClient(app)

        assert client.get("/profile/usage").status_code == 401
