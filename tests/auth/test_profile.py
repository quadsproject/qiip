"""Tests for the per-user profile surface: tokens CRUD and usage reporting."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth.dependencies import get_auth_plugin, get_sso_allowlist
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_settings
from inference_proxy.config.settings import Settings

from .conftest import FakeAllowlist, FakeAuthPluginBuilder


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


class TestTokenMintWhitelist:
    """SSO whitelist gate on POST /profile/tokens (fail closed)."""

    @staticmethod
    def _enforced_settings(test_settings: Settings) -> Settings:
        auth = test_settings.auth.model_copy(
            update={
                "sso_whitelist_url": "https://allowlist.example.com/list.json",
                "enforce_sso_whitelist": True,
            }
        )
        return test_settings.model_copy(deep=True, update={"auth": auth})

    def _signed_in_client(
        self,
        app: FastAPI,
        test_settings: Settings,
        mint_allowlist: object,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> TestClient:
        app.dependency_overrides[get_auth_plugin] = lambda: make_fake_auth_plugin()
        app.dependency_overrides[get_settings] = lambda: (
            TestTokenMintWhitelist._enforced_settings(test_settings)
        )
        # Sign in with an allow-all list; the mint-time gate uses the
        # allowlist under test afterwards.
        app.dependency_overrides[get_sso_allowlist] = lambda: FakeAllowlist(
            allowed=True
        )
        client = TestClient(app)
        assert (
            client.get(
                "/auth/callback?code=code&state=state", follow_redirects=False
            ).status_code
            == 302
        )
        app.dependency_overrides[get_sso_allowlist] = lambda: mint_allowlist
        return client

    def test_whitelisted_user_can_mint(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._signed_in_client(
            app, test_settings, FakeAllowlist(allowed=True), make_fake_auth_plugin
        )

        response = client.post("/profile/tokens", json={"name": "ci"})

        assert response.status_code == 201

    def test_denied_user_cannot_mint(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._signed_in_client(
            app, test_settings, FakeAllowlist(allowed=False), make_fake_auth_plugin
        )

        response = client.post("/profile/tokens", json={"name": "ci"})

        assert response.status_code == 403

    def test_unavailable_allowlist_blocks_mint(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._signed_in_client(
            app, test_settings, FakeAllowlist(raises=True), make_fake_auth_plugin
        )

        response = client.post("/profile/tokens", json={"name": "ci"})

        assert response.status_code == 503

    def test_missing_allowlist_blocks_mint(
        self,
        app: FastAPI,
        test_settings: Settings,
        make_fake_auth_plugin: FakeAuthPluginBuilder,
    ) -> None:
        client = self._signed_in_client(app, test_settings, None, make_fake_auth_plugin)

        response = client.post("/profile/tokens", json={"name": "ci"})

        assert response.status_code == 503
