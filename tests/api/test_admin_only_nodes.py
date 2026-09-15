"""Admin-only inference servers and admin-role access gates.

Regression coverage for:

- ``admin_only`` registration through the pool adoption path
- admin-only node scope rules in the node selector (admin-only routing)
- admin-only nodes excluded from the public ``/v1/models`` catalog
- the trimmed ``/fleet/nodes`` surface (admin-only servers and actions stripped)
- admin-role grant/revoke API and session-based admin access
- per-role dashboard page gating
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pytest_httpx import HTTPXMock

from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.auth.models import AdminUserStats
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_node_selector, get_settings
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.admin import RegisterRequest
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.routing.node_selector import NodeSelector
from tests.auth.conftest import FakeAuthPlugin


def _make_node(
    node_id: str,
    *,
    admin_only: bool = False,
    owner: str = "",
    model: str = "llama-3",
) -> Node:
    """Create an adopted (self-setup) test node."""
    return Node(
        node_id=node_id,
        endpoint="10.0.1.100:8000",
        status=NodeStatus.HEALTHY,
        model=model,
        managed=False,
        self_setup=True,
        admin_only=admin_only,
        owner=owner,
    )


def _signed_in_client(
    app: FastAPI, store: AuthStore
) -> tuple[TestClient, AdminUserStats]:
    """Sign a Google user in through the real callback; return client + user."""
    app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state", follow_redirects=False
    )
    assert response.status_code == 302
    users = store.list_users_with_stats()
    assert len(users) == 1
    return client, users[0]


class TestAdminOnlyRegistration:
    def test_admin_only_implies_self_setup(self) -> None:
        request = RegisterRequest(hostname="gpu01", admin_only=True)

        assert request.self_setup is True

    def test_admin_only_pool_registration_sets_flags(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.register_self_setup = AsyncMock(
            return_value=_make_node("gpu01", admin_only=True, model="org/model")
        )
        mock_provisioner.validate_endpoint.return_value = "http://gpu01:8000"

        response = client.post(
            "/admin/nodes/pool", json={"hostname": "gpu01", "admin_only": True}
        )

        assert response.status_code == 201
        assert response.json() == {
            "hostname": "gpu01",
            "state": "healthy",
            "model": "org/model",
            "self_setup": True,
            "admin_only": True,
            "name": "",
        }
        mock_provisioner.register_self_setup.assert_awaited_once_with(
            "gpu01", None, owner="", admin_only=True, name=""
        )


class TestAdminOnlyRoutingScope:
    def test_admin_only_nodes_require_admin_scope(
        self,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
    ) -> None:
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))

        # Admin scope (owner=None) reaches admin-only nodes.
        assert node_selector.select("secret", owner=None) is not None
        # Anonymous (owner="") and user (email) scopes never do.
        assert node_selector.select("secret", owner="") is None
        assert node_selector.select("secret", owner="alice@example.com") is None
        # A non-admin-only node stays reachable for anonymous scope.
        assert node_selector.select("public", owner="") is not None

    def test_admin_only_nodes_absent_from_public_model_catalog(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))

        response = client.get("/v1/models")

        assert response.status_code == 200
        assert [model["id"] for model in response.json()["data"]] == ["public"]

    def test_admin_scope_dependency_used_by_selector(
        self,
        app: FastAPI,
        node_selector: NodeSelector,
    ) -> None:
        assert app.dependency_overrides[get_node_selector] is not None

    def test_user_token_never_selects_admin_only_node(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """End-to-end: /v1 must forward the resolved token scope into node
        selection, so a non-admin bearer token can never reach an admin-only
        node even when the request names its model by id."""
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))
        user = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )
        token = auth_store.create_token(user.id, "ci-job").token

        response = client.post(
            "/v1/chat/completions",
            json={"model": "secret", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Never routed: an error body and no backend call for the admin-only model.
        assert response.status_code != 200
        assert len(httpx_mock.get_requests()) == 0

    def test_admin_role_token_selects_admin_only_node(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        user = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )
        auth_store.set_user_admin(user.id, True)
        token = auth_store.create_token(user.id, "ci-job").token
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            status_code=200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "secret",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

        response = client.post(
            "/v1/chat/completions",
            json={"model": "secret", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert len(httpx_mock.get_requests()) == 1

    def test_admin_role_pinned_token_stays_inside_pin(
        self,
        client: TestClient,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        httpx_mock: HTTPXMock,
    ) -> None:
        """Regression (third review): a stored pin must bind admin-role tokens
        too. create_token accepts and stores the pin, so /v1 selection has to
        honor it instead of silently routing the token to every healthy node
        (including other users' privately-owned nodes)."""
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))
        user = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )
        auth_store.set_user_admin(user.id, True)
        token = auth_store.create_token(
            user.id, "ci-job", endpoint_scope=["pub1"]
        ).token
        httpx_mock.add_response(
            url="http://10.0.1.100:8000/v1/chat/completions",
            status_code=200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "public",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

        # The admin-only node is not part of the pin: never routed.
        outside_pin = client.post(
            "/v1/chat/completions",
            json={"model": "secret", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert outside_pin.status_code != 200
        assert len(httpx_mock.get_requests()) == 0

        # The pinned node is routable, so the token still works as scoped.
        inside_pin = client.post(
            "/v1/chat/completions",
            json={"model": "public", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert inside_pin.status_code == 200
        assert len(httpx_mock.get_requests()) == 1


class TestFleetEndpoint:
    def test_fleet_node_endpoint_requires_session(self, app: FastAPI) -> None:
        assert TestClient(app).get("/fleet/nodes").status_code == 401

    def test_fleet_node_endpoint_hides_admin_only_nodes_and_actions(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))
        client, _user = _signed_in_client(app, auth_store)

        response = client.get("/fleet/nodes")

        assert response.status_code == 200
        nodes = response.json()
        assert [node["node_id"] for node in nodes] == ["pub1"]
        assert all(node["actions"] == [] for node in nodes)

    def test_admin_nodes_endpoint_requires_admin(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("pub1", model="public"))
        client, _user = _signed_in_client(app, auth_store)

        assert client.get("/admin/nodes").status_code == 401

    def test_admin_nodes_includes_admin_only_flag(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))

        response = client.get("/admin/nodes")

        assert response.status_code == 200
        nodes = {node["node_id"]: node for node in response.json()}
        assert nodes["admin_only1"]["admin_only"] is True
        assert nodes["pub1"]["admin_only"] is False

    def test_fleet_node_endpoint_strips_owner_email(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        """The non-admin fleet view must not disclose who owns a node (owner
        email is private per RFE-107), while the admin view keeps it."""
        test_registry.add(_make_node("pub1", model="public", owner="alice@example.com"))
        client, _user = _signed_in_client(app, auth_store)

        response = client.get("/fleet/nodes")
        assert response.status_code == 200
        fleet = {node["node_id"]: node for node in response.json()}
        assert fleet["pub1"]["owner"] == ""

        admin = TestClient(app).get("/admin/nodes")
        assert admin.status_code == 401  # no admin identity on the bare client


class TestAdminRoles:
    def test_grant_and_revoke_admin_role(
        self,
        client: TestClient,
        auth_store: AuthStore,
    ) -> None:
        user = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )

        assert client.post(f"/admin/users/{user.id}/admin", json={}).status_code == 204
        promoted = auth_store.get_user(user.id)
        assert promoted is not None
        assert promoted.is_admin is True

        response = client.get("/admin/users")
        row = next(entry for entry in response.json() if entry["id"] == user.id)
        assert row["is_admin"] is True

        assert client.delete(f"/admin/users/{user.id}/admin").status_code == 204
        demoted = auth_store.get_user(user.id)
        assert demoted is not None
        assert demoted.is_admin is False
        row = next(
            entry
            for entry in client.get("/admin/users").json()
            if entry["id"] == user.id
        )
        assert row["is_admin"] is False

    def test_grant_admin_unknown_user_404(self, client: TestClient) -> None:
        assert client.post("/admin/users/999999/admin", json={}).status_code == 404
        assert client.delete("/admin/users/999999/admin").status_code == 404

    def test_admin_session_granted_admin_api_access(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, user = _signed_in_client(app, auth_store)

        assert client.get("/admin/metrics").status_code == 401

        auth_store.set_user_admin(user.id, True)

        assert client.get("/admin/metrics").status_code == 200

    def test_demoted_user_loses_admin_access_on_next_request(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        """The documented security invariant: a demoted user loses admin
        access on their next request (role is re-read from the store, never
        cached in the session)."""
        client, user = _signed_in_client(app, auth_store)

        auth_store.set_user_admin(user.id, True)
        assert client.get("/admin/metrics").status_code == 200
        assert 'href="/dashboard/admin"' in client.get("/dashboard").text

        auth_store.set_user_admin(user.id, False)
        assert client.get("/admin/metrics").status_code == 401
        assert 'href="/dashboard/admin"' not in client.get("/dashboard").text


class TestSessionIsolation:
    """A signed-in Google user is never elevated by incidental Basic creds."""

    def test_oauth_session_not_elevated_by_cached_basic(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        import base64

        client, _user = _signed_in_client(app, auth_store)
        basic = base64.b64encode(b"test-admin:test-password").decode()

        response = client.get("/dashboard", headers={"Authorization": f"Basic {basic}"})

        assert response.status_code == 200
        assert 'href="/dashboard/admin"' not in response.text
        assert ">Logout</button>" in response.text

    def test_admin_api_not_elevated_by_cached_basic(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        """Signed-in non-admin + cached Basic => 401 on admin JSON endpoints."""
        import base64

        client, _user = _signed_in_client(app, auth_store)
        basic = base64.b64encode(b"test-admin:test-password").decode()

        response = client.get(
            "/admin/nodes", headers={"Authorization": f"Basic {basic}"}
        )

        assert response.status_code == 401

    def test_navbar_shows_identity(self, app: FastAPI, auth_store: AuthStore) -> None:
        import base64

        client, user = _signed_in_client(app, auth_store)

        signed_in = client.get("/dashboard")
        assert user.email in signed_in.text
        assert "Local Admin" not in signed_in.text

        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )
        local = client.get("/dashboard")
        assert "Local Admin" in local.text

        anonymous_basic = TestClient(app).get(
            "/dashboard",
            headers={
                "Authorization": "Basic "
                + base64.b64encode(b"test-admin:test-password").decode()
            },
        )
        assert "Local Admin" in anonymous_basic.text

    def test_oauth_switch_drops_local_admin_privileges(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client = TestClient(app)
        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )
        assert client.get("/admin/metrics").status_code == 200

        # Same browser now signs in via OAuth as a non-admin Google user.
        app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
        response = client.get(
            "/auth/callback?code=code&state=state", follow_redirects=False
        )
        assert response.status_code == 302

        assert client.get("/admin/metrics").status_code == 401
        dashboard = client.get("/dashboard")
        assert 'href="/dashboard/admin"' not in dashboard.text
        assert ">Logout</button>" in dashboard.text


class TestDashboardRoles:
    def test_anonymous_gets_signin_page(
        self,
        app: FastAPI,
        test_settings: Settings,
    ) -> None:
        # With the OAuth integration configured both sign-in options render;
        # with OAuth off the Google button is hidden (see the Basic-only
        # regression in test_admin_auth.py).
        enabled = test_settings.model_copy(
            deep=True,
            update={
                "oauth": test_settings.oauth.model_copy(
                    update={
                        "client_id": "test-client",
                        "client_secret": SecretStr("test-secret"),
                        "redirect_uri": "https://gateway.example.com/auth/callback",
                    }
                )
            },
        )
        app.dependency_overrides[get_settings] = lambda: enabled

        response = TestClient(app).get("/dashboard")

        assert response.status_code == 200
        assert "Sign in with Local Admin" in response.text
        assert "Sign in with Google Auth" in response.text

    def test_user_sees_fleet_but_not_admin_pages(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, _user = _signed_in_client(app, auth_store)

        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200
        assert "Node Fleet" in dashboard.text
        assert 'href="/dashboard/admin"' not in dashboard.text

        tokens = client.get("/dashboard/tokens")
        assert tokens.status_code == 200
        assert "Sign in with Local Admin" in tokens.text
        assert "Administrator access required" in tokens.text

    def test_admin_role_sees_admin_pages(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, user = _signed_in_client(app, auth_store)
        auth_store.set_user_admin(user.id, True)

        tokens = client.get("/dashboard/tokens")
        assert tokens.status_code == 200
        assert "QIIP - Token Management" in tokens.text

        admin_page = client.get("/dashboard/admin")
        assert admin_page.status_code == 200
        assert "Admin-only Inference Servers" in admin_page.text
        # The admin page carries the users table with grant/revoke actions
        # (restored: role management lives here AND on the token dashboard).
        assert 'id="admin-user-body"' in admin_page.text
        assert '<th scope="col">Admin</th>' in admin_page.text
        assert '<th scope="col">Admin</th>' in tokens.text

    def test_logout_visible_for_local_admin_session(
        self,
        app: FastAPI,
    ) -> None:
        client = TestClient(app)
        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )

        response = client.get("/dashboard")
        assert response.status_code == 200
        assert ">Logout</button>" in response.text

    def test_logout_visible_for_oauth_user_any_role(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, _user = _signed_in_client(app, auth_store)
        # Non-admin Google user (role does not matter for the Logout control).
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert ">Logout</button>" in response.text

    def test_logout_clears_local_admin_and_returns_to_signin(
        self,
        app: FastAPI,
    ) -> None:
        client = TestClient(app)
        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )
        assert ">Logout</button>" in client.get("/dashboard").text

        response = client.post("/auth/logout", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/dashboard"

        dashboard = client.get("/dashboard")
        assert "Sign in with Local Admin" in dashboard.text
        assert ">Logout</button>" not in dashboard.text

    def test_local_admin_form_login(self, app: FastAPI) -> None:
        """The sign-in form signs in via a session cookie, no Basic popup."""
        client = TestClient(app)

        # Direct GET: no challenge popup, just a redirect to the sign-in page.
        direct = client.get("/auth/local-admin", follow_redirects=False)
        assert direct.status_code == 302
        assert direct.headers["location"] == "/dashboard"

        # Wrong credentials: 401 with an explicit error message.
        bad = client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "wrong"},
        )
        assert bad.status_code == 401
        assert bad.json()["detail"] == "Invalid username or password"

        # Correct credentials: session cookie established; dashboard reachable.
        ok = client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
            follow_redirects=False,
        )
        assert ok.status_code == 302
        assert ok.headers["location"] == "/dashboard"

        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200
        assert "Node Fleet" in dashboard.text

    def test_local_admin_session_grants_admin_api(
        self,
        app: FastAPI,
    ) -> None:
        """A form-signed-in local admin can use the admin JSON API by session."""
        client = TestClient(app)
        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )

        assert client.get("/admin/metrics").status_code == 200


class TestFleetReadOnlySurface:
    """Read-only per-node detail surface for signed-in non-admins."""

    def test_fleet_node_detail_returns_visible_node(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("pub1", model="public"))
        test_registry.add(_make_node("mine1", model="mine", owner="alice@example.com"))
        client, _user = _signed_in_client(app, auth_store)

        for node_id in ("pub1", "mine1"):
            response = client.get(f"/fleet/nodes/{node_id}")
            assert response.status_code == 200
            assert response.json()["node_id"] == node_id
            assert response.json()["actions"] == []

    def test_fleet_node_detail_hides_invisible_nodes(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        """The read-only detail obeys the same visibility contract as the
        fleet list: admin-only, other users' owned, and absent nodes 404."""
        test_registry.add(_make_node("admin_only1", admin_only=True, model="secret"))
        test_registry.add(_make_node("private1", model="gpt", owner="bob@example.com"))
        client, _user = _signed_in_client(app, auth_store)

        assert client.get("/fleet/nodes/admin_only1").status_code == 404
        assert client.get("/fleet/nodes/private1").status_code == 404
        assert client.get("/fleet/nodes/ghost").status_code == 404

    def test_fleet_node_tasks_and_logs_are_scoped_to_visible_nodes(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node("pub1", model="public"))
        test_registry.add(_make_node("private1", model="gpt", owner="bob@example.com"))
        mock_provisioner.list_tasks_raw.return_value = [
            (
                json.dumps(
                    {
                        "hostname": "pub1",
                        "current_step": "installing",
                        "started_at": "2026-09-15T12:00:00Z",
                        "updated_at": "2026-09-15T12:00:01Z",
                    }
                ).encode(),
                None,
            ),
        ]
        buffer = ProvisioningLogBuffer()
        buffer.create("pub1")
        buffer.append("pub1", "info", "driver installed")
        buffer.mark_complete("pub1")
        mock_provisioner.log_buffer = buffer
        client, _user = _signed_in_client(app, auth_store)

        tasks = client.get("/fleet/nodes/pub1/tasks")
        assert tasks.status_code == 200
        assert [task["hostname"] for task in tasks.json()] == ["pub1"]
        logs = client.get("/fleet/nodes/pub1/logs")
        assert logs.status_code == 200
        assert logs.headers["content-type"].startswith("text/event-stream")

        # Invisible nodes get nothing at all on the read-only surface.
        assert client.get("/fleet/nodes/private1/tasks").status_code == 404
        assert client.get("/fleet/nodes/private1/logs").status_code == 404
        assert client.get("/fleet/nodes/private1").status_code == 404

    def test_fleet_tasks_and_logs_redact_credentials(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        """Regression (sjug review): failed engine-start commands embed
        HF_TOKEN in error/log text; the read-only surface must redact it for
        non-admin viewers, including retained buffered entries."""
        test_registry.add(_make_node("pub1", model="public"))
        secret = "hf_0123456789abcdef"
        mock_provisioner.list_tasks_raw.return_value = [
            (
                json.dumps(
                    {
                        "hostname": "pub1",
                        "current_step": "engine-start",
                        "failed_step": "engine-start",
                        "error": f"command failed: export HF_TOKEN={secret}",
                        "started_at": "2026-09-15T12:00:00Z",
                        "updated_at": "2026-09-15T12:00:01Z",
                    }
                ).encode(),
                None,
            ),
        ]
        buffer = ProvisioningLogBuffer()
        buffer.create("pub1")
        buffer.append("pub1", "error", f"HF_TOKEN={secret} launch failed")
        buffer.mark_complete("pub1")
        mock_provisioner.log_buffer = buffer
        client, _user = _signed_in_client(app, auth_store)

        tasks = client.get("/fleet/nodes/pub1/tasks")
        assert tasks.status_code == 200
        assert secret not in tasks.text
        assert "HF_TOKEN=[REDACTED]" in tasks.text

        logs = client.get("/fleet/nodes/pub1/logs")
        assert logs.status_code == 200
        log_text = logs.text
        assert secret not in log_text
        assert "HF_TOKEN=[REDACTED]" in log_text


class TestLocalAdminLoginJSONOnly:
    """The JSON-only CSRF boundary and body validation on /auth/local-admin."""

    def test_text_plain_body_rejected(self, app: FastAPI) -> None:
        client = TestClient(app)
        response = client.post(
            "/auth/local-admin",
            content='{"username": "test-admin", "password": "test-password"}',
            headers={"Content-Type": "text/plain"},
        )

        assert response.status_code == 415
        assert response.json()["detail"] == (
            "Login requires Content-Type: application/json"
        )

    def test_non_object_json_rejected(self, app: FastAPI) -> None:
        client = TestClient(app)
        for body in ("[]", "null", '"x"'):
            response = client.post(
                "/auth/local-admin",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422


class TestNodeDetailReadOnly:
    def test_non_admin_gets_read_only_detail_page(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, _user = _signed_in_client(app, auth_store)

        response = client.get("/dashboard/nodes/gpu01")

        assert response.status_code == 200
        assert "READ_ONLY = true" in response.text
        # In-page sign-in must never be shown for a still-signed-in user.
        assert "Sign in with Local Admin" not in response.text

    def test_admin_gets_operational_detail_page(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        client, user = _signed_in_client(app, auth_store)
        auth_store.set_user_admin(user.id, True)

        response = client.get("/dashboard/nodes/gpu01")

        assert response.status_code == 200
        assert "READ_ONLY = false" in response.text

    def test_anonymous_gets_signin_page(
        self,
        app: FastAPI,
    ) -> None:
        response = TestClient(app).get("/dashboard/nodes/gpu01")

        assert response.status_code == 200
        assert "Sign in with Local Admin" in response.text


class TestSelfRevocation:
    def test_self_revoke_marks_response_and_keeps_session(
        self,
        app: FastAPI,
        auth_store: AuthStore,
    ) -> None:
        """Revoking your own admin role must never pop the native Basic
        dialog: the session stays valid, the dashboard keeps serving the
        trimmed fleet view, and admin pages fall back to the in-page sign-in.
        """
        client, user = _signed_in_client(app, auth_store)
        auth_store.set_user_admin(user.id, True)
        assert client.get("/dashboard/tokens").status_code == 200

        response = client.delete(f"/admin/users/{user.id}/admin")

        assert response.status_code == 204
        assert response.headers["x-qiip-self-revoked"] == "true"
        assert "www-authenticate" not in response.headers
        # Session survives; admin pages now render the in-page sign-in.
        assert client.get("/dashboard").status_code == 200
        tokens_page = client.get("/dashboard/tokens")
        assert tokens_page.status_code == 200
        assert "Sign in with Local Admin" in tokens_page.text
        assert "www-authenticate" not in tokens_page.headers

    def test_revoking_other_user_does_not_mark_self(
        self,
        client: TestClient,
        auth_store: AuthStore,
    ) -> None:
        user = auth_store.upsert_google_user(
            google_sub="sub-bob",
            email="bob@example.com",
            name="Bob",
            picture="",
        )

        response = client.delete(f"/admin/users/{user.id}/admin")

        assert response.status_code == 204
        assert "x-qiip-self-revoked" not in response.headers
