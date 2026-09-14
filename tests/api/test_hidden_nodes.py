"""Hidden inference servers and admin-role access gates.

Regression coverage for:

- ``hidden`` registration through the pool adoption path
- hidden-node scope rules in the node selector (admin-only routing)
- hidden nodes excluded from the public ``/v1/models`` catalog
- the trimmed ``/fleet/nodes`` surface (hidden servers and actions stripped)
- admin-role grant/revoke API and session-based admin access
- per-role dashboard page gating
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.auth.models import User
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import get_node_selector
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.admin import RegisterRequest
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.routing.node_selector import NodeSelector
from tests.auth.conftest import FakeAuthPlugin


def _make_node(
    node_id: str,
    *,
    hidden: bool = False,
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
        hidden=hidden,
        owner=owner,
    )


def _signed_in_client(app: FastAPI, store: AuthStore) -> tuple[TestClient, User]:
    """Sign a Google user in through the real callback; return client + user."""
    app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state", follow_redirects=False
    )
    assert response.status_code == 302
    users = store.list_users()
    assert len(users) == 1
    return client, users[0]


class TestHiddenRegistration:
    def test_hidden_implies_self_setup(self) -> None:
        request = RegisterRequest(hostname="gpu01", hidden=True)

        assert request.self_setup is True

    def test_hidden_pool_registration_sets_flags(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.register_self_setup = AsyncMock(
            return_value=_make_node("gpu01", hidden=True, model="org/model")
        )
        mock_provisioner.validate_endpoint.return_value = "http://gpu01:8000"

        response = client.post(
            "/admin/nodes/pool", json={"hostname": "gpu01", "hidden": True}
        )

        assert response.status_code == 201
        assert response.json() == {
            "hostname": "gpu01",
            "state": "healthy",
            "model": "org/model",
            "self_setup": True,
            "hidden": True,
            "name": "",
        }
        mock_provisioner.register_self_setup.assert_awaited_once_with(
            "gpu01", None, owner="", hidden=True, name=""
        )


class TestHiddenRoutingScope:
    def test_hidden_nodes_require_admin_scope(
        self,
        test_registry: NodeRegistry,
        node_selector: NodeSelector,
    ) -> None:
        test_registry.add(_make_node("hidden1", hidden=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))

        # Admin scope (owner=None) reaches hidden nodes.
        assert node_selector.select("secret", owner=None) is not None
        # Anonymous (owner="") and user (email) scopes never do.
        assert node_selector.select("secret", owner="") is None
        assert node_selector.select("secret", owner="alice@example.com") is None
        # A non-hidden node stays reachable for anonymous scope.
        assert node_selector.select("public", owner="") is not None

    def test_hidden_nodes_absent_from_public_model_catalog(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("hidden1", hidden=True, model="secret"))
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


class TestFleetEndpoint:
    def test_fleet_node_endpoint_requires_session(self, app: FastAPI) -> None:
        assert TestClient(app).get("/fleet/nodes").status_code == 401

    def test_fleet_node_endpoint_hides_hidden_nodes_and_actions(
        self,
        app: FastAPI,
        auth_store: AuthStore,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("hidden1", hidden=True, model="secret"))
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

    def test_admin_nodes_includes_hidden_flag(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node("hidden1", hidden=True, model="secret"))
        test_registry.add(_make_node("pub1", model="public"))

        response = client.get("/admin/nodes")

        assert response.status_code == 200
        nodes = {node["node_id"]: node for node in response.json()}
        assert nodes["hidden1"]["hidden"] is True
        assert nodes["pub1"]["hidden"] is False


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
    def test_anonymous_gets_signin_page(self, app: FastAPI) -> None:
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
        assert "Hidden Inference Servers" in admin_page.text
        assert "Admin Users" in admin_page.text

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
