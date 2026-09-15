"""Authentication and trust-boundary tests for the administrative surface."""

from __future__ import annotations

import asyncio
import base64
import secrets
from typing import Any
from unittest.mock import ANY, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import SecretStr

import inference_proxy.api.admin as admin_module
import inference_proxy.config.dependencies as dependencies
from inference_proxy.auth.dependencies import get_auth_store
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_request_metrics,
    get_settings,
    get_unified_node_service,
)
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.main import create_app
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics
from inference_proxy.services.unified_nodes import UnifiedNodeService


def _basic_header(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    **kwargs: Any,
) -> httpx.Response:
    """Drive ASGI directly with a test-owned deadline."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await asyncio.wait_for(
            client.request(method, path, **kwargs),
            timeout=2,
        )


class TestBasicOnlyDeployment:
    """Regression (second review, blocking): with ``auth.session_secret`` unset
    the session middleware is absent and Starlette's ``Request.session`` raises
    ``AssertionError``; every page/route that reads it must keep working for
    HTTP Basic deployments instead of 500ing."""

    @staticmethod
    def _app_without_session_secret(
        test_settings: Settings,
        auth_store: AuthStore,
    ) -> FastAPI:
        settings = test_settings.model_copy(
            deep=True,
            update={
                "auth": test_settings.auth.model_copy(update={"session_secret": None})
            },
        )
        application = create_app(settings=settings)
        application.state.auth_store = auth_store
        application.dependency_overrides[get_auth_store] = lambda: auth_store
        application.state.request_metrics = RequestMetrics()
        application.dependency_overrides[get_request_metrics] = lambda: (
            application.state.request_metrics
        )
        application.state.registry = NodeRegistry()
        application.state.node_selector = NodeSelector(
            application.state.registry, ConnectionTracker()
        )
        application.state.circuit_breaker_registry = CircuitBreakerRegistry()
        application.dependency_overrides[get_unified_node_service] = lambda: (
            UnifiedNodeService(
                registry=application.state.registry,
                poller=None,
                cb_registry=application.state.circuit_breaker_registry,
                tracker=application.state.node_selector.tracker,
            )
        )
        return application

    async def test_dashboard_renders_without_session_secret(
        self,
        test_settings: Settings,
        auth_store: AuthStore,
    ) -> None:
        app = self._app_without_session_secret(test_settings, auth_store)

        response = await _request(app, "GET", "/dashboard")

        assert response.status_code == 200
        # Neither browser option can work here (no session_secret, OAuth off):
        # both must be gated off and the HTTP-Basic hint shown instead.
        assert "Sign in with Local Admin" not in response.text
        assert "Sign in with Google Auth" not in response.text
        assert 'class="signin-form"' not in response.text
        assert "Browser sign-in is not configured" in response.text

    async def test_admin_api_works_with_valid_basic_without_session_secret(
        self,
        test_settings: Settings,
        auth_store: AuthStore,
    ) -> None:
        app = self._app_without_session_secret(test_settings, auth_store)

        response = await _request(
            app,
            "GET",
            "/admin/metrics",
            headers=_basic_header("test-admin", "test-password"),
        )

        assert response.status_code == 200

    async def test_fleet_nodes_works_with_valid_basic_without_session_secret(
        self,
        test_settings: Settings,
        auth_store: AuthStore,
    ) -> None:
        """The fleet route runs the distinct chain require_fleet_viewer ->
        viewer_role -> get_session_user_id + require_fleet_viewer_email
        (Basic: no session user, ownership filter stays off) -- the exact
        AssertionError path this guard fixes."""
        app = self._app_without_session_secret(test_settings, auth_store)

        response = await _request(
            app,
            "GET",
            "/fleet/nodes",
            headers=_basic_header("test-admin", "test-password"),
        )

        assert response.status_code == 200
        assert response.json() == []


class TestAdminBasicAuthentication:
    @pytest.mark.parametrize(
        ("headers", "expected_status"),
        [
            ({}, 401),
            (_basic_header("wrong-user", "test-password"), 401),
            (_basic_header("test-admin", "wrong-password"), 401),
            (_basic_header("test-admin", "test-password"), 200),
        ],
    )
    async def test_admin_basic_auth_matrix(
        self,
        app: FastAPI,
        headers: dict[str, str],
        expected_status: int,
    ) -> None:
        response = await _request(app, "GET", "/admin/metrics", headers=headers)

        assert response.status_code == expected_status
        if expected_status == 401:
            assert response.headers["www-authenticate"].startswith("Basic ")

    async def test_both_credentials_are_compared_for_wrong_username(
        self,
        app: FastAPI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[bytes, bytes]] = []

        def capture_compare(left: bytes, right: bytes) -> bool:
            calls.append((left, right))
            return secrets.compare_digest(left, right)

        monkeypatch.setattr(
            dependencies,
            "compare_digest",
            capture_compare,
            raising=False,
        )

        response = await _request(
            app,
            "GET",
            "/admin/metrics",
            headers=_basic_header("wrong-user", "wrong-password"),
        )

        assert response.status_code == 401
        assert len(calls) == 2
        assert calls[0][0] == b"wrong-user"
        assert calls[1][0] == b"wrong-password"

    @pytest.mark.parametrize(
        "headers",
        [{}, _basic_header("test-admin", "wrong-password")],
    )
    async def test_setup_requires_admin_auth_before_side_effects(
        self,
        app: FastAPI,
        mock_provisioner: MagicMock,
        headers: dict[str, str],
    ) -> None:
        response = await _request(
            app,
            "POST",
            "/admin/nodes/setup",
            headers=headers,
            json={"hostname": "gpu01"},
        )

        if response.status_code != 401 and mock_provisioner.fire_background.called:
            background = mock_provisioner.fire_background.call_args.args[0]
            background.close()
            admin_module.pending_hosts.clear()
        assert response.status_code == 401
        mock_provisioner.validate_endpoint.assert_not_called()
        mock_provisioner.try_reserve_host.assert_not_awaited()
        mock_provisioner.cleanup_stale_node.assert_not_awaited()
        mock_provisioner.provision.assert_not_awaited()
        mock_provisioner.fire_background.assert_not_called()

    def test_authenticated_standalone_setup_preserves_existing_policy(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        response = client.post(
            "/admin/nodes/setup",
            json={"hostname": "gpu01", "managed": False},
        )

        assert response.status_code == 202
        mock_provisioner.validate_endpoint.assert_called_once_with("gpu01")
        background = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(background, timeout=1))
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=False,
            model=None,
            engine=ANY,
            artifact_id=None,
            vllm_params=None,
            lifecycle_lease=ANY,
            owner="",
        )

    def test_every_admin_route_declares_auth_dependency(self, app: FastAPI) -> None:
        # FastAPI 0.134 keeps included routers lazy, so inspect each included
        # router's source routes as well as routes mounted directly on the app.
        routes: list[object] = []
        for route in app.routes:
            original_router = getattr(route, "original_router", None)
            routes.extend(
                original_router.routes if original_router is not None else [route]
            )
        admin_routes = [
            route
            for route in routes
            if isinstance(route, APIRoute) and route.path.startswith("/admin")
        ]

        assert admin_routes
        required_auth = getattr(dependencies, "require_admin_auth", None)
        assert required_auth is not None
        assert all(
            any(
                dependency.call is required_auth
                for dependency in route.dependant.dependencies
            )
            for route in admin_routes
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/dashboard",
            "/dashboard/nodes/gpu01",
            "/dashboard/tokens",
            "/dashboard/users/1",
        ],
    )
    async def test_dashboard_routes_require_signin(
        self,
        app: FastAPI,
        test_settings: Settings,
        path: str,
    ) -> None:
        """Anonymous visitors get the two-option sign-in page; Basic admins pass.

        OAuth must be configured for the Google option to render (it is gated
        off with the default test settings; see TestBasicOnlyDeployment).
        """
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
        unauthenticated = await _request(app, "GET", path)
        authenticated = await _request(
            app,
            "GET",
            path,
            headers=_basic_header("test-admin", "test-password"),
        )

        assert unauthenticated.status_code == 200
        assert "Sign in with Local Admin" in unauthenticated.text
        assert "Sign in with Google Auth" in unauthenticated.text
        assert authenticated.status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/v1/models", "/chat"])
    async def test_non_admin_routes_remain_public(
        self,
        app: FastAPI,
        path: str,
    ) -> None:
        response = await _request(app, "GET", path)

        assert response.status_code == 200

    async def test_eventsource_style_log_request_uses_browser_basic_session(
        self,
        app: FastAPI,
        mock_provisioner: MagicMock,
    ) -> None:
        buffer = ProvisioningLogBuffer()
        buffer.create("gpu01")
        buffer.append("gpu01", "info", "driver installed")
        buffer.mark_complete("gpu01")
        mock_provisioner.log_buffer = buffer

        # EventSource cannot add a Bearer header. A browser Basic session sends
        # its cached Authorization header on this ordinary GET automatically.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            auth=httpx.BasicAuth("test-admin", "test-password"),
        ) as browser_session:
            response = await asyncio.wait_for(
                browser_session.get("/admin/provisioning/gpu01/logs"),
                timeout=2,
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.startswith("data: ")
        assert '"msg": "driver installed"' in response.text

    async def test_state_changing_admin_routes_require_json_before_side_effects(
        self,
        app: FastAPI,
        mock_provisioner: MagicMock,
    ) -> None:
        response = await _request(
            app,
            "POST",
            "/admin/nodes/setup",
            headers={
                **_basic_header("test-admin", "test-password"),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            content="hostname=gpu01",
        )

        assert response.status_code == 415
        assert response.json()["detail"] == (
            "Admin state-changing requests must use application/json"
        )
        mock_provisioner.validate_endpoint.assert_not_called()
        mock_provisioner.try_reserve_host.assert_not_awaited()
        mock_provisioner.fire_background.assert_not_called()
