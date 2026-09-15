"""Dependency injection providers for application configuration and services.

Settings are provided via ``@lru_cache`` so the same instance is reused
across requests.  The node registry and proxy client are stored in
``app.state`` during lifespan and exposed via ``get_registry()`` and
``get_proxy_client()`` -- per-request dependencies that read from the
current application instance.

In tests, use ``app.dependency_overrides[get_settings]``,
``app.dependency_overrides[get_registry]``, or
``app.dependency_overrides[get_proxy_client]`` to inject test-specific
instances.
"""

import base64
import binascii
from functools import lru_cache
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from inference_proxy.auth.session import get_local_admin_session, get_session_user_id
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.catalog import ModelCatalogService
from inference_proxy.huggingface.downloader import DownloadService
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.proxy.client import ProxyClient
from inference_proxy.quads.client import QUADSClient
from inference_proxy.quads.poller import QUADSPoller
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.node_selector import NodeSelector
from inference_proxy.routing.request_metrics import RequestMetrics
from inference_proxy.services.unified_nodes import UnifiedNodeService

from .settings import Settings

_JSON_ADMIN_METHODS = frozenset({"POST", "PUT", "PATCH"})
_ADMIN_AUTH_HEADERS = {
    "WWW-Authenticate": 'Basic realm="inference-proxy-admin", charset="UTF-8"'
}


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


def _credentials_match(
    username: str,
    password: str,
    settings: Settings,
) -> bool:
    """Return True when the credentials match the configured local admin."""
    username_matches = compare_digest(
        username.encode("utf-8"),
        settings.admin.username.encode("utf-8"),
    )
    password_matches = compare_digest(
        password.encode("utf-8"),
        settings.admin.password.get_secret_value().encode("utf-8"),
    )
    return username_matches and password_matches


def _basic_credentials_match(request: Request, settings: Settings) -> bool:
    """Return True when the request carries valid local-admin Basic credentials.

    Parses the ``Authorization`` header directly (no FastAPI HTTPBasic
    dependency) so page handlers and template helpers can check the local
    admin identity without declaring a credentials parameter.
    """
    header = request.headers.get("authorization", "")
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded.strip():
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return _credentials_match(username, password, settings)


def _viewer_from_session(request: Request) -> str | None:
    """Resolve the signed-in viewer role from the session cookie.

    ``"admin"`` when the signed-in user carries the admin role, ``"user"``
    for any other signed-in user, ``None`` when no valid session exists.
    The identity is re-read from the SQLite store so a stale cookie can
    never resurrect a deleted user or a demoted admin.
    """
    user_id = get_session_user_id(request)
    if user_id is None:
        return None
    store = getattr(request.app.state, "auth_store", None)
    if store is None:
        return None
    user = store.get_user(user_id)
    if user is None:
        return None
    return "admin" if user.is_admin else "user"


def viewer_role(request: Request, settings: Settings) -> str | None:
    """Return the viewer role: ``"admin"``, ``"user"``, or ``None``.

    A signed-in Google user is authoritative: their role is governed by the
    admin role only, never elevated by incidentally cached HTTP Basic admin
    credentials in the same browser. Without a Google session, the local
    admin is recognized by HTTP Basic credentials or a signed local-admin
    session (from the sign-in page form).
    """
    user_id = get_session_user_id(request)
    if user_id is not None:
        _role = _viewer_from_session(request)
    elif _basic_credentials_match(request, settings) or get_local_admin_session(
        request
    ):
        _role = "admin"
    else:
        _role = None
    return _role


def require_fleet_viewer(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """Require an authenticated fleet viewer (local admin or signed-in user)."""
    role = viewer_role(request, settings)
    if role is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers=_ADMIN_AUTH_HEADERS,
        )
    return role


def require_fleet_viewer_email(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """Return the signed-in Google user's lowercase email for fleet filtering.

    Fleet privacy (RFE-107): nodes owned by someone else are excluded from
    the non-admin fleet view, so the endpoint/model/GPU identity of another
    user's node is never disclosed. Local-admin/HTTP Basic viewers have no
    Google identity and get None (the node filter then keeps the current
    non-admin_only set). Raises 401 for anonymous callers, same gate as
    ``require_fleet_viewer``.
    """
    require_fleet_viewer(request, settings)
    user_id = get_session_user_id(request)
    if user_id is None:
        return None
    store = getattr(request.app.state, "auth_store", None)
    if store is None:
        return None
    user = store.get_user(user_id)
    return user.email.lower() if user is not None else None


def require_admin_auth(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Authenticate admin requests and enforce the JSON-only CSRF boundary.

    Accepted identities: the HTTP Basic local admin, a signed local-admin
    session, or a signed-in Google user carrying the admin role -- the same
    chain ``viewer_role`` resolves, so the HTML and JSON surfaces can never
    disagree about who is admin and both parse Basic credentials identically.

    A signed-in Google session is authoritative: same-origin browser
    requests replay cached HTTP Basic credentials automatically, so a
    non-admin session is never elevated by incidentally cached admin
    credentials.
    """
    if viewer_role(request, settings) != "admin":
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
            headers=_ADMIN_AUTH_HEADERS,
        )

    if request.method in _JSON_ADMIN_METHODS:
        media_type = request.headers.get("content-type", "").partition(";")[0].lower()
        if media_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Admin state-changing requests must use application/json",
            )


def get_registry(request: Request) -> NodeRegistry:
    """Return the node registry from the current application state.

    The registry is created during lifespan startup and stored in
    ``app.state.registry`` (per D-07).  This dependency makes it
    available to FastAPI route handlers via ``Depends(get_registry)``.
    """
    return request.app.state.registry  # type: ignore[no-any-return]


def get_proxy_client(request: Request) -> ProxyClient:
    """Return the proxy client from the current application state.

    The proxy client is created during lifespan startup and stored in
    ``app.state.proxy_client``.  This dependency makes it available to
    FastAPI route handlers via ``Depends(get_proxy_client)``.
    """
    return request.app.state.proxy_client  # type: ignore[no-any-return]


def get_circuit_breaker_registry(request: Request) -> CircuitBreakerRegistry:
    """Return the circuit breaker registry from the current application state.

    The registry is created during lifespan startup and stored in
    ``app.state.circuit_breaker_registry``.  This dependency makes it
    available to FastAPI route handlers via
    ``Depends(get_circuit_breaker_registry)``.
    """
    return request.app.state.circuit_breaker_registry  # type: ignore[no-any-return]


def get_request_metrics(request: Request) -> RequestMetrics:
    """Return the request metrics from the current application state.

    The metrics instance is created during lifespan startup and stored in
    ``app.state.request_metrics``.  This dependency makes it available to
    FastAPI route handlers via ``Depends(get_request_metrics)``.
    """
    return request.app.state.request_metrics  # type: ignore[no-any-return]


def get_node_selector(request: Request) -> NodeSelector:
    """Return the node selector from the current application state.

    The node selector is created during lifespan startup and stored in
    ``app.state.node_selector``.  This dependency makes it available to
    FastAPI route handlers via ``Depends(get_node_selector)``.
    """
    return request.app.state.node_selector  # type: ignore[no-any-return]


def get_provisioner(request: Request) -> NodeProvisioner:
    """Return the node provisioner from the current application state."""
    return request.app.state.provisioner  # type: ignore[no-any-return]


def get_quads_client(request: Request) -> QUADSClient | None:
    """Return the QUADS client, or None when QUADS is not configured (D-10)."""
    return request.app.state.quads_client  # type: ignore[no-any-return]


def get_catalog_service(request: Request) -> ModelCatalogService:
    """Return the model catalog service from the current application state."""
    return request.app.state.catalog_service  # type: ignore[no-any-return]


def get_download_service(request: Request) -> DownloadService:
    """Return the download service from the current application state."""
    return request.app.state.download_service  # type: ignore[no-any-return]


def get_llmfit_runner(request: Request) -> LLMFitRunner:
    """Return the LLMFit runner from the current application state."""
    return request.app.state.llmfit_runner  # type: ignore[no-any-return]


def get_redfish_client(request: Request) -> RedfishClient | None:
    """Return the Redfish client, or None when Redfish is not configured."""
    return request.app.state.redfish_client  # type: ignore[no-any-return]


def get_quads_poller(request: Request) -> QUADSPoller | None:
    """Return the QUADS poller, or None when QUADS is not configured.

    Phase 17 consumes this to merge QUADS hosts with etcd nodes.
    """
    return request.app.state.quads_poller  # type: ignore[no-any-return]


def get_unified_node_service(request: Request) -> UnifiedNodeService:
    """Build UnifiedNodeService from app.state components."""
    return UnifiedNodeService(
        registry=request.app.state.registry,
        poller=request.app.state.quads_poller,
        cb_registry=request.app.state.circuit_breaker_registry,
        tracker=request.app.state.node_selector.tracker,
    )
