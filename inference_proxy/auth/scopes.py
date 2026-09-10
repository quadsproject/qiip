"""Endpoint-scope computation for API tokens (RFE #107).

Pure helpers that turn a resolved ``TokenAuth`` plus settings into the
selection filters the router passes to ``NodeSelector``. Kept out of
``get_api_auth`` so authentication and authorization stay separate and
these functions stay trivially testable.
"""

from __future__ import annotations

from collections.abc import Iterable

from inference_proxy.auth.models import TokenAuth
from inference_proxy.config.settings import Settings
from inference_proxy.models.node import Node


def is_full_access(email: str, settings: Settings) -> bool:
    """Return True when *email* is on the admin full-access trust list."""
    listed = settings.auth.admin_only_tokens_full_access
    if not listed:
        return False
    normalized = email.lower()
    return any(item.lower() == normalized for item in listed)


def allowed_node_ids(
    auth: TokenAuth | None,
    settings: Settings,
) -> frozenset[str] | None:
    """Return the hostnames a token may route to (None = no pin).

    Anonymous requests and admin full-access tokens are unpinned; a
    token with an endpoint scope is pinned to its stored hostnames.
    """
    if auth is None or is_full_access(auth.user.email, settings):
        return None
    scope = auth.token.endpoint_scope
    return frozenset(scope) if scope else None


def scope_owner(auth: TokenAuth | None, settings: Settings) -> str | None:
    """Return the owner gate for selection.

    ``None`` = admin (no owner filter), ``""`` = anonymous (unowned
    nodes only), otherwise the user email (unowned + own nodes).
    """
    if auth is None:
        return ""
    if is_full_access(auth.user.email, settings):
        return None
    return auth.user.email


def pickable_endpoints(
    user_email: str,
    settings: Settings,
    nodes: Iterable[Node],
) -> list[str]:
    """Return hostnames *user_email* may pin a token to, sorted.

    A user may pin unowned nodes and nodes they own; admins may pin
    anything.
    """
    admin = is_full_access(user_email, settings)
    pickable = []
    for node in nodes:
        if admin or not node.owner or node.owner == user_email:
            pickable.append(node.node_id)
    return sorted(pickable)
