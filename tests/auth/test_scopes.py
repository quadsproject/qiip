"""Unit tests for endpoint-scope computation (RFE #107)."""

from __future__ import annotations

from datetime import UTC, datetime

from inference_proxy.auth.models import ApiToken, TokenAuth, User
from inference_proxy.auth.scopes import (
    allowed_node_ids,
    auth_scope,
    is_full_access,
    pickable_endpoints,
    scope_owner,
)
from inference_proxy.config.settings import AuthSettings, Settings
from inference_proxy.models.node import Node


def _settings(admins: list[str] | None = None) -> Settings:
    return Settings(auth=AuthSettings(admin_only_tokens_full_access=admins or []))


def _user(email: str = "alice@example.com") -> User:
    return User(
        id=1,
        google_sub="sub-1",
        email=email,
        name="Alice",
        picture="",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _auth(
    email: str = "alice@example.com", scopes: list[str] | None = None
) -> TokenAuth:
    return TokenAuth(
        user=_user(email),
        token=ApiToken(
            id=1,
            user_id=1,
            name="ci",
            prefix="qiip_abc",
            created_at=datetime.now(UTC),
            last_used_at=None,
            revoked=False,
            endpoint_scope=scopes,
        ),
    )


class TestIsFullAccess:
    def test_matches_case_insensitively(self) -> None:
        assert is_full_access("Ops@Example.com", _settings(["ops@example.com"]))

    def test_not_listed(self) -> None:
        assert not is_full_access("alice@example.com", _settings(["ops@example.com"]))

    def test_empty_list(self) -> None:
        assert not is_full_access("ops@example.com", _settings())


class TestAllowedNodeIds:
    def test_anonymous_is_unpinned(self) -> None:
        assert allowed_node_ids(None, _settings()) is None

    def test_admin_is_unpinned(self) -> None:
        auth = _auth("ops@example.com")
        assert allowed_node_ids(auth, _settings(["ops@example.com"])) is None

    def test_scoped_token_is_pinned(self) -> None:
        auth = _auth("ops@example.com", scopes=["h1", "h2"])
        assert allowed_node_ids(auth, _settings(["ops@example.com"])) is None
        assert allowed_node_ids(_auth(scopes=["h1", "h2"]), _settings()) == frozenset(
            {"h1", "h2"}
        )

    def test_unscoped_token_is_unpinned(self) -> None:
        assert allowed_node_ids(_auth(), _settings()) is None


class TestScopeOwner:
    def test_anonymous_restricts_to_unowned(self) -> None:
        assert scope_owner(None, _settings()) == ""

    def test_admin_bypasses_owner_filter(self) -> None:
        assert (
            scope_owner(_auth("ops@example.com"), _settings(["ops@example.com"]))
            is None
        )

    def test_user_gets_their_email(self) -> None:
        assert scope_owner(_auth(), _settings()) == "alice@example.com"


class TestPickableEndpoints:
    def _nodes(self) -> list[Node]:
        return [
            Node(node_id="shared-1", endpoint="10.0.0.1:8000"),
            Node(node_id="mine-1", endpoint="10.0.0.2:8000", owner="alice@example.com"),
            Node(node_id="theirs-1", endpoint="10.0.0.3:8000", owner="bob@example.com"),
        ]

    def test_user_pins_shared_and_own(self) -> None:
        got = pickable_endpoints("alice@example.com", _settings(), self._nodes())
        assert got == ["mine-1", "shared-1"]

    def test_admin_pins_everything(self) -> None:
        got = pickable_endpoints(
            "ops@example.com",
            _settings(["ops@example.com"]),
            self._nodes(),
        )
        assert got == ["mine-1", "shared-1", "theirs-1"]


class TestAuthScope:
    """Combined (allowed, owner) resolver -- one admin check per request."""

    def test_scoped_token_gets_pin_and_owner(self) -> None:
        allowed, owner = auth_scope(_auth(scopes=["gpu01"]), _settings())
        assert allowed == frozenset({"gpu01"})
        assert owner == "alice@example.com"

    def test_admin_gets_no_filters(self) -> None:
        allowed, owner = auth_scope(
            _auth(email="ops@example.com"), _settings(["ops@example.com"])
        )
        assert allowed is None
        assert owner is None

    def test_anonymous_gets_unowned_only(self) -> None:
        assert auth_scope(None, _settings()) == (None, "")

    def test_unscoped_token_gets_owner_only(self) -> None:
        allowed, owner = auth_scope(_auth(), _settings())
        assert allowed is None
        assert owner == "alice@example.com"

    def test_empty_scope_pins_to_nothing(self) -> None:
        allowed, owner = auth_scope(_auth(scopes=[]), _settings())
        assert allowed == frozenset()
        assert owner == "alice@example.com"


class TestEmptyScopeDistinctFromFull:
    """[] must mean pinned-to-nothing, never unrestricted (contra review)."""

    def test_allowed_node_ids_empty_scope_is_not_none(self) -> None:
        assert allowed_node_ids(_auth(scopes=[]), _settings()) == frozenset()

    def test_allowed_node_ids_none_scope_is_unrestricted(self) -> None:
        assert allowed_node_ids(_auth(), _settings()) is None

    def test_pickable_endpoints_matches_owner_caselessly(self) -> None:
        node = Node(
            node_id="gpu01",
            endpoint="http://gpu01:8000",
            owner="Alice@Example.com",
        )
        assert pickable_endpoints("alice@example.com", _settings(), [node]) == ["gpu01"]
