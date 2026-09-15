"""Behavioral tests for the SQLite auth store."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import TypedDict

import pytest

import inference_proxy.auth.store as store_module
from inference_proxy.auth.models import CreatedToken, User
from inference_proxy.auth.store import AuthStore, _utcnow

_GOOGLE = {
    "google_sub": "sub-123",
    "email": "alice@example.com",
    "name": "Alice",
    "picture": "https://example.com/alice.png",
}


class _AdminSeed(TypedDict):
    alice: User
    bob: User
    token_id: int
    token_raw: str


class TestUsers:
    def test_upsert_creates_user(self, auth_store: AuthStore) -> None:
        user = auth_store.upsert_google_user(**_GOOGLE)

        assert user.id == 1
        assert user.google_sub == _GOOGLE["google_sub"]
        assert user.email == _GOOGLE["email"]
        assert user.name == "Alice"
        assert user.created_at == user.updated_at

    def test_upsert_update_is_idempotent_by_sub(
        self,
        auth_store: AuthStore,
    ) -> None:
        first = auth_store.upsert_google_user(**_GOOGLE)
        second = auth_store.upsert_google_user(
            google_sub="sub-123",
            email="alice@example.com",
            name="Alice Smith",
            picture="https://example.com/alice-v2.png",
        )

        assert second.id == first.id
        assert second.name == "Alice Smith"
        assert second.picture == "https://example.com/alice-v2.png"
        assert auth_store._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1

    def test_upsert_rebinds_email_to_new_sub(
        self,
        auth_store: AuthStore,
    ) -> None:
        original = auth_store.upsert_google_user(**_GOOGLE)
        migrated = auth_store.upsert_google_user(
            google_sub="sub-456",
            email="alice@example.com",
            name="Alice",
            picture="",
        )

        assert migrated.id == original.id
        assert migrated.google_sub == "sub-456"
        assert auth_store._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1

    def test_get_user_missing_returns_none(self, auth_store: AuthStore) -> None:
        assert auth_store.get_user(999) is None

    def test_is_admin_defaults_false(self, auth_store: AuthStore) -> None:
        user = auth_store.upsert_google_user(**_GOOGLE)

        assert user.is_admin is False

    def test_set_user_admin_roundtrip(self, auth_store: AuthStore) -> None:
        user = auth_store.upsert_google_user(**_GOOGLE)

        assert auth_store.set_user_admin(user.id, True) is True
        promoted = auth_store.get_user(user.id)
        assert promoted is not None
        assert promoted.is_admin is True
        assert auth_store.set_user_admin(user.id, False) is True
        demoted = auth_store.get_user(user.id)
        assert demoted is not None
        assert demoted.is_admin is False

    def test_set_user_admin_unknown_user_is_false(
        self,
        auth_store: AuthStore,
    ) -> None:
        assert auth_store.set_user_admin(999999, True) is False

    def test_list_users_includes_all(self, auth_store: AuthStore) -> None:
        auth_store.upsert_google_user(**_GOOGLE)
        auth_store.upsert_google_user(
            google_sub="sub-bob",
            email="bob@example.com",
            name="Bob",
            picture="",
        )

        users = auth_store.list_users_with_stats()

        assert {user.email for user in users} == {
            "alice@example.com",
            "bob@example.com",
        }

    def test_pre_existing_db_migrates_is_admin(
        self,
        tmp_path: Path,
    ) -> None:
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub  TEXT    NOT NULL UNIQUE,
                email       TEXT    NOT NULL UNIQUE,
                name        TEXT    NOT NULL DEFAULT '',
                picture     TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            )
            """
        )
        now = _utcnow().isoformat()
        conn.execute(
            "INSERT INTO users (google_sub, email, name, picture, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("sub-legacy", "legacy@example.com", "", "", now, now),
        )
        conn.commit()
        conn.close()

        store = AuthStore(db_path)
        try:
            user = store.get_user(1)

            assert user is not None
            assert user.is_admin is False
            assert store.set_user_admin(user.id, True) is True
            promoted = store.get_user(user.id)
            assert promoted is not None
            assert promoted.is_admin is True
        finally:
            store.close()


class TestTokens:
    def _user(self, store: AuthStore) -> User:
        return store.upsert_google_user(**_GOOGLE)

    def test_create_token_returns_raw_secret_once(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        assert created.token.startswith("qiip_")
        assert created.name == "ci-job"
        assert created.user_id == user.id
        assert created.revoked is False
        assert created.prefix == created.token[: len("qiip_") + 8]

        # The raw secret is never stored: there is no plaintext column (AUTH-02).
        columns = [
            row[1] for row in auth_store._conn.execute("PRAGMA table_info(tokens)")
        ]
        assert "token" not in columns

    def test_create_token_stores_only_digest(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        row = auth_store._conn.execute(
            "SELECT token_hash FROM tokens WHERE id = ?", (created.id,)
        ).fetchone()
        assert row["token_hash"] != created.token
        assert len(row["token_hash"]) == 64

    def test_create_token_unknown_user_raises(
        self,
        auth_store: AuthStore,
    ) -> None:
        with pytest.raises(KeyError):
            auth_store.create_token(4242, "ghost")

    def test_get_or_create_config_token_reuses_same_key(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "s3cret")
        second = auth_store.get_or_create_config_token(user.id, "s3cret")

        assert first.token == second.token
        assert first.id == second.id
        assert first.name == "agent-config"
        assert first.token.startswith("qiip_")
        count = auth_store._conn.execute(
            "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config'"
        ).fetchone()[0]
        assert count == 1

    def test_get_or_create_config_token_rotates_after_revoke(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "s3cret")

        assert auth_store.revoke_token(user.id, first.id) is True
        second = auth_store.get_or_create_config_token(user.id, "s3cret")

        assert second.token != first.token
        assert second.id != first.id
        assert second.revoked is False

    def test_get_or_create_config_token_stable_across_store_instances(
        self,
        auth_store: AuthStore,
        tmp_path: Path,
    ) -> None:
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "s3cret")
        second_store = AuthStore(tmp_path / "qiip-test-auth-store.db")
        try:
            second = second_store.get_or_create_config_token(user.id, "s3cret")
        finally:
            second_store.close()
        assert second.token == first.token
        assert second.id == first.id

    def test_get_or_create_config_token_preserves_legacy_random(
        self,
        auth_store: AuthStore,
    ) -> None:
        """Regression (sjug review): a pre-reuse database holds a user-created
        token named 'agent-config' with a random value. The generated-key
        path must keep it active (it is indistinguishable from a user pin)
        and mint a purpose-marked derived key beside it."""
        user = self._user(auth_store)
        legacy = auth_store.create_token(user.id, "agent-config")

        derived = auth_store.get_or_create_config_token(user.id, "s3cret")

        assert derived.token != legacy.token
        assert derived.name == "agent-config"
        # The pre-reuse row is preserved and still resolvable.
        assert auth_store.resolve_token(legacy.token) is not None
        # Exactly one purpose-marked active key exists (the derived one).
        active = auth_store._conn.execute(
            "SELECT COUNT(*) FROM tokens "
            "WHERE user_id = ? AND purpose = 'agent-config' AND revoked = 0",
            (user.id,),
        ).fetchone()[0]
        assert active == 1
        again = auth_store.get_or_create_config_token(user.id, "s3cret")
        assert again.token == derived.token

    def test_list_tokens_returns_newest_first(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        first = auth_store.create_token(user.id, "a")
        second = auth_store.create_token(user.id, "b")

        tokens = auth_store.list_tokens(user.id)
        assert [token.name for token in tokens] == ["b", "a"]
        assert {token.id for token in tokens} == {first.id, second.id}

    def test_list_tokens_ignores_other_users(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        auth_store.create_token(user.id, "a")
        other = auth_store.upsert_google_user(
            google_sub="sub-other",
            email="bob@example.com",
            name="Bob",
            picture="",
        )

        assert auth_store.list_tokens(other.id) == []

    def test_revoke_token_and_resolve(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        assert auth_store.resolve_token(created.token) is not None
        assert auth_store.revoke_token(user.id, created.id) is True
        # Second revoke of the same token reports nothing happened.
        assert auth_store.revoke_token(user.id, created.id) is False
        assert auth_store.resolve_token(created.token) is None

    def test_resolve_token_rejects_malformed_and_unknown(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        assert auth_store.resolve_token("") is None
        assert auth_store.resolve_token("not-a-qiip-token") is None
        assert auth_store.resolve_token("qiip_totally-unknown") is None
        assert auth_store.resolve_token(created.token) is not None

    def test_resolve_token_joins_user(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")
        resolved = auth_store.resolve_token(created.token)

        assert resolved is not None
        assert resolved.user.id == user.id
        assert resolved.user.email == _GOOGLE["email"]
        assert resolved.token.id == created.id

    def test_resolve_token_keeps_user_id_distinct_from_token_id(
        self,
        auth_store: AuthStore,
    ) -> None:
        # A token shared, id, name and created_at column names with users, so
        # a `tokens.*, users.*` join makes row["id"] ambiguous. Seed the store
        # so the token id differs from the user id, then confirm the resolved
        # user is the real account (not a phantom built from the token row).
        user = self._user(auth_store)  # user id 1
        auth_store.create_token(user.id, "decoy")  # consumes token id 1
        created = auth_store.create_token(user.id, "ci-job")  # token id 2
        resolved = auth_store.resolve_token(created.token)

        assert resolved is not None
        assert resolved.user.id == user.id
        assert resolved.user.email == _GOOGLE["email"]
        assert resolved.user.name == _GOOGLE["name"]
        assert resolved.token.id == created.id

        # Usage recorded through the resolved identity lands on the real user.
        auth_store.record_usage(
            user_id=resolved.user.id,
            token_id=resolved.token.id,
            model="llama-3",
            endpoint="/v1/chat/completions",
            total_tokens=7,
        )
        assert auth_store.get_usage_totals(user.id).request_count == 1


class TestUsage:
    def _user(self, store: AuthStore) -> User:
        return store.upsert_google_user(**_GOOGLE)

    def test_record_and_summarize_usage(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        auth_store.record_usage(
            user_id=user.id,
            token_id=created.id,
            model="llama-3",
            endpoint="/v1/chat/completions",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        auth_store.record_usage(
            user_id=user.id,
            token_id=created.id,
            model="llama-3",
            endpoint="/v1/chat/completions",
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
        )

        summary = auth_store.get_usage_summary(user.id)
        assert len(summary) == 1
        assert summary[0].token_name == "ci-job"
        assert summary[0].model == "llama-3"
        assert summary[0].request_count == 2
        assert summary[0].prompt_tokens == 15
        assert summary[0].completion_tokens == 25
        assert summary[0].total_tokens == 40

        totals = auth_store.get_usage_totals(user.id)
        assert totals.request_count == 2
        assert totals.total_tokens == 40

    def test_usage_stamped_last_used_and_grouped_by_model(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")

        auth_store.record_usage(
            user_id=user.id,
            token_id=created.id,
            model="model-a",
            endpoint="/v1/completions",
            total_tokens=5,
        )
        auth_store.record_usage(
            user_id=user.id,
            token_id=created.id,
            model="model-b",
            endpoint="/v1/completions",
            total_tokens=7,
        )

        summary = auth_store.get_usage_summary(user.id)
        assert {row.model for row in summary} == {"model-a", "model-b"}
        assert auth_store.resolve_token(created.token) is not None  # still live
        tokens = auth_store.list_tokens(user.id)
        assert tokens[0].last_used_at is not None

    def test_totals_empty_for_fresh_user(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)

        totals = auth_store.get_usage_totals(user.id)

        assert totals.request_count == 0
        assert totals.total_tokens == 0
        assert auth_store.get_usage_summary(user.id) == []

    def test_usage_for_unknown_token_survives_token_deletion(
        self,
        auth_store: AuthStore,
    ) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "ci-job")
        auth_store.record_usage(
            user_id=user.id,
            token_id=created.id,
            model="model-a",
            endpoint="/v1/chat/completions",
            total_tokens=9,
        )

        # Simulate token removal (ON DELETE SET NULL) and confirm the usage
        # rows remain attributable to the user.
        auth_store._conn.execute("DELETE FROM tokens WHERE id = ?", (created.id,))
        auth_store._conn.commit()

        summary = auth_store.get_usage_summary(user.id)
        assert summary[0].token_id is None
        assert summary[0].token_name is None
        assert summary[0].total_tokens == 9


class TestStoreLifecycle:
    def test_opens_creating_parent_directory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "qiip.db"

        store = AuthStore(db_path)

        assert db_path.is_file()
        store.close()

    def test_close_is_idempotent(self, auth_store: AuthStore) -> None:
        auth_store.close()
        auth_store.close()


class TestEndpointScope:
    """Per-token endpoint scoping (RFE #107)."""

    def _user(self, store: AuthStore) -> User:
        return store.upsert_google_user(**_GOOGLE)

    def test_create_with_scope_roundtrips(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(
            user.id, "pinned", endpoint_scope=["host-a", "host-b"]
        )

        assert created.endpoint_scope == ["host-a", "host-b"]
        resolved = auth_store.resolve_token(created.token)
        assert resolved is not None
        assert resolved.token.endpoint_scope == ["host-a", "host-b"]
        listed = auth_store.list_tokens(user.id)
        assert listed[0].endpoint_scope == ["host-a", "host-b"]

    def test_unscoped_token_is_none(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.create_token(user.id, "open")

        assert created.endpoint_scope is None
        resolved = auth_store.resolve_token(created.token)
        assert resolved is not None
        assert resolved.token.endpoint_scope is None

    def test_existing_db_gets_scope_column(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "old.db"
        # Simulate a pre-scope database: create the old tokens table shape
        # by pointing a fresh store at a hand-built schema.
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                google_sub  TEXT    NOT NULL UNIQUE,
                email       TEXT    NOT NULL UNIQUE,
                name        TEXT    NOT NULL DEFAULT '',
                picture     TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tokens (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name         TEXT    NOT NULL,
                token_hash   TEXT    NOT NULL UNIQUE,
                prefix       TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                last_used_at TEXT,
                revoked      INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO users (google_sub, email, name, picture, created_at, updated_at) "
            "VALUES ('sub-1', 'alice@example.com', 'Alice', '', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        store = AuthStore(db_path)
        columns = [row[1] for row in store._conn.execute("PRAGMA table_info(tokens)")]
        assert "endpoint_scope" in columns
        user = store.get_user(1)
        assert user is not None
        created = store.create_token(user.id, "migrated", endpoint_scope=["h1"])
        assert created.endpoint_scope == ["h1"]
        store.close()


class TestScopeSerialization:
    """endpoint_scope NULL vs "[]" roundtrip (contra review)."""

    def test_dump_distinguishes_none_from_empty(self) -> None:
        from inference_proxy.auth.store import _dump_scopes

        assert _dump_scopes(None) is None
        assert _dump_scopes([]) == "[]"

    def test_empty_scope_roundtrips_as_empty(self) -> None:
        from inference_proxy.auth.store import _load_scopes

        assert _load_scopes("[]") == []
        assert _load_scopes(None) is None

    def test_malformed_scope_fails_closed(self) -> None:
        from inference_proxy.auth.store import _load_scopes

        assert _load_scopes("not json") == []
        assert _load_scopes('{"a": 1}') == []
        assert _load_scopes("42") == []
        assert _load_scopes("[1, 2]") == []


class TestAdminSurfaces:
    """Cross-user queries backing the admin token dashboard (RFE #113)."""

    def _seed(self, store: AuthStore) -> _AdminSeed:
        """Create two users, one token each, and usage for Alice."""
        alice = store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )
        bob = store.upsert_google_user(
            google_sub="sub-bob",
            email="bob@example.com",
            name="Bob",
            picture="",
        )
        token = store.create_token(alice.id, "ci")
        store.create_token(bob.id, "gh")
        store.record_usage(
            user_id=alice.id,
            token_id=token.id,
            model="llama-3",
            endpoint="/v1/chat/completions",
            prompt_tokens=4,
            completion_tokens=6,
            total_tokens=10,
        )
        return {
            "alice": alice,
            "bob": bob,
            "token_id": token.id,
            "token_raw": token.token,
        }

    def test_list_users_with_stats_counts(self, auth_store: AuthStore) -> None:
        self._seed(auth_store)

        stats = auth_store.list_users_with_stats()

        by_email = {row.email: row for row in stats}
        assert by_email["alice@example.com"].token_count == 1
        assert by_email["alice@example.com"].active_token_count == 1
        assert by_email["alice@example.com"].request_count == 1
        assert by_email["alice@example.com"].prompt_tokens == 4
        assert by_email["alice@example.com"].completion_tokens == 6
        assert by_email["alice@example.com"].total_tokens == 10
        assert by_email["bob@example.com"].token_count == 1
        assert by_email["bob@example.com"].request_count == 0
        assert by_email["bob@example.com"].total_tokens == 0

    def test_list_users_empty_store(self, auth_store: AuthStore) -> None:
        assert auth_store.list_users_with_stats() == []

    def test_list_all_tokens_joins_user(self, auth_store: AuthStore) -> None:
        seeded = self._seed(auth_store)
        bob = seeded["bob"]

        tokens = auth_store.list_all_tokens()

        assert len(tokens) == 2
        assert {token.user_email for token in tokens} == {
            "alice@example.com",
            "bob@example.com",
        }
        bob_token = next(token for token in tokens if token.user_id == bob.id)
        assert bob_token.user_name == "Bob"
        assert bob_token.prefix.startswith("qiip_")
        assert bob_token.request_count == 0
        assert bob_token.total_tokens == 0
        alice_token = next(token for token in tokens if token.user_id != bob.id)
        assert alice_token.request_count == 1
        assert alice_token.prompt_tokens == 4
        assert alice_token.completion_tokens == 6
        assert alice_token.total_tokens == 10
        # AUTH-02: no raw secret is ever exposed by the admin view either.
        assert not hasattr(bob_token, "token")

    def test_revoke_any_token_by_id(self, auth_store: AuthStore) -> None:
        seeded = self._seed(auth_store)

        assert auth_store.revoke_any_token(seeded["token_id"]) is True
        assert auth_store.revoke_any_token(seeded["token_id"]) is False
        assert auth_store.resolve_token(seeded["token_raw"]) is None
        assert auth_store.revoke_any_token(4242) is False

    def test_usage_totals_all_spans_users(self, auth_store: AuthStore) -> None:
        self._seed(auth_store)

        totals = auth_store.get_usage_totals_all()

        assert totals.request_count == 1
        assert totals.prompt_tokens == 4
        assert totals.completion_tokens == 6
        assert totals.total_tokens == 10

    def test_usage_totals_all_empty(self, auth_store: AuthStore) -> None:
        assert auth_store.get_usage_totals_all().total_tokens == 0

    def test_user_usage_timeline_groups_by_day(self, auth_store: AuthStore) -> None:
        seeded = self._seed(auth_store)
        alice = seeded["alice"]
        token = auth_store.list_tokens(alice.id)[0]
        auth_store.record_usage(
            user_id=alice.id,
            token_id=token.id,
            model="llama-3",
            endpoint="/v1/completions",
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )

        timeline = auth_store.get_user_usage_timeline(alice.id)

        assert len(timeline) == 1
        row = timeline[0]
        assert row.day == _utcnow().date().isoformat()
        assert row.request_count == 2
        assert row.prompt_tokens == 5
        assert row.completion_tokens == 8
        assert row.total_tokens == 13

    def test_user_usage_timeline_honors_day_window(self, auth_store: AuthStore) -> None:
        seeded = self._seed(auth_store)
        alice = seeded["alice"]

        assert auth_store.get_user_usage_timeline(alice.id, days=-1) == []

    def test_user_usage_timeline_empty(self, auth_store: AuthStore) -> None:
        alice = auth_store.upsert_google_user(
            google_sub="sub-alice",
            email="alice@example.com",
            name="Alice",
            picture="",
        )

        assert auth_store.get_user_usage_timeline(alice.id) == []


class TestConfigToken:
    """The reusable agent-config token: stable key, revoke rotates it."""

    def _user(self, store: AuthStore) -> User:
        return store.upsert_google_user(**_GOOGLE)

    def test_reuses_same_raw_and_row(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "session-secret")
        second = auth_store.get_or_create_config_token(user.id, "session-secret")

        assert first.token == second.token
        assert first.id == second.id

    def test_same_raw_across_store_instances(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        created = auth_store.get_or_create_config_token(user.id, "session-secret")

        reopened = AuthStore(auth_store._db_path)
        try:
            again = reopened.get_or_create_config_token(user.id, "session-secret")
        finally:
            reopened.close()
        assert again.token == created.token
        assert again.id == created.id

    def test_revoke_rotates_the_key(self, auth_store: AuthStore) -> None:
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "session-secret")
        assert auth_store.revoke_token(user.id, first.id) is True

        second = auth_store.get_or_create_config_token(user.id, "session-secret")
        assert second.token != first.token
        assert second.id != first.id
        assert auth_store.resolve_token(first.token) is None
        assert auth_store.resolve_token(second.token) is not None

    def test_secret_rotation_revokes_stale_derived_key(
        self,
        auth_store: AuthStore,
    ) -> None:
        """Rotating ``auth.session_secret`` must invalidate previously
        distributed agent-config keys, not leave them resolvable."""
        user = self._user(auth_store)
        first = auth_store.get_or_create_config_token(user.id, "secret-v1")

        second = auth_store.get_or_create_config_token(user.id, "secret-v2")

        assert second.token != first.token
        assert second.id != first.id
        # The stale derived key stops resolving after the secret changes.
        assert auth_store.resolve_token(first.token) is None
        assert auth_store.resolve_token(second.token) is not None
        # Exactly one active agent-config row remains.
        active = auth_store._conn.execute(
            "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config' AND revoked = 0"
        ).fetchone()[0]
        assert active == 1

    def test_preserves_pre_reuse_same_named_user_token(
        self,
        auth_store: AuthStore,
    ) -> None:
        """Regression (sjug review): a user-created token named
        'agent-config' (random value, pre-reuse deployments) is not a
        generated key. The downloader mints a purpose-marked derived key
        alongside it and the user token keeps working — it must never be
        revoked or replaced by the generated-key path."""
        user = self._user(auth_store)
        user_token = auth_store.create_token(
            user.id, "agent-config", endpoint_scope=["pub1"]
        )

        derived = auth_store.get_or_create_config_token(user.id, "session-secret")

        assert derived.token != user_token.token
        assert derived.id != user_token.id
        # The user's same-named token stays active and keeps its scope.
        resolved = auth_store.resolve_token(user_token.token)
        assert resolved is not None
        assert resolved.token.endpoint_scope == ["pub1"]
        # Exactly one purpose-marked active key exists.
        active = auth_store._conn.execute(
            "SELECT COUNT(*) FROM tokens "
            "WHERE user_id = ? AND purpose = 'agent-config' AND revoked = 0",
            (user.id,),
        ).fetchone()[0]
        assert active == 1


class _StaleRows:
    """Result stand-in with a sqlite3.Row-compatible fetchall/fetchone."""

    def __init__(self, rows: tuple[object, ...]) -> None:
        self._rows = rows

    def fetchall(self) -> tuple[object, ...]:
        return self._rows

    def fetchone(self) -> object | None:
        return self._rows[0] if self._rows else None


class _StaleSnapshotConn:
    """Serve a pre-race (stale) rows snapshot for the first agent-config
    read, then delegate everything else to the real connection.

    Simulates a second process whose ``SELECT`` happened before the winner's
    ``INSERT`` (the exact timing the multi-process race produces).
    """

    _SNAPSHOT_SQL = """
SELECT * FROM tokens
 WHERE user_id = ?
   AND (purpose = ? OR (purpose IS NULL AND name = ?))
 ORDER BY id
"""

    def __init__(self, real: sqlite3.Connection, snapshot: tuple[object, ...]) -> None:
        self._real = real
        self._snapshot = snapshot
        self._served = False

    def execute(
        self,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor | _StaleRows:
        if not self._served and sql == self._SNAPSHOT_SQL:
            self._served = True
            return _StaleRows(self._snapshot)
        return self._real.execute(sql, params)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class TestConfigTokenRaceRecovery:
    """Concurrent mint recovery: the IntegrityError path must return a usable
    key (the winner's, or the next generation), never 500. Reachable only
    with multiple AuthStore instances on one DB (the in-process lock
    serializes a single store)."""

    def test_concurrent_mint_returns_winner_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = tmp_path / "race.db"
        store_a = AuthStore(db)
        store_b = AuthStore(db)
        user = store_a.upsert_google_user(**_GOOGLE)

        barrier = threading.Barrier(2)
        original = store_module._derive_config_token
        counter = 0
        counter_lock = threading.Lock()

        def synchronized(secret: str, user_id: int, generation: int) -> str:
            nonlocal counter
            with counter_lock:
                counter += 1
                call_no = counter
            if call_no <= 2:
                # The two pre-insert derivations (one per process) coincide;
                # recovery derivations pass straight through.
                with suppress(threading.BrokenBarrierError):
                    barrier.wait(timeout=5)
            return original(secret, user_id, generation)

        monkeypatch.setattr(store_module, "_derive_config_token", synchronized)

        results: dict[str, CreatedToken] = {}

        def mint(label: str, store: AuthStore) -> None:
            results[label] = store.get_or_create_config_token(user.id, "s3cret")

        threads = [
            threading.Thread(target=mint, args=("a", store_a)),
            threading.Thread(target=mint, args=("b", store_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        # Both callers end up with the same stable agent-config key.
        assert results["a"].token == results["b"].token
        auth = store_a.resolve_token(results["a"].token)
        assert auth is not None
        assert auth.user.id == user.id
        active = store_a._conn.execute(
            "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config' AND revoked = 0"
        ).fetchone()[0]
        assert active == 1

    def test_concurrent_mint_with_revoked_winner_mints_next_generation(
        self,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / "race-revoked.db"
        store = AuthStore(db)
        user = store.upsert_google_user(**_GOOGLE)
        # Generator-0 key created and then revoked -- the "winner" row is gone
        # by the time the loser re-queries after its failed insert.
        first = store.get_or_create_config_token(user.id, "s3cret")
        assert store.revoke_token(user.id, first.id) is True

        # Second process: its rows snapshot predates the winner's insert, so
        # it derives generation 0 -> collides with the (now revoked) row.
        stale = AuthStore(db)
        stale._conn = _StaleSnapshotConn(stale._conn, ())  # type: ignore[assignment]
        recovered = stale.get_or_create_config_token(user.id, "s3cret")

        assert recovered.token != first.token
        auth = stale.resolve_token(recovered.token)
        assert auth is not None
        assert auth.user.id == user.id
        active = stale._conn.execute(
            "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config' AND revoked = 0"
        ).fetchone()[0]
        assert active == 1

    def test_concurrent_mint_never_raises_on_repeated_collisions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three workers race the same generation with the same secret: the
        partial unique index forces the losers into the recovery branch,
        exactly one active row remains, and every caller receives the same
        winner key (no 500, no duplicate key)."""
        db = tmp_path / "race-triple.db"
        stores = [AuthStore(db) for _ in range(3)]
        user = stores[0].upsert_google_user(**_GOOGLE)

        barrier = threading.Barrier(3)
        original = store_module._derive_config_token
        counter = 0
        counter_lock = threading.Lock()

        def synchronized(secret: str, user_id: int, generation: int) -> str:
            nonlocal counter
            with counter_lock:
                counter += 1
                call_no = counter
            if call_no <= 3:
                with suppress(threading.BrokenBarrierError):
                    barrier.wait(timeout=5)
            return original(secret, user_id, generation)

        monkeypatch.setattr(store_module, "_derive_config_token", synchronized)

        results: dict[str, CreatedToken] = {}

        def mint(label: str, store: AuthStore) -> None:
            results[label] = store.get_or_create_config_token(user.id, "s3cret")

        threads = [
            threading.Thread(target=mint, args=(f"s{i}", store))
            for i, store in enumerate(stores)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        active = (
            stores[0]
            ._conn.execute(
                "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config' AND revoked = 0"
            )
            .fetchone()[0]
        )
        assert active == 1
        # Every caller resolved to the same active key.
        for i, store in enumerate(stores):
            auth = store.resolve_token(results[f"s{i}"].token)
            assert auth is not None
            assert auth.user.id == user.id

    def test_mixed_secret_rotation_keeps_one_active_key(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression (third review): workers holding different session secrets
        (rolling rotation) must not both mint active agent-config rows. The
        partial unique index forces the new-secret worker into recovery, which
        revokes the stale-secret winner and mints the next generation -- one
        active key, old key dead, no churn."""
        db = tmp_path / "rotation-race.db"
        store_v1 = AuthStore(db)
        user = store_v1.upsert_google_user(**_GOOGLE)
        first = store_v1.get_or_create_config_token(user.id, "secret-v1")

        # Second worker with the rotated secret and a stale rows snapshot: it
        # derives generation 0 (collides with the index), recovery sees a
        # mismatched-secret winner, revokes it, mints generation 1.
        store_v2 = AuthStore(db)
        store_v2._conn = _StaleSnapshotConn(store_v2._conn, ())  # type: ignore[assignment]
        rotated = store_v2.get_or_create_config_token(user.id, "secret-v2")

        assert rotated.token != first.token
        assert store_v1.resolve_token(rotated.token) is not None
        auth = store_v2.resolve_token(rotated.token)
        assert auth is not None
        assert auth.user.id == user.id
        assert store_v2.resolve_token(first.token) is None
        active = store_v2._conn.execute(
            "SELECT COUNT(*) FROM tokens WHERE name = 'agent-config' AND revoked = 0"
        ).fetchone()[0]
        assert active == 1

    def test_migrate_preserves_legacy_agent_config_named_tokens(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression (sjug review): users could create their own tokens named
        'agent-config' in earlier builds. The migration must preserve those
        rows untouched — including their endpoint pins — and only generated
        (purpose-marked) keys are governed by the one-active invariant."""
        db = tmp_path / "legacy-name.db"
        store = AuthStore(db)
        user = store.upsert_google_user(**_GOOGLE)
        # A user-created token that happens to be named 'agent-config' (pinned).
        legacy = store.create_token(user.id, "agent-config", endpoint_scope=["pub1"])
        # A generated key from an earlier build (no purpose marker yet).
        generated = store.get_or_create_config_token(user.id, "s3cret")
        # Simulate a pre-marker database: clear any purpose markers so both
        # rows look like legacy rows, then reopen (migration runs).
        store._conn.execute("UPDATE tokens SET purpose = NULL")
        store._conn.commit()

        reopened = AuthStore(db)

        # The pinned user token is untouched and still resolves with its scope.
        auth = reopened.resolve_token(legacy.token)
        assert auth is not None
        assert auth.token.endpoint_scope == ["pub1"]
        # The generated row is also preserved as a plain legacy token.
        assert reopened.resolve_token(generated.token) is not None
        # New downloads reuse the legacy generated key (hash match proves it
        # is generated) and backfill its purpose marker; legacy rows stay
        # valid. The user-created token is untouched.
        fresh = reopened.get_or_create_config_token(user.id, "s3cret")
        assert fresh.token == generated.token
        assert reopened.resolve_token(generated.token) is not None
        active = reopened._conn.execute(
            "SELECT COUNT(*) FROM tokens "
            "WHERE user_id = ? AND purpose = 'agent-config' AND revoked = 0",
            (user.id,),
        ).fetchone()[0]
        assert active == 1

    def test_generated_config_keys_carry_purpose_marker(
        self,
        tmp_path: Path,
    ) -> None:
        """Minted keys are marked with purpose='agent-config' and never touch
        same-named user tokens."""
        db = tmp_path / "purpose.db"
        store = AuthStore(db)
        user = store.upsert_google_user(**_GOOGLE)
        store.create_token(user.id, "agent-config", endpoint_scope=["pub1"])

        token = store.get_or_create_config_token(user.id, "s3cret")
        row = store._conn.execute(
            "SELECT purpose, name, endpoint_scope FROM tokens WHERE id = ?",
            (token.id,),
        ).fetchone()
        assert row is not None
        assert row["purpose"] == "agent-config"
        assert row["name"] == "agent-config"
        # The user-created same-named token is untouched (still active).
        legacy = store._conn.execute(
            "SELECT COUNT(*) FROM tokens "
            "WHERE user_id = ? AND name = 'agent-config' AND purpose IS NULL AND revoked = 0",
            (user.id,),
        ).fetchone()[0]
        assert legacy == 1
