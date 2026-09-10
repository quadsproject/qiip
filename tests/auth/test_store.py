"""Behavioral tests for the SQLite auth store."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_proxy.auth.models import User
from inference_proxy.auth.store import AuthStore

_GOOGLE = {
    "google_sub": "sub-123",
    "email": "alice@example.com",
    "name": "Alice",
    "picture": "https://example.com/alice.png",
}


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
