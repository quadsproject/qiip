"""SQLite-backed persistence for users, API tokens, and token usage.

Per AUTH-01: user records are keyed by the Google ``sub`` claim (stable
per-account, never reissued) and emails stay unique so an account maps to
exactly one row.

Per AUTH-02: API tokens are stored as SHA-256 digests. A database read
(backup, console, etcd-style dump) never yields a usable credential, and a
compromised digest cannot be replayed because the source of randomness is
``secrets.token_urlsafe``.

Concurrency: a single SQLite connection guarded by a re-entrant lock. The
write volume is one row per authenticated inference request, which is far
below anything WAL-mode SQLite cannot absorb on one process.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from inference_proxy.auth._constants import TOKEN_PREFIX
from inference_proxy.auth.models import (
    AdminTokenView,
    AdminUserStats,
    ApiToken,
    CreatedToken,
    TokenAuth,
    TokenUsage,
    UsageTimelineRow,
    UsageTotals,
    User,
)

logger = structlog.get_logger()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    google_sub  TEXT    NOT NULL UNIQUE,
    email       TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL DEFAULT '',
    picture     TEXT    NOT NULL DEFAULT '',
    is_admin    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    token_hash      TEXT    NOT NULL UNIQUE,
    prefix          TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    last_used_at    TEXT,
    revoked         INTEGER NOT NULL DEFAULT 0,
    endpoint_scope  TEXT
);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_id          INTEGER REFERENCES tokens(id) ON DELETE SET NULL,
    model             TEXT    NOT NULL DEFAULT '',
    endpoint          TEXT    NOT NULL DEFAULT '',
    request_count     INTEGER NOT NULL DEFAULT 1,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_token ON usage(token_id);
CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage(user_id, created_at);
"""


def _utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def _iso(when: datetime) -> str:
    """Render a datetime as a round-trippable UTC ISO-8601 string."""
    return when.astimezone(UTC).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string back into an aware datetime."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def _hash_token(raw: str) -> str:
    """Return the SHA-256 digest of a raw bearer token (AUTH-02)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_token() -> str:
    """Mint a fresh bearer token with 256 bits of entropy."""
    return f"{TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


# Reusable agent-config token (AUTH-02-compatible): the raw value is never
# stored -- it is deterministically re-derived from the session secret, the
# user id, and a generation counter tracked by the token-row history. The
# same config key therefore works across browsers and machines, and a revoke
# rotates it (the next download derives the next generation).
_CONFIG_TOKEN_NAME = "agent-config"
_CONFIG_TOKEN_INFO = b"qiip-agent-config-token-v1"


def _derive_config_token(secret: str, user_id: int, generation: int) -> str:
    """Derive the stable raw value for the user's agent-config token."""
    material = f"{_CONFIG_TOKEN_INFO.decode()}:{user_id}:{generation}"
    digest = hmac.new(
        secret.encode("utf-8"),
        material.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{TOKEN_PREFIX}{base64.urlsafe_b64encode(digest).decode().rstrip('=')}"


def _dump_scopes(scopes: list[str] | None) -> str | None:
    """Serialize an endpoint scope to its TEXT column value.

    ``None`` (full access) becomes NULL; an empty list is serialized as
    ``"[]"`` so "pinned to nothing" stays distinct from "unrestricted".
    """
    if scopes is None:
        return None
    return json.dumps(scopes)


def _load_scopes(raw: str | None) -> list[str] | None:
    """Parse the endpoint_scope column.

    ``None`` (NULL) means full access; a malformed or non-list value is
    parsed as ``[]`` (pinned to nothing) so a corrupt row never silently
    widens a pinned token to unrestricted access.
    """
    if raw is None:
        return None
    try:
        scopes = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(scopes, list) or not all(
        isinstance(item, str) for item in scopes
    ):
        return []
    return scopes


class AuthStore:
    """Thread-safe SQLite store for identities, tokens, and usage.

    All methods are synchronous and guarded by an internal re-entrant
    lock. Async callers should hop off the event loop with
    ``await asyncio.to_thread(store.method, ...)``.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
        logger.info("auth store opened", db_path=str(db_path))

    def _migrate(self) -> None:
        """Apply additive migrations to pre-existing databases.

        ``CREATE TABLE IF NOT EXISTS`` handles fresh databases; tables
        created before a column existed need a guarded ALTER here.
        """
        token_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(tokens)").fetchall()
        }
        if "endpoint_scope" not in token_columns:
            self._conn.execute("ALTER TABLE tokens ADD COLUMN endpoint_scope TEXT")
        user_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "is_admin" not in user_columns:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def upsert_google_user(
        self,
        *,
        google_sub: str,
        email: str,
        name: str,
        picture: str,
    ) -> User:
        """Create or refresh a user row keyed by the OIDC ``sub`` claim.

        Falls back to the email-unique index when a Google account appears
        under a new ``sub`` (rare account-migration case): the existing row
        is rebound to the new subject.
        """
        now = _iso(_utcnow())
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM users WHERE google_sub = ?", (google_sub,)
            ).fetchone()
            if existing is None:
                try:
                    self._conn.execute(
                        """
                        INSERT INTO users
                            (google_sub, email, name, picture, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (google_sub, email, name, picture, now, now),
                    )
                except sqlite3.IntegrityError:
                    self._conn.execute(
                        """
                        UPDATE users
                           SET google_sub = ?, name = ?, picture = ?, updated_at = ?
                         WHERE email = ?
                        """,
                        (google_sub, name, picture, now, email),
                    )
            else:
                self._conn.execute(
                    """
                    UPDATE users
                       SET email = ?, name = ?, picture = ?, updated_at = ?
                     WHERE google_sub = ?
                    """,
                    (email, name, picture, now, google_sub),
                )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
            ).fetchone()
            user = self._user_from_row(row)
            if user is None:  # pragma: no cover - defensive
                raise RuntimeError("user row vanished after upsert")
        return user

    def get_user(self, user_id: int) -> User | None:
        """Return the user row, or None when it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._user_from_row(row)

    def list_users(self) -> list[User]:
        """Return every user row, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [user for row in rows if (user := self._user_from_row(row))]

    def set_user_admin(self, user_id: int, is_admin: bool) -> bool:
        """Grant or revoke the admin role; False when the user does not exist."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE users SET is_admin = ?, updated_at = ? WHERE id = ?",
                (1 if is_admin else 0, _iso(_utcnow()), user_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # API tokens
    # ------------------------------------------------------------------

    def create_token(
        self,
        user_id: int,
        name: str,
        endpoint_scope: list[str] | None = None,
    ) -> CreatedToken:
        """Mint a token, store its digest, and return the raw secret once.

        *endpoint_scope* is an optional list of node hostnames the token
        may route to; ``None`` means full access. Raises ``KeyError`` when
        *user_id* does not exist.
        """
        raw = _generate_token()
        now = _iso(_utcnow())
        with self._lock:
            exists = self._conn.execute(
                "SELECT id FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(user_id)
            cursor = self._conn.execute(
                """
                INSERT INTO tokens
                    (user_id, name, token_hash, prefix, created_at, endpoint_scope)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    name,
                    _hash_token(raw),
                    raw[: len(TOKEN_PREFIX) + 8],
                    now,
                    _dump_scopes(endpoint_scope),
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM tokens WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        token = self._token_from_row(row)
        if token is None:  # pragma: no cover - defensive
            raise RuntimeError("token row vanished after insert")
        return CreatedToken(**token.model_dump(), token=raw)

    def get_or_create_config_token(self, user_id: int, secret: str) -> CreatedToken:
        """Return the user's reusable agent-config token, minting on first use.

        One stable raw value is shared by every agent-config download
        (across servers, browsers, and machines). The value is derived
        deterministically from *secret* + user id + a generation counter;
        only its digest is stored. Generation equals the number of prior
        ``agent-config`` rows, so revoking the token rotates the value: the
        next download derives (and stores) a fresh one. A pre-existing row
        minted with a random secret (pre-reuse deployments) is detected by
        hash mismatch and superseded by a derived token.

        Raises ``KeyError`` when *user_id* does not exist.
        """
        now = _iso(_utcnow())
        with self._lock:
            exists = self._conn.execute(
                "SELECT id FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(user_id)
            rows = self._conn.execute(
                "SELECT * FROM tokens WHERE user_id = ? AND name = ? ORDER BY id",
                (user_id, _CONFIG_TOKEN_NAME),
            ).fetchall()
            active = next((r for r in reversed(rows) if not r["revoked"]), None)
            if active is not None:
                generation = next(
                    i for i, r in enumerate(rows) if r["id"] == active["id"]
                )
                raw = _derive_config_token(secret, user_id, generation)
                if _hash_token(raw) == active["token_hash"]:
                    token = self._token_from_row(active)
                    if token is not None:  # pragma: no cover - defensive
                        return CreatedToken(**token.model_dump(), token=raw)

            generation = len(rows)
            raw = _derive_config_token(secret, user_id, generation)
            try:
                cursor = self._conn.execute(
                    """
                    INSERT INTO tokens
                        (user_id, name, token_hash, prefix, created_at, endpoint_scope)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        user_id,
                        _CONFIG_TOKEN_NAME,
                        _hash_token(raw),
                        raw[: len(TOKEN_PREFIX) + 8],
                        now,
                    ),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM tokens WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
            except sqlite3.IntegrityError:
                # Lost a concurrent mint race: return the winner's row.
                self._conn.rollback()
                row = self._conn.execute(
                    """
                    SELECT * FROM tokens
                     WHERE user_id = ? AND name = ? AND revoked = 0
                     ORDER BY id DESC LIMIT 1
                    """,
                    (user_id, _CONFIG_TOKEN_NAME),
                ).fetchone()
                if row is None:  # pragma: no cover - defensive
                    raise
                raw = _derive_config_token(
                    secret,
                    user_id,
                    next(i for i, r in enumerate(rows) if r["id"] == row["id"]),
                )
        token = self._token_from_row(row)
        if token is None:  # pragma: no cover - defensive
            raise RuntimeError("token row vanished after insert")
        return CreatedToken(**token.model_dump(), token=raw)

    def list_tokens(self, user_id: int) -> list[ApiToken]:
        """Return all token rows for a user, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM tokens
                 WHERE user_id = ?
                 ORDER BY created_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [token for row in rows if (token := self._token_from_row(row))]

    def revoke_token(self, user_id: int, token_id: int) -> bool:
        """Revoke a token, returning True if a live token was revoked."""
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE tokens
                   SET revoked = 1
                 WHERE id = ? AND user_id = ? AND revoked = 0
                """,
                (token_id, user_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def resolve_token(self, raw: str) -> TokenAuth | None:
        """Resolve a raw bearer token to its user + token row.

        Returns None for unknown, malformed, or revoked tokens. The raw
        value is never stored or logged (AUTH-02).
        """
        if not raw or not str(raw).startswith(TOKEN_PREFIX):
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tokens.*, users.*
                  FROM tokens
                  JOIN users ON users.id = tokens.user_id
                 WHERE tokens.token_hash = ? AND tokens.revoked = 0
                """,
                (_hash_token(raw),),
            ).fetchone()
        if row is None:
            return None
        user = self._user_from_row(row)
        token = self._token_from_row(row)
        if user is None or token is None:  # pragma: no cover - defensive
            return None
        return TokenAuth(user=user, token=token)

    # ------------------------------------------------------------------
    # Usage tracking (AUTH-04)
    # ------------------------------------------------------------------

    def record_usage(
        self,
        *,
        user_id: int,
        token_id: int,
        model: str,
        endpoint: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        request_count: int = 1,
    ) -> None:
        """Append one usage row and stamp the token's last-used time."""
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO usage
                    (user_id, token_id, model, endpoint, request_count,
                     prompt_tokens, completion_tokens, total_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    token_id,
                    model,
                    endpoint,
                    request_count,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE tokens SET last_used_at = ? WHERE id = ?",
                (now, token_id),
            )
            self._conn.commit()

    def get_usage_summary(self, user_id: int) -> list[TokenUsage]:
        """Aggregate usage per (token, model, endpoint) for a user."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t.id           AS token_id,
                       t.name         AS token_name,
                       t.created_at   AS token_created_at,
                       u.model,
                       u.endpoint,
                       SUM(u.request_count)     AS request_count,
                       SUM(u.prompt_tokens)     AS prompt_tokens,
                       SUM(u.completion_tokens) AS completion_tokens,
                       SUM(u.total_tokens)      AS total_tokens
                  FROM usage u
                  LEFT JOIN tokens t ON t.id = u.token_id
                 WHERE u.user_id = ?
                 GROUP BY u.token_id, u.model, u.endpoint
                 ORDER BY t.created_at DESC, u.model
                """,
                (user_id,),
            ).fetchall()
        return [
            TokenUsage(
                user_id=user_id,
                token_id=row["token_id"],
                token_name=row["token_name"],
                model=row["model"],
                endpoint=row["endpoint"],
                request_count=row["request_count"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
            )
            for row in rows
        ]

    def get_usage_totals(self, user_id: int) -> UsageTotals:
        """Return headline request/token sums across all of a user's usage."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(request_count), 0)     AS request_count,
                       COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(total_tokens), 0)      AS total_tokens
                  FROM usage
                 WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return UsageTotals(
            request_count=row["request_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
        )

    # ------------------------------------------------------------------
    # Admin surfaces (RFE #113)
    # ------------------------------------------------------------------

    def list_users_with_stats(self) -> list[AdminUserStats]:
        """Return every user plus token and usage counts, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT u.id, u.email, u.name, u.picture, u.is_admin, u.created_at,
                       (SELECT COUNT(*) FROM tokens t
                          WHERE t.user_id = u.id) AS token_count,
                       (SELECT COUNT(*) FROM tokens t
                          WHERE t.user_id = u.id AND t.revoked = 0)
                           AS active_token_count,
                       COALESCE((SELECT SUM(request_count) FROM usage w
                                   WHERE w.user_id = u.id), 0) AS request_count,
                       COALESCE((SELECT SUM(prompt_tokens) FROM usage w
                                   WHERE w.user_id = u.id), 0) AS prompt_tokens,
                       COALESCE((SELECT SUM(completion_tokens) FROM usage w
                                   WHERE w.user_id = u.id), 0) AS completion_tokens,
                       COALESCE((SELECT SUM(total_tokens) FROM usage w
                                   WHERE w.user_id = u.id), 0) AS total_tokens
                  FROM users u
                 ORDER BY u.created_at DESC, u.id DESC
                """
            ).fetchall()
        return [
            AdminUserStats(
                id=row["id"],
                email=row["email"],
                name=row["name"],
                picture=row["picture"],
                is_admin=bool(row["is_admin"]),
                created_at=_parse_iso(row["created_at"]),
                token_count=row["token_count"],
                active_token_count=row["active_token_count"],
                request_count=row["request_count"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
            )
            for row in rows
        ]

    def list_all_tokens(self) -> list[AdminTokenView]:
        """Return every token with its owner identity and usage, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tokens.*, users.email AS user_email, users.name AS user_name,
                       COALESCE(agg.request_count, 0)     AS request_count,
                       COALESCE(agg.prompt_tokens, 0)     AS prompt_tokens,
                       COALESCE(agg.completion_tokens, 0) AS completion_tokens,
                       COALESCE(agg.total_tokens, 0)      AS total_tokens
                  FROM tokens
                  JOIN users ON users.id = tokens.user_id
                  LEFT JOIN (
                      SELECT token_id,
                             SUM(request_count)     AS request_count,
                             SUM(prompt_tokens)     AS prompt_tokens,
                             SUM(completion_tokens) AS completion_tokens,
                             SUM(total_tokens)      AS total_tokens
                        FROM usage
                       GROUP BY token_id
                  ) agg ON agg.token_id = tokens.id
                 ORDER BY tokens.created_at DESC, tokens.id DESC
                """
            ).fetchall()
        return [
            AdminTokenView(
                id=row["id"],
                user_id=row["user_id"],
                user_email=row["user_email"],
                user_name=row["user_name"],
                name=row["name"],
                prefix=row["prefix"],
                created_at=_parse_iso(row["created_at"]),
                last_used_at=_parse_iso(row["last_used_at"]),
                revoked=bool(row["revoked"]),
                endpoint_scope=_load_scopes(row["endpoint_scope"]),
                request_count=row["request_count"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
            )
            for row in rows
        ]

    def get_user_usage_timeline(
        self, user_id: int, days: int = 30
    ) -> list[UsageTimelineRow]:
        """Return per-day usage aggregates for a user, newest day first."""
        cutoff = _iso(_utcnow() - timedelta(days=days))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT substr(created_at, 1, 10)                AS day,
                       COALESCE(SUM(request_count), 0)          AS request_count,
                       COALESCE(SUM(prompt_tokens), 0)          AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0)      AS completion_tokens,
                       COALESCE(SUM(total_tokens), 0)           AS total_tokens
                  FROM usage
                 WHERE user_id = ? AND created_at >= ?
                 GROUP BY day
                 ORDER BY day DESC
                """,
                (user_id, cutoff),
            ).fetchall()
        return [
            UsageTimelineRow(
                day=row["day"],
                request_count=row["request_count"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
            )
            for row in rows
        ]

    def get_usage_totals_all(self) -> UsageTotals:
        """Return headline request/token sums across the whole store."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(request_count), 0)     AS request_count,
                       COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(total_tokens), 0)      AS total_tokens
                  FROM usage
                """
            ).fetchone()
        return UsageTotals(
            request_count=row["request_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
        )

    def revoke_any_token(self, token_id: int) -> bool:
        """Revoke any token by id (admin path), True when a live token was revoked."""
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE tokens
                   SET revoked = 1
                 WHERE id = ? AND revoked = 0
                """,
                (token_id,),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Row handling
    # ------------------------------------------------------------------

    @staticmethod
    def _user_from_row(row: sqlite3.Row | None) -> User | None:
        """Convert a (possibly joined) users row to a User model."""
        if row is None:
            return None
        return User(
            id=row["id"],
            google_sub=row["google_sub"],
            email=row["email"],
            name=row["name"],
            picture=row["picture"],
            is_admin=bool(row["is_admin"]),
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
        )

    @staticmethod
    def _token_from_row(row: sqlite3.Row | None) -> ApiToken | None:
        """Convert a (possibly joined) tokens row to an ApiToken model."""
        if row is None:
            return None
        return ApiToken(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            prefix=row["prefix"],
            created_at=_parse_iso(row["created_at"]),
            last_used_at=_parse_iso(row["last_used_at"]),
            revoked=bool(row["revoked"]),
            endpoint_scope=_load_scopes(row["endpoint_scope"]),
        )
