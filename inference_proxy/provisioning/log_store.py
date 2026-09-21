"""Transactional, bounded attempt logs, also uploaded to nodes (stdlib only).

A record and its retrieval cursor commit together. Prefix rotation never resets
sequence numbers, so replay remains idempotent even after retained rows expire.
SQLite uses full synchronous commits and WAL for concurrent readers. New
databases enable full auto-vacuum; existing stores with vacuum disabled require
an explicit rebuild to enable it. Allow space for database/index pages and the
WAL in addition to payloads; long-lived readers may delay checkpoints.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - node Python 3.9


class AttemptLogStore:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 268_435_456,
        attempt_max_bytes: int = 33_554_432,
        max_attempts: int = 1000,
        retention_days: float = 30,
        max_record_bytes: int = 16_384,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.attempt_max_bytes = attempt_max_bytes
        self.max_attempts = max_attempts
        self.retention_days = retention_days
        self.max_record_bytes = max_record_bytes
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create with restrictive permissions before SQLite opens the file.
        path.touch(mode=0o600, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA auto_vacuum=FULL")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, hostname TEXT NOT NULL,
                    created REAL NOT NULL, metadata TEXT NOT NULL,
                    next_seq INTEGER NOT NULL DEFAULT 0,
                    remote_cursor INTEGER NOT NULL DEFAULT 0,
                    dropped INTEGER NOT NULL DEFAULT 0,
                    bytes INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS records (
                    attempt TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL, payload TEXT NOT NULL, bytes INTEGER NOT NULL,
                    PRIMARY KEY (attempt, seq)
                );
                CREATE TABLE IF NOT EXISTS statistics (
                    name TEXT PRIMARY KEY, value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO statistics VALUES ('evicted_attempts', 0);
                CREATE INDEX IF NOT EXISTS attempts_retention
                ON attempts(created DESC, id DESC)
                WHERE json_extract(metadata,'$.status') != 'running';
                CREATE TRIGGER IF NOT EXISTS record_bytes_insert
                AFTER INSERT ON records BEGIN
                    UPDATE statistics SET value=value+NEW.bytes WHERE name='retained_bytes';
                END;
                CREATE TRIGGER IF NOT EXISTS record_bytes_delete
                AFTER DELETE ON records BEGIN
                    UPDATE statistics SET value=value-OLD.bytes WHERE name='retained_bytes';
                END;
            """)
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute(
                    "SELECT value FROM statistics WHERE name='retained_bytes'"
                ).fetchone()
                is None
            ):
                # One-time backfill for databases created before byte accounting.
                db.execute(
                    "INSERT INTO statistics SELECT 'retained_bytes', coalesce(sum(bytes),0) FROM records"
                )
            self._prune(db)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(
        self,
        hostname: str,
        *,
        engine: str = "unknown",
        model: str | None = None,
        bundle_version: str = "unknown",
        operation: str = "provision",
        attempt_id: str | None = None,
    ) -> str:
        attempt_id = attempt_id or uuid.uuid4().hex
        metadata: dict[str, Any] = dict(
            attempt_id=attempt_id,
            hostname=hostname,
            engine=engine,
            model=model,
            bundle_version=bundle_version,
            operation=operation,
            started_at=timestamp(),
            stage="pending",
            status="running",
            failure_summary=None,
            sources={},
            issues=[],
            finished_at=None,
        )
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO attempts(id,hostname,created,metadata) VALUES(?,?,?,?)",
                (attempt_id, hostname, time.time(), json.dumps(metadata)),
            )
            self._prune(db)
        return attempt_id

    def get(self, attempt_id: str) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            return self._manifest(row)

    @staticmethod
    def _manifest(row: sqlite3.Row) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(row["metadata"])
        result.update(
            next_seq=row["next_seq"],
            remote_cursor=row["remote_cursor"],
            dropped_records=row["dropped"],
            retained_bytes=row["bytes"],
        )
        result["incomplete"] = bool(result["issues"] or row["dropped"])
        return result

    def history(
        self, hostname: str, *, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        self.prune()
        with self._db() as db:
            db.execute("BEGIN")
            attempts = [
                self._manifest(row)
                for row in db.execute(
                    "SELECT * FROM attempts WHERE hostname=? ORDER BY created DESC LIMIT ? OFFSET ?",
                    (hostname, limit, offset),
                )
            ]
            total = db.execute(
                "SELECT count(*) FROM attempts WHERE hostname=?", (hostname,)
            ).fetchone()[0]
            evicted = db.execute(
                "SELECT value FROM statistics WHERE name='evicted_attempts'"
            ).fetchone()[0]
        return dict(attempts=attempts, total=total, evicted_attempts=evicted)

    def attempts_metadata_by_host(
        self, hostname: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(attempt_id, metadata)`` pairs for *hostname*, newest first."""
        with self._db() as db:
            rows = db.execute(
                "SELECT id, metadata FROM attempts WHERE hostname=? ORDER BY created DESC",
                (hostname,),
            ).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

    def latest_running(self, hostname: str, exclude: str | None = None) -> str | None:
        """Return the newest running attempt id for *hostname*.

        Used by the node recorder to name the holder when the host-scoped
        mutation lock is busy.
        """
        with self._db() as db:
            row = db.execute(
                "SELECT id FROM attempts WHERE hostname=? AND id != ? "
                "AND json_extract(metadata,'$.status')='running' "
                "ORDER BY created DESC LIMIT 1",
                (hostname, exclude or ""),
            ).fetchone()
            return row["id"] if row is not None else None

    def pending_hosts(self) -> list[str]:
        """Return hostnames whose newest attempt is not terminal-evidenced.

        Reconciliation candidates: the newest attempt per host is still
        running, was interrupted, or failed without mirrored terminal remote
        phases. Relaunch operations are excluded (owned by relaunch recovery).
        """
        with self._db() as db:
            rows = db.execute(
                "SELECT hostname FROM attempts a "
                "GROUP BY hostname "
                "HAVING json_extract(("
                "SELECT metadata FROM attempts b WHERE b.hostname=a.hostname "
                "ORDER BY b.created DESC LIMIT 1),'$.status') IN "
                "('running','interrupted','failed') "
                "ORDER BY MAX(created) DESC"
            ).fetchall()
            return [row["hostname"] for row in rows]

    def update_phase(self, attempt_id: str, phase: str, **changes: Any) -> None:
        """Atomically merge a node worker's phase state with current metadata."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT metadata FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            metadata = json.loads(row[0])
            metadata.setdefault("phases", {}).setdefault(phase, {}).update(changes)
            self._update(db, attempt_id, metadata)

    def update(self, attempt_id: str, **fields: Any) -> None:
        if isinstance(fields.get("failure_summary"), str):
            fields["failure_summary"] = fields["failure_summary"][:8192]
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._update(db, attempt_id, fields)

    @staticmethod
    def _update(
        db: sqlite3.Connection, attempt_id: str, fields: dict[str, Any]
    ) -> None:
        row = db.execute(
            "SELECT metadata FROM attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        metadata = json.loads(row[0])
        changes = dict(fields)
        # Remote pages may arrive while a gateway warning is being recorded.
        # Merge warnings under the writer lock rather than replacing a stale list.
        if "issues" in changes:
            issues = metadata["issues"]
            for message in changes.pop("issues"):
                message = message[:2048]
                if message not in issues:
                    issues.append(message)
            metadata["issues"] = issues[-32:]
        metadata.update(changes)
        db.execute(
            "UPDATE attempts SET metadata=? WHERE id=?",
            (json.dumps(metadata), attempt_id),
        )

    def issue(
        self, attempt_id: str, message: str, *, source: str | None = None
    ) -> None:
        message = message[:2048]
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT metadata FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            metadata = json.loads(row[0])
            # Bounded and stable: repeat failed reconnects cannot grow metadata forever.
            if message not in metadata["issues"]:
                metadata["issues"] = (metadata["issues"] + [message])[-32:]
            if source:
                metadata["sources"][source] = "unavailable"
            self._update(db, attempt_id, metadata)

    def append(
        self,
        attempt_id: str,
        msg: str,
        *,
        level: str = "info",
        source: str = "gateway",
        stage: str | None = None,
        ts: str | None = None,
        remote_seq: int | None = None,
        stream: str | None = None,
        journal_cursor: str | None = None,
    ) -> dict[str, Any] | None:
        records = self.append_many(
            attempt_id,
            [
                dict(
                    msg=msg,
                    level=level,
                    source=source,
                    stage=stage,
                    ts=ts,
                    remote_seq=remote_seq,
                    stream=stream,
                    journal_cursor=journal_cursor,
                )
            ],
        )
        return records[0] if records else None

    def append_many(
        self, attempt_id: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Commit a bounded batch and its cursors in one transaction."""
        if not records:
            return []
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            added = []
            for record in records:
                entry = self._append(db, attempt_id, **record)
                if entry is not None:
                    added.append(entry)
            self._apply_budgets(db, attempt_id)
            return added

    def _append(
        self,
        db: sqlite3.Connection,
        attempt_id: str,
        msg: str,
        *,
        level: str = "info",
        source: str = "gateway",
        stage: str | None = None,
        ts: str | None = None,
        remote_seq: int | None = None,
        stream: str | None = None,
        journal_cursor: str | None = None,
    ) -> dict[str, Any] | None:
        row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        if remote_seq is not None and remote_seq < row["remote_cursor"]:
            return None
        metadata = json.loads(row["metadata"])
        if journal_cursor is not None:
            if metadata.get("journal_cursor") == journal_cursor:
                return None
            metadata["journal_cursor"] = journal_cursor
        if remote_seq is not None and remote_seq > row["remote_cursor"]:
            gap = f"Remote records {row['remote_cursor']}..{remote_seq - 1} unavailable (rotation or interrupted collection)"
            metadata["issues"] = (metadata["issues"] + [gap])[-32:]
        raw = msg.encode("utf-8", errors="replace")
        truncated = len(raw) > self.max_record_bytes
        if truncated:
            msg = (
                raw[: self.max_record_bytes - 16].decode("utf-8", errors="ignore")
                + " ... [truncated]"
            )
            if "Record truncated by byte limit" not in metadata["issues"]:
                metadata["issues"] = (
                    metadata["issues"] + ["Record truncated by byte limit"]
                )[-32:]
        seq = row["next_seq"]
        entry = dict(
            hostname=metadata["hostname"],
            attempt_id=attempt_id,
            engine=metadata["engine"],
            model=metadata["model"],
            bundle_version=metadata["bundle_version"],
            seq=seq,
            remote_seq=remote_seq,
            ts=ts or timestamp(),
            level=level,
            msg=msg,
            stream=stream,
            source=source,
            stage=stage or metadata["stage"],
            truncated=truncated,
        )
        payload = json.dumps(entry, ensure_ascii=False)
        size = len(payload.encode("utf-8"))
        metadata["sources"][source] = "collected"
        db.execute(
            "INSERT INTO records VALUES(?,?,?,?)", (attempt_id, seq, payload, size)
        )
        db.execute(
            "UPDATE attempts SET next_seq=?, remote_cursor=?, bytes=bytes+?, metadata=? WHERE id=?",
            (
                seq + 1,
                remote_seq + 1 if remote_seq is not None else row["remote_cursor"],
                size,
                json.dumps(metadata),
                attempt_id,
            ),
        )
        return entry

    def _apply_budgets(self, db: sqlite3.Connection, attempt_id: str) -> None:
        self._rotate(db, attempt_id, self.attempt_max_bytes)
        total = db.execute(
            "SELECT value FROM statistics WHERE name='retained_bytes'"
        ).fetchone()[0]
        if total <= self.max_bytes:
            return
        for candidate in db.execute(
            "SELECT id, bytes FROM attempts WHERE bytes>0 ORDER BY created"
        ).fetchall():
            if total <= self.max_bytes:
                break
            budget = max(0, candidate["bytes"] - (total - self.max_bytes))
            total -= self._rotate(db, candidate["id"], budget)
        return

    @staticmethod
    def _rotate(db: sqlite3.Connection, attempt_id: str, budget: int) -> int:
        size = db.execute(
            "SELECT bytes FROM attempts WHERE id=?", (attempt_id,)
        ).fetchone()[0]
        removed = 0
        count = 0
        last = -1
        if size > budget:
            for row in db.execute(
                "SELECT seq,bytes FROM records WHERE attempt=? ORDER BY seq",
                (attempt_id,),
            ):
                removed += row["bytes"]
                count += 1
                last = row["seq"]
                if size - removed <= budget:
                    break
            db.execute(
                "DELETE FROM records WHERE attempt=? AND seq<=?", (attempt_id, last)
            )
            db.execute(
                "UPDATE attempts SET bytes=bytes-?, dropped=dropped+? WHERE id=?",
                (removed, count, attempt_id),
            )
        return removed

    def _prune(self, db: sqlite3.Connection) -> None:
        cutoff = time.time() - self.retention_days * 86400
        removed = db.execute(
            "DELETE FROM attempts WHERE json_extract(metadata,'$.status') != 'running' "
            "AND (created < ? OR id IN (SELECT id FROM attempts "
            "WHERE json_extract(metadata,'$.status') != 'running' "
            "ORDER BY created DESC, id DESC LIMIT -1 OFFSET ?))",
            (cutoff, self.max_attempts),
        ).rowcount
        if removed:
            db.execute(
                "UPDATE statistics SET value=value+? WHERE name='evicted_attempts'",
                (removed,),
            )

    def prune(self) -> None:
        """Acquire a write lock only when a retention candidate exists."""
        with self._db() as db:
            cutoff = time.time() - self.retention_days * 86400
            expired = db.execute(
                "SELECT 1 FROM attempts WHERE json_extract(metadata,'$.status') != 'running' "
                "AND created < ? LIMIT 1",
                (cutoff,),
            ).fetchone()
            excess = db.execute(
                "SELECT 1 FROM attempts WHERE json_extract(metadata,'$.status') != 'running' "
                "ORDER BY created DESC, id DESC LIMIT 1 OFFSET ?",
                (self.max_attempts,),
            ).fetchone()
            if expired or excess:
                db.execute("BEGIN IMMEDIATE")
                self._prune(db)

    def read(
        self,
        attempt_id: str,
        *,
        after: int = 0,
        query: str = "",
        source: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        attempt = self.get(attempt_id)
        with self._db() as db:
            # instr is a literal substring search; '%' and '_' are not wildcards.
            rows = db.execute(
                "SELECT payload FROM records WHERE attempt=? AND seq>=? AND seq<? "
                "AND instr(lower(json_extract(payload,'$.msg')),lower(?))>0 "
                "AND (?='' OR json_extract(payload,'$.source')=?) ORDER BY seq LIMIT ?",
                (
                    attempt_id,
                    after,
                    attempt["next_seq"],
                    query,
                    source,
                    source,
                    limit + 1,
                ),
            ).fetchall()
        records = [json.loads(r[0]) for r in rows[:limit]]
        more = len(rows) > limit
        return dict(
            attempt=attempt,
            records=records,
            has_more=more,
            next_offset=records[-1]["seq"] + 1 if more else attempt["next_seq"],
        )

    def interrupt_running(self) -> None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT id,metadata FROM attempts").fetchall():
                metadata = json.loads(row["metadata"])
                if metadata["status"] == "running":
                    metadata.update(
                        status="interrupted",
                        failure_summary="Gateway stopped before attempt completion; collect remote logs to recover evidence",
                    )
                    metadata["issues"] = (
                        metadata["issues"] + ["Gateway collection interrupted"]
                    )[-32:]
                    self._update(db, row["id"], metadata)
