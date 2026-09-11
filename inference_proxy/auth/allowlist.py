"""SSO whitelist: per-domain user allowlist fetched from an HTTPS JSON URL.

Document shape (per-domain, so usernames inside an allowed domain can be
filtered):

    {"example.com": ["alice", "bob"], "lab.example.com": ["carol"]}

or, when ``sso_whitelist_default_domain`` is configured, a flat list of
usernames:

    ["alice", "bob", "carol"]

Bare entries in a flat list resolve to ``<username>@<default_domain>``;
full email entries are used as-is. A flat list without a default domain
fails closed (unresolvable usernames are never silently ignored).

The allowlist is a *filter*, not a revocation mechanism: tokens minted
while a user was allowed are re-checked against the cache at use time by
``get_api_auth`` (fail closed).

Caching: the fetched document lives in an in-memory cache with a
schedule-based freshness window (``poll_interval`` hourly, or daily at a
configured wall-clock ``poll_time``); the first check after the window
refreshes it, and stale data is never served past the window. Failures
enter a short cooldown so an outage does not serialize a full fetch behind
the lock for every request. An optional flat-file cache (``cache_file``)
persists the last successful document for operator inspection and warm
start; the file is only authoritative within the freshness window and is
atomically replaced on refresh.

Local overrides (``extra_users`` emails, ``extra_domains`` catch-all) are
merged on top of the fetched document so specific accounts or domains can
be granted without touching the remote payload.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import structlog
from pydantic import TypeAdapter, ValidationError

logger = structlog.get_logger()

_MAX_ALLOWLIST_BYTES = 1_048_576  # 1 MiB
_RETRY_COOLDOWN_SECONDS = 30
_FETCH_DEADLINE_SECONDS = 30
_POLL_TIME_PATTERN = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")

_WhitelistDocument: TypeAdapter[dict[str, list[str]] | list[str]] = TypeAdapter(
    dict[str, list[str]] | list[str]
)


class AllowlistUnavailableError(Exception):
    """Raised when the whitelist document cannot be fetched or parsed."""


class AllowlistProtocol(Protocol):
    """The minimal allowlist contract used by auth dependencies (testable)."""

    async def is_allowed(self, email: str) -> bool: ...


async def enforce_allowlist(email: str, allowlist: AllowlistProtocol | None) -> bool:
    """Return True when *email* is allowed; raise when the allowlist is not.

    An unconfigured allowlist (``None``) and an unavailable one both raise
    :class:`AllowlistUnavailableError` so call sites fail closed uniformly.
    """
    if allowlist is None:
        raise AllowlistUnavailableError("whitelist is not configured")
    return await allowlist.is_allowed(email)


def email_domain(email: str) -> str:
    """Return the lowercase domain of *email*, or ``""`` when absent."""
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().lower()


def _normalize_flat_entries(
    entries: list[str], default_domain: str | None
) -> dict[str, frozenset[str]]:
    """Resolve a flat username/email list into the per-domain map.

    Bare usernames require *default_domain*; without one the document is
    commented out as unresolvable and the allowlist fails closed.
    """
    domains: dict[str, set[str]] = {}
    for entry in entries:
        name = entry.strip().lower()
        if not name:
            continue
        if "@" in name:
            username, domain = name.rsplit("@", 1)
            if not domain or not username:
                raise AllowlistUnavailableError(
                    "flat list contains malformed email entries"
                )
            domains.setdefault(domain, set()).add(username)
        elif default_domain:
            domains.setdefault(default_domain, set()).add(name)
        else:
            raise AllowlistUnavailableError(
                "flat username list requires sso_whitelist_default_domain"
            )
    return {domain: frozenset(users) for domain, users in domains.items()}


def _normalize_document(
    raw: object, default_domain: str | None = None
) -> dict[str, frozenset[str]]:
    """Validate a raw JSON document and normalize domain/usernames."""
    document = _WhitelistDocument.validate_python(raw)
    if isinstance(document, list):
        return _normalize_flat_entries(document, default_domain)
    return {
        domain.strip().lower(): frozenset(
            username.strip().lower() for username in users if username.strip()
        )
        for domain, users in document.items()
        if domain.strip()
    }


class SSOAllowlist:
    """Cached per-domain allowlist with schedule-based refresh.

    *refresh_interval* is ``"hourly"`` (top of the next hour) or
    ``"daily"`` with *poll_time* ``"HH:MM"`` in the server's local time.
    Concurrent callers share one in-flight refresh (an asyncio lock guards
    the cache). Stale data is never served past the window: the refresh
    happens before answering, and a refresh failure raises instead of
    returning the previous document.
    """

    def __init__(
        self,
        url: str,
        poll_interval: str,
        poll_time: str | None,
        client: httpx.AsyncClient,
        *,
        cache_file: Path | None = None,
        extra_users: tuple[str, ...] = (),
        extra_domains: tuple[str, ...] = (),
        default_domain: str | None = None,
    ) -> None:
        self._url = url
        self._poll_interval = poll_interval
        self._poll_time = poll_time
        self._client = client
        self._cache_file = cache_file
        self._default_domain = (default_domain or "").strip().lower() or None
        self._lock = asyncio.Lock()
        self._domains: dict[str, frozenset[str]] = {}
        self._extra_users = frozenset(
            user.strip().lower() for user in extra_users if user.strip()
        )
        self._extra_domains = frozenset(
            domain.strip().lower() for domain in extra_domains if domain.strip()
        )
        self._retry_after = 0.0
        self._next_refresh = self._next_refresh_at(time.time())
        self._loaded = False
        self._load_seed()

    async def is_allowed(self, email: str) -> bool:
        """Return True when *email* is listed, overridden, or in an allowed domain."""
        normalized = email.strip().lower()
        if normalized in self._extra_users:
            return True
        domain = email_domain(normalized)
        if not domain:
            return False
        if domain in self._extra_domains:
            return True
        username = normalized.rsplit("@", 1)[0]
        users = (await self._current()).get(domain)
        return users is not None and username in users

    async def _current(self) -> dict[str, frozenset[str]]:
        if self._fresh():
            return self._domains
        self._reject_during_cooldown()
        async with self._lock:
            if self._fresh():
                return self._domains
            self._reject_during_cooldown()
            try:
                self._domains = await asyncio.wait_for(
                    self._fetch(), timeout=_FETCH_DEADLINE_SECONDS
                )
            except TimeoutError:
                self._retry_after = time.monotonic() + _RETRY_COOLDOWN_SECONDS
                raise AllowlistUnavailableError("allowlist fetch timed out") from None
            except AllowlistUnavailableError:
                self._retry_after = time.monotonic() + _RETRY_COOLDOWN_SECONDS
                raise
            self._loaded = True
            self._next_refresh = self._next_refresh_at(time.time())
            self._write_cache()
            return self._domains

    def _fresh(self) -> bool:
        return self._loaded and time.time() < self._next_refresh

    def _reject_during_cooldown(self) -> None:
        if time.monotonic() < self._retry_after:
            raise AllowlistUnavailableError("allowlist temporarily unavailable")

    def _next_refresh_at(self, now: float) -> float:
        """Return the earliest wall-clock time the document may be refreshed."""
        current = datetime.fromtimestamp(now)
        if self._poll_interval == "daily":
            match = _POLL_TIME_PATTERN.fullmatch(self._poll_time or "")
            if match is None:  # pragma: no cover - settings validate this
                return current.timestamp() + 86_400
            hour, minute = int(match.group(1)), int(match.group(2))
            boundary = current.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if boundary <= current:
                boundary += timedelta(days=1)
        else:
            boundary = current.replace(minute=0, second=0, microsecond=0)
            if boundary <= current:
                boundary += timedelta(hours=1)
        return boundary.timestamp()

    async def _fetch(self) -> dict[str, frozenset[str]]:
        try:
            async with self._client.stream("GET", self._url) as response:
                if response.status_code != 200:
                    raise AllowlistUnavailableError(
                        f"allowlist returned HTTP {response.status_code}"
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_ALLOWLIST_BYTES:
                        raise AllowlistUnavailableError("allowlist document too large")
        except AllowlistUnavailableError:
            raise
        except httpx.HTTPError:
            raise AllowlistUnavailableError("allowlist fetch failed") from None
        try:
            raw: Any = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AllowlistUnavailableError(
                "allowlist document is invalid JSON"
            ) from exc
        try:
            return _normalize_document(raw, self._default_domain)
        except (ValidationError, TypeError) as exc:
            raise AllowlistUnavailableError(
                "allowlist document is invalid JSON"
            ) from exc

    def _load_seed(self) -> None:
        """Load the flat-file cache as a cold-start seed (never overrides a fetch).

        The seed is only served inside its own freshness window: the next
        refresh is derived from the file's mtime (clamped to now), so an
        aged file triggers an immediate refetch instead of being served as
        fresh for a full window after a restart.
        """
        if self._cache_file is None or not self._cache_file.is_file():
            return
        try:
            mtime = self._cache_file.stat().st_mtime
            raw = json.loads(self._cache_file.read_text(encoding="utf-8"))
            self._domains = _normalize_document(raw)
            self._loaded = True
            self._next_refresh = max(self._next_refresh_at(mtime), time.time())
            logger.info(
                "sso whitelist cache file loaded",
                path=str(self._cache_file),
                age_seconds=round(time.time() - mtime),
            )
        except (OSError, ValueError, TypeError):
            logger.warning(
                "sso whitelist cache file unreadable; ignoring",
                path=str(self._cache_file),
                exc_info=True,
            )

    def _write_cache(self) -> None:
        """Atomically persist the last successful document for warm start."""
        if self._cache_file is None:
            return
        payload = json.dumps(
            {domain: sorted(users) for domain, users in sorted(self._domains.items())}
        )
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_file.with_name(f"{self._cache_file.name}.tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self._cache_file)
        except OSError:
            logger.warning(
                "sso whitelist cache file write failed",
                path=str(self._cache_file),
                exc_info=True,
            )
