"""SSO whitelist: per-domain user allowlist fetched from an HTTPS JSON URL.

Document shape (per-domain, so usernames inside an allowed domain can be
filtered):

    {"example.com": ["alice", "bob"], "lab.example.com": ["carol"]}

The allowlist is a *filter*, not a revocation mechanism: tokens minted
while a user was allowed are re-checked against the cached list at use time
by ``get_api_auth`` (fail closed). Fetch policy: dedicated short-timeout
client, no redirects (the configured URL is the final URL), a 1 MiB body
cap, and an overall fetch deadline; any fetch/parse failure raises
:class:`AllowlistUnavailableError` so callers fail closed instead of
serving stale data. Failures enter a short cooldown so an outage does not
serialize a full fetch behind the lock for every request.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

import httpx
import structlog
from pydantic import TypeAdapter, ValidationError

logger = structlog.get_logger()

_MAX_ALLOWLIST_BYTES = 1_048_576  # 1 MiB
_RETRY_COOLDOWN_SECONDS = 30

_WhitelistDocument = TypeAdapter(dict[str, list[str]])


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


class SSOAllowlist:
    """Cached per-domain allowlist loaded from an HTTPS JSON URL.

    Concurrent callers share one in-flight refresh (an asyncio lock guards
    the cache). Stale data is never served: when the cached document is
    older than *refresh_seconds*, the refresh happens before answering, and
    a refresh failure raises instead of returning the previous document. A
    fetch failure is followed by a short cooldown during which checks fail
    immediately without touching the network.
    """

    def __init__(
        self,
        url: str,
        refresh_seconds: int,
        client: httpx.AsyncClient,
    ) -> None:
        self._url = url
        self._refresh_seconds = refresh_seconds
        self._client = client
        self._lock = asyncio.Lock()
        self._domains: dict[str, frozenset[str]] = {}
        self._loaded_at = 0.0
        self._retry_after = 0.0

    async def is_allowed(self, email: str) -> bool:
        """Return True when *email*'s username is listed for its domain."""
        domain = email_domain(email)
        if not domain:
            return False
        username = email.rsplit("@", 1)[0].strip().lower()
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
                    self._fetch(),
                    timeout=min(self._refresh_seconds, _RETRY_COOLDOWN_SECONDS),
                )
            except TimeoutError:
                self._retry_after = time.monotonic() + _RETRY_COOLDOWN_SECONDS
                raise AllowlistUnavailableError("allowlist fetch timed out") from None
            except AllowlistUnavailableError:
                self._retry_after = time.monotonic() + _RETRY_COOLDOWN_SECONDS
                raise
            self._loaded_at = time.monotonic()
            return self._domains

    def _fresh(self) -> bool:
        return self._loaded_at > 0 and (
            time.monotonic() - self._loaded_at < self._refresh_seconds
        )

    def _reject_during_cooldown(self) -> None:
        if time.monotonic() < self._retry_after:
            raise AllowlistUnavailableError("allowlist temporarily unavailable")

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
            document = _WhitelistDocument.validate_python(raw)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            TypeError,
        ) as exc:
            raise AllowlistUnavailableError(
                "allowlist document is invalid JSON"
            ) from exc
        return {
            domain.strip().lower(): frozenset(
                username.strip().lower() for username in users if username.strip()
            )
            for domain, users in document.items()
            if domain.strip()
        }
