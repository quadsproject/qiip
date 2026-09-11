"""Unit tests for the SSO whitelist service (per-domain JSON allowlist)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_httpx import HTTPXMock

from inference_proxy.auth.allowlist import (
    AllowlistUnavailableError,
    SSOAllowlist,
    email_domain,
)

_URL = "https://allowlist.example.com/users.json"
_DOCUMENT = {"example.com": ["alice", "bob"]}


@pytest.fixture
async def allowlist() -> AsyncIterator[SSOAllowlist]:
    """Yield an allowlist backed by a real client; pytest_httpx intercepts."""
    client = httpx.AsyncClient(follow_redirects=False)
    yield SSOAllowlist(_URL, "hourly", None, client)
    await client.aclose()


@pytest.fixture
async def allowlist_with_domain() -> AsyncIterator[SSOAllowlist]:
    """Yield an allowlist with a default domain for flat username lists."""
    client = httpx.AsyncClient(follow_redirects=False)
    yield SSOAllowlist(_URL, "hourly", None, client, default_domain="example.com")
    await client.aclose()


class TestEmailDomain:
    def test_domain_is_lowercased(self) -> None:
        assert email_domain("Alice@Example.COM") == "example.com"
        assert email_domain("no-at-sign") == ""


class TestSSOAllowlist:
    async def test_allowed_user(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)

        assert await allowlist.is_allowed("alice@example.com") is True

    async def test_domain_and_user_matched_case_insensitively(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json={"Example.COM": ["  Alice  "]})

        assert await allowlist.is_allowed("ALICE@example.com") is True

    async def test_unknown_user_denied(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)

        assert await allowlist.is_allowed("carol@example.com") is False

    async def test_unknown_domain_denied(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)

        assert await allowlist.is_allowed("alice@other.com") is False

    async def test_malformed_email_denied(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        assert await allowlist.is_allowed("not-an-email") is False

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            '["alice"]',
            '{"example.com": "alice"}',
            '{"example.com": [1, 2]}',
        ],
    )
    async def test_invalid_document_fails_closed(
        self,
        allowlist: SSOAllowlist,
        httpx_mock: HTTPXMock,
        payload: str,
    ) -> None:
        httpx_mock.add_response(url=_URL, text=payload)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_http_error_fails_closed(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, status_code=500)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_oversized_document_fails_closed(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=_URL,
            text='{"example.com": ["' + "x" * 1_048_600 + '"]}',
        )

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_invalid_utf8_fails_closed(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, content=b"\xff\xfe")

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")


class TestSSOAllowlistCache:
    async def test_cache_used_within_window(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)

        assert await allowlist.is_allowed("alice@example.com") is True
        assert await allowlist.is_allowed("bob@example.com") is True
        assert len(httpx_mock.get_requests()) == 1

    async def test_empty_document_is_cached(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json={})

        assert await allowlist.is_allowed("alice@example.com") is False
        assert await allowlist.is_allowed("bob@example.com") is False
        assert len(httpx_mock.get_requests()) == 1

    async def test_stale_cache_refreshes(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)
        httpx_mock.add_response(
            url=_URL,
            json={"example.com": ["alice", "bob", "carol"]},
        )
        assert await allowlist.is_allowed("carol@example.com") is False
        allowlist._next_refresh = time.time() - 1  # force stale
        assert await allowlist.is_allowed("carol@example.com") is True
        assert len(httpx_mock.get_requests()) == 2

    async def test_stale_cache_failure_does_not_serve_stale_data(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)
        assert await allowlist.is_allowed("alice@example.com") is True
        allowlist._next_refresh = time.time() - 1  # force stale
        httpx_mock.add_response(url=_URL, status_code=503)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_failure_cooldown_avoids_retry_storm(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, status_code=500)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")
        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("bob@example.com")
        assert len(httpx_mock.get_requests()) == 1


class TestSSOAllowlistSchedule:
    def test_hourly_boundary_rolls_to_next_hour(self) -> None:
        allowlist = SSOAllowlist(_URL, "hourly", None, MagicMock())
        now = datetime.now().replace(minute=30, second=0, microsecond=0).timestamp()
        expected = datetime.fromtimestamp(now).replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=1)

        assert allowlist._next_refresh_at(now) == expected.timestamp()

    def test_daily_boundary_rolls_to_tomorrow_when_passed(self) -> None:
        now = datetime.now().replace(hour=23, minute=30, second=0, microsecond=0)
        allowlist = SSOAllowlist(_URL, "daily", "05:00", MagicMock())
        expected = now.replace(hour=5, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )

        assert allowlist._next_refresh_at(now.timestamp()) == expected.timestamp()

    def test_daily_boundary_stays_today_when_future(self) -> None:
        now = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        allowlist = SSOAllowlist(_URL, "daily", "05:00", MagicMock())
        expected = now.replace(hour=5, minute=0, second=0, microsecond=0)

        assert allowlist._next_refresh_at(now.timestamp()) == expected.timestamp()


class TestSSOAllowlistCacheFile:
    async def test_seed_loaded_without_network(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        cache = tmp_path / "whitelist.json"
        cache.write_text('{"example.com": ["alice"]}', encoding="utf-8")
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(_URL, "hourly", None, client, cache_file=cache)

        assert await allowlist.is_allowed("alice@example.com") is True
        assert len(httpx_mock.get_requests()) == 0
        await client.aclose()

    async def test_refresh_overwrites_cache_file(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        cache = tmp_path / "whitelist.json"
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(_URL, "hourly", None, client, cache_file=cache)
        httpx_mock.add_response(url=_URL, json={"example.com": ["alice", "bob"]})

        assert await allowlist.is_allowed("bob@example.com") is True
        assert cache.exists()
        assert json.loads(cache.read_text(encoding="utf-8")) == {
            "example.com": ["alice", "bob"]
        }
        await client.aclose()

    async def test_stale_seed_triggers_immediate_refetch(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        cache = tmp_path / "whitelist.json"
        cache.write_text('{"example.com": ["alice"]}', encoding="utf-8")
        old = time.time() - 72 * 3600
        os.utime(cache, (old, old))
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(_URL, "daily", "05:00", client, cache_file=cache)
        httpx_mock.add_response(url=_URL, json={"example.com": ["alice", "bob"]})

        assert await allowlist.is_allowed("alice@example.com") is True
        assert len(httpx_mock.get_requests()) == 1
        await client.aclose()

    async def test_corrupt_seed_ignored(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        cache = tmp_path / "whitelist.json"
        cache.write_text("not json", encoding="utf-8")
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(_URL, "hourly", None, client, cache_file=cache)
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)

        assert await allowlist.is_allowed("alice@example.com") is True
        assert len(httpx_mock.get_requests()) == 1
        await client.aclose()


class TestSSOAllowlistExtras:
    async def test_extra_user_allowed_without_document(
        self, httpx_mock: HTTPXMock
    ) -> None:
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(
            _URL,
            "hourly",
            None,
            client,
            extra_users=("carol@example.com", " DAVE@Other.com "),
        )

        assert await allowlist.is_allowed("carol@example.com") is True
        assert await allowlist.is_allowed("dave@other.com") is True
        assert len(httpx_mock.get_requests()) == 0
        await client.aclose()

    async def test_extra_domain_catches_all_users(self, httpx_mock: HTTPXMock) -> None:
        client = httpx.AsyncClient(follow_redirects=False)
        allowlist = SSOAllowlist(
            _URL, "hourly", None, client, extra_domains=("Lab.example.com",)
        )

        assert await allowlist.is_allowed("anyone@lab.example.com") is True
        assert len(httpx_mock.get_requests()) == 0
        await client.aclose()


class TestFlatUsernameList:
    """Flat username JSON with sso_whitelist_default_domain (RFE)."""

    async def test_empty_flat_list_denies_all(
        self,
        allowlist_with_domain: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json=[])

        assert await allowlist_with_domain.is_allowed("alice@example.com") is False

    async def test_bare_usernames_resolve_to_default_domain(
        self,
        allowlist_with_domain: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json=["alice", "  Bob  "])

        assert await allowlist_with_domain.is_allowed("alice@example.com") is True
        assert await allowlist_with_domain.is_allowed("BOB@example.com") is True
        assert await allowlist_with_domain.is_allowed("carol@example.com") is False
        assert await allowlist_with_domain.is_allowed("alice@other.com") is False

    async def test_full_emails_in_flat_list_used_as_is(
        self,
        allowlist_with_domain: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json=["alice", "bob@lab.example.com"])

        assert await allowlist_with_domain.is_allowed("alice@example.com") is True
        assert await allowlist_with_domain.is_allowed("bob@lab.example.com") is True
        assert await allowlist_with_domain.is_allowed("bob@example.com") is False

    async def test_flat_list_without_default_domain_fails_closed(
        self,
        allowlist: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json=["alice"])

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_malformed_entry_fails_closed(
        self,
        allowlist_with_domain: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json=["@nope"])

        with pytest.raises(AllowlistUnavailableError):
            await allowlist_with_domain.is_allowed("alice@example.com")

    async def test_per_domain_document_still_supported_with_domain_set(
        self,
        allowlist_with_domain: SSOAllowlist,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(url=_URL, json={"other.com": ["carol"]})

        assert await allowlist_with_domain.is_allowed("carol@other.com") is True
        assert await allowlist_with_domain.is_allowed("alice@example.com") is False
