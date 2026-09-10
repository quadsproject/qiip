"""Unit tests for the SSO whitelist service (per-domain JSON allowlist)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

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
    yield SSOAllowlist(_URL, 300, client)
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
            "[]",
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

    async def test_cache_used_within_ttl(
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

    async def test_failure_cooldown_avoids_retry_storm(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, status_code=500)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")
        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("bob@example.com")
        assert len(httpx_mock.get_requests()) == 1

    async def test_invalid_utf8_fails_closed(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, content=b"\xff\xfe")

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")

    async def test_stale_cache_refreshes(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)
        httpx_mock.add_response(
            url=_URL,
            json={"example.com": ["alice", "bob", "carol"]},
        )
        assert await allowlist.is_allowed("carol@example.com") is False
        allowlist._loaded_at = time.monotonic() - 301  # force stale
        assert await allowlist.is_allowed("carol@example.com") is True
        assert len(httpx_mock.get_requests()) == 2

    async def test_stale_cache_failure_does_not_serve_stale_data(
        self, allowlist: SSOAllowlist, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=_URL, json=_DOCUMENT)
        assert await allowlist.is_allowed("alice@example.com") is True
        allowlist._loaded_at = time.monotonic() - 301  # force stale
        httpx_mock.add_response(url=_URL, status_code=503)

        with pytest.raises(AllowlistUnavailableError):
            await allowlist.is_allowed("alice@example.com")
