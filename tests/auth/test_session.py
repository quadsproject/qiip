"""Tests for the signed-cookie session helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from starlette.requests import Request

from inference_proxy.auth.session import (
    clear_session_user,
    get_session_user_id,
    set_session_user,
)


class _SessionRequest:
    """Minimal request stand-in exposing a mutable session mapping."""

    def __init__(self, session: dict[str, object] | None = None) -> None:
        self.session: dict[str, object] = {} if session is None else dict(session)


def _as_request(request: _SessionRequest) -> Request:
    return cast(Request, request)


class TestSessionHelpers:
    def test_set_and_read_roundtrip(self) -> None:
        request = _SessionRequest()

        set_session_user(_as_request(request), 42, ttl_seconds=3600)

        assert get_session_user_id(_as_request(request)) == 42

    def test_expired_session_returns_none_and_clears(self) -> None:
        request = _SessionRequest()
        set_session_user(_as_request(request), 42, ttl_seconds=-100)

        assert get_session_user_id(_as_request(request)) is None
        assert "qiip_user_id" not in request.session
        assert "qiip_exp" not in request.session

    def test_missing_session_scope_returns_none(self) -> None:
        request = SimpleNamespace()  # no .session at all

        assert get_session_user_id(cast(Request, request)) is None

    def test_non_int_payload_returns_none(self) -> None:
        request = _SessionRequest({"qiip_user_id": "not-an-int", "qiip_exp": 1_000_000})

        assert get_session_user_id(_as_request(request)) is None

    def test_clear_removes_identity(self) -> None:
        request = _SessionRequest()
        set_session_user(_as_request(request), 7, ttl_seconds=3600)

        clear_session_user(_as_request(request))

        assert get_session_user_id(_as_request(request)) is None
        assert request.session == {}
