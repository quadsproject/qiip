"""Signed-cookie session helpers for the browser UI.

Sessions ride on Starlette's ``SessionMiddleware`` (an itsdangerous-signed,
client-side cookie). Only the user id and an expiry timestamp live in the
cookie; the authoritative identity is re-read from the SQLite store on every
request so a stale cookie can never resurrect a deleted user.

The cookie is signed with ``auth.session_secret``; without that secret the
session middleware is not installed and every session read here returns
None (AUTH-02).
"""

from __future__ import annotations

import time

from starlette.requests import Request

_SESSION_USER_KEY = "qiip_user_id"
_SESSION_EXPIRY_KEY = "qiip_exp"


def set_session_user(request: Request, user_id: int, ttl_seconds: int) -> None:
    """Sign *user_id* into the request session for *ttl_seconds*."""
    request.session[_SESSION_USER_KEY] = user_id
    request.session[_SESSION_EXPIRY_KEY] = int(time.time()) + ttl_seconds


def clear_session_user(request: Request) -> None:
    """Drop the user identity and expiry from the session cookie."""
    request.session.pop(_SESSION_USER_KEY, None)
    request.session.pop(_SESSION_EXPIRY_KEY, None)


def get_session_user_id(request: Request) -> int | None:
    """Return the signed-in user id, applying the stored expiry.

    Returns None when the session middleware is not installed, the cookie
    carries no valid user id, or the session has expired (the expired
    identity is cleared so the cookie self-heals on the next request).
    """
    try:
        user_id = request.session.get(_SESSION_USER_KEY)
        expiry = request.session.get(_SESSION_EXPIRY_KEY)
    except AttributeError:
        return None
    if not isinstance(user_id, int) or not isinstance(expiry, int):
        return None
    if expiry < time.time():
        clear_session_user(request)
        return None
    return user_id
