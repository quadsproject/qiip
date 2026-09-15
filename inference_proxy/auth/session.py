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
_SESSION_LOCAL_ADMIN_KEY = "qiip_local_admin"
_SESSION_EXPIRY_KEY = "qiip_exp"


def set_session_user(request: Request, user_id: int, ttl_seconds: int) -> None:
    """Sign *user_id* into the request session for *ttl_seconds*.

    The identity is exclusive: signing in a Google user clears any local
    admin marker, so a single session is never both (which would otherwise
    elevate a non-admin Google user to the admin surface).
    """
    clear_local_admin_session(request)
    request.session[_SESSION_USER_KEY] = user_id
    request.session[_SESSION_EXPIRY_KEY] = int(time.time()) + ttl_seconds


def clear_session_user(request: Request) -> None:
    """Drop the user identity and expiry from the session cookie."""
    request.session.pop(_SESSION_USER_KEY, None)
    request.session.pop(_SESSION_EXPIRY_KEY, None)


def set_local_admin_session(request: Request, ttl_seconds: int) -> None:
    """Mark the request session as the local admin for *ttl_seconds*.

    The signed-in identity lives in the session cookie, so the admin stays
    signed in across navigations without relying on the browser caching
    HTTP Basic credentials (which the previous challenge flow depended on).
    The identity is exclusive: local admin sign-in clears any Google user
    marker.
    """
    clear_session_user(request)
    request.session[_SESSION_LOCAL_ADMIN_KEY] = True
    request.session[_SESSION_EXPIRY_KEY] = int(time.time()) + ttl_seconds


def clear_local_admin_session(request: Request) -> None:
    """Drop the local-admin marker from the session cookie."""
    request.session.pop(_SESSION_LOCAL_ADMIN_KEY, None)
    request.session.pop(_SESSION_EXPIRY_KEY, None)


def _session_state(request: Request, key: str) -> tuple[object, object] | None:
    """Return ``(marker, expiry)`` for *key*, or None when session middleware
    is not installed (no ``SessionMiddleware`` on the app; AUTH-02)."""
    try:
        return (request.session.get(key), request.session.get(_SESSION_EXPIRY_KEY))
    except AttributeError:
        return None


def get_local_admin_session(request: Request) -> bool:
    """Return True when the session was established by the local admin login.

    Applies the shared expiry; an expired marker is ignored (and cleared so
    the cookie self-heals). Returns False when the session middleware is not
    installed.
    """
    state = _session_state(request, _SESSION_LOCAL_ADMIN_KEY)
    if state is None:
        return False
    marker, expiry = state
    if marker is not True:
        return False
    if not isinstance(expiry, int) or expiry < time.time():
        clear_local_admin_session(request)
        return False
    return True


def get_session_user_id(request: Request) -> int | None:
    """Return the signed-in user id, applying the stored expiry.

    Returns None when the session middleware is not installed, the cookie
    carries no valid user id, or the session has expired (the expired
    identity is cleared so the cookie self-heals on the next request).
    """
    state = _session_state(request, _SESSION_USER_KEY)
    if state is None:
        return None
    user_id, expiry = state
    if not isinstance(user_id, int) or not isinstance(expiry, int):
        return None
    if expiry < time.time():
        clear_session_user(request)
        return None
    return user_id
