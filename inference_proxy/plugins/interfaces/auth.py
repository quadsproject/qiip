"""Interface for SSO identity-provider plugins."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import RedirectResponse

from inference_proxy.auth._constants import GOOGLE_ISSUER
from inference_proxy.plugins.base import BasePlugin

_CALLBACK_CODE_PATTERN = re.compile(r"[a-z_]{1,32}")


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    """Verified identity claims returned by an SSO provider.

    ``issuer`` is the OIDC ``iss`` of the provider that issued the subject
    claim; user rows are keyed by (issuer, sub) so two providers can issue
    the same opaque ``sub`` without colliding. It defaults to the Google
    issuer so an external plugin built before the field existed keeps
    producing Google keyed rows (no TypeError, review #231).
    """

    sub: str
    email: str
    email_verified: bool
    issuer: str = GOOGLE_ISSUER
    name: str = ""
    picture: str = ""


class AuthCallbackError(Exception):
    """Provider callback failure carrying a short redirect error code.

    The code must be a safe lowercase token because it is embedded into the
    profile redirect URL; rejecting invalid codes keeps a future provider's
    error strings out of the Location header and logs.
    """

    def __init__(self, code: str) -> None:
        if _CALLBACK_CODE_PATTERN.fullmatch(code) is None:
            raise ValueError(f"invalid auth callback error code: {code!r}")
        super().__init__(code)
        self.code = code


class AuthPlugin(BasePlugin, ABC):
    """Abstract SSO identity provider plugin.

    The plugin owns the provider round trip (authorization redirect and
    code exchange) and returns a normalized :class:`AuthIdentity`; generic
    policy (email verification, hosted-domain allowlist, session, user
    upsert) stays in the auth router so it is provider-neutral.

    ``label`` is the human-readable provider name shown on the sign-in
    page (e.g. "Google Auth"); ``name`` stays the plugin's stable
    identifier and is used in the ``/auth/login?provider=`` link.
    """

    label: str = ""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when the provider has full credentials."""

    @abstractmethod
    async def start_login(
        self, request: Request, redirect_uri: str
    ) -> RedirectResponse:
        """Begin the provider authorization flow."""

    @abstractmethod
    async def complete_login(self, request: Request) -> AuthIdentity:
        """Exchange the provider callback for a verified identity.

        Raises ``AuthCallbackError`` with a short error code when the
        provider rejects or fails the exchange.
        """

    def verify_local_credentials(
        self, username: str, password: str, admin_username: str
    ) -> str | None:
        """Return the canonical local-account email for valid local credentials.

        Providers that manage a local user list (e.g. ``auth.internal_oidc``)
        check ``username``/``password`` against it; providers without a local
        list (Google) return ``None``. Entries matching ``admin_username`` are
        never local accounts: qiip's configured admin password wins, so the
        auth router checks the admin credentials before calling this.
        """
        return None
