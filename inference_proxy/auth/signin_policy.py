"""Shared sign-in policy gates for every authentication path.

The OIDC callback and the local sign-in form both funnel through
``check_signin_policy`` so a local account cannot bypass the same gates an
OIDC login must pass: the email-verification requirement, the OAuth
``allowed_domains`` list, and the optional SSO allowlist (the admin
full-access trust list short-circuits the whitelist). Policy failures
return a short error code the caller surfaces; the allowlist-fetch failure
is folded in too (fail closed with a single code).
"""

from __future__ import annotations

from inference_proxy.auth.allowlist import (
    AllowlistProtocol,
    AllowlistUnavailableError,
    email_domain,
    enforce_allowlist,
)
from inference_proxy.auth.scopes import is_full_access
from inference_proxy.config.settings import Settings


async def check_signin_policy(
    email: str,
    issuer: str,
    email_verified: bool,
    settings: Settings,
    allowlist: AllowlistProtocol | None,
) -> str | None:
    """Return the first failing policy code, or None when the email passes.

    Codes: ``unverified_email``, ``domain_not_allowed``,
    ``allowlist_unavailable``, ``not_whitelisted``. The admin full-access
    list bypasses the whitelist gate (never the domain gate); the grant is
    keyed by (email, issuer) so a local identity cannot claim a Google
    trust-list email (review #231).
    """
    if settings.auth.require_email_verification and not email_verified:
        return "unverified_email"
    allowed_domains = [domain.lower() for domain in settings.oauth.allowed_domains]
    if allowed_domains and email_domain(email) not in allowed_domains:
        return "domain_not_allowed"
    if settings.auth.enforce_sso_whitelist and not is_full_access(
        email, issuer, settings
    ):
        try:
            allowed = await enforce_allowlist(email, allowlist)
        except AllowlistUnavailableError:
            return "allowlist_unavailable"
        if not allowed:
            return "not_whitelisted"
    return None
