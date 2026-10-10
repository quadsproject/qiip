"""Unit tests for the shared sign-in policy gates."""

from __future__ import annotations

from inference_proxy.auth._constants import GOOGLE_ISSUER
from inference_proxy.auth.allowlist import AllowlistUnavailableError
from inference_proxy.auth.signin_policy import check_signin_policy
from inference_proxy.config.settings import Settings

_LOCAL_ISSUER = "https://inference-proxy.localdomain/oidc"


class _FakeAllowlist:
    def __init__(self, allowed: bool, *, unavailable: bool = False) -> None:
        self._allowed = allowed
        self._unavailable = unavailable

    async def is_allowed(self, email: str) -> bool:
        if self._unavailable:
            raise AllowlistUnavailableError("fetch failed")
        return self._allowed


def _with(
    test_settings: Settings,
    **auth: object,
) -> Settings:
    # The root ``test_settings`` fixture inherits the developer's ``.env``
    # (Settings is a BaseSettings with ``env_file=".env"``), so pin the
    # default policy surface explicitly: empty ``allowed_domains`` means the
    # documented "any Google account is accepted" behavior (Settings docs),
    # independent of the local clone's OAuth env.
    return test_settings.model_copy(
        deep=True,
        update={
            "auth": test_settings.auth.model_copy(update=auth),
            "oauth": test_settings.oauth.model_copy(update={"allowed_domains": []}),
        },
    )


async def test_unverified_email_rejected_by_default(
    test_settings: Settings,
) -> None:
    code = await check_signin_policy(
        "alice@example.com", GOOGLE_ISSUER, False, test_settings, None
    )
    assert code == "unverified_email"


async def test_verified_email_passes_default_policy(
    test_settings: Settings,
) -> None:
    code = await check_signin_policy(
        "alice@example.com", GOOGLE_ISSUER, True, _with(test_settings), None
    )
    assert code is None


async def test_domain_gate_uses_oauth_allowed_domains(
    test_settings: Settings,
) -> None:
    settings = test_settings.model_copy(
        deep=True,
        update={
            "oauth": test_settings.oauth.model_copy(
                update={"allowed_domains": ["corp.example.com"]}
            )
        },
    )
    assert (
        await check_signin_policy(
            "alice@corp.example.com", GOOGLE_ISSUER, True, settings, None
        )
        is None
    )
    assert (
        await check_signin_policy(
            "alice@elsewhere.com", GOOGLE_ISSUER, True, settings, None
        )
        == "domain_not_allowed"
    )


async def test_whitelist_denies_non_admin(test_settings: Settings) -> None:
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
    )
    code = await check_signin_policy(
        "alice@example.com",
        GOOGLE_ISSUER,
        True,
        settings,
        _FakeAllowlist(allowed=False),
    )
    assert code == "not_whitelisted"


async def test_whitelist_allows_member(test_settings: Settings) -> None:
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
    )
    code = await check_signin_policy(
        "alice@example.com",
        GOOGLE_ISSUER,
        True,
        settings,
        _FakeAllowlist(allowed=True),
    )
    assert code is None


async def test_full_access_bypasses_whitelist(test_settings: Settings) -> None:
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
        admin_only_tokens_full_access=["alice@example.com"],
    )
    code = await check_signin_policy(
        "alice@example.com", GOOGLE_ISSUER, True, settings, None
    )
    assert code is None


async def test_full_access_google_list_does_not_bypass_for_local_issuer(
    test_settings: Settings,
) -> None:
    """A local identity asserting a Google trust-list email still faces the
    whitelist (review #231): grants are keyed by (email, issuer)."""
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
        admin_only_tokens_full_access=["alice@example.com"],
    )
    code = await check_signin_policy(
        "alice@example.com",
        _LOCAL_ISSUER,
        True,
        settings,
        _FakeAllowlist(allowed=False),
    )
    assert code == "not_whitelisted"


async def test_full_access_by_issuer_bypasses_whitelist(
    test_settings: Settings,
) -> None:
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
        admin_only_tokens_full_access_by_issuer={_LOCAL_ISSUER: ["alice@example.com"]},
    )
    code = await check_signin_policy(
        "alice@example.com", _LOCAL_ISSUER, True, settings, None
    )
    assert code is None


async def test_whitelist_unavailable_fails_closed(test_settings: Settings) -> None:
    settings = _with(
        test_settings,
        enforce_sso_whitelist=True,
        sso_whitelist_url="https://allowlist.example.com/users.json",
    )
    code = await check_signin_policy(
        "alice@example.com",
        GOOGLE_ISSUER,
        True,
        settings,
        _FakeAllowlist(allowed=True, unavailable=True),
    )
    assert code == "allowlist_unavailable"
