"""Local OpenID Connect auth plugin (opt-in, test/small-install oriented).

Speaks to a network-local OpenID Connect provider (the vendored toy OIDC
provider under ``oidc-provider/``) instead of ``accounts.google.com``. It is
opt-in: ``initialize`` refuses unless ``plugins.config`` marks it enabled and
supplies ``server_metadata_url``, and the OAuth client credentials come from
the standard ``INFERENCE_PROXY_OAUTH__*`` settings (the same triple Google
uses). With no configuration the plugin is never loaded, so the sign-in page
is unchanged from the Google-only deployment.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import cast

import httpx
import structlog
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Request
from fastapi.responses import RedirectResponse

from inference_proxy.plugins.interfaces.auth import (
    AuthCallbackError,
    AuthIdentity,
    AuthPlugin,
)
from inference_proxy.plugins.manager import PluginManager

logger = structlog.get_logger()

# The OAuth client registration name inside this plugin's private registry.
# authlib scopes its transient state in the request session by this name
# (``_state_<name>_<state>``), so it cannot collide with the Google plugin's
# own registry (registered as ``google``).
_CLIENT_NAME = "oidc"

# Standard OIDC metadata suffix; the issuer is the metadata URL without it
# (true for both the vendored provider and common OIDC deployments).
_METADATA_SUFFIX = "/.well-known/openid-configuration"
_DEFAULT_DOMAIN = "localdomain"


class InternalOidcPlugin(AuthPlugin):
    """OpenID Connect sign-in against a network-local provider.

    Also validates the provider's local user list for the sign-in page's
    ``Local Login`` form when ``users_file`` is configured: qiip checks a
    username/password against the same plaintext list the toy provider serves
    (admin username is never a local account; qiip's admin password wins).
    """

    name = "internal_oidc"
    version = "1.0.0"
    description = "OpenID Connect sign-in via a local provider"
    author = "QUADS project"
    label = "Local Auth"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self._client: OAuth | None = None
        self._server_metadata_url: str = ""
        raw_users_file = self.config.get("users_file")
        self._users_file: Path | None = (
            Path(str(raw_users_file)).expanduser() if raw_users_file else None
        )
        # Provider email domain, resolved from its discovery document so the
        # Local Login form and the OIDC flow mint identical emails (single
        # source; the old ``domain`` config key is gone).
        self._domain: str | None = None

    def initialize(self, plugin_manager: PluginManager | None = None) -> bool:
        """Build the OIDC client when opted in with credentials + provider URL.

        Opt-in by default: without ``enabled: true`` in the plugin config the
        plugin refuses to load so a default deployment keeps the Google-only
        sign-in page.
        """
        if not super().initialize(plugin_manager):
            return False
        if not self.config.get("enabled", False):
            logger.info(
                "internal_oidc auth plugin disabled (opt-in; set "
                "plugins.config auth.internal_oidc.enabled: true)"
            )
            return False
        if plugin_manager is None:
            return False
        metadata_url = str(self.config.get("server_metadata_url") or "")
        if not metadata_url:
            logger.info(
                "internal_oidc auth plugin disabled (server_metadata_url not configured)"
            )
            return False
        oauth_settings = plugin_manager.settings.oauth
        client_secret = oauth_settings.client_secret
        if not oauth_settings.enabled or client_secret is None:
            logger.info(
                "internal_oidc auth plugin disabled (OAuth credentials not configured)"
            )
            return False
        oauth = OAuth()
        oauth.register(
            name=_CLIENT_NAME,
            client_id=oauth_settings.client_id,
            client_secret=client_secret.get_secret_value(),
            server_metadata_url=metadata_url,
            client_kwargs={"scope": "openid email profile"},
        )
        self._client = oauth
        self._server_metadata_url = metadata_url
        self._fetch_email_domain()
        return True

    def _fetch_email_domain(self) -> str | None:
        """Read ``user_email_domain`` from the provider discovery document.

        The vendored provider publishes its ``OIDC_DOMAIN`` there, so the
        Local Login form and the OIDC flow mint identical emails for the
        same bare username from one source (review #231). A missing or
        unfetchable document leaves the plain ``localdomain`` fallback;
        providers that only ever mint full emails are unaffected.
        """
        try:
            response = httpx.get(self._server_metadata_url, timeout=10.0)
            response.raise_for_status()
            raw = response.json().get("user_email_domain")
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "internal oidc discovery unavailable; bare usernames use the "
                "default email domain",
                url=self._server_metadata_url,
                error=str(exc),
            )
            self._domain = None
            return None
        domain = str(raw).strip() if isinstance(raw, str) else ""
        if not domain:
            logger.warning(
                "internal oidc discovery has no user_email_domain; bare "
                "usernames use the default email domain",
                url=self._server_metadata_url,
            )
            self._domain = None
            return None
        self._domain = domain
        return domain

    def _email_domain(self) -> str:
        """Return the provider email domain, or the default when unknown."""
        return self._domain or _DEFAULT_DOMAIN

    def is_configured(self) -> bool:
        """Return True when the provider client was built during initialize."""
        return self._client is not None

    async def start_login(
        self, request: Request, redirect_uri: str
    ) -> RedirectResponse:
        """Start the local provider Authorization Code flow (302)."""
        return cast(
            RedirectResponse,
            await self._require_client().oidc.authorize_redirect(request, redirect_uri),
        )

    async def complete_login(self, request: Request) -> AuthIdentity:
        """Exchange the callback code for claims and return a normalized identity."""
        try:
            token = await self._require_client().oidc.authorize_access_token(request)
        except OAuthError as exc:
            logger.warning(
                "internal oidc callback rejected",
                error=exc.error,
                description=exc.description,
            )
            raise AuthCallbackError("login_failed") from exc

        userinfo = token.get("userinfo") or {}
        sub = userinfo.get("sub")
        email = userinfo.get("email")
        issuer = userinfo.get("iss")
        if not isinstance(sub, str) or not sub:
            logger.warning("internal oidc callback missing subject", userinfo=userinfo)
            raise AuthCallbackError("no_profile")
        if not isinstance(email, str) or not email:
            logger.warning("internal oidc callback missing email", userinfo=userinfo)
            raise AuthCallbackError("no_profile")
        if not isinstance(issuer, str) or not issuer:
            logger.warning("internal oidc callback missing issuer", userinfo=userinfo)
            raise AuthCallbackError("no_profile")
        name = userinfo.get("name")
        picture = userinfo.get("picture")
        return AuthIdentity(
            sub=sub,
            email=email,
            email_verified=userinfo.get("email_verified") is True,
            issuer=issuer,
            name=name if isinstance(name, str) else "",
            picture=picture if isinstance(picture, str) else "",
        )

    @property
    def issuer(self) -> str:
        """Return the provider issuer (metadata URL without the OIDC suffix).

        Used to key users created by the Local Login form with the same
        (issuer, sub) identity the provider flow produces.
        """
        return self._server_metadata_url.removesuffix(_METADATA_SUFFIX)

    def verify_local_credentials(
        self, username: str, password: str, admin_username: str
    ) -> str | None:
        """Return the canonical email for valid local credentials, else None.

        Reads the configured ``users_file`` (``username:password`` per line,
        ``#`` comments), mirroring the toy provider's rules: bare names are
        matched literally and mapped to ``name@domain``; entries that already
        contain ``@`` are used as-is. An entry whose username equals the
        configured admin username is never a local account — it is ignored
        because qiip's admin password wins (checked by the caller first).
        """
        if self._users_file is None:
            return None
        try:
            lines = self._users_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning(
                "local oauth users file unreadable", users_file=str(self._users_file)
            )
            return None
        users: dict[str, str] = {}
        for line in lines:
            name, _, secret = line.partition(":")
            name = name.strip().lower()
            if not name or name.startswith("#") or not secret:
                continue
            if name == admin_username.strip().lower():
                continue  # admin username is never a local account
            users[name] = secret
        name = username.strip().lower()
        stored = users.get(name)
        if stored is None:
            return None
        # compare_digest only accepts ASCII str or bytes: non-ASCII passwords
        # (UTF-8) must be compared as bytes or they raise TypeError.
        if not hmac.compare_digest(stored.encode("utf-8"), password.encode("utf-8")):
            return None
        return name if "@" in name else f"{name}@{self._email_domain()}"

    def _require_client(self) -> OAuth:
        if self._client is None:
            raise RuntimeError("internal_oidc plugin is not configured")
        return self._client
