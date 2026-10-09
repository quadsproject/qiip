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

from typing import cast

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


class InternalOidcPlugin(AuthPlugin):
    """OpenID Connect sign-in against a network-local provider."""

    name = "internal_oidc"
    version = "1.0.0"
    description = "OpenID Connect sign-in via a local provider"
    author = "QUADS project"
    label = "Local Auth"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self._client: OAuth | None = None
        self._server_metadata_url: str = ""

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
            logger.info("internal_oidc auth plugin disabled (server_metadata_url not configured)")
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
        return True

    def is_configured(self) -> bool:
        """Return True when the provider client was built during initialize."""
        return self._client is not None

    async def start_login(
        self, request: Request, redirect_uri: str
    ) -> RedirectResponse:
        """Start the local provider Authorization Code flow (302)."""
        return cast(
            RedirectResponse,
            await self._require_client().oidc.authorize_redirect(
                request, redirect_uri
            ),
        )

    async def complete_login(self, request: Request) -> AuthIdentity:
        """Exchange the callback code for claims and return a normalized identity."""
        try:
            token = await self._require_client().oidc.authorize_access_token(
                request
            )
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

    def _require_client(self) -> OAuth:
        if self._client is None:
            raise RuntimeError("internal_oidc plugin is not configured")
        return self._client
