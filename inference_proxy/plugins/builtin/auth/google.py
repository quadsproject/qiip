"""Google OAuth (OpenID Connect) auth plugin.

``build_google_oauth`` registers the Google provider on a fresh ``OAuth``
registry. authlib resolves Google's authorization, token, and JWKS endpoints
from the provider's OpenID Configuration discovery document at call time, so
no hard-coded endpoint is maintained here.

The OAuth client stores its transient ``state`` in the request session,
which is why ``auth.session_secret`` is mandatory before OAuth can be
enabled (the settings layer enforces this).
"""

from __future__ import annotations

from typing import cast

import structlog
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Request
from fastapi.responses import RedirectResponse

from inference_proxy.config.settings import OAuthSettings
from inference_proxy.plugins.interfaces.auth import (
    AuthCallbackError,
    AuthIdentity,
    AuthPlugin,
)
from inference_proxy.plugins.manager import PluginManager

logger = structlog.get_logger()

_GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


def build_google_oauth(settings: OAuthSettings) -> OAuth:
    """Register and return a Google OAuth client for the given settings."""
    if not settings.enabled or settings.client_secret is None:
        raise ValueError(
            "cannot build OAuth client from a disabled or partial OAuth settings"
        )
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        server_metadata_url=_GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


class GoogleAuthPlugin(AuthPlugin):
    """Google OAuth 2.0 / OpenID Connect sign-in provider."""

    name = "google"
    version = "1.0.0"
    description = "Google OAuth (OpenID Connect) sign-in"
    author = "QUADS project"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        super().__init__(config)
        self._client: OAuth | None = None

    def initialize(self, plugin_manager: PluginManager | None = None) -> bool:
        """Build the Google client when the credential triple is complete."""
        if not super().initialize(plugin_manager):
            return False
        if plugin_manager is None:
            return False
        oauth_settings = plugin_manager.settings.oauth
        if not oauth_settings.enabled:
            logger.info(
                "google auth plugin disabled (OAuth credentials not configured)"
            )
            return False
        self._client = build_google_oauth(oauth_settings)
        return True

    def is_configured(self) -> bool:
        """Return True when the provider client was built during initialize."""
        return self._client is not None

    async def start_login(
        self, request: Request, redirect_uri: str
    ) -> RedirectResponse:
        """Start the Google Authorization Code flow (302 to Google)."""
        return cast(
            RedirectResponse,
            await self._require_client().google.authorize_redirect(
                request, redirect_uri
            ),
        )

    async def complete_login(self, request: Request) -> AuthIdentity:
        """Exchange the callback code for claims and return a normalized identity."""
        try:
            token = await self._require_client().google.authorize_access_token(request)
        except OAuthError as exc:
            logger.warning(
                "oauth callback rejected",
                error=exc.error,
                description=exc.description,
            )
            raise AuthCallbackError("login_failed") from exc

        userinfo = token.get("userinfo") or {}
        sub = userinfo.get("sub")
        email = userinfo.get("email")
        if not isinstance(sub, str) or not sub:
            logger.warning("oauth callback missing subject", userinfo=userinfo)
            raise AuthCallbackError("no_profile")
        if not isinstance(email, str) or not email:
            logger.warning("oauth callback missing email", userinfo=userinfo)
            raise AuthCallbackError("no_profile")
        name = userinfo.get("name")
        picture = userinfo.get("picture")
        return AuthIdentity(
            sub=sub,
            email=email,
            email_verified=userinfo.get("email_verified") is True,
            name=name if isinstance(name, str) else "",
            picture=picture if isinstance(picture, str) else "",
        )

    def _require_client(self) -> OAuth:
        if self._client is None:
            raise RuntimeError("google auth plugin is not configured")
        return self._client
