"""Google OAuth client construction (authlib, RFC 6749 + OpenID Connect).

``build_google_oauth`` registers the Google provider on a fresh ``OAuth``
registry. authlib resolves Google's authorization/token/JWKS endpoints from
the provider's OpenID Configuration discovery document at call time, so no
hard-coded endpoint is maintained here.

The OAuth client stores its transient ``state`` in the request session, which
is why ``auth.session_secret`` is mandatory before OAuth can be enabled (the
settings layer enforces this).
"""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth

from inference_proxy.config.settings import OAuthSettings

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
