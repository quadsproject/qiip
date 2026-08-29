"""User identity, API-token, and token-usage persistence (AUTH-01..AUTH-04).

Per AUTH-01: user records are keyed by the Google ``sub`` claim (stable
per-account, never reissued); API tokens are stored as SHA-256 digests so
a database read never exposes a usable credential (AUTH-02).

Per AUTH-03: /v1 inference requests may authenticate with a bearer token;
enforcement is config-gated so existing public deployments are not broken.
Per AUTH-04: token-minted requests record OpenAI ``usage`` into the store
for per-token/per-model usage reporting on the profile page.
"""

from __future__ import annotations

from inference_proxy.auth.models import (
    ApiToken,
    CreatedToken,
    CreateTokenRequest,
    PublicToken,
    PublicUser,
    TokenAuth,
    TokenUsage,
    User,
)
from inference_proxy.auth.session import (
    clear_session_user,
    get_session_user_id,
    set_session_user,
)
from inference_proxy.auth.store import AuthStore

__all__ = [
    "ApiToken",
    "AuthStore",
    "CreatedToken",
    "CreateTokenRequest",
    "PublicToken",
    "PublicUser",
    "TokenAuth",
    "TokenUsage",
    "User",
    "clear_session_user",
    "get_session_user_id",
    "set_session_user",
]
