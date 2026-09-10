"""Domain models for user identities, API tokens, and token usage.

The model boundary matches the SQLite schema in ``auth/store.py``. All
records are immutable so store rows can be freely shared between threads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class User(BaseModel):
    """A Google-authenticated user row.

    ``google_sub`` is the OIDC subject claim: the stable, per-account
    identifier that never changes even when the user's email does.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    google_sub: str
    email: str
    name: str
    picture: str
    created_at: datetime
    updated_at: datetime


class PublicUser(BaseModel):
    """User fields safe to expose to the browser (no OIDC subject)."""

    model_config = ConfigDict(frozen=True)

    id: int
    email: str
    name: str
    picture: str


class ApiToken(BaseModel):
    """A stored API-token row. Never contains the raw token secret."""

    model_config = ConfigDict(frozen=True)

    id: int
    user_id: int
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked: bool
    endpoint_scope: list[str] | None = None


class CreatedToken(ApiToken):
    """The token row plus the one-time raw secret (AUTH-02).

    The full ``token`` value is returned exactly once, at creation.
    """

    token: str


class PublicToken(BaseModel):
    """Token row safe to list over the wire (never the raw secret)."""

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked: bool
    endpoint_scope: list[str] | None = None

    @classmethod
    def from_token(cls, token: ApiToken) -> PublicToken:
        """Strip the internal fields when serializing a token row."""
        return cls(
            id=token.id,
            name=token.name,
            prefix=token.prefix,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            revoked=token.revoked,
            endpoint_scope=token.endpoint_scope,
        )


class CreateTokenRequest(BaseModel):
    """Request body for POST /profile/tokens."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=100)
    endpoints: list[str] | None = None

    @field_validator("name")
    @classmethod
    def name_has_no_surrounding_whitespace(cls, value: str) -> str:
        """Reject blank or whitespace-padded token names."""
        if value != value.strip():
            raise ValueError("token name must not have surrounding whitespace")
        return value

    @field_validator("endpoints")
    @classmethod
    def endpoints_are_hostnames(cls, value: list[str] | None) -> list[str] | None:
        """Require clean hostnames; an empty list is rejected (None = full)."""
        if value is None:
            return None
        if not value:
            raise ValueError("endpoint scope must not be empty")
        cleaned: list[str] = []
        for item in value:
            normalized = item.strip()
            if (
                not normalized
                or any(ord(char) < 32 or char.isspace() for char in normalized)
                or any(char in normalized for char in "/:@")
            ):
                raise ValueError("endpoint scope entries must be hostnames")
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned


class TokenUsage(BaseModel):
    """Aggregated usage for one (token, model, endpoint) group."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    token_id: int | None
    token_name: str | None
    model: str
    endpoint: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class UsageTotals(BaseModel):
    """Headline sums across every token in the profile."""

    model_config = ConfigDict(frozen=True)

    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenAuth(BaseModel):
    """Authenticated caller context for a proxied /v1 request (AUTH-03)."""

    model_config = ConfigDict(frozen=True)

    user: User
    token: ApiToken
