"""Domain models for user identities, API tokens, and token usage.

The model boundary matches the SQLite schema in ``auth/store.py``. All
records are immutable so store rows can be freely shared between threads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inference_proxy.models.admin import _HOSTNAME_RE


class User(BaseModel):
    """A Google-authenticated user row.

    ``google_sub`` is the OIDC subject claim: the stable, per-account
    identifier that never changes even when the user's email does.
    ``is_admin`` is the admin-role designation granted by an existing
    admin (HTTP Basic holder) through the admin page.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    google_sub: str
    email: str
    name: str
    picture: str
    is_admin: bool = False
    created_at: datetime
    updated_at: datetime


class PublicUser(BaseModel):
    """User fields safe to expose to the browser (no OIDC subject)."""

    model_config = ConfigDict(frozen=True)

    id: int
    email: str
    name: str
    picture: str
    is_admin: bool = False


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
    # Model ids the token may request on /v1 (None = unrestricted).
    model_scope: list[str] | None = None
    purpose: str | None = None


class SetupLink(BaseModel):
    """A live short-lived link to a rendered harness setup script."""

    user_id: int
    token_id: int
    harness: str
    models: list[str]
    expires_at: datetime


class CreatedToken(ApiToken):
    """The token row plus the one-time raw secret (AUTH-02).

    The full ``token`` value is returned exactly once, at creation -- except
    for the derived ``agent-config`` key, which is returned by every
    ``get_or_create_config_token`` call (see :mod:`inference_proxy.auth.store`).
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
    model_scope: list[str] | None = None

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
            model_scope=token.model_scope,
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
        """Require registered-node hostnames; empty list rejected (None = full)."""
        if value is None:
            return None
        if not value:
            raise ValueError("endpoint scope must not be empty")
        cleaned: list[str] = []
        for item in value:
            normalized = item.strip()
            if not _HOSTNAME_RE.fullmatch(normalized):
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


class AdminUserStats(BaseModel):
    """Per-user aggregate counts for the admin token dashboard (RFE #113)."""

    model_config = ConfigDict(frozen=True)

    id: int
    email: str
    name: str
    picture: str
    is_admin: bool = False
    created_at: datetime
    token_count: int
    active_token_count: int
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float = 0.0


class AdminTokenView(BaseModel):
    """Token row plus owner identity and usage counts (RFE #113)."""

    model_config = ConfigDict(frozen=True)

    id: int
    user_id: int
    user_email: str
    user_name: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked: bool
    endpoint_scope: list[str] | None = None
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AdminUserDetail(BaseModel):
    """Full user view for the admin drill-down surface (RFE #113)."""

    model_config = ConfigDict(frozen=True)

    user: PublicUser
    tokens: list[AdminTokenView]
    usage: list[TokenUsage]
    totals: UsageTotals
    timeline: list[UsageTimelineRow]
    estimated_cost_usd: float = 0.0
    model_label: str = ""


class UsageTimelineRow(BaseModel):
    """One day of aggregated usage for a user (AUTH-04 rows by created_at)."""

    model_config = ConfigDict(frozen=True)

    day: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BillingSummary(BaseModel):
    """Global usage totals plus premium-equivalent cost (RFE #113)."""

    model_config = ConfigDict(frozen=True)

    totals: UsageTotals
    estimated_cost_usd: float
    model_label: str


class TokenAuth(BaseModel):
    """Authenticated caller context for a proxied /v1 request (AUTH-03)."""

    model_config = ConfigDict(frozen=True)

    user: User
    token: ApiToken
