"""Shared constants for the auth subsystem."""

from __future__ import annotations

# Every minted API token starts with this prefix so bearer tokens are
# trivial to identify in proxy logs and clients can recognize theirs.
TOKEN_PREFIX = "qiip_"

# The OIDC issuer of Google accounts. User rows created before the issuer
# column existed are backfilled with this value (see AuthStore._migrate),
# and scope grants in ``admin_only_tokens_full_access`` are scoped to it.
GOOGLE_ISSUER = "https://accounts.google.com"
