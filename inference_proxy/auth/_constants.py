"""Shared constants for the auth subsystem."""

from __future__ import annotations

# Every minted API token starts with this prefix so bearer tokens are
# trivial to identify in proxy logs and clients can recognize theirs.
TOKEN_PREFIX = "qiip_"
