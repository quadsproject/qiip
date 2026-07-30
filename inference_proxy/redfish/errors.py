"""Redfish error types and human-readable error mapping (DIAG-03).

Maps common Redfish Base registry MessageIds to operator-friendly text.
Falls back to the raw ``Message`` field from ``@Message.ExtendedInfo``,
truncated to 200 chars for safety.
"""

from __future__ import annotations

from typing import cast

import httpx


class RedfishError(Exception):
    """Raised when a Redfish BMC operation fails."""

    def __init__(self, human_message: str) -> None:
        self.human_message = human_message
        super().__init__(human_message)


REDFISH_ERROR_MAP: dict[str, str] = {
    "ActionNotSupported": "This action is not supported by the BMC",
    "ActionParameterNotSupported": "This action parameter is not supported",
    "ResourceNotFound": "BMC resource not found -- check system ID",
    "InternalError": "BMC internal error -- retry or check BMC health",
    "ServiceTemporarilyUnavailable": "BMC is temporarily busy -- retry later",
    "NoOperation": "No change needed -- system already in requested state",
    "InsufficientPrivilege": "BMC credentials lack permission for this action",
    "PropertyValueTypeError": "Invalid parameter value in request",
}


def extract_error_message(exc: Exception) -> str:
    """Extract a human-readable message from a Redfish error.

    For ``httpx.HTTPStatusError``, parses the Redfish JSON body and maps
    known ``MessageId`` keys via ``REDFISH_ERROR_MAP``.  Unknown IDs fall
    back to the ``Message`` field.  All other exceptions return ``str(exc)``
    truncated to 200 characters.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.json()
            ext_info = body.get("error", {}).get("@Message.ExtendedInfo", [])
            if ext_info:
                msg_id = ext_info[0].get("MessageId", "")
                key = msg_id.rsplit(".", 1)[-1] if msg_id else ""
                if key in REDFISH_ERROR_MAP:
                    return REDFISH_ERROR_MAP[key]
                return cast(str, ext_info[0].get("Message", str(exc)))[:200]
            return cast(str, body.get("error", {}).get("message", str(exc)))[:200]
        except Exception:
            pass
    return str(exc)[:200]
