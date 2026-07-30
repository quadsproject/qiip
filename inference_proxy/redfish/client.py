"""Redfish BMC client for power state queries and power actions.

Thin async wrapper over httpx, mirroring the QUADSClient pattern:
constructor-injected ``httpx.AsyncClient``, typed ``RedfishError``
exception, async methods.

D-03: check-before-act -- skip BMC POST when already in desired state.
D-04: post-action polling -- poll PowerState until target or timeout.
D-05: ``verify=False`` on the dedicated httpx.AsyncClient (caller's concern).
D-06: httpx uses httpcore, not urllib3 -- no InsecureRequestWarning emitted.
"""

from __future__ import annotations

import asyncio
from typing import cast

import httpx
import structlog

from inference_proxy.redfish.errors import RedfishError, extract_error_message

logger = structlog.get_logger()

_ACTION_TARGET_STATE: dict[str, str] = {
    "On": "On",
    "ForceOff": "Off",
    "GracefulRestart": "On",
    "ForceRestart": "On",
}


class RedfishClient:
    """Async client for Redfish BMC power management.

    Args:
        http_client: Pre-built ``httpx.AsyncClient`` (lifecycle managed externally).
        bmc_host_template: Template for resolving BMC hostname (D-01).
        system_id: Redfish system ID (default ``"1"``).
        poll_timeout: Seconds to wait for power state transition (D-04).
        poll_interval: Seconds between power state polls.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        bmc_host_template: str,
        system_id: str,
        poll_timeout: float = 60.0,
        poll_interval: float = 5.0,
    ) -> None:
        self._client = http_client
        self._bmc_host_template = bmc_host_template
        self._system_id = system_id
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval

    def _resolve_bmc_host(self, hostname: str) -> str:
        """Resolve BMC hostname from template (D-01)."""
        return self._bmc_host_template.format(hostname=hostname)

    async def get_power_state(self, hostname: str) -> str:
        """Query current power state from BMC.

        Returns one of: On, Off, PoweringOn, PoweringOff.
        """
        bmc = self._resolve_bmc_host(hostname)
        url = f"https://{bmc}/redfish/v1/Systems/{self._system_id}"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            msg = extract_error_message(exc)
            logger.error(
                "redfish_get_power_state_failed",
                hostname=hostname,
                bmc_host=bmc,
                error=msg,
            )
            raise RedfishError(msg) from exc
        # B12 tracks runtime validation and typed error mapping for this field.
        return cast(str, resp.json()["PowerState"])

    async def power_action(
        self, hostname: str, action: str, *, timeout: float | None = None
    ) -> str:
        """Issue a power action with check-before-act (D-03) and polling (D-04).

        Returns the final PowerState after the action completes or times out.
        """
        if action not in _ACTION_TARGET_STATE:
            raise RedfishError(f"Unsupported action: {action}")
        target = _ACTION_TARGET_STATE[action]
        current = await self.get_power_state(hostname)
        if current == target:
            logger.info(
                "redfish_power_action_skipped",
                hostname=hostname,
                action=action,
                state=current,
            )
            return current
        await self._post_reset(hostname, action)
        return await self._poll_power_state(
            hostname, target, timeout or self._poll_timeout
        )

    async def _post_reset(self, hostname: str, action: str) -> None:
        """POST a ComputerSystem.Reset action to the BMC."""
        bmc = self._resolve_bmc_host(hostname)
        url = f"https://{bmc}/redfish/v1/Systems/{self._system_id}/Actions/ComputerSystem.Reset"
        try:
            resp = await self._client.post(url, json={"ResetType": action})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            msg = extract_error_message(exc)
            logger.error(
                "redfish_post_reset_failed",
                hostname=hostname,
                bmc_host=bmc,
                action=action,
                error=msg,
            )
            raise RedfishError(msg) from exc
        logger.info("redfish_reset_issued", hostname=hostname, action=action)

    async def _poll_power_state(
        self, hostname: str, target: str, timeout: float
    ) -> str:
        """Poll PowerState until target reached or timeout (D-04)."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            state = await self.get_power_state(hostname)
            if state == target:
                return state
            await asyncio.sleep(self._poll_interval)
        raise RedfishError(f"Power state did not reach {target} within {timeout}s")
