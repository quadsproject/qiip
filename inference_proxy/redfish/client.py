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
import json

import httpx
import structlog

from inference_proxy.models.endpoint import (
    EndpointPolicy,
    EndpointValidationError,
    parse_endpoint,
)
from inference_proxy.redfish.errors import (
    RedfishDestinationError,
    RedfishError,
    extract_error_message,
)

logger = structlog.get_logger()

_ACTION_TARGET_STATE: dict[str, str] = {
    "On": "On",
    "ForceOff": "Off",
    "GracefulRestart": "On",
    "ForceRestart": "On",
}
_IDEMPOTENT_ACTIONS = frozenset({"On", "ForceOff"})
_ACTION_TRANSITIONAL_STATE = {
    "On": "PoweringOn",
    "ForceOff": "PoweringOff",
}
_VALID_POWER_STATES = frozenset({"On", "Off", "PoweringOn", "PoweringOff"})


def _monotonic() -> float:
    """Return the event loop's monotonic clock for deterministic tests."""
    return asyncio.get_running_loop().time()


async def _sleep(delay: float) -> None:
    """Sleep between polls through a testable module seam."""
    await asyncio.sleep(delay)


class RedfishClient:
    """Async client for Redfish BMC power management.

    Args:
        http_client: Pre-built ``httpx.AsyncClient`` (lifecycle managed externally).
        bmc_host_template: Template for resolving BMC hostname (D-01).
        system_id: Redfish system ID (default ``"1"``).
        hostname_policy: Trust policy for caller-supplied node hostnames.
        auth: Credentials applied only after destination validation.
        poll_timeout: Seconds to wait for power state transition (D-04).
        poll_interval: Seconds between power state polls.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        bmc_host_template: str,
        system_id: str,
        *,
        hostname_policy: EndpointPolicy,
        auth: httpx.Auth,
        poll_timeout: float = 60.0,
        poll_interval: float = 5.0,
    ) -> None:
        self._client = http_client
        self._bmc_host_template = bmc_host_template
        self._system_id = system_id
        self._hostname_policy = hostname_policy
        self._auth = auth
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval

    def _resolve_bmc_host(self, hostname: str) -> str:
        """Validate the node hostname, then resolve its BMC destination."""
        try:
            normalized = self._hostname_policy.normalize_hostname(hostname)
        except EndpointValidationError as exc:
            raise RedfishDestinationError(str(exc)) from exc

        rendered = self._bmc_host_template.format(hostname=normalized)
        try:
            endpoint = parse_endpoint(f"https://{rendered}:443")
        except EndpointValidationError as exc:
            raise RedfishDestinationError(
                f"BMC host template produced an invalid destination: {rendered!r}"
            ) from exc
        return endpoint.host

    async def get_power_state(self, hostname: str) -> str:
        """Query current power state from BMC.

        Returns one of: On, Off, PoweringOn, PoweringOff.
        """
        bmc = self._resolve_bmc_host(hostname)
        url = f"https://{bmc}/redfish/v1/Systems/{self._system_id}"
        try:
            resp = await self._client.get(url, auth=self._auth)
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
        return self._parse_power_state(resp)

    @staticmethod
    def _parse_power_state(response: httpx.Response) -> str:
        """Validate the successful Redfish response without hiding code bugs."""
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RedfishError("BMC returned malformed JSON for PowerState") from exc

        if not isinstance(payload, dict) or "PowerState" not in payload:
            raise RedfishError("BMC response is missing PowerState")
        state = payload["PowerState"]
        if not isinstance(state, str) or state not in _VALID_POWER_STATES:
            raise RedfishError(f"BMC returned unsupported PowerState: {state!r}")
        return state

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
        if action in _IDEMPOTENT_ACTIONS and current == target:
            logger.info(
                "redfish_power_action_skipped",
                hostname=hostname,
                action=action,
                state=current,
            )
            return current
        if current == _ACTION_TRANSITIONAL_STATE.get(action):
            logger.info(
                "redfish_power_transition_in_progress",
                hostname=hostname,
                action=action,
                state=current,
            )
        else:
            await self._post_reset(hostname, action)
        return await self._poll_power_state(
            hostname,
            target,
            timeout if timeout is not None else self._poll_timeout,
        )

    async def _post_reset(self, hostname: str, action: str) -> None:
        """POST a ComputerSystem.Reset action to the BMC."""
        bmc = self._resolve_bmc_host(hostname)
        url = f"https://{bmc}/redfish/v1/Systems/{self._system_id}/Actions/ComputerSystem.Reset"
        try:
            resp = await self._client.post(
                url,
                json={"ResetType": action},
                auth=self._auth,
            )
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
        """Poll through transient BMC faults and perform a deadline probe."""
        deadline = _monotonic() + timeout
        last_error: RedfishError | None = None

        while _monotonic() < deadline:
            try:
                state = await self.get_power_state(hostname)
            except RedfishError as exc:
                last_error = exc
                logger.warning(
                    "redfish_power_poll_transient_error",
                    hostname=hostname,
                    target=target,
                    error=exc.human_message,
                )
            else:
                if state == target:
                    return state

            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            await _sleep(min(self._poll_interval, remaining))

        # A state change at the deadline is observable only through this final
        # probe; omitting it can report a timeout after the BMC reached target.
        try:
            state = await self.get_power_state(hostname)
        except RedfishError as exc:
            last_error = exc
        else:
            if state == target:
                return state

        message = f"Power state did not reach {target} within {timeout}s"
        if last_error is not None:
            message = f"{message}; last BMC error: {last_error.human_message}"
        raise RedfishError(message)
