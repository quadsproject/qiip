"""QUADS schedule enforcer — auto-teardown nodes with upcoming schedules.

Periodically queries QUADS availability with a lookahead window and
tears down any inference node that will lose availability within that
window.  QUADS is treated as the authoritative source of host
scheduling.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import structlog

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import NodeStatus
from inference_proxy.provisioning.host_lifecycle import HostLifecycleLease
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import QUADSClient

logger = structlog.get_logger()

_ACTIONABLE_STATUSES = frozenset({NodeStatus.HEALTHY, NodeStatus.UNHEALTHY})


class ScheduleEnforcer:
    """Tears down nodes that QUADS shows as unavailable within the lookahead window.

    Args:
        client: QUADS API client (injected, DIP).
        registry: In-memory node registry.
        provisioner: Node provisioner for firing teardowns.
        lookahead_hours: How far ahead to check for schedules.
        check_interval: Seconds between enforcement checks.
    """

    def __init__(
        self,
        client: QUADSClient,
        registry: NodeRegistry,
        provisioner: NodeProvisioner,
        lookahead_hours: int = 24,
        check_interval: int = 300,
    ) -> None:
        self._client = client
        self._registry = registry
        self._provisioner = provisioner
        self._lookahead_hours = lookahead_hours
        self._interval = check_interval
        self._teardown_initiated: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    @property
    def teardown_initiated(self) -> set[str]:
        return set(self._teardown_initiated)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _loop(self) -> None:
        await self._enforce_once()
        while True:
            await asyncio.sleep(self._interval)
            await self._enforce_once()

    async def _enforce_once(self) -> None:
        self._prune_completed()

        end = datetime.now(tz=UTC) + timedelta(hours=self._lookahead_hours)
        try:
            available = await self._client.get_available(end=end)
        except Exception:
            logger.warning("schedule_enforcer_poll_failed", exc_info=True)
            return

        available_set = set(available)
        for node in self._registry.get_all():
            if not node.managed:
                continue
            if node.status not in _ACTIONABLE_STATUSES:
                continue
            if node.node_id in available_set:
                continue
            if node.node_id in self._teardown_initiated:
                continue
            lease = await self._provisioner.try_reserve_host(node.node_id)
            if lease is None:
                logger.debug(
                    "schedule_enforcer_host_busy",
                    hostname=node.node_id,
                )
                continue
            logger.info(
                "schedule_enforcer_teardown",
                hostname=node.node_id,
                reason="quads_schedule_conflict",
                lookahead_hours=self._lookahead_hours,
            )
            self._teardown_initiated.add(node.node_id)

            async def _teardown(
                hostname: str = node.node_id,
                lifecycle_lease: HostLifecycleLease = lease,
            ) -> None:
                try:
                    await self._provisioner.teardown(
                        hostname,
                        lifecycle_lease=lifecycle_lease,
                    )
                finally:
                    lifecycle_lease.release()

            background = _teardown()
            try:
                task = self._provisioner.fire_background(background)

                def _release_lease(
                    _task: asyncio.Task[None],
                    lifecycle_lease: HostLifecycleLease = lease,
                ) -> None:
                    lifecycle_lease.release()

                task.add_done_callback(_release_lease)
            except Exception:
                background.close()
                self._teardown_initiated.discard(node.node_id)
                lease.release()
                raise

    def _prune_completed(self) -> None:
        """Remove hostnames from tracking once they leave the registry."""
        registered = {n.node_id for n in self._registry.get_all()}
        self._teardown_initiated &= registered
