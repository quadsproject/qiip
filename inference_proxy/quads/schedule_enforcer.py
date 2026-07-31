"""QUADS schedule enforcer — auto-teardown nodes with upcoming schedules.

Periodically queries QUADS availability with a lookahead window and
tears down any inference node that will lose availability within that
window.  QUADS is treated as the authoritative source of host
scheduling.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from time import monotonic

import structlog

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import NodeStatus
from inference_proxy.provisioning.host_lifecycle import HostLifecycleLease
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import (
    QUADSClient,
    availability_window_end,
    canonical_hostname,
)

logger = structlog.get_logger()

_ACTIONABLE_STATUSES = frozenset(
    {NodeStatus.HEALTHY, NodeStatus.UNHEALTHY, NodeStatus.DRAINING}
)
_MAX_RETRY_BACKOFF_SECONDS = 3600.0


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
        self._retry_attempts: dict[str, int] = {}
        self._retry_after: dict[str, float] = {}
        self._retry_escalated: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    @property
    def teardown_initiated(self) -> set[str]:
        return set(self._teardown_initiated)

    @property
    def teardown_retry_attempts(self) -> dict[str, int]:
        return dict(self._retry_attempts)

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

        end = availability_window_end(self._lookahead_hours)
        try:
            available = await self._client.get_available(end=end)
        except Exception:
            logger.warning("schedule_enforcer_poll_failed", exc_info=True)
            return

        available_set = {canonical_hostname(hostname) for hostname in available}
        for node in self._registry.get_all():
            if not node.managed:
                continue
            if node.status not in _ACTIONABLE_STATUSES:
                continue
            retry_pending = node.node_id in self._retry_attempts
            if canonical_hostname(node.node_id) in available_set:
                if node.status == NodeStatus.DRAINING and retry_pending:
                    # Once this enforcer has started draining a host, finish
                    # the owned teardown even if the schedule later changes.
                    pass
                else:
                    self._clear_teardown_retry(node.node_id)
                    continue
            if node.node_id in self._teardown_initiated:
                continue
            retry_after = self._retry_after.get(node.node_id)
            if retry_after is not None:
                if monotonic() < retry_after:
                    continue
                self._retry_after.pop(node.node_id, None)
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
            self._schedule_teardown(node.node_id, lease)

    def _schedule_teardown(
        self,
        hostname: str,
        lifecycle_lease: HostLifecycleLease,
    ) -> None:
        """Run one owned teardown with retry-aware failure handling."""
        failure_recorded = False

        async def _teardown() -> None:
            nonlocal failure_recorded
            try:
                await self._provisioner.teardown(
                    hostname,
                    lifecycle_lease=lifecycle_lease,
                )
            except asyncio.CancelledError as exc:
                failure_recorded = True
                self._record_teardown_failure(hostname, exc)
                raise
            except Exception as exc:
                # This owner consumes the failure because it must update
                # host-specific retry state. PR 10's generic background
                # observer therefore has no duplicate exception to log.
                failure_recorded = True
                self._record_teardown_failure(hostname, exc)
            else:
                self._clear_teardown_retry(hostname)
            finally:
                lifecycle_lease.release()

        background = _teardown()
        try:
            task = self._provisioner.fire_background(background)

            def _release_lease(done_task: asyncio.Task[None]) -> None:
                # A task cancelled before its coroutine starts never reaches
                # the wrapper's cancellation handler.
                if done_task.cancelled() and not failure_recorded:
                    self._record_teardown_failure(
                        hostname,
                        RuntimeError("teardown task cancelled before start"),
                    )
                lifecycle_lease.release()

            task.add_done_callback(_release_lease)
        except Exception as exc:
            background.close()
            self._record_teardown_failure(hostname, exc)
            lifecycle_lease.release()

    def _record_teardown_failure(
        self,
        hostname: str,
        error: BaseException,
    ) -> None:
        """Schedule a bounded-backoff retry for an owned teardown failure."""
        self._teardown_initiated.discard(hostname)
        attempts = self._retry_attempts.get(hostname, 0) + 1
        self._retry_attempts[hostname] = attempts
        base_delay = max(1.0, float(self._interval))
        retry_delay = min(
            base_delay * (2 ** min(attempts - 1, 16)),
            _MAX_RETRY_BACKOFF_SECONDS,
        )
        self._retry_after[hostname] = monotonic() + retry_delay

        if (
            retry_delay == _MAX_RETRY_BACKOFF_SECONDS
            and hostname not in self._retry_escalated
        ):
            self._retry_escalated.add(hostname)
            logger.error(
                "schedule_enforcer_teardown_requires_operator",
                hostname=hostname,
                attempt=attempts,
                retry_delay_seconds=retry_delay,
                error=str(error),
            )
        else:
            logger.warning(
                "schedule_enforcer_teardown_retry_scheduled",
                hostname=hostname,
                attempt=attempts,
                retry_delay_seconds=retry_delay,
                error=str(error),
            )

    def _clear_teardown_retry(self, hostname: str) -> None:
        self._retry_attempts.pop(hostname, None)
        self._retry_after.pop(hostname, None)
        self._retry_escalated.discard(hostname)

    def _prune_completed(self) -> None:
        """Remove hostnames from tracking once they leave the registry."""
        registered = {n.node_id for n in self._registry.get_all()}
        self._teardown_initiated &= registered
        for hostname in set(self._retry_attempts) - registered:
            self._clear_teardown_retry(hostname)
