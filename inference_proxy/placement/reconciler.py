"""Automatic fixed-ratio placement of catalog profiles onto free GPUs.

One pass gathers four facts (which catalog files exist, which hosts QUADS shows
free for the whole scheduling window, which placements automation already
holds, and what the node registry says), asks the pure planner what to do, and
starts at most a few provisions.

qiip is deployed as a single gateway, and that is what this supports. Within
it, three things keep a host from being provisioned twice:

* the host lifecycle lease every manual setup, relaunch and teardown also
  takes. It is in-process: it coordinates this gateway with its own operators,
  nothing more;
* a persistent, revision-checked claim per host, so this gateway after a
  restart sees the host as taken even when the node's own leased key expired;
* before every launch, first or retry, a check on the host itself that no
  earlier provisioning command is still running. A cancelled gateway task does
  not always stop its remote command, and an operator can reset a claim while
  one runs, so a launch that cannot verify this is blocked and reported. A host
  with no claim yet is then left out of planning for one backoff period, so the
  planner gives its share to another free host instead of picking it again.

``placement.max_attempts`` bounds the attempts since the last successful
provision: consecutive failures plus the one in flight. A success, or adopting
a node that turned out to be serving, starts the count again.

The claim's compare-and-swap and heartbeat also stop two gateways from
claiming one host, and a holder that loses or cannot refresh its claim stops
its own provision (remote worker included) at half the stale period. That is a
safeguard against an accidental second gateway, not support for running two.

Automation only ever retries or cleans up a node that carries its own claim id.
A node an operator set up, owns, adopted or relabelled is left alone.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import structlog

from inference_proxy.config.settings import PlacementSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.artifacts import GGUFArtifactIndex
from inference_proxy.models.node import (
    InferenceEngine,
    Node,
    NodePlacement,
    NodeStatus,
)
from inference_proxy.models.quads import QUADSHost
from inference_proxy.placement.catalog import (
    BUILTIN_PROFILES,
    CatalogResolution,
    MissingArtifact,
    ModelProfile,
    ResolvedProfile,
    resolve_catalog,
)
from inference_proxy.placement.claims import (
    ClaimState,
    ClaimStore,
    PlacementClaim,
    StoredClaim,
    utcnow,
)
from inference_proxy.placement.planner import (
    Candidate,
    PlannedProfile,
    plan_placements,
)
from inference_proxy.provisioning.provisioner import (
    NodeProvisioner,
    ProvisioningCapacityError,
    ProvisioningIdentity,
)
from inference_proxy.quads.client import (
    QUADSClient,
    availability_window_end,
    canonical_hostname,
)
from inference_proxy.quads.poller import QUADSPoller

logger = structlog.get_logger()

# A process-list probe must not inherit the long setup command deadlines.
_REMOTE_PROBE_TIMEOUT_SECONDS = 10.0

# Node states that mean a placement's server was registered and is being
# health-checked. Automation never rebuilds one of these.
_SERVING_NODE_STATUSES = frozenset(
    {
        NodeStatus.HEALTHY,
        NodeStatus.UNHEALTHY,
        NodeStatus.DRAINING,
        NodeStatus.RELAUNCHING,
        NodeStatus.RELAUNCH_FAILED,
    }
)
# Node states automation may clear before retrying its own failed placement.
_RETRYABLE_NODE_STATUSES = frozenset(
    {NodeStatus.FAILED, NodeStatus.PROVISIONING, NodeStatus.UNKNOWN}
)


@dataclass(frozen=True)
class SkippedHost:
    """A GPU host automation looked at and deliberately left alone."""

    hostname: str
    reason: str


@dataclass(frozen=True)
class PlacementStatus:
    """The outcome of the most recent pass, for the admin surfaces."""

    enabled: bool
    holder: str
    last_run_at: datetime | None = None
    error: str = ""
    targets: dict[str, int] = field(default_factory=dict)
    held: dict[str, int] = field(default_factory=dict)
    unfilled: dict[str, int] = field(default_factory=dict)
    missing_artifacts: tuple[MissingArtifact, ...] = ()
    claims: tuple[PlacementClaim, ...] = ()
    unreadable_claims: tuple[str, ...] = ()
    started: tuple[str, ...] = ()
    skipped: tuple[SkippedHost, ...] = ()


@dataclass
class _OwnedClaim:
    """The revision this gateway last wrote, shared with the heartbeat task."""

    stored: StoredClaim
    refreshed_at: datetime
    lost: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PlacementReconciler:
    """Periodic loop in the idiom of ``QUADSPoller`` and ``ScheduleEnforcer``."""

    def __init__(
        self,
        *,
        settings: PlacementSettings,
        quads_client: QUADSClient,
        quads_poller: QUADSPoller,
        registry: NodeRegistry,
        provisioner: NodeProvisioner,
        artifact_index: GGUFArtifactIndex,
        claims: ClaimStore,
        lookahead_hours: int,
        profiles: Sequence[ModelProfile] = BUILTIN_PROFILES,
        clock: Callable[[], datetime] = utcnow,
        holder: str | None = None,
    ) -> None:
        self._settings = settings
        self._quads = quads_client
        self._poller = quads_poller
        self._registry = registry
        self._provisioner = provisioner
        self._artifacts = artifact_index
        self._claims = claims
        self._lookahead_hours = lookahead_hours
        self._profiles = tuple(profiles)
        self._clock = clock
        self._holder = holder or uuid.uuid4().hex
        self.started_at = clock()
        self._task: asyncio.Task[None] | None = None
        self._owned: dict[str, _OwnedClaim] = {}
        self._fencing: set[asyncio.Task[None]] = set()
        self._pass_lock = asyncio.Lock()
        # Unclaimed hosts a launch was refused on: when to ask again, and why.
        # In memory on purpose: a restarted gateway simply asks the host again.
        self._deferred: dict[str, tuple[datetime, str]] = {}
        self._status = PlacementStatus(enabled=settings.enabled, holder=self._holder)

    @property
    def status(self) -> PlacementStatus:
        return self._status

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return self._profiles

    @property
    def launch_blockers(self) -> dict[str, str]:
        """Last observed host blockers, retained until a fresh check succeeds."""
        return {host: reason for host, (_, reason) in self._deferred.items()}

    def start(self) -> None:
        if not self._settings.enabled:
            return
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
        while True:
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("placement_pass_failed", exc_info=True)
            await asyncio.sleep(self._settings.interval_seconds)

    async def live_claims(self) -> tuple[tuple[PlacementClaim, ...], tuple[str, ...]]:
        """Read the durable claims now, whether or not the loop is enabled.

        Disabling placement stops new decisions. It does not make the hosts
        automation already placed, or failed to place, disappear.
        """
        stored, unreadable = await self._claims.list()
        return tuple(item.claim for item in stored), tuple(unreadable)

    def node_status(self, hostname: str) -> str | None:
        node = self._registry.get(hostname)
        return node.status.value if node is not None else None

    async def resolve_catalog(self) -> CatalogResolution:
        """Match the catalog against the shared cache, off the event loop."""
        scan = await asyncio.to_thread(self._artifacts.scan)
        return resolve_catalog(self._profiles, scan.artifacts)

    async def reconcile_once(self) -> PlacementStatus:
        """Run one full pass. Never raises for an expected outage."""
        async with self._pass_lock:
            status = await self._reconcile()
            self._status = status
            return status

    async def _reconcile(self) -> PlacementStatus:
        now = self._clock()
        base = PlacementStatus(
            enabled=self._settings.enabled, holder=self._holder, last_run_at=now
        )

        def failed(
            message: str,
            *,
            missing_artifacts: tuple[MissingArtifact, ...] = (),
            claims: tuple[PlacementClaim, ...] = (),
        ) -> PlacementStatus:
            logger.warning("placement_pass_skipped", reason=message)
            return replace(
                base,
                error=message,
                missing_artifacts=missing_artifacts,
                claims=claims,
            )

        try:
            resolution = await self.resolve_catalog()
        except Exception as exc:
            return failed(f"could not scan the GGUF cache: {exc}")
        try:
            claims, unreadable = await self._claims.list()
        except Exception as exc:
            return failed(
                f"could not read placement claims: {exc}",
                missing_artifacts=resolution.missing,
            )
        known_claims = tuple(item.claim for item in claims)
        if self._poller.last_sync is None:
            return failed(
                "QUADS inventory has not synchronized yet",
                missing_artifacts=resolution.missing,
                claims=known_claims,
            )
        try:
            window_end = availability_window_end(self._lookahead_hours)
            available = {
                canonical_hostname(host)
                for host in await self._quads.get_available(end=window_end)
            }
        except Exception as exc:
            return failed(
                f"QUADS availability is unknown: {exc}",
                missing_artifacts=resolution.missing,
                claims=known_claims,
            )

        settled: list[StoredClaim] = []
        for stored in claims:
            kept = await self._settle_claim(stored, available, now)
            if kept is not None:
                settled.append(kept)

        held: dict[str, int] = {}
        for item in settled:
            if item.claim.counts_toward_ratio:
                held[item.claim.profile_id] = held.get(item.claim.profile_id, 0) + 1

        planned = self._planned_profiles(resolution)
        hosts = {canonical_hostname(host.hostname): host for host in self._poller.hosts}
        blocked = {item.claim.hostname for item in settled} | set(unreadable)
        candidates, skipped = self._candidates(available, blocked, planned, now)
        budget = self._settings.max_concurrent - sum(
            1
            for item in settled
            if item.claim.state is ClaimState.PROVISIONING
            and item.claim.holder == self._holder
        )
        if budget > 0:
            candidates, unreachable = await self._ready_candidates(candidates)
            skipped.extend(unreachable)
        plan = plan_placements(planned, held, candidates)
        started: list[str] = []
        classes = {item.hostname: item.gpu_class for item in candidates}

        # Retries first: they keep the profile already chosen for that host.
        for stored in settled:
            if budget <= 0:
                break
            claim = stored.claim
            if (
                claim.state is not ClaimState.FAILED
                or (claim.retry_at is not None and claim.retry_at > now)
                or claim.hostname not in available
            ):
                continue
            # A retry is a placement decision like any other: the same policy
            # applies, evaluated now, not as it stood when the host was claimed.
            refusal = self._policy_refusal(
                claim.hostname,
                hosts.get(claim.hostname),
                planned,
                profile_id=claim.profile_id,
                profile_version=claim.profile_version,
                gpu_class=claim.gpu_class,
            )
            if refusal is not None:
                logger.info(
                    "placement_claim_released",
                    hostname=claim.hostname,
                    reason=refusal,
                )
                if await self._release_failed_claim(stored):
                    skipped.append(SkippedHost(claim.hostname, refusal))
                continue
            resolved = resolution.get(claim.profile_id)
            if resolved is None:
                continue  # Its files are missing for now; keep the claim.
            if await self._launch(resolved, claim.hostname, claim.gpu_class, stored):
                started.append(claim.hostname)
                budget -= 1

        for assignment in plan.assignments:
            if budget <= 0:
                break
            resolved = resolution.get(assignment.profile_id)
            if resolved is None:
                continue
            if await self._launch(
                resolved, assignment.hostname, classes[assignment.hostname], None
            ):
                started.append(assignment.hostname)
                budget -= 1
            elif assignment.hostname in self._deferred:
                skipped.append(
                    SkippedHost(
                        assignment.hostname, self._deferred[assignment.hostname][1]
                    )
                )

        final_claims, final_unreadable = await self._claims.list()
        # Report what is held now, including what this pass just started.
        held_now = {profile.profile_id: 0 for profile in self._profiles}
        for item in final_claims:
            if item.claim.counts_toward_ratio:
                held_now[item.claim.profile_id] = (
                    held_now.get(item.claim.profile_id, 0) + 1
                )
        unfilled_now = {
            key: owed
            for key, target in plan.targets.items()
            if (owed := target - held_now.get(key, 0)) > 0
        }
        return PlacementStatus(
            enabled=self._settings.enabled,
            holder=self._holder,
            last_run_at=now,
            targets=dict(plan.targets),
            held=held_now,
            unfilled=unfilled_now,
            missing_artifacts=resolution.missing,
            claims=tuple(item.claim for item in final_claims),
            unreadable_claims=tuple(final_unreadable),
            started=tuple(started),
            skipped=tuple(skipped),
        )

    def _planned_profiles(self, resolution: CatalogResolution) -> list[PlannedProfile]:
        planned: list[PlannedProfile] = []
        for profile in self._profiles:
            classes = {gpu.key for gpu in profile.gpu_classes}
            if self._settings.require_qualified_gpu:
                classes &= set(profile.qualified_gpus)
            planned.append(
                PlannedProfile(
                    profile_id=profile.profile_id,
                    weight=self._settings.ratios.get(profile.profile_id, 0),
                    gpu_classes=frozenset(classes),
                    preferred_gpu_class=profile.preferred_gpu_class,
                    placeable=resolution.get(profile.profile_id) is not None
                    and bool(classes),
                )
            )
        return planned

    def _gpu_class(self, host: QUADSHost) -> str | None:
        for profile in self._profiles:
            for gpu in profile.gpu_classes:
                if gpu.quads_model_token.lower() in host.gpu_model.lower():
                    return gpu.key
        return None

    def _policy_refusal(
        self,
        hostname: str,
        host: QUADSHost | None,
        planned: Sequence[PlannedProfile],
        *,
        profile_id: str | None = None,
        profile_version: int | None = None,
        gpu_class: str | None = None,
    ) -> str | None:
        """Why policy forbids placing on *hostname* now, or ``None``.

        The single rule for a fresh placement and for a retry. With a
        *profile_id* it also checks that this profile may still be placed on
        this host's GPU class; without one, that some profile may.
        """
        if hostname in set(self._settings.exclude_hosts):
            return "excluded by placement.exclude_hosts"
        if self._settings.only_hosts and hostname not in self._settings.only_hosts:
            return "not listed in placement.only_hosts"
        if host is None:
            return "no longer in the QUADS GPU inventory"
        actual_class = self._gpu_class(host)
        if actual_class is None:
            return "its GPU product is not in the profile catalog"
        if host.gpu_count != 1:
            return (
                f"{host.gpu_count} GPUs: catalog profiles support exactly "
                "one GPU per host"
            )
        if gpu_class is not None and gpu_class != actual_class:
            return f"inventory now reports {actual_class}, not {gpu_class}"
        usable = [
            profile
            for profile in planned
            if profile.weight > 0
            and actual_class in profile.gpu_classes
            and (profile_id is None or profile.profile_id == profile_id)
        ]
        if profile_id is None:
            if not any(profile.placeable for profile in usable):
                return f"no placeable profile is qualified for {actual_class}"
            return None
        if not usable:
            return (
                f"profile {profile_id} is no longer weighted and qualified "
                f"for {actual_class}"
            )
        current = next(
            (item for item in self._profiles if item.profile_id == profile_id), None
        )
        if current is None or current.version != profile_version:
            return f"profile {profile_id} changed version"
        return None

    def _candidates(
        self,
        available: set[str],
        blocked: set[str],
        planned: Sequence[PlannedProfile],
        now: datetime,
    ) -> tuple[list[Candidate], list[SkippedHost]]:
        candidates: list[Candidate] = []
        skipped: list[SkippedHost] = []
        inventory = {canonical_hostname(host.hostname) for host in self._poller.hosts}
        for hostname in [name for name in self._deferred if name not in inventory]:
            del self._deferred[hostname]
        for host in sorted(self._poller.hosts, key=lambda item: item.hostname):
            hostname = canonical_hostname(host.hostname)
            gpu_class = self._gpu_class(host)
            if gpu_class is None:
                continue  # Not a GPU product any profile targets.
            if hostname in blocked:
                continue  # Already claimed; reported through the claim list.
            node = self._registry.get(hostname)
            reason = self._policy_refusal(hostname, host, planned)
            if reason is None:
                if hostname not in available:
                    reason = "not free for the whole QUADS scheduling window"
                elif node is not None and not _is_free_pool_node(node):
                    reason = f"already has a {node.status.value} node record"
                elif self._provisioner.host_operation_in_progress(hostname):
                    reason = "another lifecycle operation is in progress"
                elif hostname in self._deferred:
                    ask_again_at, why = self._deferred[hostname]
                    if ask_again_at > now:
                        reason = why
                    else:
                        del self._deferred[hostname]
            if reason is not None:
                skipped.append(SkippedHost(hostname, reason))
            else:
                candidates.append(Candidate(hostname, gpu_class))
        return candidates, skipped

    async def _settle_claim(
        self, stored: StoredClaim, available: set[str], now: datetime
    ) -> StoredClaim | None:
        """Bring one claim in line with reality. ``None``: the claim is gone."""
        claim = stored.claim
        node = self._registry.get(claim.hostname)
        if (
            node is not None
            and not _is_free_pool_node(node)
            and (
                node.placement is None
                or node.placement.claim_id != claim.claim_id
                or node.owner
                or node.self_setup
                or not node.managed
            )
        ):
            # Somebody set this host up by hand. It is theirs now.
            logger.info("placement_claim_superseded", hostname=claim.hostname)
            return await self._drop(stored)

        serving = (
            node is not None
            and node.placement is not None
            and node.placement.claim_id == claim.claim_id
            and node.status in _SERVING_NODE_STATUSES
        )
        if claim.state is ClaimState.PROVISIONING:
            if claim.holder == self._holder:
                if claim.hostname in self._owned:
                    return stored
                # This gateway holds the claim but runs no provision for it:
                # the final state write failed. Nothing is in flight.
            else:
                stale_after = timedelta(seconds=self._settings.claim_stale_seconds)
                if now - claim.heartbeat_at < stale_after:
                    return stored
        if serving and claim.state in (ClaimState.PROVISIONING, ClaimState.FAILED):
            # The node this claim produced is registered and serving: the
            # provision succeeded even though its final claim write did not.
            return await self._rewrite(
                stored,
                state=ClaimState.ACTIVE,
                attempts=0,
                holder=self._holder,
                retry_at=None,
                last_error="",
                now=now,
            )
        if claim.state is ClaimState.PROVISIONING:
            # The abandoned attempt was counted when it started, so it spends
            # budget like any other failure.
            spent = claim.attempts >= self._settings.max_attempts
            return await self._rewrite(
                stored,
                state=ClaimState.EXHAUSTED if spent else ClaimState.FAILED,
                holder=self._holder,
                retry_at=None if spent else now,
                last_error="the provisioning gateway stopped refreshing this claim",
                now=now,
            )

        if claim.hostname not in available:
            # QUADS needs the host back. The schedule enforcer tears down a
            # healthy node; a failed one is only a record, cleared here.
            if node is None:
                return await self._drop(stored)
            if node.status in _RETRYABLE_NODE_STATUSES:
                if not await self._clear_own_node(claim):
                    return stored
                return await self._drop(stored)
            return stored

        if claim.state is ClaimState.ACTIVE and node is None:
            # The node's leased key is gone while QUADS still shows the host
            # free. The server may well be running; re-provisioning is
            # idempotent, and waiting one stale period avoids racing a
            # registration that is merely late. The last provision succeeded,
            # so this starts a new attempt budget (claims written before a
            # success reset the count may still carry an old one).
            return await self._rewrite(
                stored,
                state=ClaimState.FAILED,
                attempts=0,
                retry_at=now + timedelta(seconds=self._settings.claim_stale_seconds),
                last_error="the node record disappeared",
                now=now,
            )
        return stored

    async def _release_failed_claim(self, stored: StoredClaim) -> bool:
        """Give up a failed claim: clear its own node record, then the claim.

        A node record that is no longer this claim's (someone owns it, adopted
        it or set it up again) is left exactly as it is.
        """
        node = self._registry.get(stored.claim.hostname)
        if (
            node is not None
            and _is_own_failed_node(node, stored.claim)
            and not await self._clear_own_node(stored.claim)
        ):
            return False
        return await self._claims.delete(stored)

    @staticmethod
    def _may_replace(node: Node | None, existing: StoredClaim | None) -> bool:
        """Whether a launch may provision over *node*: free, or its own failure."""
        return (
            node is None
            or _is_free_pool_node(node)
            or (existing is not None and _is_own_failed_node(node, existing.claim))
        )

    async def _remote_work_blocker(self, hostname: str) -> str | None:
        """Why a launch must wait for the host itself, or ``None``."""
        try:
            async with asyncio.timeout(_REMOTE_PROBE_TIMEOUT_SECONDS):
                running = await self._provisioner.remote_lifecycle_processes(hostname)
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            return f"could not verify that no earlier provisioning is running: {detail}"
        if running:
            return "an earlier provisioning command is still running on the host"
        return None

    async def _ready_candidates(
        self, candidates: list[Candidate]
    ) -> tuple[list[Candidate], list[SkippedHost]]:
        """Apportion profiles against hosts we can actually launch on.

        A failed first launch otherwise leaves its assigned profile missing
        while healthy hosts permanently receive the wrong small-fleet mix.
        Launch still repeats its check under its own lease to close the gap
        between this snapshot and the actual mutation.
        """
        limit = asyncio.Semaphore(8)

        async def check(candidate: Candidate) -> str | None:
            async with limit:
                hostname = candidate.hostname
                lease = await self._provisioner.try_reserve_host(hostname)
                if lease is None:
                    return "host lifecycle operation in progress"
                try:
                    if not self._may_replace(self._registry.get(hostname), None):
                        return "node changed before readiness check"
                    blocker = await self._remote_work_blocker(hostname)
                    if blocker is not None:
                        reason = f"blocked: {blocker}"
                        self._deferred[hostname] = (
                            self._clock()
                            + timedelta(seconds=self._settings.retry_backoff_seconds),
                            reason,
                        )
                        return reason
                    if not self._may_replace(self._registry.get(hostname), None):
                        return "node changed during readiness check"
                    return None
                finally:
                    lease.release()

        reasons = await asyncio.gather(*(check(candidate) for candidate in candidates))
        ready, skipped = [], []
        for candidate, reason in zip(candidates, reasons, strict=True):
            if reason is None:
                ready.append(candidate)
            else:
                skipped.append(SkippedHost(candidate.hostname, reason))
        return ready, skipped

    async def _drop(self, stored: StoredClaim) -> StoredClaim | None:
        if await self._claims.delete(stored):
            return None
        return stored

    async def _rewrite(
        self, stored: StoredClaim, *, now: datetime, **changes: object
    ) -> StoredClaim:
        claim = stored.claim.model_copy(update={**changes, "updated_at": now})
        revision = await self._claims.update(stored, claim)
        if revision is None:
            return stored
        return StoredClaim(claim, revision)

    async def _clear_own_node(self, claim: PlacementClaim) -> bool:
        """Remove a failed node record, but only one this claim produced."""
        lease = await self._provisioner.try_reserve_host(claim.hostname)
        if lease is None:
            return False
        try:
            node = self._registry.get(claim.hostname)
            if node is None:
                return True
            if not _is_own_failed_node(node, claim):
                return False
            await self._provisioner.cleanup_stale_node(claim.hostname)
            return True
        except Exception:
            logger.warning(
                "placement_cleanup_failed", hostname=claim.hostname, exc_info=True
            )
            return False
        finally:
            lease.release()

    async def _launch(
        self,
        resolved: ResolvedProfile,
        hostname: str,
        gpu_class: str,
        existing: StoredClaim | None,
    ) -> bool:
        """Claim *hostname* and start provisioning it. ``False``: not started."""
        lease = await self._provisioner.try_reserve_host(hostname)
        if lease is None:
            return False
        handed_over = False
        try:
            # Re-check under the lease: an operator may have acted since the
            # snapshot this pass planned from.
            if not self._may_replace(self._registry.get(hostname), existing):
                return False
            if (
                existing is not None
                and existing.claim.attempts >= self._settings.max_attempts
            ):
                # The budget is checked here, before anything touches the host,
                # so no path into a launch can spend more than was configured:
                # not a recovered claim, and not a limit lowered since.
                await self._rewrite(
                    existing,
                    now=self._clock(),
                    state=ClaimState.EXHAUSTED,
                    retry_at=None,
                    last_error=existing.claim.last_error
                    or "the attempt limit was reached",
                )
                return False
            blocker = await self._remote_work_blocker(hostname)
            if blocker is not None:
                # An earlier attempt may still be running on the host: a
                # cancelled gateway task does not always stop the remote
                # command, and a reset claim leaves no trace of it here. Never
                # start a second one on top of it.
                wait = timedelta(seconds=self._settings.retry_backoff_seconds)
                if existing is not None:
                    await self._rewrite(
                        existing,
                        now=self._clock(),
                        retry_at=self._clock() + wait,
                        last_error=f"blocked: {blocker}",
                    )
                else:
                    # No attempt was made, so no claim is written. The host
                    # sits out of planning instead, which hands its share to
                    # another free host until this one can be asked again.
                    self._deferred[hostname] = (
                        self._clock() + wait,
                        f"blocked: {blocker}",
                    )
                    logger.warning(
                        "placement_launch_blocked", hostname=hostname, reason=blocker
                    )
                return False
            self._deferred.pop(hostname, None)
            # The probe awaited. Writers that do not take the lifecycle lease
            # (health checks, the etcd watch) may have changed the record, so
            # the decision to replace it is made on what is there now.
            node = self._registry.get(hostname)
            if not self._may_replace(node, existing):
                return False
            now = self._clock()
            profile = resolved.profile
            claim = PlacementClaim(
                claim_id=existing.claim.claim_id if existing else uuid.uuid4().hex,
                hostname=hostname,
                profile_id=profile.profile_id,
                profile_version=profile.version,
                gpu_class=gpu_class,
                state=ClaimState.PROVISIONING,
                attempts=(existing.claim.attempts if existing else 0) + 1,
                holder=self._holder,
                created_at=existing.claim.created_at if existing else now,
                updated_at=now,
                heartbeat_at=now,
            )
            revision = (
                await self._claims.update(existing, claim)
                if existing is not None
                else await self._claims.create(claim)
            )
            if revision is None:
                return False  # Another gateway claimed the host first.
            owned = _OwnedClaim(StoredClaim(claim, revision), refreshed_at=now)
            if node is not None and not _is_free_pool_node(node):
                await self._provisioner.cleanup_stale_node(hostname)

            request = profile.runtime_request(
                reserve_mib=self._settings.reserve_mib,
                draft_artifact_id=resolved.draft_artifact_id,
                gpu_class=gpu_class,
            )
            placement = NodePlacement(
                profile_id=profile.profile_id,
                profile_version=profile.version,
                claim_id=claim.claim_id,
            )

            async def run() -> None:
                try:
                    await self._run_owned(
                        owned,
                        self._provisioner.provision(
                            hostname,
                            managed=True,
                            engine=InferenceEngine.LLAMA_CPP,
                            artifact_id=resolved.artifact_id,
                            llamacpp_request=request,
                            lifecycle_lease=lease,
                            placement=placement,
                        ),
                    )
                finally:
                    # The provisioner owns the lease once it starts. This
                    # idempotent release covers a cancel before that point.
                    lease.release()

            background = run()
            try:
                self._provisioner.fire_background(
                    background,
                    provisioning_hostname=hostname,
                    provisioning_identity=ProvisioningIdentity(
                        engine=InferenceEngine.LLAMA_CPP,
                        artifact_id=resolved.artifact_id,
                    ),
                )
            except ProvisioningCapacityError:
                background.close()
                await self._finish(
                    owned, error="provisioning capacity reached", count_attempt=False
                )
                return False
            except Exception:
                background.close()
                await self._finish(
                    owned, error="could not schedule provisioning", count_attempt=False
                )
                raise
            self._owned[hostname] = owned
            handed_over = True
            logger.info(
                "placement_started",
                hostname=hostname,
                profile_id=profile.profile_id,
                attempt=claim.attempts,
            )
            return True
        finally:
            if not handed_over:
                lease.release()

    async def _run_owned(
        self, owned: _OwnedClaim, provision: Coroutine[object, object, None]
    ) -> None:
        """Provision while keeping the claim alive; record how it ended."""
        hostname = owned.stored.claim.hostname
        parent = asyncio.current_task()
        heartbeat = asyncio.create_task(self._heartbeat(owned, parent))
        error: str | None = None
        try:
            await provision
        except asyncio.CancelledError:
            error = (
                "claim ownership was lost during provisioning"
                if owned.lost
                else "provisioning was cancelled"
            )
            raise
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            self._owned.pop(hostname, None)
            if not owned.lost:
                await asyncio.shield(self._finish(owned, error=error))

    async def _heartbeat(
        self, owned: _OwnedClaim, parent: asyncio.Task[object] | None
    ) -> None:
        interval = max(5.0, self._settings.claim_stale_seconds / 4)
        while not owned.lost:
            await asyncio.sleep(interval)
            await self._heartbeat_once(owned, parent)

    async def _heartbeat_once(
        self, owned: _OwnedClaim, parent: asyncio.Task[object] | None
    ) -> None:
        """Refresh the claim, or stop provisioning once it may no longer be ours.

        Ownership is lost outright when the compare-and-swap fails. It is also
        given up when the claim could not be refreshed for half the stale
        period: another gateway may take a claim over after the full period,
        and this one must have stopped, remote command included, before then.
        """
        async with owned.lock:
            if owned.lost:
                return
            now = self._clock()
            claim = owned.stored.claim.model_copy(
                update={"heartbeat_at": now, "updated_at": now}
            )
            reason: str | None = None
            try:
                revision = await self._claims.update(owned.stored, claim)
            except Exception as exc:
                logger.warning(
                    "placement_heartbeat_failed",
                    hostname=claim.hostname,
                    exc_info=True,
                )
                limit = timedelta(seconds=self._settings.claim_stale_seconds / 2)
                if now - owned.refreshed_at < limit:
                    return
                reason = f"the claim could not be refreshed for {limit}: {exc}"
            else:
                if revision is not None:
                    owned.stored = StoredClaim(claim, revision)
                    owned.refreshed_at = now
                    return
                reason = "another gateway took the claim over"
            owned.lost = True
            logger.error("placement_claim_lost", hostname=claim.hostname, reason=reason)
            self._fence(claim.hostname, parent)

    def _fence(self, hostname: str, parent: asyncio.Task[object] | None) -> None:
        """Stop this gateway's provision of *hostname*, remote command included.

        ``cancel_active_provision`` is the explicit cancellation teardown uses:
        unlike a bare task cancel it also stops the detached remote worker.
        """

        async def fence() -> None:
            try:
                await self._provisioner.cancel_active_provision(hostname)
            except Exception:
                logger.error("placement_fence_failed", hostname=hostname, exc_info=True)
                if parent is not None and not parent.done():
                    parent.cancel()

        task = asyncio.create_task(fence())
        self._fencing.add(task)
        task.add_done_callback(self._fencing.discard)

    async def _finish(
        self,
        owned: _OwnedClaim,
        *,
        error: str | None,
        count_attempt: bool = True,
    ) -> None:
        async with owned.lock:
            now = self._clock()
            claim = owned.stored.claim
            attempts = claim.attempts if count_attempt else max(0, claim.attempts - 1)
            if error is None:
                # The budget counts attempts since the last success.
                attempts = 0
                changes: dict[str, object] = {
                    "state": ClaimState.ACTIVE,
                    "retry_at": None,
                    "last_error": "",
                }
            elif attempts >= self._settings.max_attempts:
                changes = {
                    "state": ClaimState.EXHAUSTED,
                    "retry_at": None,
                    "last_error": error[:2000],
                }
            else:
                backoff = self._settings.retry_backoff_seconds * 2 ** max(
                    0, attempts - 1
                )
                changes = {
                    "state": ClaimState.FAILED,
                    "retry_at": now + timedelta(seconds=backoff),
                    "last_error": error[:2000],
                }
            updated = claim.model_copy(
                update={**changes, "attempts": attempts, "updated_at": now}
            )
            try:
                revision = await self._claims.update(owned.stored, updated)
            except Exception:
                logger.error(
                    "placement_claim_finish_failed",
                    hostname=claim.hostname,
                    exc_info=True,
                )
                return
            if revision is None:
                owned.lost = True
                logger.error("placement_claim_lost", hostname=claim.hostname)
                return
            owned.stored = StoredClaim(updated, revision)
            logger.info(
                "placement_finished",
                hostname=claim.hostname,
                profile_id=claim.profile_id,
                state=updated.state.value,
                attempts=attempts,
            )

    async def reset_claim(self, hostname: str) -> bool:
        """Operator action: forget a failed or exhausted claim so it is retried."""
        claims, _ = await self._claims.list()
        for stored in claims:
            if stored.claim.hostname != hostname:
                continue
            if stored.claim.state not in (ClaimState.FAILED, ClaimState.EXHAUSTED):
                raise ValueError(
                    f"claim is {stored.claim.state.value}; only a failed or "
                    "exhausted claim can be reset"
                )
            if not await self._release_failed_claim(stored):
                raise ValueError("the host is busy; try again when it is idle")
            return True
        return False


def _is_own_failed_node(node: Node, claim: PlacementClaim) -> bool:
    """A failed or interrupted node record that this claim, and nobody else, owns.

    Automation clears or replaces only such a record. One that has since been
    given an owner, adopted, made unmanaged or set up again is a person's.
    """
    return (
        node.placement is not None
        and node.placement.claim_id == claim.claim_id
        and node.managed
        and not node.self_setup
        and not node.owner
        and node.status in _RETRYABLE_NODE_STATUSES
    )


def _is_free_pool_node(node: Node) -> bool:
    """A host an operator added to the pool and nobody has set up or owns."""
    return (
        node.status is NodeStatus.AVAILABLE
        and not node.self_setup
        and not node.owner
        and node.placement is None
    )
