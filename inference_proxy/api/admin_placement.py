"""Admin API for automatic profile placement.

``GET /admin/placement`` reports, in one document, what placement is aiming
for and where it stands: configured ratios, per-profile targets and held
hosts, every claim with its last error, catalog files missing from the shared
cache, hosts that were looked at and skipped (with the reason), and per-model
demand from the existing request counters and token-usage table.

Demand is reported, never acted on: ratios are fixed configuration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from inference_proxy.auth.dependencies import get_auth_store
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.dependencies import (
    get_request_metrics,
    get_settings,
    require_admin_auth,
)
from inference_proxy.config.settings import Settings
from inference_proxy.placement.catalog import CATALOG_VERSION
from inference_proxy.placement.claims import ClaimState, PlacementClaim
from inference_proxy.placement.reconciler import PlacementReconciler
from inference_proxy.routing.request_metrics import RequestMetrics

admin_placement_router = APIRouter(
    prefix="/admin/placement",
    tags=["admin"],
    dependencies=[Depends(require_admin_auth)],
)


class PlacementProfileStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    version: int
    display_name: str
    model: str
    weight: int
    # Hosts this profile should hold, of the automation-owned fleet.
    target: int
    # Every claim that occupies a host: serving + pending + failed-but-retrying.
    held: int
    # Claimed hosts whose node is registered and healthy right now.
    serving: int
    # Claimed hosts still being provisioned. Not capacity yet.
    pending: int
    # Claimed hosts whose last attempt failed, or whose attempts are used up.
    failed: int
    unfilled: int
    artifacts_present: bool
    qualified_gpus: tuple[str, ...]


class MissingArtifactStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    role: str
    repo_id: str
    revision: str
    filename: str
    size_bytes: int


class SkippedHostStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    hostname: str
    reason: str


class ModelUsageStatus(BaseModel):
    """What was served per model. A lower bound on demand, not demand itself.

    ``requests_since_restart`` counts requests this gateway process routed to
    a backend since ``counters_started_at``; it resets on every restart.
    ``recorded_*`` sums the persistent usage table, which only has rows for
    authenticated requests that completed. Neither includes requests that were
    refused, found no healthy node or failed, so a model that is absent or at
    zero may still be in demand. Tokens are work done, not GPU time occupied
    and not queue depth.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    requests_since_restart: int
    recorded_requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class PlacementStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: bool
    enabled: bool
    catalog_version: int
    last_run_at: datetime | None
    error: str
    profiles: tuple[PlacementProfileStatus, ...]
    missing_artifacts: tuple[MissingArtifactStatus, ...]
    claims: tuple[PlacementClaim, ...]
    unreadable_claims: tuple[str, ...]
    skipped_hosts: tuple[SkippedHostStatus, ...]
    usage: tuple[ModelUsageStatus, ...]
    counters_started_at: datetime | None
    usage_note: str = (
        "Served requests and tokens only: refused, unroutable and failed requests "
        "are not counted, so zero does not mean no demand. Request counters reset "
        "when the gateway restarts. Nothing here changes placement; ratios are fixed."
    )


def get_placement_reconciler(request: Request) -> PlacementReconciler | None:
    """Return the reconciler, or None when QUADS is not configured."""
    reconciler = getattr(request.app.state, "placement_reconciler", None)
    return reconciler if isinstance(reconciler, PlacementReconciler) else None


def _usage(
    request_metrics: RequestMetrics, store: AuthStore
) -> tuple[ModelUsageStatus, ...]:
    live = request_metrics.get_per_model()
    recorded = store.get_usage_by_model()
    return tuple(
        ModelUsageStatus(
            model=model,
            requests_since_restart=live.get(model, 0),
            recorded_requests=recorded[model].request_count if model in recorded else 0,
            prompt_tokens=recorded[model].prompt_tokens if model in recorded else 0,
            completion_tokens=(
                recorded[model].completion_tokens if model in recorded else 0
            ),
            total_tokens=recorded[model].total_tokens if model in recorded else 0,
        )
        for model in sorted(set(live) | set(recorded))
    )


@admin_placement_router.get("")
async def get_placement_status(
    reconciler: Annotated[
        PlacementReconciler | None, Depends(get_placement_reconciler)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    request_metrics: Annotated[RequestMetrics, Depends(get_request_metrics)],
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> PlacementStatusResponse:
    """Return placement targets, claims, missing files and model demand."""
    usage = _usage(request_metrics, store)
    if reconciler is None:
        return PlacementStatusResponse(
            available=False,
            enabled=settings.placement.enabled,
            catalog_version=CATALOG_VERSION,
            last_run_at=None,
            error="automatic placement needs QUADS (quads.base_url)",
            profiles=(),
            missing_artifacts=(),
            claims=(),
            unreadable_claims=(),
            skipped_hosts=(),
            usage=usage,
            counters_started_at=None,
        )
    status = reconciler.status
    missing = status.missing_artifacts
    error = status.error
    if status.last_run_at is None:
        # Placement is disabled, or its first pass has not run: the catalog can
        # still be checked against the cache so missing files are visible.
        try:
            missing = (await reconciler.resolve_catalog()).missing
        except Exception as exc:
            error = f"could not scan the GGUF cache: {exc}"
    # Claims are durable state. Read them now, so a disabled or restarted
    # gateway still shows every host automation placed or failed to place.
    try:
        claims, unreadable = await reconciler.live_claims()
    except Exception as exc:
        claims, unreadable = status.claims, status.unreadable_claims
        error = error or f"could not read placement claims: {exc}"
    counts: dict[str, dict[str, int]] = {}
    for claim in claims:
        bucket = counts.setdefault(
            claim.profile_id, {"held": 0, "serving": 0, "pending": 0, "failed": 0}
        )
        if claim.counts_toward_ratio:
            bucket["held"] += 1
        if claim.state is ClaimState.PROVISIONING:
            bucket["pending"] += 1
        elif claim.state is ClaimState.ACTIVE:
            if reconciler.node_status(claim.hostname) == "healthy":
                bucket["serving"] += 1
        else:
            bucket["failed"] += 1
    missing_profiles = {item.profile_id for item in missing}
    return PlacementStatusResponse(
        available=True,
        enabled=status.enabled,
        catalog_version=CATALOG_VERSION,
        last_run_at=status.last_run_at,
        error=error,
        profiles=tuple(
            PlacementProfileStatus(
                profile_id=profile.profile_id,
                version=profile.version,
                display_name=profile.display_name,
                model=profile.target.repo_id,
                weight=settings.placement.ratios.get(profile.profile_id, 0),
                target=status.targets.get(profile.profile_id, 0),
                held=counts.get(profile.profile_id, {}).get("held", 0),
                serving=counts.get(profile.profile_id, {}).get("serving", 0),
                pending=counts.get(profile.profile_id, {}).get("pending", 0),
                failed=counts.get(profile.profile_id, {}).get("failed", 0),
                unfilled=status.unfilled.get(profile.profile_id, 0),
                artifacts_present=profile.profile_id not in missing_profiles,
                qualified_gpus=profile.qualified_gpus,
            )
            for profile in reconciler.profiles
        ),
        missing_artifacts=tuple(
            MissingArtifactStatus(
                profile_id=item.profile_id,
                role=item.role.value,
                repo_id=item.ref.repo_id,
                revision=item.ref.revision,
                filename=item.ref.filename,
                size_bytes=item.ref.size_bytes,
            )
            for item in missing
        ),
        claims=claims,
        unreadable_claims=unreadable,
        skipped_hosts=tuple(
            SkippedHostStatus(hostname=item.hostname, reason=item.reason)
            for item in status.skipped
        ),
        usage=usage,
        counters_started_at=reconciler.started_at,
    )


@admin_placement_router.delete("/claims/{hostname}", status_code=204)
async def reset_placement_claim(
    hostname: str,
    reconciler: Annotated[
        PlacementReconciler | None, Depends(get_placement_reconciler)
    ],
) -> None:
    """Forget a failed or exhausted claim so the host is considered again."""
    if reconciler is None:
        raise HTTPException(status_code=404, detail="Automatic placement is off")
    try:
        removed = await reconciler.reset_claim(hostname.strip().lower().rstrip("."))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="No such placement claim")
