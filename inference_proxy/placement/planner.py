"""Pure, deterministic fixed-ratio placement.

Nothing here reads a clock, a network or etcd. The reconciler gathers the
inputs, and this module answers one question: given the placements automation
already holds and the hosts that are free, which free host gets which profile.

Placements already held are never moved. Ratios therefore steer only where new
capacity goes; an over-represented profile simply receives no more hosts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class PlannedProfile:
    """The planner's view of one catalog profile, in catalog order."""

    profile_id: str
    weight: int
    gpu_classes: frozenset[str]
    preferred_gpu_class: str | None = None
    # False when a file is missing or the profile is otherwise unplaceable.
    # Its share is then left unplaced, not handed to another profile: a host
    # given away could never be taken back without moving a healthy placement.
    placeable: bool = True


@dataclass(frozen=True)
class Candidate:
    """One free, eligible single-GPU host."""

    hostname: str
    gpu_class: str


@dataclass(frozen=True)
class Assignment:
    hostname: str
    profile_id: str


@dataclass(frozen=True)
class PlacementPlan:
    """Targets for the whole automation-owned fleet plus the new assignments."""

    targets: Mapping[str, int]
    held: Mapping[str, int]
    assignments: tuple[Assignment, ...]
    # Profile id -> hosts it is still owed but could not be given this round.
    unfilled: Mapping[str, int]


def apportion(total: int, profiles: Sequence[PlannedProfile]) -> dict[str, int]:
    """Split *total* hosts by weight: largest remainder, catalog-order ties.

    With at least as many hosts as weighted profiles, every weighted profile
    gets one host: the host is taken from whichever profile holds the most,
    the later one on a tie. With fewer hosts than that, the first profiles in
    catalog order get one host each.
    """
    weighted = [profile for profile in profiles if profile.weight > 0]
    targets = {profile.profile_id: 0 for profile in profiles}
    if total <= 0 or not weighted:
        return targets
    if total < len(weighted):
        # A fleet too small for one host each: catalog order decides who is
        # served first, whatever the weights.
        for profile in weighted[:total]:
            targets[profile.profile_id] = 1
        return targets
    weight_sum = sum(profile.weight for profile in weighted)
    shares = [Fraction(total * profile.weight, weight_sum) for profile in weighted]
    for profile, share in zip(weighted, shares, strict=True):
        targets[profile.profile_id] = share.numerator // share.denominator
    leftover = total - sum(targets.values())
    by_remainder = sorted(
        range(len(weighted)),
        key=lambda index: (-(shares[index] % 1), index),
    )
    for index in by_remainder[:leftover]:
        targets[weighted[index].profile_id] += 1

    if total >= len(weighted):
        for profile in weighted:
            if targets[profile.profile_id] > 0:
                continue
            donor = max(
                range(len(weighted)),
                key=lambda index: (targets[weighted[index].profile_id], index),
            )
            targets[weighted[donor].profile_id] -= 1
            targets[profile.profile_id] += 1
    return targets


def plan_placements(
    profiles: Sequence[PlannedProfile],
    held: Mapping[str, int],
    candidates: Sequence[Candidate],
) -> PlacementPlan:
    """Assign free hosts to profiles to approach the configured ratios.

    *held* counts, per profile, every placement automation already owns,
    including ones still being provisioned, so an in-flight placement is never
    assigned twice. The ratio denominator is those plus the free hosts.
    """
    ordered = sorted(set(candidates), key=lambda item: item.hostname)
    if len({item.hostname for item in ordered}) != len(ordered):
        raise ValueError("a host was offered with two GPU classes")
    known = {profile.profile_id for profile in profiles}
    held_counts = {
        profile.profile_id: max(0, held.get(profile.profile_id, 0))
        for profile in profiles
    }
    # Placements of a profile that left the catalog still occupy hosts.
    retired = sum(max(0, count) for key, count in held.items() if key not in known)
    total = sum(held_counts.values()) + len(ordered)
    targets = apportion(total, profiles)
    deficits = {key: max(0, targets[key] - held_counts[key]) for key in targets}

    # Decide how many of the free hosts each profile receives.
    grants = {key: 0 for key in targets}
    remaining = len(ordered)
    order = {profile.profile_id: index for index, profile in enumerate(profiles)}
    placeable = {
        profile.profile_id: profile for profile in profiles if profile.placeable
    }
    while remaining > 0:
        open_profiles = [key for key in placeable if deficits[key] - grants[key] > 0]
        if not open_profiles:
            break
        chosen = min(
            open_profiles,
            key=lambda key: (-(deficits[key] - grants[key]), order[key]),
        )
        grants[chosen] += 1
        remaining -= 1

    # Match grants to compatible hosts. If a constrained profile needs a host
    # already chosen for a flexible profile, move only that tentative assignment
    # along an augmenting path. Existing held placements are never touched.
    assigned: dict[str, PlannedProfile] = {}
    pending = dict(grants)

    def take(profile: PlannedProfile, visited: set[str]) -> bool:
        contested = {
            other.preferred_gpu_class
            for other in profiles
            if other.profile_id != profile.profile_id
            and pending.get(other.profile_id, 0) > 0
        }
        usable = sorted(
            (item for item in ordered if item.gpu_class in profile.gpu_classes),
            key=lambda item: (
                item.gpu_class != profile.preferred_gpu_class,
                item.gpu_class in contested,
                item.hostname,
            ),
        )
        # Prefer a free host over displacing an earlier profile's preference.
        for host in usable:
            if host.hostname not in assigned and host.hostname not in visited:
                assigned[host.hostname] = profile
                return True
        for host in usable:
            if host.hostname in visited:
                continue
            visited.add(host.hostname)
            if take(assigned[host.hostname], visited):
                assigned[host.hostname] = profile
                return True
        return False

    for profile in profiles:
        while pending.get(profile.profile_id, 0) > 0:
            pending[profile.profile_id] -= 1
            if not take(profile, set()):
                break

    assignments = sorted(
        (Assignment(host, profile.profile_id) for host, profile in assigned.items()),
        key=lambda item: (order[item.profile_id], item.hostname),
    )
    unfilled = dict(deficits)
    for assignment in assignments:
        unfilled[assignment.profile_id] -= 1

    return PlacementPlan(
        targets=targets,
        held={**held_counts, **({"(retired)": retired} if retired else {})},
        assignments=tuple(assignments),
        unfilled={key: value for key, value in unfilled.items() if value > 0},
    )
