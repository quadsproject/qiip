"""Fixed-ratio placement: worked fleet cases plus randomized invariants."""

from __future__ import annotations

import random

import pytest

from inference_proxy.placement.planner import (
    Candidate,
    PlannedProfile,
    apportion,
    plan_placements,
)

BOTH = frozenset({"l4", "a30"})
Q38 = PlannedProfile("q38", 60, BOTH, preferred_gpu_class="a30")
Q36 = PlannedProfile("q36", 20, BOTH)
MUSE = PlannedProfile("muse", 20, BOTH)
PROFILES = [Q38, Q36, MUSE]


def _l4(count: int) -> list[Candidate]:
    return [Candidate(f"l4-{index:02d}", "l4") for index in range(count)]


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),  # fewer than three: Qwen3.8 first,
        (2, (1, 1, 0)),  # then Qwen3.6
        (3, (1, 1, 1)),  # one each from three hosts up
        (4, (2, 1, 1)),
        (5, (3, 1, 1)),
        (8, (5, 2, 1)),  # 4.8 / 1.6 / 1.6: the tie goes to catalog order
        (10, (6, 2, 2)),
        (100, (60, 20, 20)),
    ],
)
def test_apportion_matches_the_agreed_rounding(
    total: int, expected: tuple[int, int, int]
) -> None:
    targets = apportion(total, PROFILES)

    assert (targets["q38"], targets["q36"], targets["muse"]) == expected


FOUR = [
    PlannedProfile("q38", 9, BOTH, preferred_gpu_class="a30"),
    PlannedProfile("q36", 2, BOTH),
    PlannedProfile("muse", 2, BOTH),
    PlannedProfile("gemma", 2, BOTH),
]


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (1, (1, 0, 0, 0)),  # a short fleet is served in catalog order:
        (2, (1, 1, 0, 0)),  # Qwen3.8, Qwen3.6, Muse, then Gemma
        (3, (1, 1, 1, 0)),
        (4, (1, 1, 1, 1)),  # one each from four hosts up
        (5, (2, 1, 1, 1)),
        (8, (5, 1, 1, 1)),
        (10, (6, 2, 1, 1)),  # 1.33 each for the three: the tie goes to catalog order
        (15, (9, 2, 2, 2)),
        (30, (18, 4, 4, 4)),
    ],
)
def test_four_profiles_at_nine_two_two_two(
    total: int, expected: tuple[int, int, int, int]
) -> None:
    targets = apportion(total, FOUR)

    assert tuple(targets[key] for key in ("q38", "q36", "muse", "gemma")) == expected


def test_a_short_fleet_ignores_weights_and_follows_catalog_order() -> None:
    heavy_last = [
        PlannedProfile("first", 1, BOTH),
        PlannedProfile("second", 1, BOTH),
        PlannedProfile("third", 98, BOTH),
    ]

    assert apportion(2, heavy_last) == {"first": 1, "second": 1, "third": 0}


def test_eight_free_l4_hosts_become_five_two_one() -> None:
    plan = plan_placements(PROFILES, {}, _l4(8))

    counts = {key: 0 for key in ("q38", "q36", "muse")}
    for assignment in plan.assignments:
        counts[assignment.profile_id] += 1
    assert counts == {"q38": 5, "q36": 2, "muse": 1}
    assert len({item.hostname for item in plan.assignments}) == 8
    assert not plan.unfilled


def test_qwen38_takes_the_a30_hosts_first() -> None:
    candidates = [*_l4(8), Candidate("a30-0", "a30"), Candidate("a30-1", "a30")]

    plan = plan_placements(PROFILES, {}, candidates)

    by_host = {item.hostname: item.profile_id for item in plan.assignments}
    assert by_host["a30-0"] == by_host["a30-1"] == "q38"
    assert sum(1 for value in by_host.values() if value == "q38") == 6


def test_other_profiles_leave_the_a30_to_qwen38_while_it_is_still_owed() -> None:
    # One A30 and two L4 hosts free; Qwen3.8 is owed exactly one.
    candidates = [Candidate("a30-0", "a30"), *_l4(2)]

    plan = plan_placements(PROFILES, {}, candidates)

    by_host = {item.hostname: item.profile_id for item in plan.assignments}
    assert by_host == {"a30-0": "q38", "l4-00": "q36", "l4-01": "muse"}


def test_in_flight_placements_are_counted_so_nothing_is_assigned_twice() -> None:
    # Five Qwen3.8 placements are already held (some still provisioning).
    plan = plan_placements(PROFILES, {"q38": 5}, _l4(3))

    assert sorted(item.profile_id for item in plan.assignments) == [
        "muse",
        "q36",
        "q36",
    ]


def test_an_over_represented_profile_gets_nothing_and_nothing_is_moved() -> None:
    plan = plan_placements(PROFILES, {"q38": 8}, _l4(2))

    assert sorted(item.profile_id for item in plan.assignments) == ["muse", "q36"]
    assert plan.held["q38"] == 8


def test_a_profile_with_missing_files_keeps_its_share_unplaced() -> None:
    muse = PlannedProfile("muse", 20, BOTH, placeable=False)

    plan = plan_placements([Q38, Q36, muse], {}, _l4(8))

    assert sorted(item.profile_id for item in plan.assignments).count("muse") == 0
    assert len(plan.assignments) == 7
    assert plan.unfilled == {"muse": 1}


def test_a_profile_not_qualified_for_a_gpu_class_never_lands_on_it() -> None:
    q38_l4_only = PlannedProfile("q38", 60, frozenset({"l4"}))

    plan = plan_placements([q38_l4_only, Q36, MUSE], {}, [Candidate("a30-0", "a30")])

    assert plan.assignments == ()
    assert plan.unfilled == {"q38": 1}


def test_zero_weight_profiles_are_never_placed() -> None:
    plan = plan_placements([Q38, PlannedProfile("q36", 0, BOTH), MUSE], {}, _l4(4))

    assert {item.profile_id for item in plan.assignments} == {"q38", "muse"}


def test_a_host_offered_with_two_gpu_classes_is_an_error() -> None:
    with pytest.raises(ValueError, match="two GPU classes"):
        plan_placements(PROFILES, {}, [Candidate("h", "l4"), Candidate("h", "a30")])


def test_randomized_invariants() -> None:
    rng = random.Random(20260919)
    for _ in range(400):
        weights = [rng.randint(0, 9) for _ in range(3)]
        if not any(weights):
            weights[0] = 1
        profiles = [
            PlannedProfile("a", weights[0], BOTH, preferred_gpu_class="a30"),
            PlannedProfile("b", weights[1], BOTH, placeable=rng.random() > 0.2),
            PlannedProfile("c", weights[2], frozenset({"l4"})),
        ]
        held = {key: rng.randint(0, 4) for key in "abc" if rng.random() > 0.5}
        candidates = [
            Candidate(f"h{index:02d}", rng.choice(["l4", "a30"]))
            for index in range(rng.randint(0, 12))
        ]
        shuffled = candidates[:]
        rng.shuffle(shuffled)

        plan = plan_placements(profiles, held, candidates)

        # Input order never matters.
        assert plan_placements(profiles, held, shuffled) == plan
        total = sum(held.values()) + len(candidates)
        assert sum(plan.targets.values()) == total
        weighted = [item for item in profiles if item.weight > 0]
        if total >= len(weighted):
            assert all(plan.targets[item.profile_id] >= 1 for item in weighted)
        hosts = [item.hostname for item in plan.assignments]
        assert len(hosts) == len(set(hosts)) <= len(candidates)
        classes = {item.hostname: item.gpu_class for item in candidates}
        by_id = {item.profile_id: item for item in profiles}
        granted: dict[str, int] = {}
        for assignment in plan.assignments:
            profile = by_id[assignment.profile_id]
            assert profile.weight > 0 and profile.placeable
            assert classes[assignment.hostname] in profile.gpu_classes
            granted[profile.profile_id] = granted.get(profile.profile_id, 0) + 1
        for key, count in granted.items():
            # Never more than the profile is still owed.
            assert count <= max(0, plan.targets[key] - held.get(key, 0))


@pytest.mark.parametrize("l4_hostname", ["aaa-l4", "zzz-l4"])
def test_four_profiles_reserve_the_only_l4_for_gemma(l4_hostname: str) -> None:
    profiles = [*FOUR[:3], PlannedProfile("gemma", 2, frozenset({"l4"}))]
    candidates = [Candidate(l4_hostname, "l4")] + [
        Candidate(f"a30-{index}", "a30") for index in range(3)
    ]

    plan = plan_placements(profiles, {}, candidates)

    assert not plan.unfilled
    assert {item.profile_id for item in plan.assignments} == {
        "q38",
        "q36",
        "muse",
        "gemma",
    }
    assert (
        next(
            item.profile_id for item in plan.assignments if item.hostname == l4_hostname
        )
        == "gemma"
    )


def test_matching_can_reassign_a_chain_of_tentative_placements() -> None:
    profiles = [
        PlannedProfile("flex", 1, frozenset({"a", "b"})),
        PlannedProfile("middle", 1, frozenset({"b", "c"})),
        PlannedProfile("restricted", 1, frozenset({"a"})),
    ]
    plan = plan_placements(profiles, {}, [Candidate(k, k) for k in "abc"])
    assert not plan.unfilled
    assert {item.hostname: item.profile_id for item in plan.assignments} == {
        "a": "restricted",
        "b": "flex",
        "c": "middle",
    }
