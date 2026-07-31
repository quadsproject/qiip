"""Unit tests for CircuitBreaker and CircuitBreakerRegistry.

Tests cover CLOSED, OPEN and HALF_OPEN transitions plus registry lifecycle.
"""

from __future__ import annotations

import threading

from inference_proxy.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerState,
)

# -- CircuitBreaker tests --


class TestRecordFailure:
    """record_failure() increments failure count and trips breaker at threshold."""

    def test_single_failure_does_not_trip(self) -> None:
        breaker = CircuitBreaker(threshold=3)

        breaker.record_failure()

        assert not breaker.is_open

    def test_two_failures_does_not_trip(self) -> None:
        breaker = CircuitBreaker(threshold=3)

        breaker.record_failure()
        breaker.record_failure()

        assert not breaker.is_open

    def test_three_failures_trips_to_open(self) -> None:
        breaker = CircuitBreaker(threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open

    def test_custom_threshold_trips_at_threshold(self) -> None:
        breaker = CircuitBreaker(threshold=5)

        for _ in range(4):
            breaker.record_failure()
        assert not breaker.is_open

        breaker.record_failure()
        assert breaker.is_open


class TestRecordSuccess:
    """record_success() resets failure count and state to closed."""

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        breaker.record_success()

        # Two more failures should not trip (count was reset)
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open

    def test_success_resets_open_breaker_to_closed(self) -> None:
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        breaker.record_success()

        assert not breaker.is_open


class TestIsOpen:
    """is_open property returns the current breaker state."""

    def test_new_breaker_is_closed(self) -> None:
        breaker = CircuitBreaker()

        assert not breaker.is_open

    def test_tripped_breaker_is_open(self) -> None:
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open


class TestHalfOpen:
    """OPEN breakers admit exactly one thread-safe recovery probe."""

    def test_only_one_half_open_probe_is_admitted(self) -> None:
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure()
        start = threading.Event()
        results: list[bool] = []
        results_lock = threading.Lock()

        def attempt_probe() -> None:
            assert start.wait(timeout=1)
            admitted = breaker.try_half_open()
            with results_lock:
                results.append(admitted)

        threads = [
            threading.Thread(target=attempt_probe, daemon=True) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=1)

        assert all(not thread.is_alive() for thread in threads)
        assert results.count(True) == 1
        assert results.count(False) == 7
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        assert breaker.is_open

    def test_half_open_failure_reopens_breaker(self) -> None:
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure()
        assert breaker.try_half_open()

        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.is_open

    def test_half_open_success_closes_breaker(self) -> None:
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure()
        assert breaker.try_half_open()

        breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert not breaker.is_open


class TestReset:
    """reset() sets state to closed and clears failure count."""

    def test_reset_closes_open_breaker(self) -> None:
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        breaker.reset()

        assert not breaker.is_open

    def test_reset_clears_failure_count(self) -> None:
        breaker = CircuitBreaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        breaker.reset()

        # Three more failures needed to trip again
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open

        breaker.record_failure()
        assert breaker.is_open


# -- CircuitBreakerRegistry tests --


class TestGetOrCreate:
    """get_or_create() returns existing breaker or creates new one."""

    def test_creates_new_breaker_for_unknown_node(self) -> None:
        registry = CircuitBreakerRegistry()

        breaker = registry.get_or_create("node-1")

        assert isinstance(breaker, CircuitBreaker)
        assert not breaker.is_open

    def test_returns_same_instance_on_second_call(self) -> None:
        registry = CircuitBreakerRegistry()

        first = registry.get_or_create("node-1")
        second = registry.get_or_create("node-1")

        assert first is second

    def test_different_nodes_get_different_breakers(self) -> None:
        registry = CircuitBreakerRegistry()

        breaker_1 = registry.get_or_create("node-1")
        breaker_2 = registry.get_or_create("node-2")

        assert breaker_1 is not breaker_2

    def test_created_breaker_uses_registry_threshold(self) -> None:
        registry = CircuitBreakerRegistry(threshold=5)

        breaker = registry.get_or_create("node-1")

        # Should not trip after 4 failures with threshold=5
        for _ in range(4):
            breaker.record_failure()
        assert not breaker.is_open

        breaker.record_failure()
        assert breaker.is_open


class TestRegistryReset:
    """reset() delegates to the breaker's reset method."""

    def test_reset_resets_existing_breaker(self) -> None:
        registry = CircuitBreakerRegistry(threshold=3)
        breaker = registry.get_or_create("node-1")
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        registry.reset("node-1")

        assert not breaker.is_open

    def test_reset_nonexistent_node_is_silent(self) -> None:
        registry = CircuitBreakerRegistry()

        registry.reset("nonexistent")  # should not raise


class TestRegistryRemove:
    """remove() removes the breaker for a node_id."""

    def test_remove_clears_breaker(self) -> None:
        registry = CircuitBreakerRegistry(threshold=3)
        original = registry.get_or_create("node-1")
        for _ in range(3):
            original.record_failure()
        assert original.is_open

        registry.remove("node-1")

        assert registry.get("node-1") is None
        new_breaker = registry.get_or_create("node-1")
        assert new_breaker is not original
        assert not new_breaker.is_open
        new_breaker.record_failure()
        new_breaker.record_failure()
        assert not new_breaker.is_open

    def test_remove_nonexistent_is_silent(self) -> None:
        registry = CircuitBreakerRegistry()

        registry.remove("nonexistent")  # should not raise
