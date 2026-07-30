"""Unit tests for CircuitBreaker and CircuitBreakerRegistry.

Tests cover record_failure, record_success, is_open, reset for
CircuitBreaker, and get_or_create, reset, remove for
CircuitBreakerRegistry.
"""

from __future__ import annotations

from inference_proxy.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
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
        registry = CircuitBreakerRegistry()
        registry.get_or_create("node-1")

        registry.remove("node-1")

        # get_or_create should return a new instance
        new_breaker = registry.get_or_create("node-1")
        assert not new_breaker.is_open

    def test_remove_nonexistent_is_silent(self) -> None:
        registry = CircuitBreakerRegistry()

        registry.remove("nonexistent")  # should not raise
