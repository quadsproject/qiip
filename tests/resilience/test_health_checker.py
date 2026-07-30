"""Unit tests for the health checker background thread.

Tests cover:
- Healthy node stays healthy after successful probe
- Node marked UNHEALTHY after 3 consecutive probe failures (D-03)
- Node restored to HEALTHY after 1 successful probe with circuit breaker reset (D-04, D-08)
- Pre-set stop_event exits immediately without probing (D-11)
- HTTP exception during probing counts as failure, does not crash thread
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import httpx

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.resilience.health_checker import run_health_checker


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
) -> Node:
    """Create a Node fixture with the given parameters."""
    return Node(node_id=node_id, endpoint=endpoint, status=status)


class TestHealthyNodeStaysHealthy:
    """A probe returning 200 does not change a healthy node's status."""

    def test_healthy_node_stays_healthy(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0
        original_wait = stop_event.wait

        def stop_after_one_iteration(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                stop_event.set()
                return True
            return original_wait(timeout)

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_one_iteration  # type: ignore[assignment]
            run_health_checker(registry, cb_registry, stop_event, interval=0.01)

        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.HEALTHY


class TestUnhealthyAfterThreeFailures:
    """3 consecutive probe failures mark a node UNHEALTHY (D-03)."""

    def test_three_failures_marks_unhealthy(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0

        def stop_after_three_iterations(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 3:
                stop_event.set()
                return True
            return False

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_three_iterations  # type: ignore[assignment]
            run_health_checker(
                registry,
                cb_registry,
                stop_event,
                interval=0.01,
                failure_threshold=3,
            )

        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.UNHEALTHY

    def test_non_200_counts_as_failure(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0

        def stop_after_three_iterations(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 3:
                stop_event.set()
                return True
            return False

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_three_iterations  # type: ignore[assignment]
            run_health_checker(
                registry,
                cb_registry,
                stop_event,
                interval=0.01,
                failure_threshold=3,
            )

        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.UNHEALTHY


class TestRecoveryAfterOneSuccess:
    """UNHEALTHY node recovers to HEALTHY after 1 successful probe (D-04, D-08)."""

    def test_recovery_restores_healthy_and_resets_circuit_breaker(self) -> None:
        registry = NodeRegistry()
        # Start with an unhealthy node
        node = _make_node(status=NodeStatus.UNHEALTHY)
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()
        # Trip the circuit breaker for this node
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        stop_event = threading.Event()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0

        def stop_after_one_iteration(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                stop_event.set()
                return True
            return False

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_one_iteration  # type: ignore[assignment]
            run_health_checker(registry, cb_registry, stop_event, interval=0.01)

        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.HEALTHY
        assert not breaker.is_open


class TestStopEventExitsImmediately:
    """Pre-set stop_event causes immediate exit without probing (D-11)."""

    def test_preset_stop_event_exits_without_probing(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()

        stop_event = threading.Event()
        stop_event.set()  # Pre-set

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            run_health_checker(registry, cb_registry, stop_event, interval=0.01)

        # Should not have made any HTTP calls
        mock_client.get.assert_not_called()


class TestProbeExceptionDoesNotCrash:
    """Exception during HTTP probe counts as failure, thread continues."""

    def test_exception_counts_as_failure_thread_continues(self) -> None:
        registry = NodeRegistry()
        node = _make_node()
        registry.add(node)
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.TimeoutException("probe timed out")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0

        def stop_after_two_iterations(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 2:
                stop_event.set()
                return True
            return False

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_two_iterations  # type: ignore[assignment]
            run_health_checker(
                registry,
                cb_registry,
                stop_event,
                interval=0.01,
                failure_threshold=3,
            )

        # Thread should have completed without crashing
        # Node should still be HEALTHY (only 2 failures, threshold is 3)
        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.HEALTHY
        # But the mock was called twice (2 iterations)
        assert mock_client.get.call_count == 2


class TestProvisioningNodeSkipped:
    """PROVISIONING nodes are not probed by the health checker (D-09)."""

    def test_provisioning_node_not_probed(self) -> None:
        """Only the HEALTHY node is probed; PROVISIONING node is skipped."""
        registry = NodeRegistry()
        provisioning_node = _make_node(
            node_id="prov-1",
            endpoint="10.0.1.200:8000",
            status=NodeStatus.PROVISIONING,
        )
        healthy_node = _make_node(
            node_id="healthy-1",
            endpoint="10.0.1.100:8000",
            status=NodeStatus.HEALTHY,
        )
        registry.add(provisioning_node)
        registry.add(healthy_node)
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        iteration_count = 0

        def stop_after_one_iteration(timeout: float | None = None) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                stop_event.set()
                return True
            return False

        with patch(
            "inference_proxy.resilience.health_checker.httpx.Client",
            return_value=mock_client,
        ):
            stop_event.wait = stop_after_one_iteration  # type: ignore[assignment]
            run_health_checker(registry, cb_registry, stop_event, interval=0.01)

        # Only the healthy node was probed
        assert mock_client.get.call_count == 1
        mock_client.get.assert_called_once_with("http://10.0.1.100:8000/health")

        # Provisioning node status unchanged
        result_prov = registry.get("prov-1")
        assert result_prov is not None
        assert result_prov.status == NodeStatus.PROVISIONING
