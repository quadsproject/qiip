"""Unit tests for the health checker background thread.

Tests cover:
- Healthy node stays healthy after successful probe
- Node marked UNHEALTHY after 3 consecutive probe failures (D-03)
- Health-demoted nodes recover from liveness while open breakers require inference
- Pre-set stop_event exits immediately without probing (D-11)
- HTTP exception during probing counts as failure, does not crash thread
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerState,
)
from inference_proxy.resilience.health_checker import (
    _ConsecutiveFailures,
    _probe_all_nodes,
    run_health_checker,
)
from inference_proxy.routing.connection_tracker import ConnectionTracker


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
) -> Node:
    """Create a Node fixture with the given parameters."""
    return Node(node_id=node_id, endpoint=endpoint, status=status, model=model)


class _FailureCounts(_ConsecutiveFailures):
    """Real counter behavior without registering a removal listener."""

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self._counts = dict(counts or {})
        self._lock = threading.Lock()
        self._unregister = lambda: None


def _assert_breaker_state(
    breaker: CircuitBreaker,
    expected: CircuitBreakerState,
) -> None:
    """Assert mutable breaker state without retaining mypy's prior narrowing."""
    assert breaker.state is expected


def test_health_cycle_removes_idle_draining_node_without_request_traffic() -> None:
    """R7: periodic health work removes a ghost without request finalization."""
    registry = NodeRegistry()
    registry.add(_make_node(status=NodeStatus.DRAINING))
    tracker = ConnectionTracker()

    _probe_all_nodes(
        registry,
        CircuitBreakerRegistry(),
        MagicMock(spec=httpx.Client),
        _FailureCounts(),
        3,
        connection_tracker=tracker,
    )

    assert registry.get("node-1") is None


@pytest.mark.parametrize("status", list(NodeStatus))
@pytest.mark.parametrize(
    ("probe_succeeded", "expected_by_status", "reset_statuses"),
    [
        (
            True,
            {
                NodeStatus.HEALTHY: NodeStatus.HEALTHY,
                NodeStatus.UNHEALTHY: NodeStatus.HEALTHY,
                NodeStatus.DRAINING: NodeStatus.DRAINING,
                NodeStatus.PROVISIONING: NodeStatus.PROVISIONING,
                NodeStatus.FAILED: NodeStatus.FAILED,
                NodeStatus.UNKNOWN: NodeStatus.HEALTHY,
            },
            {NodeStatus.UNHEALTHY, NodeStatus.UNKNOWN},
        ),
        (
            False,
            {
                NodeStatus.HEALTHY: NodeStatus.UNHEALTHY,
                NodeStatus.UNHEALTHY: NodeStatus.UNHEALTHY,
                NodeStatus.DRAINING: NodeStatus.DRAINING,
                NodeStatus.PROVISIONING: NodeStatus.PROVISIONING,
                NodeStatus.FAILED: NodeStatus.FAILED,
                NodeStatus.UNKNOWN: NodeStatus.UNHEALTHY,
            },
            set(),
        ),
    ],
    ids=["success", "failure-past-threshold"],
)
def test_probe_transition_matrix(
    status: NodeStatus,
    probe_succeeded: bool,
    expected_by_status: dict[NodeStatus, NodeStatus],
    reset_statuses: set[NodeStatus],
) -> None:
    """Every status has an explicit probe transition and breaker-reset policy."""
    registry = NodeRegistry()
    registry.add(_make_node(status=status))
    cb_registry = CircuitBreakerRegistry(threshold=1)
    breaker = cb_registry.get_or_create("node-1")
    breaker.record_failure()
    assert breaker.is_open
    failures = _FailureCounts({"node-1": 2})
    client = MagicMock(spec=httpx.Client)
    client.get.return_value = MagicMock(status_code=200 if probe_succeeded else 500)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "http://10.0.1.100:8000/v1/completions"),
    )

    try:
        _probe_all_nodes(
            registry,
            cb_registry,
            client,
            failures,
            failure_threshold=3,
        )
    finally:
        failures.close()

    if status == NodeStatus.PROVISIONING:
        client.get.assert_not_called()
    else:
        client.get.assert_called_once_with("http://10.0.1.100:8000/health")

    result = registry.get("node-1")
    assert result is not None
    assert result.status == expected_by_status[status]
    if status in reset_statuses:
        assert not breaker.is_open
        client.post.assert_called_once_with(
            "http://10.0.1.100:8000/v1/completions",
            json={"model": "llama-3", "prompt": "ping", "max_tokens": 1},
            timeout=2.0,
        )
    else:
        assert breaker.is_open
        client.post.assert_not_called()


@pytest.mark.parametrize(
    ("concurrent_update", "expected_status", "expected_endpoint", "expected_model"),
    [
        (
            {"status": NodeStatus.PROVISIONING},
            NodeStatus.PROVISIONING,
            "10.0.1.100:8000",
            "llama-3",
        ),
        (
            {"endpoint": "10.0.1.200:8000"},
            NodeStatus.HEALTHY,
            "10.0.1.200:8000",
            "llama-3",
        ),
        (
            {"model": "qwen-3"},
            NodeStatus.HEALTHY,
            "10.0.1.100:8000",
            "qwen-3",
        ),
    ],
    ids=["status-provisioning", "endpoint", "model"],
)
def test_probe_result_preserves_concurrent_registry_update(
    concurrent_update: dict[str, object],
    expected_status: NodeStatus,
    expected_endpoint: str,
    expected_model: str,
) -> None:
    """A probe result updates the current node, never its stale cycle snapshot."""
    registry = NodeRegistry()
    stale_node = _make_node(status=NodeStatus.UNHEALTHY)
    registry.add(stale_node)
    cb_registry = CircuitBreakerRegistry()
    failures = _FailureCounts()
    mock_response = MagicMock(status_code=200)
    client = MagicMock(spec=httpx.Client)

    def update_during_probe(_url: str) -> MagicMock:
        current = registry.get("node-1")
        assert current is not None
        registry.add(current.model_copy(update=concurrent_update))
        return mock_response

    client.get.side_effect = update_during_probe

    try:
        _probe_all_nodes(
            registry,
            cb_registry,
            client,
            failures,
            failure_threshold=3,
        )
    finally:
        failures.close()

    result = registry.get("node-1")
    assert result is not None
    assert result.status == expected_status
    assert result.endpoint == expected_endpoint
    assert result.model == expected_model


def test_probe_failure_does_not_resurrect_removed_node() -> None:
    """A node removed while its probe is in flight remains absent."""
    registry = NodeRegistry()
    node = _make_node()
    registry.add(node)
    cb_registry = CircuitBreakerRegistry()
    failures = _FailureCounts({"node-1": 2})
    client = MagicMock(spec=httpx.Client)

    def remove_during_probe(_url: str) -> MagicMock:
        registry.remove("node-1")
        return MagicMock(status_code=500)

    client.get.side_effect = remove_during_probe

    try:
        _probe_all_nodes(
            registry,
            cb_registry,
            client,
            failures,
            failure_threshold=3,
        )
    finally:
        failures.close()

    assert registry.get("node-1") is None


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

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=mock_client,
            ),
            patch.object(stop_event, "wait", side_effect=stop_after_one_iteration),
        ):
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

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=mock_client,
            ),
            patch.object(stop_event, "wait", side_effect=stop_after_three_iterations),
        ):
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

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=mock_client,
            ),
            patch.object(stop_event, "wait", side_effect=stop_after_three_iterations),
        ):
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
    """Recovery distinguishes liveness success from inference success."""

    def test_open_breaker_recovers_only_after_successful_inference_probe(
        self,
    ) -> None:
        registry = NodeRegistry()
        registry.add(_make_node(status=NodeStatus.UNHEALTHY))
        cb_registry = CircuitBreakerRegistry()
        breaker = cb_registry.get_or_create("node-1")
        for _ in range(3):
            breaker.record_failure()
        _assert_breaker_state(breaker, CircuitBreakerState.OPEN)

        failures = _FailureCounts()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=200)

        def successful_inference(*_args: object, **_kwargs: object) -> httpx.Response:
            current = registry.get("node-1")
            assert current is not None
            assert current.status == NodeStatus.UNHEALTHY
            _assert_breaker_state(breaker, CircuitBreakerState.HALF_OPEN)
            return httpx.Response(
                200,
                request=httpx.Request("POST", "http://10.0.1.100:8000/v1/completions"),
            )

        client.post.side_effect = successful_inference
        try:
            _probe_all_nodes(
                registry,
                cb_registry,
                client,
                failures,
                failure_threshold=3,
            )
        finally:
            failures.close()

        result_node = registry.get("node-1")
        assert result_node is not None
        assert result_node.status == NodeStatus.HEALTHY
        _assert_breaker_state(breaker, CircuitBreakerState.CLOSED)
        client.post.assert_called_once_with(
            "http://10.0.1.100:8000/v1/completions",
            json={"model": "llama-3", "prompt": "ping", "max_tokens": 1},
            timeout=2.0,
        )

    @pytest.mark.parametrize(
        "probe_result",
        [
            httpx.Response(
                503,
                request=httpx.Request("POST", "http://10.0.1.100:8000/v1/completions"),
            ),
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("inference timed out"),
        ],
        ids=["server_error", "transport_error", "timeout"],
    )
    def test_failed_half_open_inference_probe_keeps_node_unhealthy(
        self,
        probe_result: httpx.Response | Exception,
    ) -> None:
        registry = NodeRegistry()
        registry.add(_make_node(status=NodeStatus.UNHEALTHY))
        cb_registry = CircuitBreakerRegistry(threshold=1)
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        failures = _FailureCounts()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=200)
        if isinstance(probe_result, Exception):
            client.post.side_effect = probe_result
        else:
            client.post.return_value = probe_result

        try:
            _probe_all_nodes(
                registry,
                cb_registry,
                client,
                failures,
                failure_threshold=3,
            )
        finally:
            failures.close()

        current = registry.get("node-1")
        assert current is not None
        assert current.status == NodeStatus.UNHEALTHY
        _assert_breaker_state(breaker, CircuitBreakerState.OPEN)

    def test_half_open_success_does_not_recover_concurrent_draining_node(
        self,
    ) -> None:
        registry = NodeRegistry()
        registry.add(_make_node(status=NodeStatus.UNHEALTHY))
        cb_registry = CircuitBreakerRegistry(threshold=1)
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        failures = _FailureCounts()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=200)

        def drain_during_inference(
            *_args: object,
            **_kwargs: object,
        ) -> httpx.Response:
            assert registry.drain("node-1")
            return httpx.Response(
                200,
                request=httpx.Request("POST", "http://10.0.1.100:8000/v1/completions"),
            )

        client.post.side_effect = drain_during_inference
        try:
            _probe_all_nodes(
                registry,
                cb_registry,
                client,
                failures,
                failure_threshold=3,
            )
        finally:
            failures.close()

        current = registry.get("node-1")
        assert current is not None
        assert current.status == NodeStatus.DRAINING
        _assert_breaker_state(breaker, CircuitBreakerState.OPEN)

    def test_health_demoted_node_recovers_without_inference_probe(self) -> None:
        registry = NodeRegistry()
        registry.add(_make_node(status=NodeStatus.UNHEALTHY))
        cb_registry = CircuitBreakerRegistry()
        breaker = cb_registry.get_or_create("node-1")
        breaker.record_failure()
        breaker.record_failure()
        failures = _FailureCounts()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=200)

        try:
            _probe_all_nodes(
                registry,
                cb_registry,
                client,
                failures,
                failure_threshold=3,
            )
        finally:
            failures.close()

        current = registry.get("node-1")
        assert current is not None
        assert current.status == NodeStatus.HEALTHY
        _assert_breaker_state(breaker, CircuitBreakerState.CLOSED)
        client.post.assert_not_called()

        breaker.record_failure()
        _assert_breaker_state(breaker, CircuitBreakerState.OPEN)

    def test_half_open_timeout_does_not_block_remaining_probe_cycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        registry = NodeRegistry()
        registry.add(
            _make_node(
                node_id="node-1",
                endpoint="10.0.1.100:8000",
                status=NodeStatus.UNHEALTHY,
            )
        )
        registry.add(
            _make_node(
                node_id="node-2",
                endpoint="10.0.1.101:8000",
                status=NodeStatus.HEALTHY,
            )
        )
        cb_registry = CircuitBreakerRegistry(threshold=1)
        cb_registry.get_or_create("node-1").record_failure()
        failures = _FailureCounts()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=200)
        never_complete = threading.Event()

        def hanging_inference(
            *_args: object,
            timeout: float | None = None,
            **_kwargs: object,
        ) -> None:
            if timeout is None:
                never_complete.wait()
            else:
                never_complete.wait(timeout)
            raise httpx.ReadTimeout("inference timed out")

        client.post.side_effect = hanging_inference
        monkeypatch.setattr(
            "inference_proxy.resilience.health_checker._HALF_OPEN_PROBE_TIMEOUT",
            0.01,
            raising=False,
        )
        completed = threading.Event()
        errors: list[BaseException] = []

        def run_cycle() -> None:
            try:
                _probe_all_nodes(
                    registry,
                    cb_registry,
                    client,
                    failures,
                    failure_threshold=3,
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(target=run_cycle, daemon=True)
        thread.start()

        assert completed.wait(timeout=0.5)
        thread.join(timeout=0.1)
        failures.close()
        assert not thread.is_alive()
        assert errors == []
        assert [call.args[0] for call in client.get.call_args_list] == [
            "http://10.0.1.100:8000/health",
            "http://10.0.1.101:8000/health",
        ]
        client.post.assert_called_once_with(
            "http://10.0.1.100:8000/v1/completions",
            json={"model": "llama-3", "prompt": "ping", "max_tokens": 1},
            timeout=0.01,
        )

    def test_removed_node_does_not_inherit_probe_failure_count(self) -> None:
        registry = NodeRegistry()
        registry.add(_make_node())
        cb_registry = CircuitBreakerRegistry()
        stop_event = threading.Event()
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = MagicMock(status_code=500)
        iteration = 0

        def replace_after_two_failures(timeout: float | None = None) -> bool:
            nonlocal iteration
            iteration += 1
            if iteration == 2:
                before_removal = registry.get("node-1")
                assert before_removal is not None
                assert before_removal.status == NodeStatus.HEALTHY
                registry.remove("node-1")
                registry.add(_make_node())
                return False
            if iteration == 3:
                stop_event.set()
                return True
            return False

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=client,
            ),
            patch.object(stop_event, "wait", side_effect=replace_after_two_failures),
        ):
            run_health_checker(
                registry,
                cb_registry,
                stop_event,
                interval=0.01,
                failure_threshold=3,
            )

        replacement = registry.get("node-1")
        assert replacement is not None
        assert replacement.status == NodeStatus.HEALTHY


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

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=mock_client,
            ),
            patch.object(stop_event, "wait", side_effect=stop_after_two_iterations),
        ):
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

        with (
            patch(
                "inference_proxy.resilience.health_checker.httpx.Client",
                return_value=mock_client,
            ),
            patch.object(stop_event, "wait", side_effect=stop_after_one_iteration),
        ):
            run_health_checker(registry, cb_registry, stop_event, interval=0.01)

        # Only the healthy node was probed
        assert mock_client.get.call_count == 1
        mock_client.get.assert_called_once_with("http://10.0.1.100:8000/health")

        # Provisioning node status unchanged
        result_prov = registry.get("prov-1")
        assert result_prov is not None
        assert result_prov.status == NodeStatus.PROVISIONING


@pytest.mark.parametrize(
    ("endpoint", "expected_url"),
    [
        ("10.0.1.100:8000", "http://10.0.1.100:8000/health"),
        ("http://10.0.1.100:8000", "http://10.0.1.100:8000/health"),
        ("https://gpu01.example.com:8443", "https://gpu01.example.com:8443/health"),
        ("http://[::1]:8000", "http://[::1]:8000/health"),
    ],
)
def test_health_probe_endpoint_normalization_matrix(
    endpoint: str,
    expected_url: str,
) -> None:
    registry = NodeRegistry()
    registry.add(_make_node(endpoint=endpoint))
    cb_registry = CircuitBreakerRegistry()
    failures = _FailureCounts()
    client = MagicMock(spec=httpx.Client)
    client.get.return_value = MagicMock(status_code=200)

    _probe_all_nodes(
        registry,
        cb_registry,
        client,
        failures,
        failure_threshold=3,
    )

    client.get.assert_called_once_with(expected_url)


def test_half_open_inference_probe_preserves_endpoint_scheme() -> None:
    registry = NodeRegistry()
    registry.add(
        _make_node(
            endpoint="https://gpu01.example.com:8443",
            status=NodeStatus.UNHEALTHY,
        )
    )
    cb_registry = CircuitBreakerRegistry(threshold=1)
    cb_registry.get_or_create("node-1").record_failure()
    failures = _FailureCounts()
    client = MagicMock(spec=httpx.Client)
    client.get.return_value = MagicMock(status_code=200)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://gpu01.example.com:8443/v1/completions"),
    )

    _probe_all_nodes(
        registry,
        cb_registry,
        client,
        failures,
        failure_threshold=3,
    )

    client.post.assert_called_once_with(
        "https://gpu01.example.com:8443/v1/completions",
        json={"model": "llama-3", "prompt": "ping", "max_tokens": 1},
        timeout=2.0,
    )
