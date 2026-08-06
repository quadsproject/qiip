"""Background health checker thread for vLLM node probing.

Runs in a dedicated ``threading.Thread`` (per D-01) started during
FastAPI lifespan startup.  Probes each registered node's ``/health``
endpoint using synchronous HTTP calls (per D-02) and updates the
``NodeRegistry`` status accordingly.

**Failure tracking** (per D-03): A node is marked UNHEALTHY after
``failure_threshold`` consecutive probe failures.

**Recovery** (per D-04): A node is restored to HEALTHY after 1
successful liveness probe when its circuit breaker is closed. An OPEN
breaker requires a successful minimal inference probe before recovery.

**Timeout** (per T-05-02): Health probes use a 5-second timeout. The
inference recovery probe has its own 2-second timeout so a wedged engine
cannot stall the serial probe cycle for the ordinary liveness budget.

Usage::

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_health_checker,
        args=(registry, cb_registry, stop_event),
        kwargs={"interval": 30.0, "failure_threshold": 3},
        daemon=True,
    )
    thread.start()

    # On shutdown:
    stop_event.set()
    thread.join(timeout=10)
"""

from __future__ import annotations

import threading

import httpx
import structlog

from inference_proxy.discovery.node_leases import NodeLeaseManager
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.endpoint import build_backend_url
from inference_proxy.models.node import NodeStatus
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing import drain_cleanup
from inference_proxy.routing.connection_tracker import ConnectionTracker

logger = structlog.get_logger()

_PROBE_TIMEOUT: float = 5.0
_HALF_OPEN_PROBE_TIMEOUT: float = 2.0
_RECOVERABLE_STATUSES = {NodeStatus.UNHEALTHY, NodeStatus.UNKNOWN}
_DEMOTABLE_STATUSES = {NodeStatus.HEALTHY, NodeStatus.UNKNOWN}


class _ConsecutiveFailures:
    """Thread-safe health-probe counters bound to registry removal."""

    def __init__(self, registry: NodeRegistry) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._unregister = registry.register_remove_listener(self.remove)

    def reset(self, node_id: str) -> None:
        """Reset the probe-failure count for *node_id*."""
        with self._lock:
            self._counts[node_id] = 0

    def increment(self, node_id: str) -> int:
        """Increment and return the probe-failure count for *node_id*."""
        with self._lock:
            count = self._counts.get(node_id, 0) + 1
            self._counts[node_id] = count
            return count

    def remove(self, node_id: str) -> None:
        """Discard the probe-failure count for a removed node."""
        with self._lock:
            self._counts.pop(node_id, None)

    def close(self) -> None:
        """Detach the counter cleanup listener from its registry."""
        self._unregister()


def run_health_checker(
    registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    stop_event: threading.Event,
    interval: float = 30.0,
    failure_threshold: int = 3,
    connection_tracker: ConnectionTracker | None = None,
    lease_manager: NodeLeaseManager | None = None,
) -> None:
    """Probe registered nodes and manage HEALTHY/UNHEALTHY transitions.

    Runs in a dedicated thread.  Stops when *stop_event* is set.

    Args:
        registry: The node registry containing nodes to probe.
        circuit_breaker_registry: Registry of per-node circuit breakers;
            reset on recovery (per D-08).
        stop_event: A ``threading.Event`` signalling graceful shutdown.
        interval: Seconds between probe cycles (default 30).
        failure_threshold: Consecutive failures before marking a node
            UNHEALTHY (default 3, per D-03).
    """
    consecutive_failures = _ConsecutiveFailures(registry)

    client = httpx.Client(timeout=_PROBE_TIMEOUT)
    try:
        while not stop_event.is_set():
            _probe_all_nodes(
                registry,
                circuit_breaker_registry,
                client,
                consecutive_failures,
                failure_threshold,
                connection_tracker=connection_tracker,
                lease_manager=lease_manager,
            )
            if stop_event.wait(timeout=interval):
                break
    finally:
        consecutive_failures.close()
        client.close()


def _probe_all_nodes(
    registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    client: httpx.Client,
    consecutive_failures: _ConsecutiveFailures,
    failure_threshold: int,
    *,
    connection_tracker: ConnectionTracker | None = None,
    lease_manager: NodeLeaseManager | None = None,
) -> None:
    """Probe every node in the registry once.

    Separated from the main loop for testability and to honour the
    Single Responsibility Principle: the loop manages timing, this
    function manages probing logic.
    """
    if connection_tracker is not None:
        drain_cleanup.sweep_drained_nodes(registry, connection_tracker)
    nodes = registry.get_all()
    for node in nodes:
        if node.status == NodeStatus.PROVISIONING:
            logger.debug("skipping_provisioning_node", node_id=node.node_id)
            continue
        _probe_node(
            node_id=node.node_id,
            endpoint=node.endpoint,
            registry=registry,
            circuit_breaker_registry=circuit_breaker_registry,
            client=client,
            consecutive_failures=consecutive_failures,
            failure_threshold=failure_threshold,
            lease_manager=lease_manager,
        )


def _probe_node(
    *,
    node_id: str,
    endpoint: str,
    registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    client: httpx.Client,
    consecutive_failures: _ConsecutiveFailures,
    failure_threshold: int,
    lease_manager: NodeLeaseManager | None = None,
) -> None:
    """Probe a single node and update its status if needed.

    Args:
        node_id: The node's unique identifier.
        endpoint: The node's HTTP endpoint (host:port).
        registry: The node registry to update on status changes.
        circuit_breaker_registry: Circuit breaker registry for resets.
        client: The synchronous HTTP client for probing.
        consecutive_failures: Mutable dict tracking per-node failure counts.
        failure_threshold: Consecutive failures before marking UNHEALTHY.
    """
    try:
        url = build_backend_url(endpoint, "/health")
        response = client.get(url)
        if response.status_code == 200:
            health_evidence = _handle_probe_success(
                node_id=node_id,
                registry=registry,
                circuit_breaker_registry=circuit_breaker_registry,
                client=client,
                consecutive_failures=consecutive_failures,
            )
            if health_evidence and lease_manager is not None:
                current = registry.get(node_id)
                if current is not None and current.endpoint == endpoint:
                    lease_manager.maintain_after_success(current)
                elif current is not None:
                    logger.debug(
                        "lease refresh withheld after endpoint changed",
                        node_id=node_id,
                        probed_endpoint=endpoint,
                        current_endpoint=current.endpoint,
                    )
        else:
            _handle_probe_failure(
                node_id=node_id,
                registry=registry,
                consecutive_failures=consecutive_failures,
                failure_threshold=failure_threshold,
                reason=f"non-200 status: {response.status_code}",
            )
    except Exception:
        _handle_probe_failure(
            node_id=node_id,
            registry=registry,
            consecutive_failures=consecutive_failures,
            failure_threshold=failure_threshold,
            reason="probe exception",
        )
        logger.debug(
            "health probe failed with exception",
            node_id=node_id,
            exc_info=True,
        )


def _handle_probe_success(
    *,
    node_id: str,
    registry: NodeRegistry,
    circuit_breaker_registry: CircuitBreakerRegistry,
    client: httpx.Client,
    consecutive_failures: _ConsecutiveFailures,
) -> bool:
    """Handle a successful health probe for a node."""
    consecutive_failures.reset(node_id)
    current = registry.get(node_id)
    if current is None:
        logger.debug("health probe succeeded", node_id=node_id)
        return False

    if current.status in {
        NodeStatus.DRAINING,
        NodeStatus.RELAUNCHING,
        NodeStatus.RELAUNCH_FAILED,
        NodeStatus.PROVISIONING,
        NodeStatus.FAILED,
    }:
        logger.debug("health probe succeeded for protected node", node_id=node_id)
        return False

    breaker = circuit_breaker_registry.get(node_id)
    if current.status == NodeStatus.HEALTHY:
        if breaker is not None and breaker.is_open:
            logger.debug(
                "healthy node has open breaker; lease refresh withheld",
                node_id=node_id,
            )
            return False
        logger.debug("health probe succeeded", node_id=node_id)
        return True

    if breaker is not None and breaker.is_open:
        if not breaker.try_half_open():
            logger.debug("half-open probe already active", node_id=node_id)
            return False
        if not current.model:
            breaker.reopen()
            logger.warning(
                "cannot probe inference recovery without a registered model",
                node_id=node_id,
            )
            return False
        try:
            response = client.post(
                build_backend_url(current.endpoint, "/v1/completions"),
                json={
                    "model": current.model,
                    "prompt": "ping",
                    "max_tokens": 1,
                },
                timeout=_HALF_OPEN_PROBE_TIMEOUT,
            )
            # Client-originated 4xx responses are neutral breaker evidence,
            # but this proxy-owned request is known-valid for the registered
            # model. Any non-success means the node failed its recovery trial.
            response.raise_for_status()
        except Exception:
            breaker.record_failure()
            logger.info(
                "half-open inference probe failed",
                node_id=node_id,
                exc_info=True,
            )
            return False

        try:
            transitioned = registry.update_status(
                node_id,
                NodeStatus.HEALTHY,
                allowed_from=_RECOVERABLE_STATUSES,
            )
        except Exception:
            breaker.reopen()
            raise
        if transitioned:
            breaker.record_success()
            logger.info("node recovered after inference probe", node_id=node_id)
        else:
            breaker.reopen()
            logger.debug(
                "half-open inference succeeded after node state changed",
                node_id=node_id,
            )
        return transitioned

    transitioned = registry.update_status(
        node_id,
        NodeStatus.HEALTHY,
        allowed_from=_RECOVERABLE_STATUSES,
    )
    if transitioned:
        logger.info("node recovered to healthy", node_id=node_id)
    else:
        logger.debug("health probe succeeded", node_id=node_id)
    return transitioned


def _handle_probe_failure(
    *,
    node_id: str,
    registry: NodeRegistry,
    consecutive_failures: _ConsecutiveFailures,
    failure_threshold: int,
    reason: str,
) -> None:
    """Handle a failed health probe for a node."""
    count = consecutive_failures.increment(node_id)
    logger.debug(
        "health probe failed",
        node_id=node_id,
        consecutive_failures=count,
        reason=reason,
    )
    if count >= failure_threshold and registry.update_status(
        node_id,
        NodeStatus.UNHEALTHY,
        allowed_from=_DEMOTABLE_STATUSES,
    ):
        logger.info(
            "node marked unhealthy",
            node_id=node_id,
            consecutive_failures=count,
            threshold=failure_threshold,
        )
