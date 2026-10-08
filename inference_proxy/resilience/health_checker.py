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

**Optional health for self-setup nodes**: A node adopted on the
OpenAI-compatible ``/v1/models`` contract (``self_setup``) treats a
missing ``/health`` endpoint (HTTP 404/405/501) as optional and falls
back to a ``/v1/models`` probe. An authoritative unhealthy response
(e.g. ``/health`` 503) is still a failure. Managed nodes require a
healthy ``/health`` response.
The same cycle refreshes the tracked model id for self-setup nodes from
``/v1/models``, so a backend that restarts with another model is followed
without a manual re-adoption.

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

import json
import threading

import httpx
import structlog

from inference_proxy.discovery.etcd_client import EtcdClient
from inference_proxy.discovery.node_leases import NodeLeaseManager
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.endpoint import build_backend_url
from inference_proxy.models.node import NodeStatus
from inference_proxy.provisioning.provisioner import served_model_id
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing import drain_cleanup
from inference_proxy.routing.connection_tracker import ConnectionTracker

logger = structlog.get_logger()

_PROBE_TIMEOUT: float = 5.0
_HALF_OPEN_PROBE_TIMEOUT: float = 2.0
_RECOVERABLE_STATUSES = {NodeStatus.UNHEALTHY, NodeStatus.UNKNOWN}
_DEMOTABLE_STATUSES = {NodeStatus.HEALTHY, NodeStatus.UNKNOWN}
# /health responses that mean the endpoint is not implemented. For a self-setup
# node (adopted on the OpenAI-compatible /v1/models contract) this is not a
# liveness failure, so the probe falls back to /v1/models. Managed nodes still
# require a healthy /health response.
_OPTIONAL_HEALTH_STATUSES = frozenset({404, 405, 501})


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
    etcd_client: EtcdClient | None = None,
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
                etcd_client=etcd_client,
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
    etcd_client: EtcdClient | None = None,
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
        if node.status in (NodeStatus.AVAILABLE, NodeStatus.PROVISIONING):
            logger.debug("skipping_non_active_node", node_id=node.node_id)
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
            etcd_client=etcd_client,
        )


def _probe_liveness(
    *,
    endpoint: str,
    client: httpx.Client,
    self_setup: bool,
) -> tuple[bool, str, str | None]:
    """Probe a node's health endpoint and report whether it is alive.

    Returns ``(alive, reason, observed_model)``.  For a self-setup node a
    missing ``/health`` endpoint (HTTP 404/405/501) is optional: the probe
    falls back to the OpenAI-compatible ``/v1/models`` contract the node was
    adopted on, and that response's primary model id is returned so the
    caller does not fetch it twice.  All other non-200 responses, and managed
    nodes, are treated as failures.  ``observed_model`` is ``None`` whenever
    no ``/v1/models`` body was parsed.
    """
    health_url = build_backend_url(endpoint, "/health")
    response = client.get(health_url)
    if response.status_code == 200:
        return True, "healthy", None
    if self_setup and response.status_code in _OPTIONAL_HEALTH_STATUSES:
        models_url = build_backend_url(endpoint, "/v1/models")
        models = client.get(models_url)
        if models.status_code == 200:
            try:
                observed = served_model_id(models.json())
            except Exception:
                observed = None
            return (
                True,
                f"missing /health ({response.status_code}); /v1/models ok",
                observed,
            )
        return (
            False,
            f"missing /health; /v1/models returned {models.status_code}",
            None,
        )
    return False, f"non-200 status: {response.status_code}", None


def _apply_model_if_current(
    registry: NodeRegistry,
    node_id: str,
    endpoint: str,
    model: str,
) -> bool:
    """Patch the model only while the probed registration is still current.

    The write is applied to the entry re-read under the registry lock, so a
    concurrent re-adoption or drain is never reverted by a stale snapshot.
    Returns whether the model is present on the current entry.
    """
    with registry.locked():
        latest = registry.get(node_id)
        if latest is None or latest.endpoint != endpoint:
            return False
        if latest.model != model:
            registry.add(latest.model_copy(update={"model": model}))
        return True


def _refresh_self_setup_model(
    *,
    node_id: str,
    endpoint: str,
    registry: NodeRegistry,
    client: httpx.Client,
    self_setup: bool,
    observed_model: str | None,
    etcd_client: EtcdClient | None = None,
) -> None:
    """Reconcile a self-setup node's tracked model with the live server.

    The backend can be restarted with a different primary model outside the
    proxy's control, so every successful liveness cycle re-reads
    ``/v1/models`` (single-model contract: the first reported id, same as
    adoption).  The in-memory registry is patched as soon as a new id is
    observed -- it is the source of truth for the breaker's half-open
    trial -- and the etcd record then follows through a revision CAS so a
    concurrent owner/name/admin-only write is never clobbered.  Every etcd
    call is isolated so a degraded etcd cannot demote a live node; if a
    write fails, the record lags the registry until the next cycle
    re-converges it, or until the restart snapshot is re-discovered.  The
    endpoint guard mirrors the lease-refresh rule: a stale observation
    from a replaced registration writes nothing.
    """
    if not self_setup:
        return
    model = observed_model
    if model is None:
        try:
            models = client.get(build_backend_url(endpoint, "/v1/models"))
        except Exception:
            logger.debug(
                "self_setup_model_refresh_fetch_failed",
                node_id=node_id,
                exc_info=True,
            )
            return
        if models.status_code != 200:
            logger.debug(
                "self_setup_model_refresh_non_200",
                node_id=node_id,
                status_code=models.status_code,
            )
            return
        try:
            model = served_model_id(models.json())
        except Exception:
            logger.debug(
                "self_setup_model_refresh_unparseable",
                node_id=node_id,
                exc_info=True,
            )
            return
    if model is None:
        return
    current = registry.get(node_id)
    if current is None or current.endpoint != endpoint:
        logger.debug(
            "model refresh withheld after endpoint changed",
            node_id=node_id,
            probed_endpoint=endpoint,
            current_endpoint=current.endpoint if current is not None else None,
        )
        return
    if current.model == model and etcd_client is None:
        return
    if etcd_client is None:
        if _apply_model_if_current(registry, node_id, endpoint, model):
            logger.info(
                "self_setup_model_refreshed_without_etcd",
                node_id=node_id,
                previous_model=current.model,
                model=model,
            )
        else:
            logger.debug(
                "model refresh withheld after endpoint changed",
                node_id=node_id,
                probed_endpoint=endpoint,
            )
        return
    key = f"{etcd_client.prefix}{node_id}"
    for _attempt in range(3):
        try:
            record = etcd_client.get_record(key)
        except Exception:
            logger.debug(
                "self_setup_model_refresh_read_failed",
                node_id=node_id,
                exc_info=True,
            )
            _apply_model_if_current(registry, node_id, endpoint, model)
            return
        if record is None:
            logger.debug(
                "self_setup_model_refresh_key_absent",
                node_id=node_id,
                key=key,
            )
            _apply_model_if_current(registry, node_id, endpoint, model)
            return
        try:
            data = json.loads(record.value)
        except Exception:
            logger.debug(
                "self_setup_model_refresh_malformed_record",
                node_id=node_id,
                exc_info=True,
            )
            _apply_model_if_current(registry, node_id, endpoint, model)
            return
        if not isinstance(data, dict):
            logger.debug(
                "self_setup_model_refresh_malformed_record",
                node_id=node_id,
            )
            _apply_model_if_current(registry, node_id, endpoint, model)
            return
        data["model"] = model
        try:
            new_revision = etcd_client.replace_if_revision(
                key,
                json.dumps(data).encode("utf-8"),
                expected_mod_revision=record.mod_revision,
                lease_id=record.lease_id,
            )
        except Exception:
            logger.debug(
                "self_setup_model_refresh_write_failed",
                node_id=node_id,
                exc_info=True,
            )
            _apply_model_if_current(registry, node_id, endpoint, model)
            return
        if new_revision is not None:
            if _apply_model_if_current(registry, node_id, endpoint, model):
                logger.info(
                    "self_setup_model_refreshed",
                    node_id=node_id,
                    previous_model=current.model,
                    model=model,
                    revision=new_revision,
                )
            else:
                logger.debug(
                    "model refresh withheld after endpoint changed",
                    node_id=node_id,
                    probed_endpoint=endpoint,
                )
            return
    _apply_model_if_current(registry, node_id, endpoint, model)
    logger.warning(
        "self_setup_model_refresh_cas_exhausted",
        node_id=node_id,
        model=model,
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
    etcd_client: EtcdClient | None = None,
) -> None:
    """Probe a single node and update its status if needed.

        A ``self_setup`` node treats a missing ``/health`` endpoint as optional and
        falls back to ``/v1/models``; managed nodes still require a healthy
        ``/health`` response (per D-03/D-04).
    After a successful liveness probe the tracked model id of a self-setup node
    is reconciled from ``/v1/models`` (see ``_refresh_self_setup_model``), and
    the breaker recovery trial below therefore uses the refreshed model.

        Args:
            node_id: The node's unique identifier.
            endpoint: The node's HTTP endpoint (host:port).
            registry: The node registry to update on status changes.
            circuit_breaker_registry: Circuit breaker registry for resets.
            client: The synchronous HTTP client for probing.
            consecutive_failures: Mutable dict tracking per-node failure counts.
            failure_threshold: Consecutive failures before marking UNHEALTHY.
    """
    current = registry.get(node_id)
    self_setup = current.self_setup if current is not None else False
    try:
        alive, reason, observed_model = _probe_liveness(
            endpoint=endpoint,
            client=client,
            self_setup=self_setup,
        )
        if alive:
            _refresh_self_setup_model(
                node_id=node_id,
                endpoint=endpoint,
                registry=registry,
                client=client,
                self_setup=self_setup,
                observed_model=observed_model,
                etcd_client=etcd_client,
            )
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
                reason=reason,
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
        NodeStatus.AVAILABLE,
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
