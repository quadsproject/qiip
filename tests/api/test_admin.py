"""Integration tests for the admin API endpoints.

Tests cover:
- GET /admin/nodes returns unified list merging QUADS + etcd (NODES-01)
- Each node has state and actions fields (NODES-02)
- POST /admin/nodes/setup dedup guard (NODES-04)
- POST /admin/nodes/setup live QUADS re-validation (NODES-05)
- GET /admin/metrics returns aggregate request counter data
- GET /admin/provisioning/tasks returns task status from etcd
- DELETE /admin/nodes/{id} returns 202 for known nodes, 404 for unknown
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock
from structlog.testing import capture_logs

from inference_proxy.api.admin import admin_router
from inference_proxy.config.dependencies import (
    get_catalog_service,
    get_llmfit_runner,
    get_provisioner,
    get_quads_client,
    get_quads_poller,
    get_redfish_client,
    get_registry,
    get_settings,
    get_unified_node_service,
    require_admin_auth,
)
from inference_proxy.config.settings import LLMFitSettings, Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.llmfit.errors import LLMFitParseError, LLMFitTimeoutError
from inference_proxy.llmfit.runner import LLMFitRunner
from inference_proxy.models.endpoint import EndpointPolicy, EndpointValidationError
from inference_proxy.models.llmfit import (
    GGUFSource,
    LLMFitResult,
    ModelRecommendation,
    SystemInfo,
)
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppCacheType,
    LlamaCppRuntimeRequest,
    LlamaCppRuntimeState,
    LlamaCppSizingMode,
    Node,
    NodeStatus,
)
from inference_proxy.models.quads import QUADSHost
from inference_proxy.provisioning.provisioner import (
    BackgroundOperation,
    ProvisioningCapacityError,
    ProvisioningError,
    ProvisioningIdentity,
    RelaunchPreconditionError,
    RelaunchValidationError,
)
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHConnectionError,
)
from inference_proxy.quads.client import QUADSConnectionError
from inference_proxy.redfish.client import RedfishClient
from inference_proxy.redfish.errors import RedfishError
from inference_proxy.resilience.circuit_breaker import CircuitBreakerRegistry
from inference_proxy.routing.connection_tracker import ConnectionTracker
from inference_proxy.routing.request_metrics import RequestMetrics
from inference_proxy.services.unified_nodes import UnifiedNodeService


def _make_node(
    node_id: str = "node-1",
    endpoint: str = "10.0.1.100:8000",
    status: NodeStatus = NodeStatus.HEALTHY,
    model: str = "llama-3",
    managed: bool = True,
    engine: InferenceEngine = InferenceEngine.VLLM,
    artifact_id: str | None = None,
    llamacpp_runtime: LlamaCppRuntimeState | None = None,
) -> Node:
    """Create a test node with sensible defaults."""
    return Node(
        node_id=node_id,
        endpoint=endpoint,
        status=status,
        model=model,
        managed=managed,
        engine=engine,
        artifact_id=artifact_id,
        llamacpp_runtime=llamacpp_runtime,
    )


def _llamacpp_runtime() -> LlamaCppRuntimeState:
    return LlamaCppRuntimeState.model_validate(
        {
            "requested": {"sizing": "auto", "fit_target_mib": 512},
            "effective": {
                "train_context": 262144,
                "context_per_slot": 12544,
                "slot_context_limit": 12544,
                "slots": 1,
                "aggregate_context": 12544,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": "on",
                "kv_unified": True,
                "gpu_layers": 31,
                "total_layers": 31,
                "estimator_overrun_used": False,
            },
            "gpus": [
                {"index": 0, "total_mib": 14911, "used_mib": 14089, "free_mib": 822}
            ],
            "observed_at": "2026-08-05T20:54:07Z",
        }
    )


def _real_redfish_client(http_client: httpx.AsyncClient) -> RedfishClient:
    auth = httpx.BasicAuth("operator", "redfish-secret")
    policy = EndpointPolicy.from_values(
        allowed_hosts=["gpu01"],
        allowed_networks=["10.0.0.0/8"],
        allowed_ports=[8000],
    )
    return RedfishClient(
        http_client,
        "mgmt-{hostname}",
        "1",
        hostname_policy=policy,
        auth=auth,
    )


class TestAdminNodesPopulated:
    """GET /admin/nodes with registered nodes returns node data."""

    def test_returns_200_with_two_nodes(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Two registered nodes return 200 with a list of two AdminNodeResponse objects."""
        test_registry.add(_make_node(node_id="node-1", model="llama-3"))
        test_registry.add(
            _make_node(
                node_id="node-2",
                endpoint="10.0.1.101:8000",
                model="mistral-7b",
            )
        )

        response = client.get("/admin/nodes")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_each_node_has_expected_fields(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Each node contains identity, operational state, and unified fields."""
        test_registry.add(_make_node())

        response = client.get("/admin/nodes")
        data = response.json()

        assert len(data) == 1
        node = data[0]
        expected = {
            "node_id",
            "endpoint",
            "model",
            "status",
            "active_connections",
            "circuit_breaker_state",
            "engine",
            "artifact_id",
            "llamacpp_runtime",
            "state",
            "actions",
            "gpu_vendor",
            "gpu_model",
            "gpu_count",
            "managed",
            "failed_step",
            "error",
        }
        assert set(node.keys()) == expected
        assert "last_heartbeat" not in node
        assert "capabilities" not in node

    def test_managed_llamacpp_runtime_is_serialized(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(
            _make_node(
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id="a" * 64,
                llamacpp_runtime=_llamacpp_runtime(),
            )
        )

        response = client.get("/admin/nodes")

        assert response.status_code == 200
        runtime = response.json()[0]["llamacpp_runtime"]
        assert runtime == {
            "requested": {"sizing": "auto", "fit_target_mib": 512},
            "effective": {
                "train_context": 262144,
                "context_per_slot": 12544,
                "slot_context_limit": 12544,
                "slots": 1,
                "aggregate_context": 12544,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": "on",
                "kv_unified": True,
                "gpu_layers": 31,
                "total_layers": 31,
                "estimator_overrun_used": False,
            },
            "gpus": [
                {"index": 0, "total_mib": 14911, "used_mib": 14089, "free_mib": 822}
            ],
            "observed_at": "2026-08-05T20:54:07Z",
        }

    def test_mixed_statuses_all_appear(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Nodes with HEALTHY, UNHEALTHY, and DRAINING statuses all appear."""
        test_registry.add(
            _make_node(node_id="h1", status=NodeStatus.HEALTHY, model="llama-3")
        )
        test_registry.add(
            _make_node(
                node_id="u1",
                endpoint="10.0.1.101:8000",
                status=NodeStatus.UNHEALTHY,
                model="mistral-7b",
            )
        )
        test_registry.add(
            _make_node(
                node_id="d1",
                endpoint="10.0.1.102:8000",
                status=NodeStatus.DRAINING,
                model="codellama",
            )
        )

        response = client.get("/admin/nodes")
        data = response.json()

        assert len(data) == 3
        statuses = {node["status"] for node in data}
        assert statuses == {"healthy", "unhealthy", "draining"}

    def test_admin_nodes_falls_back_when_task_etcd_unavailable(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(
            _make_node(node_id="gpu01", status=NodeStatus.FAILED, managed=True)
        )
        mock_provisioner.list_tasks_raw.side_effect = RuntimeError("etcd unavailable")

        with capture_logs() as logs:
            response = client.get("/admin/nodes")

        assert response.status_code == 200
        assert response.headers["X-Inference-Proxy-Data-Degraded"] == (
            "provisioning-tasks"
        )
        assert response.json() == [
            {
                "node_id": "gpu01",
                "endpoint": "10.0.1.100:8000",
                "model": "llama-3",
                "status": "failed",
                "active_connections": 0,
                "circuit_breaker_state": "closed",
                "engine": "vllm",
                "artifact_id": None,
                "llamacpp_runtime": None,
                "state": "failed",
                "actions": ["setup", "teardown"],
                "gpu_vendor": None,
                "gpu_model": None,
                "gpu_count": None,
                "managed": True,
                "failed_step": None,
                "error": None,
            }
        ]
        assert any(
            log.get("event") == "provisioning_task_list_unavailable"
            and log.get("log_level") == "warning"
            for log in logs
        )


class TestAdminNodesEmpty:
    """GET /admin/nodes with empty registry returns empty list."""

    def test_empty_registry_returns_empty_list(
        self,
        client: TestClient,
    ) -> None:
        """Empty registry returns 200 with an empty list."""
        response = client.get("/admin/nodes")

        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestAdminNodesResponseShape:
    """GET /admin/nodes returns a flat JSON array."""

    def test_response_is_flat_array(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """The response is a flat JSON array, not wrapped in an object."""
        test_registry.add(_make_node())

        response = client.get("/admin/nodes")
        data = response.json()

        # Must be a list, not a dict/object wrapper
        assert isinstance(data, list)
        assert len(data) == 1
        assert isinstance(data[0], dict)


class TestAdminNodesEnriched:
    """GET /admin/nodes returns enriched operational state per node."""

    def test_active_connections_reflects_tracker(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        connection_tracker: ConnectionTracker,
    ) -> None:
        """active_connections matches the ConnectionTracker count."""
        test_registry.add(_make_node(node_id="node-1"))
        connection_tracker.increment("node-1")
        connection_tracker.increment("node-1")

        response = client.get("/admin/nodes")
        node = response.json()[0]

        assert node["active_connections"] == 2

    def test_circuit_breaker_state_default_closed(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """A fresh node has circuit_breaker_state 'closed'."""
        test_registry.add(_make_node(node_id="node-1"))

        response = client.get("/admin/nodes")
        node = response.json()[0]

        assert node["circuit_breaker_state"] == "closed"

    def test_circuit_breaker_state_open_after_failures(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        circuit_breaker_registry: CircuitBreakerRegistry,
    ) -> None:
        """A tripped breaker reports circuit_breaker_state 'open'."""
        test_registry.add(_make_node(node_id="node-1"))
        breaker = circuit_breaker_registry.get_or_create("node-1")
        for _ in range(3):
            breaker.record_failure()

        response = client.get("/admin/nodes")
        node = response.json()[0]

        assert node["circuit_breaker_state"] == "open"


class TestAdminMetrics:
    """GET /admin/metrics returns aggregate request counter data."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        """The metrics endpoint returns 200."""
        response = client.get("/admin/metrics")
        assert response.status_code == 200

    def test_metrics_empty_by_default(self, client: TestClient) -> None:
        """Fresh metrics returns zeroed counters."""
        response = client.get("/admin/metrics")
        data = response.json()

        assert data == {"total_requests": 0, "per_model": {}, "per_node": {}}

    def test_metrics_after_recording(
        self,
        client: TestClient,
        request_metrics: RequestMetrics,
    ) -> None:
        """Metrics reflect data recorded via RequestMetrics."""
        request_metrics.record_request("node-1", "llama-3")

        response = client.get("/admin/metrics")
        data = response.json()

        assert data["total_requests"] == 1
        assert data["per_model"] == {"llama-3": 1}
        assert data["per_node"] == {"node-1": 1}


class TestSetupEndpoint:
    """POST /admin/nodes/setup triggers provisioning."""

    def test_returns_202_with_task_id(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202
        assert response.json() == {"task_id": "gpu01"}
        assert mock_provisioner.validate_setup_configuration.call_args_list == [
            call(InferenceEngine.VLLM),
            call(InferenceEngine.VLLM),
        ]

    def test_calls_fire_background(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        mock_provisioner.fire_background.assert_called_once()
        assert mock_provisioner.fire_background.call_args.kwargs[
            "provisioning_identity"
        ] == ProvisioningIdentity(InferenceEngine.VLLM)

    def test_rejects_disallowed_endpoint_before_reserving_host(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.validate_endpoint.side_effect = EndpointValidationError(
            "add 'gpu01' to routing.allowed_endpoint_hosts"
        )

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "add 'gpu01' to routing.allowed_endpoint_hosts"
        )
        mock_provisioner.try_reserve_host.assert_not_awaited()
        mock_provisioner.fire_background.assert_not_called()

    def test_missing_nfs_export_rejected_before_reserving_host(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.validate_setup_configuration.side_effect = ProvisioningError(
            "Node provisioning requires INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT"
        )

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 400
        assert "HUGGINGFACE__NFS_EXPORT" in response.json()["detail"]
        mock_provisioner.try_reserve_host.assert_not_awaited()
        mock_provisioner.fire_background.assert_not_called()


class TestSetupModelPassthrough:
    """POST /admin/nodes/setup passes model to provisioner.provision()."""

    def test_passes_model_to_provisioner(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.provision = AsyncMock()
        response = client.post(
            "/admin/nodes/setup", json={"hostname": "gpu01", "model": "org/model"}
        )
        assert response.status_code == 202
        # fire_background receives a coroutine; await it to trigger provision
        coro = mock_provisioner.fire_background.call_args[0][0]
        asyncio.get_event_loop().run_until_complete(coro)
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=True,
            model="org/model",
            engine=ANY,
            artifact_id=None,
            lifecycle_lease=ANY,
        )

    def test_setup_without_model_defaults_none(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.provision = AsyncMock()
        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args[0][0]
        asyncio.get_event_loop().run_until_complete(coro)
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=True,
            model=None,
            engine=ANY,
            artifact_id=None,
            lifecycle_lease=ANY,
        )

    def test_unknown_llamacpp_artifact_fails_before_host_reservation(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "a" * 64
        mock_provisioner.resolve_artifact_selection.side_effect = ProvisioningError(
            f"GGUF artifact {artifact_id!r} was not found"
        )

        response = client.post(
            "/admin/nodes/setup",
            json={
                "hostname": "gpu01",
                "engine": "llama_cpp",
                "artifact_id": artifact_id,
            },
        )

        assert response.status_code == 400
        assert "was not found" in response.json()["detail"]
        mock_provisioner.try_reserve_host.assert_not_awaited()
        mock_provisioner.provision.assert_not_awaited()

    def test_implicit_llamacpp_retry_inherits_persisted_artifact(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "a" * 64
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                model="published-alias",
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id=artifact_id,
            )
        )

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=True,
            model=None,
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id=artifact_id,
            lifecycle_lease=ANY,
        )
        assert mock_provisioner.fire_background.call_args.kwargs[
            "provisioning_identity"
        ] == ProvisioningIdentity(InferenceEngine.LLAMA_CPP, artifact_id)

    def test_implicit_llamacpp_retry_inherits_persisted_sizing_policy(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "b" * 64
        request = LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.CUSTOM,
            fit_target_mib=768,
            context_per_slot=32768,
            slots=3,
            cache_type=LlamaCppCacheType.Q8_0,
        )
        runtime = _llamacpp_runtime().model_copy(update={"requested": request})
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id=artifact_id,
                llamacpp_runtime=runtime,
            )
        )

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=True,
            model=None,
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id=artifact_id,
            llamacpp_request=request,
            lifecycle_lease=ANY,
        )

    def test_implicit_vllm_retry_inherits_persisted_model(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                model="org/persisted-model",
            )
        )

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        assert mock_provisioner.provision.await_args.kwargs["model"] == (
            "org/persisted-model"
        )
        assert mock_provisioner.provision.await_args.kwargs["engine"] is (
            InferenceEngine.VLLM
        )

    def test_explicit_selection_overrides_persisted_identity(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id="a" * 64,
            )
        )

        response = client.post(
            "/admin/nodes/setup",
            json={
                "hostname": "gpu01",
                "engine": "vllm",
                "model": "org/explicit-model",
            },
        )

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        assert mock_provisioner.provision.await_args.kwargs["engine"] is (
            InferenceEngine.VLLM
        )
        assert mock_provisioner.provision.await_args.kwargs["model"] == (
            "org/explicit-model"
        )
        assert mock_provisioner.provision.await_args.kwargs["artifact_id"] is None

    def test_any_explicit_selection_field_disables_all_retry_inheritance(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id="a" * 64,
            )
        )

        response = client.post(
            "/admin/nodes/setup",
            json={"hostname": "gpu01", "model": "org/explicit-model"},
        )

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        assert mock_provisioner.provision.await_args.kwargs["engine"] is (
            InferenceEngine.VLLM
        )
        assert mock_provisioner.provision.await_args.kwargs["model"] == (
            "org/explicit-model"
        )
        assert mock_provisioner.provision.await_args.kwargs["artifact_id"] is None

    def test_legacy_llamacpp_retry_without_artifact_fails_before_reservation(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id=None,
            )
        )

        def resolve(engine: InferenceEngine, artifact_id: str | None) -> None:
            if engine is InferenceEngine.LLAMA_CPP and artifact_id is None:
                raise ProvisioningError(
                    "Persisted llama_cpp retry requires an exact artifact_id"
                )

        mock_provisioner.resolve_artifact_selection.side_effect = resolve

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 400
        assert "exact artifact_id" in response.json()["detail"]
        mock_provisioner.try_reserve_host.assert_not_awaited()

    def test_retry_retains_captured_identity_if_record_disappears_for_lease(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "b" * 64
        request = LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.CUSTOM,
            fit_target_mib=768,
            context_per_slot=32768,
            slots=3,
            cache_type=LlamaCppCacheType.Q8_0,
        )
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                engine=InferenceEngine.LLAMA_CPP,
                artifact_id=artifact_id,
                llamacpp_runtime=_llamacpp_runtime().model_copy(
                    update={"requested": request}
                ),
            )
        )
        lease = MagicMock(hostname="gpu01")

        def reserve(_hostname: str) -> MagicMock:
            test_registry.remove("gpu01")
            return lease

        mock_provisioner.try_reserve_host.side_effect = reserve

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        assert mock_provisioner.provision.await_args.kwargs["engine"] is (
            InferenceEngine.LLAMA_CPP
        )
        assert (
            mock_provisioner.provision.await_args.kwargs["artifact_id"] == artifact_id
        )
        assert mock_provisioner.provision.await_args.kwargs["llamacpp_request"] == (
            request
        )

    def test_retry_uses_newer_registration_identity_after_taking_lease(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "c" * 64
        test_registry.add(
            _make_node(
                node_id="gpu01",
                status=NodeStatus.FAILED,
                model="org/old-vllm-model",
            )
        )
        lease = MagicMock(hostname="gpu01")

        def reserve(_hostname: str) -> MagicMock:
            test_registry.add(
                _make_node(
                    node_id="gpu01",
                    status=NodeStatus.FAILED,
                    model="new-llamacpp-alias",
                    engine=InferenceEngine.LLAMA_CPP,
                    artifact_id=artifact_id,
                )
            )
            return lease

        mock_provisioner.try_reserve_host.side_effect = reserve

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        assert mock_provisioner.provision.await_args.kwargs["model"] is None
        assert mock_provisioner.provision.await_args.kwargs["engine"] is (
            InferenceEngine.LLAMA_CPP
        )
        assert (
            mock_provisioner.provision.await_args.kwargs["artifact_id"] == artifact_id
        )


class TestLlamaCppRelaunchEndpoint:
    def test_queues_typed_relaunch_as_a_non_cancellable_operation(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        artifact_id = "a" * 64
        node = _make_node(
            node_id="gpu01",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id=artifact_id,
            llamacpp_runtime=_llamacpp_runtime(),
        )
        test_registry.add(node)

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={
                "sizing": "custom",
                "fit_target_mib": 512,
                "context_per_slot": 24576,
                "slots": 2,
                "cache_type": "q8_0",
                "allow_estimator_overrun": True,
            },
        )

        assert response.status_code == 202
        assert response.json() == {"task_id": "gpu01"}
        call_kwargs = mock_provisioner.fire_background.call_args.kwargs
        assert call_kwargs["operation"] is BackgroundOperation.RELAUNCH
        assert call_kwargs["provisioning_identity"] == ProvisioningIdentity(
            InferenceEngine.LLAMA_CPP,
            artifact_id,
        )
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        request = mock_provisioner.relaunch_llamacpp.await_args.args[1]
        assert request == LlamaCppRuntimeRequest(
            sizing=LlamaCppSizingMode.CUSTOM,
            fit_target_mib=512,
            context_per_slot=24576,
            slots=2,
            cache_type=LlamaCppCacheType.Q8_0,
            allow_estimator_overrun=True,
        )
        assert type(request) is LlamaCppRuntimeRequest

    def test_rejects_unknown_request_fields(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512, "typo": 1},
        )

        assert response.status_code == 422
        mock_provisioner.try_reserve_host.assert_not_awaited()

    def test_requires_connection_tracking(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.connection_tracking_available = False

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512},
        )

        assert response.status_code == 503
        mock_provisioner.try_reserve_host.assert_not_awaited()

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            (RelaunchPreconditionError("not relaunchable"), 409),
            (RelaunchValidationError("outside limits"), 422),
        ],
    )
    def test_maps_boundary_validation_before_reservation(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
        error: Exception,
        status_code: int,
    ) -> None:
        mock_provisioner.validate_llamacpp_relaunch.side_effect = error

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512},
        )

        assert response.status_code == status_code
        mock_provisioner.try_reserve_host.assert_not_awaited()

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            (RelaunchPreconditionError("node changed"), 409),
            (RelaunchValidationError("limits changed"), 422),
        ],
    )
    def test_revalidates_after_reserving_the_host(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
        error: Exception,
        status_code: int,
    ) -> None:
        node = _make_node(
            node_id="gpu01",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="a" * 64,
            llamacpp_runtime=_llamacpp_runtime(),
        )
        test_registry.add(node)
        mock_provisioner.validate_llamacpp_relaunch.side_effect = [
            node,
            error,
        ]

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512},
        )

        assert response.status_code == status_code
        mock_provisioner.fire_background.assert_not_called()
        lease = mock_provisioner.try_reserve_host.await_args_list[0]
        assert lease.args == ("gpu01",)

    def test_busy_host_returns_conflict_without_scheduling(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = None

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512},
        )

        assert response.status_code == 409
        assert "lifecycle operation already in progress" in response.json()["detail"]
        mock_provisioner.fire_background.assert_not_called()

    def test_capacity_rejection_releases_host_reservation(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        node = _make_node(
            node_id="gpu01",
            engine=InferenceEngine.LLAMA_CPP,
            artifact_id="a" * 64,
            llamacpp_runtime=_llamacpp_runtime(),
        )
        test_registry.add(node)
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease
        mock_provisioner.fire_background.side_effect = ProvisioningCapacityError(
            active=32,
            limit=32,
        )

        response = client.post(
            "/admin/nodes/gpu01/llamacpp/relaunch",
            json={"sizing": "auto", "fit_target_mib": 512},
        )

        assert response.status_code == 429
        assert "32 active" in response.json()["detail"]
        lease.release.assert_called_once_with()


class TestTasksEndpoint:
    """GET /admin/provisioning/tasks returns task status from etcd."""

    def test_returns_tasks_from_etcd(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        task_data = {
            "hostname": "gpu01",
            "current_step": "registering",
            "started_at": "2026-07-07T12:00:00Z",
            "updated_at": "2026-07-07T12:05:00Z",
        }
        mock_provisioner.list_tasks_raw.return_value = [
            (json.dumps(task_data).encode(), {"key": b"/provisioning/gpu01"}),
        ]

        response = client.get("/admin/provisioning/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["hostname"] == "gpu01"
        assert data[0]["current_step"] == "registering"

    def test_empty_tasks_returns_empty_list(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_provisioner.list_tasks_raw.return_value = []
        response = client.get("/admin/provisioning/tasks")
        assert response.status_code == 200
        assert response.json() == []


class TestTeardownEndpoint:
    """DELETE /admin/nodes/{id} triggers teardown."""

    def test_returns_202_for_known_node(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        response = client.delete("/admin/nodes/gpu01")
        assert response.status_code == 202
        assert response.json() == {"task_id": "gpu01"}

    def test_force_param_passed(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        response = client.delete("/admin/nodes/gpu01?force=true")

        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        mock_provisioner.teardown.assert_awaited_once_with(
            "gpu01",
            force=True,
            provisioning_identity=None,
            recovery_engine=None,
            lifecycle_lease=ANY,
        )

    def test_busy_non_provision_operation_still_returns_409(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = None

        response = client.delete("/admin/nodes/gpu01")

        assert response.status_code == 409
        assert "operation already in progress" in response.json()["detail"]
        mock_provisioner.cancel_active_provision.assert_awaited_once_with("gpu01")
        mock_provisioner.fire_background.assert_not_called()

    def test_cancel_handoff_reports_when_host_is_re_reserved(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        call_order: list[str] = []

        async def cancel(_hostname: str) -> MagicMock:
            call_order.append("cancel")
            return MagicMock()

        async def reserve(_hostname: str) -> None:
            call_order.append("reserve")
            return None

        mock_provisioner.cancel_active_provision.side_effect = cancel
        mock_provisioner.try_reserve_host.side_effect = reserve

        response = client.delete("/admin/nodes/gpu01")

        assert response.status_code == 409
        assert "re-reserved" in response.json()["detail"]
        assert "retry teardown" in response.json()["detail"]
        assert call_order == ["cancel", "reserve"]
        mock_provisioner.fire_background.assert_not_called()

    def test_unknown_node_returns_404(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        response = client.delete("/admin/nodes/unknown")
        assert response.status_code == 404


# -- Unified list tests (NODES-01, NODES-02) --


class TestUnifiedNodeList:
    """GET /admin/nodes returns unified QUADS+etcd merged list."""

    def test_merged_list_with_quads(
        self,
        app: FastAPI,
        client: TestClient,
        test_registry: NodeRegistry,
        connection_tracker: ConnectionTracker,
        circuit_breaker_registry: CircuitBreakerRegistry,
    ) -> None:
        """NODES-01: Unified list includes both etcd and available QUADS hosts."""
        test_registry.add(_make_node(node_id="gpu01"))
        poller = MagicMock()
        poller.hosts = [
            QUADSHost(
                hostname="gpu01", gpu_vendor="NVIDIA", gpu_model="A100", gpu_count=4
            ),
            QUADSHost(
                hostname="gpu02", gpu_vendor="AMD", gpu_model="MI300X", gpu_count=8
            ),
        ]
        poller.available_hostnames = ["gpu01", "gpu02"]
        svc = UnifiedNodeService(
            registry=test_registry,
            poller=poller,
            cb_registry=circuit_breaker_registry,
            tracker=connection_tracker,
        )
        app.dependency_overrides[get_unified_node_service] = lambda: svc

        response = client.get("/admin/nodes")
        data = response.json()

        assert len(data) == 2
        ids = {n["node_id"] for n in data}
        assert ids == {"gpu01", "gpu02"}

    def test_each_node_has_state_and_actions(
        self,
        app: FastAPI,
        client: TestClient,
        test_registry: NodeRegistry,
        connection_tracker: ConnectionTracker,
        circuit_breaker_registry: CircuitBreakerRegistry,
    ) -> None:
        """NODES-02: Each node includes state and actions."""
        poller = MagicMock()
        poller.hosts = [
            QUADSHost(
                hostname="gpu01", gpu_vendor="NVIDIA", gpu_model="A100", gpu_count=4
            ),
        ]
        poller.available_hostnames = ["gpu01"]
        svc = UnifiedNodeService(
            registry=test_registry,
            poller=poller,
            cb_registry=circuit_breaker_registry,
            tracker=connection_tracker,
        )
        app.dependency_overrides[get_unified_node_service] = lambda: svc

        response = client.get("/admin/nodes")
        node = response.json()[0]

        assert "state" in node
        assert "actions" in node
        assert node["state"] == "available"
        assert node["actions"] == ["setup"]

    def test_no_quads_returns_etcd_only(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
    ) -> None:
        """Graceful degradation: no QUADS returns etcd-only nodes."""
        test_registry.add(_make_node(node_id="gpu01"))

        response = client.get("/admin/nodes")
        data = response.json()

        assert len(data) == 1
        assert data[0]["node_id"] == "gpu01"
        assert data[0]["state"] == "healthy"


# -- Dedup guard tests (NODES-04) --


@pytest.fixture(autouse=True)
def _clear_pending_hosts() -> None:
    """Clear the module-level pending_hosts between tests."""
    import inference_proxy.api.admin as admin_mod

    admin_mod.pending_hosts.clear()


class TestSetupDedupGuard:
    """POST /admin/nodes/setup returns 409 for duplicate requests (NODES-04)."""

    def test_returns_409_for_pending_hostname(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        import inference_proxy.api.admin as admin_mod

        admin_mod.pending_hosts.add("gpu01")

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 409

    def test_clears_pending_after_completion(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        """The real background wrapper clears pending state after success."""

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))

        import inference_proxy.api.admin as admin_mod

        assert "gpu01" not in admin_mod.pending_hosts

    @pytest.mark.parametrize("outcome", ["failure", "cancellation"])
    def test_releases_pending_and_host_lease_after_unsuccessful_background_task(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
        outcome: str,
    ) -> None:
        """Failure and cancellation cannot leave setup permanently locked."""
        error: BaseException
        if outcome == "failure":
            error = RuntimeError("provision failed")
        else:
            error = asyncio.CancelledError()
        mock_provisioner.provision.side_effect = error
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]

        with pytest.raises(type(error)):
            asyncio.run(asyncio.wait_for(coro, timeout=1))

        import inference_proxy.api.admin as admin_mod

        assert "gpu01" not in admin_mod.pending_hosts
        lease.release.assert_called()

        mock_provisioner.reset_mock()
        mock_provisioner.provision.side_effect = None
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host = AsyncMock(
            return_value=MagicMock(hostname="gpu01")
        )
        retry = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert retry.status_code == 202

    def test_schedule_failure_releases_pending_and_host_lease(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        """Failure to create the task cannot leak either setup guard."""
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease
        mock_provisioner.fire_background.side_effect = RuntimeError("no task")

        with pytest.raises(RuntimeError, match="no task"):
            client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        import inference_proxy.api.admin as admin_mod

        assert "gpu01" not in admin_mod.pending_hosts
        lease.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_capacity_returns_429_without_leaking_pending_or_host_lease(
        self,
        app: FastAPI,
        admin_auth_headers: dict[str, str],
        mock_provisioner: MagicMock,
    ) -> None:
        """S9: full global capacity rejects immediately and cleans route state."""
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease
        mock_provisioner.fire_background.side_effect = ProvisioningCapacityError(
            active=32,
            limit=32,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=admin_auth_headers,
        ) as client:
            response = await asyncio.wait_for(
                client.post(
                    "/admin/nodes/setup",
                    json={"hostname": "gpu01"},
                ),
                timeout=2,
            )

        assert response.status_code == 429
        assert response.json() == {
            "detail": (
                "Provisioning capacity reached: 32 active task(s), limit 32; "
                "retry after an existing setup finishes"
            )
        }
        assert "Retry-After" not in response.headers

        import inference_proxy.api.admin as admin_mod

        assert "gpu01" not in admin_mod.pending_hosts
        lease.release.assert_called_once()
        mock_provisioner.provision.assert_not_awaited()

    def test_task_cancelled_before_start_releases_pending_and_host_lease(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        """The task callback covers cancellation before coroutine entry."""
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease

        class NeverStartedTask:
            def __init__(self) -> None:
                self.background: Coroutine[Any, Any, None] | None = None
                self.callback: Callable[[Any], None] | None = None

            def add_done_callback(self, callback: Callable[[Any], None]) -> None:
                self.callback = callback

            def cancel_before_start(self) -> None:
                assert self.background is not None
                assert self.callback is not None
                self.background.close()
                self.callback(self)

        task = NeverStartedTask()

        def hold_background(
            background: Coroutine[Any, Any, None],
            *,
            provisioning_hostname: str,
            provisioning_identity: ProvisioningIdentity,
        ) -> NeverStartedTask:
            assert provisioning_hostname == "gpu01"
            assert provisioning_identity == ProvisioningIdentity(InferenceEngine.VLLM)
            task.background = background
            return task

        mock_provisioner.fire_background.side_effect = hold_background

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202

        import inference_proxy.api.admin as admin_mod

        assert "gpu01" in admin_mod.pending_hosts
        task.cancel_before_start()
        assert "gpu01" not in admin_mod.pending_hosts
        lease.release.assert_called_once()


class TestSetupEligibility:
    """Setup eligibility follows node state for managed and standalone hosts."""

    @pytest.mark.parametrize("managed", [True, False])
    @pytest.mark.parametrize(
        ("status", "expected_status"),
        [
            (NodeStatus.HEALTHY, 409),
            (NodeStatus.UNHEALTHY, 409),
            (NodeStatus.RELAUNCHING, 409),
            (NodeStatus.RELAUNCH_FAILED, 409),
            # Acquiring the host lease proves no provision is still running,
            # so this is a persisted record left by an interrupted process.
            (NodeStatus.PROVISIONING, 202),
            # A DRAINING node without a live host operation or connections is
            # a PR 6 reconciliation ghost and is safe to clean.
            (NodeStatus.DRAINING, 202),
            (NodeStatus.FAILED, 202),
            (NodeStatus.UNKNOWN, 202),
        ],
    )
    def test_setup_eligibility_by_status(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
        status: NodeStatus,
        expected_status: int,
        managed: bool,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", status=status))

        response = client.post(
            "/admin/nodes/setup",
            json={"hostname": "gpu01", "managed": managed},
        )

        assert response.status_code == expected_status
        if expected_status == 409:
            mock_provisioner.fire_background.assert_not_called()
            mock_provisioner.cleanup_stale_node.assert_not_awaited()
            return

        mock_provisioner.cleanup_stale_node.assert_awaited_once_with("gpu01")
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))
        mock_provisioner.provision.assert_awaited_once_with(
            "gpu01",
            managed=managed,
            model="llama-3",
            engine=ANY,
            artifact_id=None,
            lifecycle_lease=ANY,
        )

    def test_draining_node_with_connections_is_rejected_actionably(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", status=NodeStatus.DRAINING))
        mock_provisioner.connection_count.return_value = 2

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 409
        assert "2 active request" in response.json()["detail"]
        assert "wait" in response.json()["detail"].lower()
        mock_provisioner.cleanup_stale_node.assert_not_awaited()

    def test_setup_returns_409_while_host_operation_is_reserved(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", status=NodeStatus.DRAINING))
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = None

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 409
        assert "operation" in response.json()["detail"].lower()
        mock_provisioner.cleanup_stale_node.assert_not_awaited()

    @pytest.mark.parametrize(
        "status",
        [
            NodeStatus.PROVISIONING,
            NodeStatus.FAILED,
            NodeStatus.UNKNOWN,
        ],
    )
    def test_retry_clears_stale_state_before_provision(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        connection_tracker: ConnectionTracker,
        circuit_breaker_registry: CircuitBreakerRegistry,
        mock_provisioner: MagicMock,
        status: NodeStatus,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", status=status))
        connection_tracker.increment("gpu01")
        breaker = circuit_breaker_registry.get_or_create("gpu01")
        for _ in range(3):
            breaker.record_failure()

        cleanup_complete = False

        async def cleanup(hostname: str) -> None:
            nonlocal cleanup_complete
            assert hostname == "gpu01"
            test_registry.remove(hostname)
            connection_tracker.remove(hostname)
            circuit_breaker_registry.remove(hostname)
            cleanup_complete = True

        async def provision(
            hostname: str,
            *,
            managed: bool,
            model: str | None,
            engine: object = None,
            artifact_id: str | None = None,
            lifecycle_lease: object,
        ) -> None:
            assert hostname == "gpu01"
            assert managed is True
            assert model == "llama-3"
            assert artifact_id is None
            assert lifecycle_lease is not None
            assert cleanup_complete
            assert test_registry.get(hostname) is None
            assert connection_tracker.get(hostname) == 0
            assert circuit_breaker_registry.get(hostname) is None

        mock_provisioner.cleanup_stale_node.side_effect = cleanup
        mock_provisioner.provision.side_effect = provision

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202
        coro = mock_provisioner.fire_background.call_args.args[0]
        asyncio.run(asyncio.wait_for(coro, timeout=1))

    def test_retry_cleanup_failure_does_not_start_provisioning(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", status=NodeStatus.FAILED))
        lease = MagicMock(hostname="gpu01")
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = lease
        mock_provisioner.cleanup_stale_node.side_effect = RuntimeError("etcd down")

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        assert response.status_code == 503
        mock_provisioner.fire_background.assert_not_called()
        mock_provisioner.provision.assert_not_awaited()
        lease.release.assert_called_once()


# -- QUADS re-validation tests (NODES-05) --


class TestSetupQuadsRevalidation:
    """POST /admin/nodes/setup re-validates against live QUADS (NODES-05)."""

    def test_returns_503_on_quads_connection_error(
        self,
        app: FastAPI,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_quads = AsyncMock()
        mock_quads.get_available.side_effect = QUADSConnectionError("timeout")
        app.dependency_overrides[get_quads_client] = lambda: mock_quads

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 503

    def test_returns_400_for_unavailable_host(
        self,
        app: FastAPI,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_quads = AsyncMock()
        mock_quads.get_available.return_value = ["gpu99"]
        app.dependency_overrides[get_quads_client] = lambda: mock_quads

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 400

    def test_setup_checks_full_configured_availability_window(
        self,
        app: FastAPI,
        client: TestClient,
        mock_provisioner: MagicMock,
        test_settings: Settings,
    ) -> None:
        lookahead_hours = 20
        configured = test_settings.model_copy(
            update={
                "quads": test_settings.quads.model_copy(
                    update={"schedule_lookahead_hours": lookahead_hours}
                )
            }
        )
        app.dependency_overrides[get_settings] = lambda: configured
        mock_quads = AsyncMock()

        async def availability(*, end: datetime | None = None) -> list[str]:
            return ["gpu01"] if end is None else []

        mock_quads.get_available.side_effect = availability
        app.dependency_overrides[get_quads_client] = lambda: mock_quads
        before = datetime.now(tz=UTC)

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})

        after = datetime.now(tz=UTC)
        assert response.status_code == 400
        assert response.json()["detail"] == (
            "Host 'gpu01' is currently assigned or has an upcoming QUADS "
            "assignment within the configured 20-hour scheduling window"
        )
        called_end = mock_quads.get_available.await_args.kwargs["end"]
        assert before + timedelta(hours=lookahead_hours) <= called_end
        assert called_end <= after + timedelta(hours=lookahead_hours)
        mock_provisioner.fire_background.assert_not_called()
        mock_provisioner.provision.assert_not_awaited()

    def test_succeeds_for_available_host(
        self,
        app: FastAPI,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        mock_quads = AsyncMock()
        mock_quads.get_available.return_value = ["gpu01"]
        app.dependency_overrides[get_quads_client] = lambda: mock_quads

        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202

    def test_works_without_quads_configured(
        self,
        client: TestClient,
        mock_provisioner: MagicMock,
    ) -> None:
        """When QUADS not configured (None), setup proceeds without validation."""
        response = client.post("/admin/nodes/setup", json={"hostname": "gpu01"})
        assert response.status_code == 202


class TestExistingEndpointsUnchanged:
    """Existing endpoints still work after refactoring."""

    def test_metrics_still_works(self, client: TestClient) -> None:
        response = client.get("/admin/metrics")
        assert response.status_code == 200

    def test_teardown_still_works(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_provisioner: MagicMock,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        response = client.delete("/admin/nodes/gpu01")
        assert response.status_code == 202


class TestQuadsStatus:
    """GET /admin/quads/status returns poller staleness data (D-10)."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/admin/quads/status")
        assert response.status_code == 200

    def test_unavailable_when_no_poller(self, client: TestClient) -> None:
        """Default fixture has poller=None."""
        data = client.get("/admin/quads/status").json()
        assert data["status"] == "unavailable"
        assert data["last_sync"] is None
        assert data["consecutive_failures"] == 0

    def test_connected_when_zero_failures(
        self, app: FastAPI, client: TestClient
    ) -> None:
        poller = MagicMock()
        poller.last_sync = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)
        poller.consecutive_failures = 0
        app.dependency_overrides[get_quads_poller] = lambda: poller

        data = client.get("/admin/quads/status").json()
        assert data["status"] == "connected"

    def test_stale_when_one_failure(self, app: FastAPI, client: TestClient) -> None:
        poller = MagicMock()
        poller.last_sync = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)
        poller.consecutive_failures = 1
        app.dependency_overrides[get_quads_poller] = lambda: poller

        data = client.get("/admin/quads/status").json()
        assert data["status"] == "stale"

    def test_stale_when_two_failures(self, app: FastAPI, client: TestClient) -> None:
        poller = MagicMock()
        poller.last_sync = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)
        poller.consecutive_failures = 2
        app.dependency_overrides[get_quads_poller] = lambda: poller

        data = client.get("/admin/quads/status").json()
        assert data["status"] == "stale"

    def test_unavailable_when_three_failures(
        self, app: FastAPI, client: TestClient
    ) -> None:
        poller = MagicMock()
        poller.last_sync = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)
        poller.consecutive_failures = 3
        app.dependency_overrides[get_quads_poller] = lambda: poller

        data = client.get("/admin/quads/status").json()
        assert data["status"] == "unavailable"

    def test_unavailable_when_never_synced(
        self, app: FastAPI, client: TestClient
    ) -> None:
        poller = MagicMock()
        poller.last_sync = None
        poller.consecutive_failures = 0
        app.dependency_overrides[get_quads_poller] = lambda: poller

        data = client.get("/admin/quads/status").json()
        assert data["status"] == "unavailable"


# -- Power management endpoint tests (PWR-01/02/03/04, D-02/D-05/D-07) --


class TestGetPowerState:
    """GET /admin/nodes/{hostname}/power returns BMC power state."""

    def test_returns_current_state(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.get_power_state.return_value = "On"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.get("/admin/nodes/gpu01/power")
        assert response.status_code == 200
        assert response.json() == {"hostname": "gpu01", "power_state": "On"}

    def test_returns_503_when_not_configured(self, client: TestClient) -> None:
        response = client.get("/admin/nodes/gpu01/power")
        assert response.status_code == 503

    def test_normalizes_hostname(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.get_power_state.return_value = "Off"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.get("/admin/nodes/GPU01/power")
        assert response.json()["hostname"] == "gpu01"

    def test_returns_502_on_redfish_error(
        self, app: FastAPI, client: TestClient
    ) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.get_power_state.side_effect = RedfishError("BMC unreachable")
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.get("/admin/nodes/gpu01/power")
        assert response.status_code == 502
        assert "BMC unreachable" in response.json()["detail"]

    @pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
    def test_unapproved_bmc_destination_sends_no_credentials(
        self,
        app: FastAPI,
        client: TestClient,
        mock_http_client: httpx.AsyncClient,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            url="https://mgmt-attacker.example.net/redfish/v1/Systems/1",
            json={"PowerState": "On"},
        )
        app.dependency_overrides[get_redfish_client] = lambda: _real_redfish_client(
            mock_http_client
        )

        response = client.get("/admin/nodes/attacker.example.net/power")

        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]
        assert httpx_mock.get_requests() == []

    def test_approved_bmc_destination_sends_credentials(
        self,
        app: FastAPI,
        client: TestClient,
        mock_http_client: httpx.AsyncClient,
        httpx_mock: HTTPXMock,
    ) -> None:
        url = "https://mgmt-gpu01/redfish/v1/Systems/1"
        httpx_mock.add_response(url=url, json={"PowerState": "On"})
        app.dependency_overrides[get_redfish_client] = lambda: _real_redfish_client(
            mock_http_client
        )

        response = client.get("/admin/nodes/GPU01/power")

        assert response.status_code == 200
        [request] = httpx_mock.get_requests()
        assert str(request.url) == url
        assert (
            request.headers["Authorization"] == "Basic b3BlcmF0b3I6cmVkZmlzaC1zZWNyZXQ="
        )

    @pytest.mark.parametrize(
        "content",
        [
            b"{}",
            b"<html>error</html>",
            b'{"PowerState": null}',
            b'{"PowerState": 7}',
            b'{"PowerState": "Paused"}',
        ],
    )
    def test_malformed_power_state_returns_502(
        self,
        app: FastAPI,
        mock_http_client: httpx.AsyncClient,
        httpx_mock: HTTPXMock,
        admin_auth_headers: dict[str, str],
        content: bytes,
    ) -> None:
        httpx_mock.add_response(
            url="https://mgmt-gpu01/redfish/v1/Systems/1",
            content=content,
        )
        app.dependency_overrides[get_redfish_client] = lambda: _real_redfish_client(
            mock_http_client
        )

        bounded_client = TestClient(
            app,
            headers=admin_auth_headers,
            raise_server_exceptions=False,
        )
        try:
            response = bounded_client.get("/admin/nodes/gpu01/power")
        finally:
            bounded_client.close()

        assert response.status_code == 502
        assert isinstance(response.json()["detail"], str)

    def test_programming_error_returns_500_not_redfish_502(
        self,
        app: FastAPI,
        mock_http_client: httpx.AsyncClient,
        httpx_mock: HTTPXMock,
        monkeypatch: pytest.MonkeyPatch,
        admin_auth_headers: dict[str, str],
    ) -> None:
        httpx_mock.add_response(
            url="https://mgmt-gpu01/redfish/v1/Systems/1",
            json={"PowerState": "On"},
        )
        app.dependency_overrides[get_redfish_client] = lambda: _real_redfish_client(
            mock_http_client
        )

        def broken_json(_response: httpx.Response) -> object:
            raise AttributeError("programming defect")

        monkeypatch.setattr(httpx.Response, "json", broken_json)
        bounded_client = TestClient(
            app,
            headers=admin_auth_headers,
            raise_server_exceptions=False,
        )
        try:
            response = bounded_client.get("/admin/nodes/gpu01/power")
        finally:
            bounded_client.close()

        assert response.status_code == 500


class TestExecutePowerAction:
    """POST /admin/nodes/{hostname}/power executes power actions."""

    def test_power_on(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.return_value = "On"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.post("/admin/nodes/gpu01/power", json={"action": "On"})
        assert response.status_code == 200
        assert response.json() == {"hostname": "gpu01", "power_state": "On"}
        mock_redfish.power_action.assert_called_once_with("gpu01", "On")

    def test_force_off(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.return_value = "Off"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.post("/admin/nodes/gpu01/power", json={"action": "ForceOff"})
        assert response.status_code == 200
        assert response.json()["power_state"] == "Off"

    def test_graceful_restart(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.return_value = "On"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.post(
            "/admin/nodes/gpu01/power", json={"action": "GracefulRestart"}
        )
        assert response.status_code == 200
        assert response.json()["power_state"] == "On"

    def test_force_restart(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.return_value = "On"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.post(
            "/admin/nodes/gpu01/power", json={"action": "ForceRestart"}
        )
        assert response.status_code == 200
        assert response.json()["power_state"] == "On"

    def test_returns_503_when_not_configured(self, client: TestClient) -> None:
        response = client.post("/admin/nodes/gpu01/power", json={"action": "On"})
        assert response.status_code == 503

    def test_returns_422_for_invalid_action(self, client: TestClient) -> None:
        response = client.post("/admin/nodes/gpu01/power", json={"action": "Shutdown"})
        assert response.status_code == 422

    def test_normalizes_hostname(self, app: FastAPI, client: TestClient) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.return_value = "On"
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        client.post("/admin/nodes/GPU01/power", json={"action": "On"})
        mock_redfish.power_action.assert_called_once_with("gpu01", "On")

    def test_returns_502_on_redfish_error(
        self, app: FastAPI, client: TestClient
    ) -> None:
        mock_redfish = AsyncMock()
        mock_redfish.power_action.side_effect = RedfishError("Poll timeout")
        app.dependency_overrides[get_redfish_client] = lambda: mock_redfish

        response = client.post("/admin/nodes/gpu01/power", json={"action": "On"})
        assert response.status_code == 502
        assert "Poll timeout" in response.json()["detail"]


# -- Recommendation endpoint tests (API-01, API-02, API-03) --


SAMPLE_RESULT = LLMFitResult(
    system=SystemInfo(
        has_gpu=True,
        gpu_vram_gb=80.0,
        gpu_name="NVIDIA A100",
        cpu_name="AMD EPYC 7742",
        total_ram_gb=64.0,
        available_ram_gb=58.24,
        cpu_cores=16,
        backend="CUDA",
    ),
    models=[
        ModelRecommendation(
            name="llama-3.3-70b",
            score=95.2,
            fit_level="perfect",
            estimated_tps=42.5,
            memory_required_gb=43.68,
            provider="Meta",
            best_quant="4bit",
            run_mode="gpu",
            params_b=70.0,
            context_length=131072,
            utilization_pct=68.2,
            category="General",
            runtime="vLLM",
            gguf_sources=(
                GGUFSource(
                    repo="org/llama-3.3-70b-GGUF",
                    provider="publisher",
                ),
            ),
        ),
        ModelRecommendation(
            name="qwen-2.5-72b-instruct",
            score=88.7,
            fit_level="good",
            estimated_tps=38.1,
            memory_required_gb=45.2,
            provider="Alibaba",
            best_quant="4bit",
            run_mode="gpu",
            params_b=72.0,
            context_length=131072,
            utilization_pct=72.5,
            category="General",
            runtime="vLLM",
        ),
    ],
)


class TestRecommendations:
    """GET /admin/nodes/{hostname}/recommendations happy path (API-01, API-02)."""

    def test_returns_200_with_models(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 200
        data = response.json()
        assert data["hostname"] == "gpu01"
        assert "system" in data
        assert data["system"]["gpu_name"] == "NVIDIA A100"
        assert len(data["models"]) == 2
        assert data["models"][0]["runtime"] == "vllm"
        assert data["models"][0]["gguf_sources"] == [
            {
                "repo": "org/llama-3.3-70b-GGUF",
                "provider": "publisher",
            }
        ]

    def test_response_includes_hostname(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="node42.example.com"))
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/node42.example.com/recommendations")

        assert response.json()["hostname"] == "node42.example.com"

    def test_response_includes_hardware(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")
        system = response.json()["system"]

        assert system["gpu_name"] == "NVIDIA A100"
        assert system["gpu_vram_gb"] == 80.0
        assert system["backend"] == "CUDA"

    def test_invalid_hostname_returns_400(
        self,
        client: TestClient,
    ) -> None:
        # '@' is not in the allowed hostname regex, triggers _validated_hostname 400
        response = client.get("/admin/nodes/host@evil/recommendations")
        assert response.status_code == 400


class TestRecommendationErrors:
    """Error scenarios for GET /admin/nodes/{hostname}/recommendations (API-03, D-01)."""

    def test_timeout_returns_502(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.side_effect = LLMFitTimeoutError("gpu01", 60.0)

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 502
        data = response.json()
        assert data["error_type"] == "timeout"
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0

    async def test_install_timeout_returns_structured_502(
        self,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        ssh = MagicMock()
        ssh.run = AsyncMock(
            side_effect=[
                RemoteCommandError("gpu01", "llmfit recommend", 127),
                TimeoutError(),
            ]
        )
        runner = LLMFitRunner(
            ssh_client=ssh,
            settings=LLMFitSettings(timeout=0.01),
        )
        provisioner = MagicMock()
        provisioner.validate_endpoint.return_value = "http://gpu01:8000"
        application = FastAPI()
        application.include_router(admin_router)

        async def no_auth() -> None:
            return None

        async def use_runner() -> LLMFitRunner:
            return runner

        async def use_registry() -> NodeRegistry:
            return test_registry

        async def use_provisioner() -> MagicMock:
            return provisioner

        async def no_poller() -> None:
            return None

        application.dependency_overrides[require_admin_auth] = no_auth
        application.dependency_overrides[get_llmfit_runner] = use_runner
        application.dependency_overrides[get_registry] = use_registry
        application.dependency_overrides[get_provisioner] = use_provisioner
        application.dependency_overrides[get_quads_poller] = no_poller

        transport = httpx.ASGITransport(
            app=application,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as bounded_client:
            response = await asyncio.wait_for(
                bounded_client.get("/admin/nodes/gpu01/recommendations"),
                timeout=2,
            )

        assert response.status_code == 502
        assert response.json()["error_type"] == "timeout"

    def test_parse_error_returns_502(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.side_effect = LLMFitParseError(
            "invalid JSON", raw_output="not-json-garbage"
        )

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 502
        data = response.json()
        assert data["error_type"] == "parse_error"
        assert "parse" in data["detail"].lower()

    def test_ssh_connection_error_returns_502(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.side_effect = SSHConnectionError(
            "gpu01", "connection refused"
        )

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 502
        data = response.json()
        assert data["error_type"] == "connection_error"
        assert "connection" in data["detail"].lower()

    def test_command_error_returns_502(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.side_effect = RemoteCommandError(
            "gpu01", "llmfit recommend", exit_status=127, stderr="not found"
        )

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 502
        data = response.json()
        assert data["error_type"] == "ssh_error"
        assert "127" in data["detail"]

    def test_raw_output_not_exposed(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
        test_registry: NodeRegistry,
    ) -> None:
        """D-01: raw_output must never appear in the API response body."""
        test_registry.add(_make_node(node_id="gpu01"))
        mock_llmfit_runner.recommend.side_effect = LLMFitParseError(
            "bad json", raw_output="SECRET_RAW_CONTENT_MARKER"
        )

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 502
        assert "SECRET_RAW_CONTENT_MARKER" not in response.text


class TestRecommendationTargetPolicy:
    """Recommendations may SSH only to trusted, actionable node targets (S6)."""

    def test_unknown_target_is_rejected_before_runner(
        self,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
    ) -> None:
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT
        response = client.get("/admin/nodes/gpu-unknown/recommendations")

        assert response.status_code == 404
        mock_llmfit_runner.recommend.assert_not_awaited()

    @pytest.mark.parametrize("managed", [True, False])
    def test_registered_target_is_accepted(
        self,
        client: TestClient,
        test_registry: NodeRegistry,
        mock_llmfit_runner: MagicMock,
        managed: bool,
    ) -> None:
        test_registry.add(_make_node(node_id="gpu01", managed=managed))
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 200
        mock_llmfit_runner.recommend.assert_awaited_once_with("gpu01")

    def test_currently_available_quads_target_is_accepted(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
    ) -> None:
        poller = MagicMock()
        poller.hosts = [
            QUADSHost(
                hostname="GPU01",
                gpu_vendor="NVIDIA",
                gpu_model="A100",
                gpu_count=8,
            )
        ]
        poller.available_hostnames = ["gpu01"]
        app.dependency_overrides[get_quads_poller] = lambda: poller
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 200
        mock_llmfit_runner.recommend.assert_awaited_once_with("gpu01")

    def test_quads_inventory_target_that_is_unavailable_is_rejected(
        self,
        app: FastAPI,
        client: TestClient,
        mock_llmfit_runner: MagicMock,
    ) -> None:
        poller = MagicMock()
        poller.hosts = [
            QUADSHost(
                hostname="gpu01",
                gpu_vendor="NVIDIA",
                gpu_model="A100",
                gpu_count=8,
            )
        ]
        poller.available_hostnames = []
        app.dependency_overrides[get_quads_poller] = lambda: poller
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 404
        mock_llmfit_runner.recommend.assert_not_awaited()

    def test_quads_available_target_outside_allowlist_is_rejected(
        self,
        app: FastAPI,
        client: TestClient,
        mock_provisioner: MagicMock,
        mock_llmfit_runner: MagicMock,
    ) -> None:
        poller = MagicMock()
        poller.hosts = [
            QUADSHost(
                hostname="gpu01",
                gpu_vendor="NVIDIA",
                gpu_model="A100",
                gpu_count=8,
            )
        ]
        poller.available_hostnames = ["gpu01"]
        app.dependency_overrides[get_quads_poller] = lambda: poller
        mock_provisioner.validate_endpoint.side_effect = EndpointValidationError(
            "host is not allowed"
        )
        mock_llmfit_runner.recommend.return_value = SAMPLE_RESULT

        response = client.get("/admin/nodes/gpu01/recommendations")

        assert response.status_code == 404
        mock_provisioner.validate_endpoint.assert_called_once_with("gpu01")
        mock_llmfit_runner.recommend.assert_not_awaited()


# -- Model catalog endpoint tests (CAT-02) --


class TestModelCatalog:
    """GET /admin/models/catalog returns NFS model catalog."""

    def test_catalog_returns_models(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        from inference_proxy.huggingface.catalog import (
            CatalogEntry,
            ModelCatalogResponse,
        )

        mock_catalog = MagicMock()
        mock_catalog.list_models = AsyncMock(
            return_value=ModelCatalogResponse(
                models=[
                    CatalogEntry(repo_id="meta-llama/Llama-3.1-8B-Instruct"),
                    CatalogEntry(repo_id="mistralai/Mistral-7B-v0.1"),
                ]
            )
        )
        app.dependency_overrides[get_catalog_service] = lambda: mock_catalog

        response = client.get("/admin/models/catalog")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        repo_ids = {m["repo_id"] for m in data["models"]}
        assert repo_ids == {
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-v0.1",
        }

    def test_catalog_empty(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        from inference_proxy.huggingface.catalog import ModelCatalogResponse

        mock_catalog = MagicMock()
        mock_catalog.list_models = AsyncMock(
            return_value=ModelCatalogResponse(models=[])
        )
        app.dependency_overrides[get_catalog_service] = lambda: mock_catalog

        response = client.get("/admin/models/catalog")

        assert response.status_code == 200
        assert response.json() == {
            "models": [],
            "gguf_artifacts": [],
            "incomplete_count": 0,
            "unverifiable_count": 0,
            "invalid_artifact_count": 0,
            "cache_warning_count": 0,
        }

    def test_catalog_surfaces_degraded_cache_counts(
        self,
        app: FastAPI,
        client: TestClient,
    ) -> None:
        from inference_proxy.huggingface.catalog import ModelCatalogResponse

        mock_catalog = MagicMock()
        mock_catalog.list_models = AsyncMock(
            return_value=ModelCatalogResponse(
                models=[],
                incomplete_count=2,
                unverifiable_count=3,
                invalid_artifact_count=4,
                cache_warning_count=5,
            )
        )
        app.dependency_overrides[get_catalog_service] = lambda: mock_catalog

        response = client.get("/admin/models/catalog")

        assert response.status_code == 200
        assert response.headers["X-Inference-Proxy-Data-Degraded"] == ("model-catalog")
        assert response.json() == {
            "models": [],
            "gguf_artifacts": [],
            "incomplete_count": 2,
            "unverifiable_count": 3,
            "invalid_artifact_count": 4,
            "cache_warning_count": 5,
        }
