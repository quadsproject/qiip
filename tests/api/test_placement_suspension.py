"""Manual teardown persists intent before starting remote work."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import NodeStatus
from tests.placement.fakes import node


def test_teardown_suspends_before_background_work(
    app: FastAPI,
    client: TestClient,
    test_registry: NodeRegistry,
    mock_provisioner: MagicMock,
) -> None:
    test_registry.add(node("l4-00", NodeStatus.HEALTHY))
    response = client.delete("/admin/nodes/L4-00.")
    assert response.status_code == 202
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]
    mock_provisioner.teardown.assert_not_awaited()
    asyncio.run(mock_provisioner.fire_background.call_args.args[0])
    assert client.delete("/admin/placement/suspensions/l4-00").status_code == 204
    assert client.get("/admin/placement").json()["suspended_hosts"] == []


def test_write_failure_prevents_teardown(
    app: FastAPI,
    client: TestClient,
    test_registry: NodeRegistry,
    mock_provisioner: MagicMock,
) -> None:
    test_registry.add(node("l4-00", NodeStatus.HEALTHY))
    app.state.placement_suspensions._etcd.fail = True
    assert client.delete("/admin/nodes/l4-00").status_code == 503
    mock_provisioner.fire_background.assert_not_called()
    assert test_registry.get("l4-00") is not None


@pytest.mark.parametrize(
    "path", ["/admin/nodes/missing", "/admin/placement/suspensions/l4-00"]
)
def test_actions_require_admin(app: FastAPI, path: str) -> None:
    assert TestClient(app).delete(path).status_code == 401


def test_resume_busy_host_preserves_suspension(
    app: FastAPI,
    client: TestClient,
    mock_provisioner: MagicMock,
) -> None:
    asyncio.run(app.state.placement_suspensions.suspend("l4-00"))
    mock_provisioner.try_reserve_host.side_effect = None
    mock_provisioner.try_reserve_host.return_value = None
    assert client.delete("/admin/placement/suspensions/l4-00").status_code == 409
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]


@pytest.mark.parametrize("self_setup", [False, True])
def test_rejected_teardown_does_not_suspend(
    app: FastAPI,
    client: TestClient,
    test_registry: NodeRegistry,
    mock_provisioner: MagicMock,
    self_setup: bool,
) -> None:
    if self_setup:
        test_registry.add(node("l4-00", NodeStatus.HEALTHY, self_setup=True))
    response = client.delete("/admin/nodes/l4-00")
    assert response.status_code == (409 if self_setup else 404)
    assert client.get("/admin/placement").json()["suspended_hosts"] == []
    mock_provisioner.cancel_provision.assert_not_awaited()
    mock_provisioner.try_reserve_host.assert_not_awaited()
    mock_provisioner.fire_background.assert_not_called()


def test_failed_teardown_keeps_suspension(
    app: FastAPI,
    client: TestClient,
    test_registry: NodeRegistry,
    mock_provisioner: MagicMock,
) -> None:
    test_registry.add(node("l4-00", NodeStatus.HEALTHY))
    mock_provisioner.teardown.side_effect = RuntimeError("SSH unavailable")
    assert client.delete("/admin/nodes/l4-00").status_code == 202
    with pytest.raises(RuntimeError, match="SSH unavailable"):
        asyncio.run(mock_provisioner.fire_background.call_args.args[0])
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]


def test_resume_storage_failure_is_not_reported_as_success(
    app: FastAPI,
    client: TestClient,
) -> None:
    asyncio.run(app.state.placement_suspensions.suspend("l4-00"))
    app.state.placement_suspensions._etcd.fail = True
    assert client.delete("/admin/placement/suspensions/l4-00").status_code == 503
    assert client.get("/admin/placement").status_code == 503
    app.state.placement_suspensions._etcd.fail = False
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]


def test_resume_during_cancel_handoff_aborts_teardown(
    app: FastAPI,
    client: TestClient,
    mock_provisioner: MagicMock,
) -> None:
    from inference_proxy.models.node import InferenceEngine
    from inference_proxy.provisioning.provisioner import ProvisioningIdentity

    record = object()
    mock_provisioner.active_provision.return_value = record
    lease = MagicMock()
    mock_provisioner.try_reserve_host.side_effect = None
    mock_provisioner.try_reserve_host.return_value = lease

    async def cancel(host: str, expected: object) -> ProvisioningIdentity:
        assert expected is record
        assert await app.state.placement_suspensions.list() == {"l4-00"}
        await app.state.placement_suspensions.resume(host)
        return ProvisioningIdentity(InferenceEngine.LLAMA_CPP)

    mock_provisioner.cancel_provision.side_effect = cancel
    response = client.delete("/admin/nodes/L4-00.")
    assert response.status_code == 409
    assert "resumed" in response.json()["detail"]
    mock_provisioner.fire_background.assert_not_called()
    lease.release.assert_called_once_with()
    assert client.get("/admin/placement").json()["suspended_hosts"] == []


@pytest.mark.parametrize("failure", ["cancel", "lease", "verification"])
def test_post_suspension_failure_retains_opt_out(
    app: FastAPI,
    client: TestClient,
    mock_provisioner: MagicMock,
    failure: str,
) -> None:
    from inference_proxy.models.node import InferenceEngine
    from inference_proxy.provisioning.provisioner import ProvisioningIdentity

    mock_provisioner.active_provision.return_value = object()
    lease = MagicMock()
    mock_provisioner.try_reserve_host.side_effect = None
    mock_provisioner.try_reserve_host.return_value = lease

    async def cancel(host: str, record: object) -> ProvisioningIdentity:
        assert await app.state.placement_suspensions.list() == {"l4-00"}
        if failure == "cancel":
            raise RuntimeError("cancel failed")
        if failure == "verification":
            # The same store served the successful write and now fails its
            # post-cancellation verification read.
            app.state.placement_suspensions._etcd.fail = True
        return ProvisioningIdentity(InferenceEngine.LLAMA_CPP)

    mock_provisioner.cancel_provision.side_effect = cancel
    if failure == "lease":
        mock_provisioner.try_reserve_host.side_effect = RuntimeError("reserve failed")
    response = client.delete("/admin/nodes/l4-00")
    assert response.status_code == 503
    assert "retry teardown or resume" in response.json()["detail"]
    mock_provisioner.fire_background.assert_not_called()
    if failure == "verification":
        lease.release.assert_called_once_with()
    else:
        lease.release.assert_not_called()
    app.state.placement_suspensions._etcd.fail = False
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]


def test_stale_operation_is_conflict_and_keeps_suspension(
    client: TestClient,
    mock_provisioner: MagicMock,
) -> None:
    from inference_proxy.provisioning.provisioner import (
        ProvisioningOperationChangedError,
    )

    mock_provisioner.active_provision.return_value = object()
    mock_provisioner.cancel_provision.side_effect = ProvisioningOperationChangedError(
        "changed"
    )
    response = client.delete("/admin/nodes/l4-00")
    assert response.status_code == 409
    assert "automatic placement is suspended" in response.json()["detail"]
    mock_provisioner.try_reserve_host.assert_not_awaited()
    mock_provisioner.fire_background.assert_not_called()
    assert client.get("/admin/placement").json()["suspended_hosts"] == ["l4-00"]


@pytest.mark.parametrize(
    "reason", ["no_force", "endpoint", "busy", "registration_race"]
)
def test_rejected_recovery_never_suspends(
    client: TestClient,
    test_registry: NodeRegistry,
    mock_provisioner: MagicMock,
    reason: str,
) -> None:
    from inference_proxy.models.endpoint import EndpointValidationError

    query = "?force=true&recovery_engine=vllm"
    if reason == "no_force":
        query = "?recovery_engine=vllm"
    elif reason == "endpoint":
        mock_provisioner.validate_endpoint.side_effect = EndpointValidationError(
            "blocked"
        )
    elif reason == "busy":
        mock_provisioner.try_reserve_host.side_effect = None
        mock_provisioner.try_reserve_host.return_value = None
    else:

        async def reserve(host: str) -> MagicMock:
            test_registry.add(node(host, NodeStatus.HEALTHY))
            return MagicMock()

        mock_provisioner.try_reserve_host.side_effect = reserve
    response = client.delete("/admin/nodes/L4-00." + query)
    assert response.status_code in (400, 409)
    assert client.get("/admin/placement").json()["suspended_hosts"] == []
    mock_provisioner.cancel_provision.assert_not_awaited()
    mock_provisioner.fire_background.assert_not_called()
