"""GET /admin/placement and the claim reset action."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth.store import AuthStore
from inference_proxy.config.settings import PlacementSettings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.artifacts import GGUFArtifactIndex
from inference_proxy.placement.catalog import BUILTIN_PROFILES
from inference_proxy.placement.claims import ClaimStore
from inference_proxy.placement.reconciler import PlacementReconciler
from inference_proxy.provisioning.provisioner import NodeProvisioner
from inference_proxy.quads.client import QUADSClient
from inference_proxy.quads.poller import QUADSPoller
from inference_proxy.routing.request_metrics import RequestMetrics
from tests.placement.fakes import (
    FakeArtifactIndex,
    FakeEtcd,
    FakePoller,
    FakeProvisioner,
    FakeQuads,
    catalog_artifacts,
    l4_host,
)


def _reconciler(*, artifacts: bool = True, enabled: bool = True) -> PlacementReconciler:
    hosts = [l4_host(f"l4-{index:02d}") for index in range(3)]
    registry = NodeRegistry()
    return PlacementReconciler(
        settings=PlacementSettings(enabled=enabled),
        quads_client=cast(QUADSClient, FakeQuads([h.hostname for h in hosts])),
        quads_poller=cast(QUADSPoller, FakePoller(hosts)),
        registry=registry,
        provisioner=cast(NodeProvisioner, FakeProvisioner(registry)),
        artifact_index=cast(
            GGUFArtifactIndex,
            FakeArtifactIndex(catalog_artifacts() if artifacts else []),
        ),
        claims=ClaimStore(FakeEtcd()),
        lookahead_hours=24,
        profiles=tuple(
            p.model_copy(update={"qualified_gpus": ("l4",)}) for p in BUILTIN_PROFILES
        ),
    )


def test_without_quads_the_endpoint_says_why(client: TestClient) -> None:
    body = client.get("/admin/placement").json()

    assert body["available"] is False
    assert "QUADS" in body["error"]
    assert body["profiles"] == []


def test_requires_admin_auth(app: FastAPI) -> None:
    assert TestClient(app).get("/admin/placement").status_code == 401


def test_disabled_placement_still_reports_missing_catalog_files(
    app: FastAPI, client: TestClient
) -> None:
    app.state.placement_reconciler = _reconciler(artifacts=False, enabled=False)

    body = client.get("/admin/placement").json()

    assert body["available"] is True and body["enabled"] is False
    assert body["last_run_at"] is None
    assert len(body["missing_artifacts"]) == 6
    assert {item["filename"] for item in body["missing_artifacts"]} >= {
        "Qwen3.8-27B-UD-Q4_K_S.gguf",
        "Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf",
    }
    assert all(not item["artifacts_present"] for item in body["profiles"])
    assert [item["weight"] for item in body["profiles"]] == [9, 2, 2, 2]


@pytest.mark.asyncio
async def test_status_after_a_pass_with_per_model_demand(
    app: FastAPI,
    client: TestClient,
    request_metrics: RequestMetrics,
    auth_store: AuthStore,
) -> None:
    reconciler = _reconciler()
    app.state.placement_reconciler = reconciler
    for _ in range(3):  # two provisions at a time, then a settled report
        await reconciler.reconcile_once()
        await cast(FakeProvisioner, reconciler._provisioner).drain()
    request_metrics.record_request("l4-00", "unsloth/Qwen3.8-27B-GGUF")
    user = auth_store.upsert_google_user(
        google_sub="s", email="a@example.com", name="A", picture=""
    )
    token = auth_store.create_token(user.id, "t")
    auth_store.record_usage(
        user_id=user.id,
        token_id=token.id,
        model="unsloth/Qwen3.8-27B-GGUF",
        endpoint="/v1/chat/completions",
        prompt_tokens=1000,
        completion_tokens=250,
        total_tokens=1250,
    )

    body = client.get("/admin/placement").json()

    assert body["error"] == ""
    assert [
        (
            p["profile_id"],
            p["target"],
            p["held"],
            p["serving"],
            p["pending"],
            p["failed"],
        )
        for p in body["profiles"]
    ] == [
        ("qwen3.8-27b-24g", 1, 1, 1, 0, 0),
        ("qwen3.6-35b-a3b-24g", 1, 1, 1, 0, 0),
        ("muse-glimmer-30b-24g", 1, 1, 1, 0, 0),
        # Three hosts for four profiles: catalog order leaves Gemma unplaced.
        ("gemma-4-31b-24g", 0, 0, 0, 0, 0),
    ]
    assert body["counters_started_at"] is not None
    assert "zero does not mean no demand" in body["usage_note"]
    assert {claim["state"] for claim in body["claims"]} == {"active"}
    assert body["missing_artifacts"] == []
    assert body["usage"] == [
        {
            "model": "unsloth/Qwen3.8-27B-GGUF",
            "requests_since_restart": 1,
            "recorded_requests": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 250,
            "total_tokens": 1250,
        }
    ]


@pytest.mark.asyncio
async def test_claim_reset_status_codes(app: FastAPI, client: TestClient) -> None:
    assert client.delete("/admin/placement/claims/l4-00").status_code == 404
    reconciler = _reconciler()
    app.state.placement_reconciler = reconciler
    provisioner = cast(FakeProvisioner, reconciler._provisioner)
    provisioner.failures["l4-00"] = ["boom"]
    await reconciler.reconcile_once()
    await provisioner.drain()

    assert client.delete("/admin/placement/claims/unknown").status_code == 404
    assert client.delete("/admin/placement/claims/l4-01").status_code == 409  # active
    assert client.delete("/admin/placement/claims/L4-00.").status_code == 204
    assert client.delete("/admin/placement/claims/l4-00").status_code == 404


@pytest.mark.asyncio
async def test_disabling_placement_does_not_hide_durable_claims(
    app: FastAPI, client: TestClient
) -> None:
    """Claims outlive the loop: a disabled or restarted gateway still shows them."""
    enabled = _reconciler()
    provisioner = cast(FakeProvisioner, enabled._provisioner)
    provisioner.failures["l4-00"] = ["boom"]
    provisioner.hold = True
    await enabled.reconcile_once()
    await asyncio.sleep(0)
    provisioner.gates["l4-01"].set()
    for _ in range(20):
        await asyncio.sleep(0.01)

    # The same etcd and registry, seen by a gateway restarted with placement off.
    disabled = _reconciler(enabled=False)
    disabled._claims = enabled._claims
    disabled._registry = enabled._registry
    app.state.placement_reconciler = disabled

    body = client.get("/admin/placement").json()

    assert body["enabled"] is False and body["last_run_at"] is None
    assert sorted(claim["hostname"] for claim in body["claims"]) == ["l4-00", "l4-01"]
    qwen = body["profiles"][0]
    states = {claim["hostname"]: claim["state"] for claim in body["claims"]}
    assert states["l4-00"] in ("provisioning", "failed")
    assert qwen["held"] + body["profiles"][1]["held"] == 2
    provisioner.gates.setdefault("l4-00", asyncio.Event()).set()
    await provisioner.drain()
