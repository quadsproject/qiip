"""Harness labels match onboarding while preserving request model IDs."""

import json

import pytest
import yaml

from inference_proxy.models.display import model_display_name
from inference_proxy.onboarding.harness import get_harness

MODELS = {
    "unsloth/Qwen3.8-27B-GGUF": "Qwen3.8-27B",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/Muse-Glimmer-30B-GGUF": "Muse-Glimmer-30B",
    "unsloth/gemma-4-31B-it-GGUF": "gemma-4-31B-it",
}


@pytest.mark.parametrize("harness_id", ["pi", "omp", "opencode"])
def test_configured_labels_preserve_raw_ids(harness_id: str) -> None:
    harness = get_harness(harness_id)
    assert harness is not None
    rendered = harness.render("https://qiip.example", "test-token", list(MODELS))
    if harness_id == "omp":
        items = yaml.safe_load(rendered)["providers"]["qiip"]["models"]
    elif harness_id == "pi":
        items = json.loads(rendered)["providers"]["qiip"]["models"]
    else:
        config = json.loads(rendered)
        assert config["model"] == "qiip/" + next(iter(MODELS))
        items = [
            {"id": key, **value}
            for key, value in config["provider"]["qiip"]["models"].items()
        ]
    assert {item["id"]: item["name"] for item in items} == MODELS


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("custom-model", "custom-model"),
        ("publisher/custom-model", "custom-model"),
        ("publisher/model-gguf", "model"),
        ("publisher/model-mtp-gguf", "model"),
        ("publisher/-GGUF", "publisher/-GGUF"),
        ("publisher/model-MTP", "model-MTP"),
        ("publisher/GGUF-model", "GGUF-model"),
    ],
)
def test_clean_name_preserves_model_variants(raw: str, expected: str) -> None:
    assert model_display_name(raw) == expected


@pytest.mark.parametrize("model,label", MODELS.items())
def test_admin_response_supplies_shared_display_name(model: str, label: str) -> None:
    from inference_proxy.models.admin import AdminNodeResponse

    response = AdminNodeResponse(
        node_id="host",
        endpoint="http://host:8080",
        model=model,
        status="healthy",
        active_connections=0,
        circuit_breaker_state="closed",
    ).model_dump()
    assert response["model"] == model
    assert response["model_display_name"] == label
