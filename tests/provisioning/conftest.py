"""Shared provisioning test doubles.

The vLLM serving gate (issue #134) performs SSH process inspection and real
HTTP requests. Most provisioning tests exercise the surrounding sequence, so
they stub the gate with successful evidence. Tests that own the gate mark
themselves with ``real_vllm_serving``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from inference_proxy.models.node import VllmServingVerification
from inference_proxy.provisioning.provisioner import NodeProvisioner


def _serving_evidence(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    *,
    max_model_len: int | None = None,
) -> VllmServingVerification:
    return VllmServingVerification(
        served_model=model,
        runtime_version="0.26.0",
        model_max_model_len=max_model_len,
        process_id=4242,
        process_age_seconds=12,
        task="chat",
        endpoint="/v1/chat/completions",
        latency_ms=1.5,
        prompt_tokens=3,
        completion_tokens=1,
        finish_reason="length",
        verified_at=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def stub_vllm_serving_verification(request: pytest.FixtureRequest) -> Any:
    """Stub the serving gate unless the test explicitly owns it."""
    if request.node.get_closest_marker("real_vllm_serving") is not None:
        yield
        return

    async def verify(
        hostname: str,
        *,
        expected_model: str,
        started_at: datetime,
        expected_max_model_len: int | None = None,
        revision_fallback: str | None = None,
    ) -> VllmServingVerification:
        return _serving_evidence(expected_model, max_model_len=expected_max_model_len)

    with patch.object(
        NodeProvisioner,
        "_verify_vllm_serving",
        new=AsyncMock(side_effect=verify),
    ):
        yield
