"""Unit tests for the settings-driven ASGI serve launcher."""

from __future__ import annotations

import pytest

from inference_proxy.config.dependencies import get_settings
from inference_proxy.serve import uvicorn_kwargs


def test_uvicorn_kwargs_use_server_defaults() -> None:
    get_settings.cache_clear()

    kwargs = uvicorn_kwargs()

    assert kwargs["app"] == "inference_proxy.main:create_app"
    assert kwargs["factory"] is True
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 5000
    assert kwargs["workers"] == 1
    assert kwargs["limit_concurrency"] == 150
    assert kwargs["log_level"] == "info"
    assert kwargs["timeout_graceful_shutdown"] == 30
    # The app's own request log redacts setup-link ids; uvicorn's would not.
    assert kwargs["access_log"] is False
    # A single worker runs without a supervisor, so the request limit must
    # not be passed (it would terminate the process instead of recycling).
    assert "limit_max_requests" not in kwargs
    assert "limit_max_requests_jitter" not in kwargs


def test_uvicorn_kwargs_follow_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("INFERENCE_PROXY_SERVER__WORKERS", "4")
    monkeypatch.setenv("INFERENCE_PROXY_SERVER__PORT", "5055")
    monkeypatch.setenv("INFERENCE_PROXY_SERVER__MAX_REQUESTS", "2500")

    kwargs = uvicorn_kwargs()

    assert kwargs["workers"] == 4
    assert kwargs["port"] == 5055
    assert kwargs["limit_max_requests"] == 2500
    assert kwargs["limit_max_requests_jitter"] == 500
