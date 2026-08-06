"""Static asset generation tests for browser-safe template deployment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_STATIC_DIR = Path(__file__).resolve().parents[2] / "inference_proxy" / "static"


def _versioned_path(path: str) -> str:
    digest = hashlib.sha256((_STATIC_DIR / path).read_bytes()).hexdigest()[:12]
    return f"/static/{path}?v={digest}"


@pytest.mark.parametrize(
    ("route", "assets"),
    [
        (
            "/dashboard",
            (
                "css/dashboard.css",
                "js/config_download.js",
                "js/setup_selection.js",
                "js/dashboard.js",
            ),
        ),
        (
            "/dashboard/nodes/test-node",
            (
                "css/dashboard.css",
                "js/config_download.js",
                "js/setup_selection.js",
                "js/llamacpp_relaunch.js",
                "js/node_detail.js",
            ),
        ),
        (
            "/chat",
            (
                "css/dashboard.css",
                "css/chat.css",
                "vendor/marked-18.0.7/marked.umd.js",
                "vendor/dompurify-3.4.12/purify.min.js",
                "js/chat.js",
            ),
        ),
    ],
)
def test_templates_bind_each_asset_url_to_its_contents(
    client: TestClient,
    route: str,
    assets: tuple[str, ...],
) -> None:
    """A fresh HTML shell cannot reuse a cached script from an older deploy."""
    response = client.get(route)

    assert response.status_code == 200
    for asset in assets:
        assert _versioned_path(asset) in response.text
