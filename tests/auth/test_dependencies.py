"""Unit tests for auth FastAPI dependencies (store/plugin resolution)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException

from inference_proxy.auth.dependencies import get_auth_plugin, get_auth_store
from inference_proxy.auth.store import AuthStore

from .conftest import FakeAuthPlugin


def _app_with_state(
    **state_kwargs: object,
) -> tuple[FastAPI, MagicMock]:
    app = FastAPI()
    for key, value in state_kwargs.items():
        setattr(app.state, key, value)
    request = MagicMock()
    request.app = app
    return app, request


class TestGetAuthStore:
    def test_missing_store_raises_503(self) -> None:
        _app, request = _app_with_state()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_store(request)

        assert exc_info.value.status_code == 503

    def test_present_store_returned(self, tmp_path: Path) -> None:
        store = AuthStore(tmp_path / "auth.db")
        _app, request = _app_with_state(auth_store=store)

        assert get_auth_store(request) is store
        store.close()


class TestGetAuthPlugin:
    def test_missing_plugin_raises_404(self) -> None:
        _app, request = _app_with_state()

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request)

        assert exc_info.value.status_code == 404

    def test_unconfigured_present_plugin_raises_404(self) -> None:
        plugin = FakeAuthPlugin(error="access_denied")
        _app, request = _app_with_state(auth_plugin=plugin)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_plugin(request)

        assert exc_info.value.status_code == 404

    def test_present_plugin_returned(self) -> None:
        plugin = FakeAuthPlugin()
        _app, request = _app_with_state(auth_plugin=plugin)

        assert get_auth_plugin(request) is plugin
