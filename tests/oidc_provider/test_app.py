"""Security and flow tests for the vendored toy OIDC provider."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

_PROVIDER_APP = Path(__file__).resolve().parents[2] / "oidc-provider" / "app.py"
_CLIENT_ID = "my-qiip-client"
_CLIENT_SECRET = "s3cr3t-test"
_REDIRECT_URI = "https://proxy.example.com/auth/callback"
_BASE = "https://oidc.localdomain/oidc"
_DOMAIN = "somelab.example.com"


@pytest.fixture(scope="module")
def provider(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Load the provider once with test-local files and credentials.

    The module reads its configuration from the environment at import time
    (users/key/db are file paths), so the fixture points those at a fresh
    temp dir before the import and restores the environment afterwards.
    """
    root = tmp_path_factory.mktemp("oidc-provider")
    users = root / "users.txt"
    users.write_text("alice:secret\nbob:pa55wörd\n")
    values = {
        "OIDC_KEY_FILE": str(root / "key.pem"),
        "OIDC_USERS_FILE": str(users),
        "OIDC_CLIENT_ID": _CLIENT_ID,
        "OIDC_CLIENT_SECRET": _CLIENT_SECRET,
        "OIDC_REDIRECT_URI": _REDIRECT_URI,
        "OIDC_DOMAIN": _DOMAIN,
        "OIDC_BASE_URL": _BASE,
    }
    for key, value in values.items():
        os.environ[key] = value
    try:
        spec = importlib.util.spec_from_file_location(
            "oidc_provider_app", _PROVIDER_APP
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key in values:
            os.environ.pop(key, None)
    return module


@pytest.fixture()
def client(provider: ModuleType) -> TestClient:
    return TestClient(provider.app)


def _valid_authorize_params(**overrides: str) -> dict[str, str]:
    params = {
        "response_type": "code",
        "client_id": _CLIENT_ID,
        "redirect_uri": _REDIRECT_URI,
        "state": "st4te",
        "scope": "openid email",
        "nonce": "n0nce",
    }
    params.update(overrides)
    return params


def test_discovery_publishes_user_email_domain(client: TestClient) -> None:
    doc = client.get("/oidc/.well-known/openid-configuration").json()

    assert doc["issuer"] == _BASE
    assert doc["user_email_domain"] == _DOMAIN


def test_authorize_renders_form_fields_escaped(client: TestClient) -> None:
    """state/scope/nonce are attacker-controlled: they must be HTML-escaped
    (review #231 reproduced an unescaped payload)."""
    response = client.get(
        "/oidc/authorize",
        params=_valid_authorize_params(
            state='"><script>alert(1)</script>',
            scope='"><img src=x onerror=alert(2)>',
        ),
    )

    assert response.status_code == 200
    # html.escape(quote=True) renders the quote as &quot; and the tags as
    # entities, so the payload round-trips as inert text only.
    assert "&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert 'value="<script>alert(1)</script>' not in response.text
    assert "<img src=x onerror=" not in response.text


def test_authorize_bad_client_id_is_400_not_redirect(client: TestClient) -> None:
    response = client.get(
        "/oidc/authorize",
        params=_valid_authorize_params(
            client_id="attacker", redirect_uri="https://evil.example/x"
        ),
    )

    assert response.status_code == 400
    assert "unauthorized_client" in response.text
    assert "Location" not in response.headers


def test_authorize_bad_redirect_uri_is_400_not_redirect(client: TestClient) -> None:
    response = client.get(
        "/oidc/authorize",
        params=_valid_authorize_params(redirect_uri="https://evil.example/x"),
    )

    assert response.status_code == 400
    assert "redirect_uri" in response.text
    assert "Location" not in response.headers


def test_authorize_bad_response_type_is_400_not_redirect(client: TestClient) -> None:
    response = client.get(
        "/oidc/authorize", params=_valid_authorize_params(response_type="token")
    )

    assert response.status_code == 400
    assert "unsupported_response_type" in response.text
    assert "Location" not in response.headers


def test_login_bad_credentials_escapes_fields(client: TestClient) -> None:
    response = client.post(
        "/oidc/login",
        data={
            "username": "alice",
            "password": "wrong",
            "client_id": _CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "state": '"><script>alert(1)</script>',
            "nonce": "",
            "scope": "openid",
        },
    )

    assert response.status_code == 401
    assert "&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert 'value="<script>alert(1)</script>' not in response.text


def test_login_unregistered_redirect_is_400_before_password_check(
    client: TestClient,
) -> None:
    response = client.post(
        "/oidc/login",
        data={
            "username": "alice",
            "password": "secret",  # correct, but the callback is not registered
            "client_id": _CLIENT_ID,
            "redirect_uri": "https://evil.example/x",
            "state": "",
            "nonce": "",
            "scope": "openid",
        },
    )

    assert response.status_code == 400
    assert "Location" not in response.headers


def _sign_in(client: TestClient, *, username: str, password: str) -> str:
    """POST /oidc/login with a registered pair; return the issued code."""
    response = client.post(
        "/oidc/login",
        data={
            "username": username,
            "password": password,
            "client_id": _CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "state": "st4te",
            "nonce": "n0nce",
            "scope": "openid",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"].startswith(f"{_REDIRECT_URI}?")
    from urllib.parse import parse_qs, urlsplit

    return parse_qs(urlsplit(response.headers["location"]).query)["code"][0]


def test_login_success_and_token_exchange(client: TestClient) -> None:
    code = _sign_in(client, username="alice", password="secret")

    response = client.post(
        "/oidc/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _REDIRECT_URI,
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "Bearer"
    assert payload["expires_in"] == 600


def test_token_requires_matching_redirect_uri(client: TestClient) -> None:
    code = _sign_in(client, username="alice", password="secret")

    missing = client.post(
        "/oidc/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        },
    )
    assert missing.status_code == 400
    assert missing.json()["error"] == "invalid_grant"

    wrong = client.post(
        "/oidc/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://evil.example/x",
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        },
    )
    assert wrong.status_code == 400
    assert wrong.json()["error"] == "invalid_grant"


def test_non_ascii_password_signs_in(client: TestClient) -> None:
    """compare_digest on str raises TypeError for non-ASCII (review #231);
    the provider must compare UTF-8 bytes."""
    code = _sign_in(client, username="bob", password="pa55wörd")

    response = client.post(
        "/oidc/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _REDIRECT_URI,
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        },
    )
    assert response.status_code == 200

    id_token = response.json()["id_token"]
    import base64
    import json

    payload = json.loads(
        base64.urlsafe_b64decode(id_token.split(".")[1] + "==").decode("utf-8")
    )
    assert payload["email"] == "bob@somelab.example.com"
    assert payload["email_verified"] is True
