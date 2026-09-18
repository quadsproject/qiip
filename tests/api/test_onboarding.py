"""Onboarding flow for normal users: single token, model scope, setup links."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_proxy.auth import store as store_module
from inference_proxy.auth.dependencies import get_auth_plugin
from inference_proxy.auth.store import AuthStore
from inference_proxy.config.settings import Settings
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.models.node import Node, NodeStatus
from inference_proxy.onboarding.harness import HARNESSES, get_harness
from inference_proxy.onboarding.script import render_setup_script
from tests.auth.conftest import FakeAuthPlugin

SECRET = "test-session-secret"
MODEL_A = "org/model-a"
MODEL_B = "org/model-b"


def _node(node_id: str, model: str, **extra: object) -> Node:
    return Node(
        node_id=node_id,
        endpoint="10.0.1.100:8000",
        status=NodeStatus.HEALTHY,
        model=model,
        managed=True,
        **extra,
    )


@pytest.fixture
def user_client(
    app: FastAPI, auth_store: AuthStore, test_registry: NodeRegistry
) -> TestClient:
    """A signed-in normal user with two healthy models on the fleet."""
    test_registry.add(_node("gpu01", MODEL_A))
    test_registry.add(_node("gpu02", MODEL_B))
    app.dependency_overrides[get_auth_plugin] = lambda: FakeAuthPlugin()
    client = TestClient(app)
    response = client.get(
        "/auth/callback?code=code&state=state", follow_redirects=False
    )
    assert response.headers["location"] == "/start"
    return client


def _mint(
    client: TestClient, models: list[str], name: str = "my laptop"
) -> dict[str, Any]:
    response = client.post("/onboarding/token", json={"name": name, "models": models})
    assert response.status_code == 201, response.text
    token: dict[str, Any] = response.json()["token"]
    return token


def _link(client: TestClient, harness: str, models: list[str]) -> dict[str, Any]:
    response = client.post(
        "/onboarding/setup-link", json={"harness": harness, "models": models}
    )
    assert response.status_code == 201, response.text
    link: dict[str, Any] = response.json()
    return link


def _legacy_token(store: AuthStore, name: str = "old") -> str:
    """Seed a pre-onboarding (random, unscoped) token; return its raw value."""
    user_id = store.list_users_with_stats()[0].id
    return store.create_token(user_id, name).token


def _script_token(script: str) -> str:
    start = script.index("qiip_")
    end = min(script.index(ch, start) for ch in '"\n' if ch in script[start:])
    return script[start:end]


def _assert_dead_link(client: TestClient, link_id: str) -> None:
    """A dead link serves the explain-and-exit-1 script, never a token."""
    response = client.get(f"/s/{link_id}")
    assert response.status_code == 200
    text = response.text
    assert "qiip_" not in text
    assert "expired" in text
    assert "exit 1" in text


class TestStartPage:
    def test_anonymous_gets_signin(self, app: FastAPI) -> None:
        response = TestClient(app).get("/start")
        assert response.status_code == 200
        assert "Google Auth" in response.text or "Local Admin" in response.text

    def test_local_admin_is_sent_to_dashboard(self, app: FastAPI) -> None:
        client = TestClient(app)
        client.post(
            "/auth/local-admin",
            json={"username": "test-admin", "password": "test-password"},
        )
        response = client.get("/start", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/dashboard"

    def test_state_for_new_user(self, user_client: TestClient) -> None:
        assert "start.js" in user_client.get("/start").text
        state = user_client.get("/onboarding/state").json()
        assert state["token"] is None
        assert state["models"] == [MODEL_A, MODEL_B]
        harnesses = {h["id"]: h for h in state["harnesses"]}
        assert harnesses["opencode"]["available"] is True
        assert harnesses["opencode"]["multi_model"] is True
        assert harnesses["codex"]["multi_model"] is False

    @pytest.mark.parametrize(
        "asset", ["css/dashboard.css", "css/start.css", "js/start.js", "img/qiip.svg"]
    )
    def test_assets_are_content_versioned(
        self, user_client: TestClient, asset: str
    ) -> None:
        import hashlib

        static = Path(__file__).resolve().parents[2] / "inference_proxy" / "static"
        digest = hashlib.sha256((static / asset).read_bytes()).hexdigest()[:12]
        assert f"/static/{asset}?v={digest}" in user_client.get("/start").text

    def test_state_requires_sign_in(self, app: FastAPI) -> None:
        assert TestClient(app).get("/onboarding/state").status_code == 401

    def test_unreachable_models_are_not_offered(
        self, user_client: TestClient, test_registry: NodeRegistry
    ) -> None:
        test_registry.add(_node("gpu03", "org/admin-model", admin_only=True))
        test_registry.add(_node("gpu04", "org/private", owner="bob@example.com"))
        test_registry.add(
            _node("gpu05", "org/draining").model_copy(
                update={"status": NodeStatus.DRAINING}
            )
        )

        assert user_client.get("/onboarding/state").json()["models"] == [
            MODEL_A,
            MODEL_B,
        ]


class TestSingleToken:
    def test_mint_replaces_every_previous_token(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        legacy = _legacy_token(auth_store)
        first = _mint(user_client, [MODEL_A])
        second = _mint(user_client, [MODEL_B], name="desktop")

        user_id = auth_store.list_users_with_stats()[0].id
        active = [t for t in auth_store.list_tokens(user_id) if not t.revoked]
        assert [t.id for t in active] == [second["id"]]
        assert auth_store.resolve_token(legacy) is None
        assert first["id"] != second["id"]
        state = user_client.get("/onboarding/state").json()
        assert state["token"]["name"] == "desktop"
        assert state["token"]["models"] == [MODEL_B]

    def test_mint_never_returns_the_raw_secret(self, user_client: TestClient) -> None:
        body = user_client.post(
            "/onboarding/token", json={"name": "x", "models": [MODEL_A]}
        ).text
        assert "qiip_" in body  # the display prefix only
        assert len(json.loads(body)["token"]["prefix"]) < 20
        assert '"token":"qiip_' not in body

    @pytest.mark.parametrize(
        ("payload", "status"),
        [
            ({"name": "x", "models": ["org/unknown"]}, 400),
            ({"name": "x", "models": []}, 422),
            ({"name": "   ", "models": [MODEL_A]}, 422),
            ({"name": "", "models": [MODEL_A]}, 422),
        ],
    )
    def test_mint_validation(
        self, user_client: TestClient, payload: dict[str, object], status: int
    ) -> None:
        response = user_client.post("/onboarding/token", json=payload)
        assert response.status_code == status

    def test_edit_models(self, user_client: TestClient) -> None:
        _mint(user_client, [MODEL_A])
        response = user_client.put(
            "/onboarding/token/models", json={"models": [MODEL_A, MODEL_B]}
        )
        assert response.json()["token"]["models"] == [MODEL_A, MODEL_B]
        assert (
            user_client.put(
                "/onboarding/token/models", json={"models": ["org/nope"]}
            ).status_code
            == 400
        )
        assert (
            user_client.put("/onboarding/token/models", json={"models": []}).status_code
            == 422
        )

    def test_edit_keeps_a_scoped_model_that_went_offline(
        self, user_client: TestClient, test_registry: NodeRegistry
    ) -> None:
        _mint(user_client, [MODEL_A, MODEL_B])
        test_registry.remove("gpu02")
        response = user_client.put(
            "/onboarding/token/models", json={"models": [MODEL_A, MODEL_B]}
        )
        assert response.status_code == 200

    def test_edit_without_token_is_404(self, user_client: TestClient) -> None:
        response = user_client.put(
            "/onboarding/token/models", json={"models": [MODEL_A]}
        )
        assert response.status_code == 404


class TestModelScope:
    def test_scoped_token_lists_only_its_models(self, user_client: TestClient) -> None:
        _mint(user_client, [MODEL_A])
        raw = _script_token(
            user_client.get(
                _link(user_client, "pi", [MODEL_A])["url"].replace(
                    "http://testserver", ""
                )
            ).text
        )
        listed = TestClient(user_client.app).get(
            "/v1/models", headers={"Authorization": f"Bearer {raw}"}
        )
        assert [m["id"] for m in listed.json()["data"]] == [MODEL_A]
        anonymous = TestClient(user_client.app).get("/v1/models")
        assert len(anonymous.json()["data"]) == 2

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
    def test_out_of_scope_model_is_refused(
        self, user_client: TestClient, path: str
    ) -> None:
        _mint(user_client, [MODEL_A])
        url = _link(user_client, "pi", [MODEL_A])["url"].replace(
            "http://testserver", ""
        )
        raw = _script_token(user_client.get(url).text)
        body: dict[str, object] = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if "chat" in path
            else {"prompt": "hi"}
        )
        body["model"] = MODEL_B
        response = TestClient(user_client.app).post(
            path, json=body, headers={"Authorization": f"Bearer {raw}"}
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "model_not_permitted"

    def test_unscoped_legacy_token_is_not_restricted(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        legacy = _legacy_token(auth_store)
        listed = TestClient(user_client.app).get(
            "/v1/models", headers={"Authorization": f"Bearer {legacy}"}
        )
        assert len(listed.json()["data"]) == 2


class TestSetupLinks:
    def test_link_serves_a_script_with_a_working_token(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        token = _mint(user_client, [MODEL_A, MODEL_B])
        link = _link(user_client, "opencode", [MODEL_A, MODEL_B])

        # No -f: an expired link must still pipe its explanation into bash.
        assert link["command"] == f"curl -sSL {link['url']} | bash"
        link_id = link["url"].rsplit("/", 1)[1]
        assert len(link_id) == 12
        # Fetched without any session: the link id is the credential.
        response = TestClient(user_client.app).get(f"/s/{link_id}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        resolved = auth_store.resolve_token(_script_token(response.text))
        assert resolved is not None
        assert resolved.token.id == token["id"]

    def test_link_and_token_are_never_stored_in_the_clear(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        _mint(user_client, [MODEL_A])
        link = _link(user_client, "pi", [MODEL_A])
        link_id = link["url"].rsplit("/", 1)[1]
        raw = _script_token(user_client.get(f"/s/{link_id}").text)
        dump = "\n".join(auth_store._conn.iterdump())
        assert link_id not in dump
        assert raw not in dump

    def test_export_widens_the_token_scope(self, user_client: TestClient) -> None:
        _mint(user_client, [MODEL_A])
        _link(user_client, "pi", [MODEL_B])
        state = user_client.get("/onboarding/state").json()
        assert state["token"]["models"] == [MODEL_A, MODEL_B]

    def test_expired_link_serves_nothing_useful(
        self,
        user_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mint(user_client, [MODEL_A])
        link_id = _link(user_client, "pi", [MODEL_A])["url"].rsplit("/", 1)[1]
        real_now = store_module._utcnow
        monkeypatch.setattr(
            store_module, "_utcnow", lambda: real_now() + timedelta(minutes=16)
        )
        _assert_dead_link(user_client, link_id)

    def test_replacing_the_token_kills_old_links(self, user_client: TestClient) -> None:
        _mint(user_client, [MODEL_A])
        link_id = _link(user_client, "pi", [MODEL_A])["url"].rsplit("/", 1)[1]
        _mint(user_client, [MODEL_A], name="new")
        _assert_dead_link(user_client, link_id)

    @pytest.mark.parametrize("link_id", ["aaaaaaaaaaaa", "A" * 12, "a" * 40])
    def test_unknown_link_serves_the_failing_script(
        self, app: FastAPI, link_id: str
    ) -> None:
        _assert_dead_link(TestClient(app), link_id)

    def test_dead_link_fails_the_advertised_pipeline(self, app: FastAPI) -> None:
        """A 404 would reach bash as empty stdin and exit 0 with nothing
        configured; the dead-link body must make the pipeline fail loudly."""
        body = TestClient(app).get("/s/aaaaaaaaaaaa").text
        result = subprocess.run(["bash"], input=body, text=True, capture_output=True)
        assert result.returncode == 1
        assert "expired" in result.stderr

    def test_link_id_is_not_logged(
        self, user_client: TestClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _mint(user_client, [MODEL_A])
        link_id = _link(user_client, "pi", [MODEL_A])["url"].rsplit("/", 1)[1]
        capsys.readouterr()
        response = user_client.get(f"/s/{link_id}")
        assert response.headers["referrer-policy"] == "no-referrer"
        logged = capsys.readouterr()
        assert "/s/[redacted]" in logged.out + logged.err
        assert link_id not in logged.out + logged.err

    def test_single_model_harness_rejects_two_models(
        self, user_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mint(user_client, [MODEL_A, MODEL_B])
        monkeypatch.setattr(
            "inference_proxy.api.onboarding._served_routes",
            lambda: {"/v1/responses", "/v1/chat/completions"},
        )
        response = user_client.post(
            "/onboarding/setup-link",
            json={"harness": "codex", "models": [MODEL_A, MODEL_B]},
        )
        assert response.status_code == 400
        assert "exactly one model" in response.json()["detail"]

    def test_harness_needing_an_unserved_route_is_refused(
        self, user_client: TestClient
    ) -> None:
        _mint(user_client, [MODEL_A])
        for harness in ("claude", "codex", "nope"):
            response = user_client.post(
                "/onboarding/setup-link",
                json={"harness": harness, "models": [MODEL_A]},
            )
            assert response.status_code == 400

    def test_legacy_token_cannot_be_exported(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        _legacy_token(auth_store)
        state = user_client.get("/onboarding/state").json()
        assert state["token"]["exportable"] is False
        response = user_client.post(
            "/onboarding/setup-link", json={"harness": "pi", "models": [MODEL_A]}
        )
        assert response.status_code == 409

    def test_rotated_session_secret_blocks_export(
        self, tmp_path: Path, auth_store: AuthStore
    ) -> None:
        user = auth_store.upsert_google_user(
            google_sub="s", email="a@example.com", name="A", picture=""
        )
        created = auth_store.create_personal_token(user.id, "x", [MODEL_A], SECRET)
        assert auth_store.reveal_personal_token(created.id, SECRET) == created.token
        assert auth_store.reveal_personal_token(created.id, "rotated") is None


class TestSetupScripts:
    @pytest.mark.parametrize("harness_id", [h.id for h in HARNESSES])
    def test_script_writes_private_config_and_is_rerunnable(
        self, harness_id: str, tmp_path: Path
    ) -> None:
        harness = get_harness(harness_id)
        assert harness is not None
        models = [MODEL_A, MODEL_B] if harness.multi_model else [MODEL_A]
        script = render_setup_script(
            harness, base_url="https://qiip.example", token="qiip_secret", models=models
        )
        env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
        for _ in range(2):
            result = subprocess.run(
                ["bash"], input=script, text=True, env=env, capture_output=True
            )
            assert result.returncode == 0, result.stderr
        target = tmp_path / harness.config_path
        text = target.read_text()
        assert "qiip_secret" in text
        assert MODEL_A in text
        assert "https://qiip.example" in text
        assert target.stat().st_mode & 0o777 == 0o600
        assert text.count("qiip_secret") == 1
        leftovers = [p.name for p in target.parent.iterdir() if ".qiip." in p.name]
        assert leftovers == []

    def test_json_merge_keeps_other_settings(self, tmp_path: Path) -> None:
        harness = get_harness("claude")
        assert harness is not None
        target = tmp_path / harness.config_path
        target.parent.mkdir(parents=True)
        target.write_text('{"permissions": {"allow": ["Bash"]}, "env": {"FOO": "1"}}')
        script = render_setup_script(
            harness, base_url="https://q", token="qiip_t", models=[MODEL_A]
        )
        subprocess.run(
            ["bash"],
            input=script,
            text=True,
            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
            check=True,
            capture_output=True,
        )
        merged = json.loads(target.read_text())
        assert merged["permissions"] == {"allow": ["Bash"]}
        assert merged["env"]["FOO"] == "1"
        assert merged["env"]["ANTHROPIC_MODEL"] == MODEL_A
        assert (tmp_path / (harness.config_path + ".bak")).exists()

    def test_codex_merge_keeps_user_config_and_replaces_model(
        self, tmp_path: Path
    ) -> None:
        import tomllib

        harness = get_harness("codex")
        assert harness is not None
        target = tmp_path / harness.config_path
        target.parent.mkdir(parents=True)
        target.write_text('model = "old"\napproval_policy = "never"\n\n[tui]\nx = 1\n')
        script = render_setup_script(
            harness, base_url="https://q", token="qiip_t", models=[MODEL_A]
        )
        env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
        for _ in range(2):
            subprocess.run(
                ["bash"],
                input=script,
                text=True,
                env=env,
                check=True,
                capture_output=True,
            )
        parsed = tomllib.loads(target.read_text())
        assert parsed["model"] == MODEL_A
        assert parsed["model_provider"] == "qiip"
        assert parsed["approval_policy"] == "never"
        assert parsed["tui"] == {"x": 1}
        assert parsed["model_providers"]["qiip"]["base_url"] == "https://q/v1"

    def test_hostile_model_id_cannot_inject_shell(self, tmp_path: Path) -> None:
        harness = get_harness("pi")
        assert harness is not None
        evil = "$(touch pwned)`touch pwned`'; touch pwned; '"
        script = render_setup_script(
            harness, base_url="https://q", token="qiip_t", models=[evil]
        )
        subprocess.run(
            ["bash"],
            input=script,
            text=True,
            cwd=tmp_path,
            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
            check=True,
            capture_output=True,
        )
        assert not (tmp_path / "pwned").exists()
        written = json.loads((tmp_path / harness.config_path).read_text())
        assert written["providers"]["qiip"]["models"] == [{"id": evil}]


class TestReviewRegressions:
    """Findings from the independent reviews of the onboarding flow."""

    @pytest.mark.parametrize("name", ["bypass", "agent-config"])
    def test_normal_user_cannot_mint_on_the_legacy_surface(
        self, user_client: TestClient, auth_store: AuthStore, name: str
    ) -> None:
        token = _mint(user_client, [MODEL_A])
        response = user_client.post("/profile/tokens", json={"name": name})
        assert response.status_code == 403
        user_id = auth_store.list_users_with_stats()[0].id
        active = [t for t in auth_store.list_tokens(user_id) if not t.revoked]
        assert [t.id for t in active] == [token["id"]]

    def test_google_admin_is_kept_out_of_the_flow(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        user_id = auth_store.list_users_with_stats()[0].id
        pinned = auth_store.create_token(user_id, "pinned", ["gpu01"])
        auth_store.set_user_admin(user_id, True)

        page = user_client.get("/start", follow_redirects=False)
        assert page.status_code == 302
        assert page.headers["location"] == "/dashboard"
        for method, path, body in [
            ("GET", "/onboarding/state", None),
            ("POST", "/onboarding/token", {"name": "x", "models": [MODEL_A]}),
            ("PUT", "/onboarding/token/models", {"models": [MODEL_A]}),
            ("POST", "/onboarding/setup-link", {"harness": "pi", "models": [MODEL_A]}),
        ]:
            assert user_client.request(method, path, json=body).status_code == 403
        assert auth_store.resolve_token(pinned.token) is not None

    def test_deleted_user_cookie_gets_signin_not_the_shell(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        auth_store._conn.execute("DELETE FROM users")
        auth_store._conn.commit()
        page = user_client.get("/start")
        assert "start.js" not in page.text

    def test_legacy_token_scope_cannot_be_narrowed(
        self, user_client: TestClient, auth_store: AuthStore
    ) -> None:
        raw = _legacy_token(auth_store)
        response = user_client.put(
            "/onboarding/token/models", json={"models": [MODEL_A]}
        )
        assert response.status_code == 409
        resolved = auth_store.resolve_token(raw)
        assert resolved is not None
        assert resolved.token.model_scope is None

    def test_in_scope_model_passes_the_scope_check(
        self, user_client: TestClient
    ) -> None:
        _mint(user_client, [MODEL_A])
        url = _link(user_client, "pi", [MODEL_A])["url"].replace(
            "http://testserver", ""
        )
        raw = _script_token(user_client.get(url).text)
        response = TestClient(user_client.app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"model": MODEL_A, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {raw}"},
        )
        # The fake backend is unreachable; what matters is that it got past scope.
        assert response.status_code != 403

    def test_one_active_personal_token_is_a_database_invariant(
        self, auth_store: AuthStore
    ) -> None:
        import sqlite3

        user = auth_store.upsert_google_user(
            google_sub="s", email="a@example.com", name="A", picture=""
        )
        auth_store.create_personal_token(user.id, "x", [MODEL_A], SECRET)
        with pytest.raises(sqlite3.IntegrityError):
            auth_store._conn.execute(
                "INSERT INTO tokens (user_id, name, token_hash, prefix, created_at,"
                " purpose) VALUES (?, 'dup', 'h', 'p', 'now', 'personal')",
                (user.id,),
            )
        auth_store._conn.rollback()

    def test_pre_onboarding_database_migrates(self, tmp_path: Path) -> None:
        import sqlite3

        path = tmp_path / "legacy.db"
        first = AuthStore(path)
        user = first.upsert_google_user(
            google_sub="s", email="a@example.com", name="A", picture=""
        )
        legacy = first.create_token(user.id, "old")
        first.close()
        conn = sqlite3.connect(path)
        conn.execute("DROP INDEX idx_tokens_user_active_personal")
        conn.execute("DROP TABLE setup_links")
        conn.execute("ALTER TABLE tokens DROP COLUMN model_scope")
        conn.execute("ALTER TABLE tokens DROP COLUMN derive_nonce")
        conn.commit()
        conn.close()

        store = AuthStore(path)
        try:
            assert store.resolve_token(legacy.token) is not None
            created = store.create_personal_token(user.id, "new", [MODEL_A], SECRET)
            assert store.reveal_personal_token(created.id, SECRET) == created.token
            link_id, _ = store.create_setup_link(
                user.id, created.id, "pi", [MODEL_A], 60
            )
            assert store.resolve_setup_link(link_id) is not None
        finally:
            store.close()

    def test_rerun_keeps_the_original_backup(self, tmp_path: Path) -> None:
        harness = get_harness("pi")
        assert harness is not None
        target = tmp_path / harness.config_path
        target.parent.mkdir(parents=True)
        target.write_text('{"mine": true}')
        script = render_setup_script(
            harness, base_url="https://q", token="qiip_t", models=[MODEL_A]
        )
        env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
        for _ in range(2):
            subprocess.run(
                ["bash"],
                input=script,
                text=True,
                env=env,
                check=True,
                capture_output=True,
            )
        backup = tmp_path / (harness.config_path + ".bak")
        assert json.loads(backup.read_text()) == {"mine": True}

    def test_model_named_like_the_heredoc_delimiter_is_not_offered(
        self, user_client: TestClient, test_registry: NodeRegistry
    ) -> None:
        test_registry.add(_node("gpu09", "QIIP_CONFIG_EOF"))
        assert (
            "QIIP_CONFIG_EOF"
            not in user_client.get("/onboarding/state").json()["models"]
        )
        harness = get_harness("pi")
        assert harness is not None
        with pytest.raises(ValueError):
            render_setup_script(
                harness, base_url="https://q", token="t", models=["QIIP_CONFIG_EOF"]
            )


class TestPublicBaseUrl:
    """The curl line and the written configs must not downgrade to http."""

    @staticmethod
    def _settings(
        test_settings: Settings, redirect_uri: str | None, hosts: list[str]
    ) -> Settings:
        oauth = test_settings.oauth.model_copy(
            update={"redirect_uri": redirect_uri, "allowed_redirect_hosts": hosts}
        )
        return test_settings.model_copy(update={"oauth": oauth})

    @pytest.mark.parametrize(
        ("redirect_uri", "hosts", "request_url", "expected"),
        [
            (None, [], "http://qiip.example:5000/x", "http://qiip.example:5000"),
            # Untrusted proxy hop degraded the scheme: configured https wins.
            (
                "https://qiip.example/auth/callback",
                [],
                "http://qiip.example/x",
                "https://qiip.example",
            ),
            # Allowlisted alternate name keeps its own host.
            (
                "https://qiip.example/auth/callback",
                ["alt.example"],
                "http://alt.example/x",
                "https://alt.example",
            ),
            # A never-trusted Host header cannot steer the command.
            (
                "https://qiip.example/auth/callback",
                [],
                "http://evil.example/x",
                "https://qiip.example",
            ),
            # Plain-http dev setups are left alone.
            (
                "http://localhost:5000/auth/callback",
                [],
                "http://localhost:5000/x",
                "http://localhost:5000",
            ),
        ],
    )
    def test_public_base_url(
        self,
        test_settings: Settings,
        redirect_uri: str | None,
        hosts: list[str],
        request_url: str,
        expected: str,
    ) -> None:
        from urllib.parse import urlsplit

        from starlette.requests import Request

        from inference_proxy.api.onboarding import public_base_url

        parts = urlsplit(request_url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "scheme": parts.scheme,
                "path": parts.path,
                "query_string": b"",
                "headers": [(b"host", parts.netloc.encode())],
                "server": (parts.hostname, port),
            }
        )
        settings = self._settings(test_settings, redirect_uri, hosts)
        assert public_base_url(request, settings) == expected
