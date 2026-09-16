"""Unit tests for the modular YAML conf-dir settings source."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from inference_proxy.config.settings import Settings

CONF_DIR_ENV = "INFERENCE_PROXY_CONF_DIR"
_CONF_FILES = ("qiip.yml", "auth.yml", "plugins.yml")


def _conf_dir(tmp_path: Path, **files: str) -> Path:
    for name, content in files.items():
        (tmp_path / f"{name}.yml").write_text(content, encoding="utf-8")
    return tmp_path


def _expected_settings() -> set[str]:
    expected: set[str] = set()
    for group_name, group_field in Settings.model_fields.items():
        model = group_field.annotation
        assert isinstance(model, type)
        assert issubclass(model, BaseModel)
        expected.update(
            f"{group_name}.{field_name}" for field_name in model.model_fields
        )
    return expected


class TestYamlConfDirLoading:
    def test_loads_modular_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(
            tmp_path,
            qiip="""
etcd:
  endpoints:
    - http://etcd.example.com:2379
  node_prefix: /prod-nodes/
routing:
  max_attempts: 5
provisioning:
  vllm_port: 8000
""",
            auth="""
auth:
  session_cookie: qiip_prod
  session_ttl_seconds: 3600
""",
            plugins="""
plugins:
  external_dir: /etc/qiip/plugins
  disabled:
    - auth.google
  config:
    myplugin:
      api_key: secretkey
""",
        )
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))

        settings = Settings(_env_file=None)

        assert settings.etcd.endpoints == ["http://etcd.example.com:2379"]
        assert settings.etcd.node_prefix == "/prod-nodes/"
        assert settings.routing.max_attempts == 5
        assert settings.auth.session_cookie == "qiip_prod"
        assert settings.auth.session_ttl_seconds == 3600
        assert settings.plugins.external_dir == Path("/etc/qiip/plugins")
        assert settings.plugins.disabled == ["auth.google"]
        assert settings.plugins.config == {"myplugin": {"api_key": "secretkey"}}

    def test_env_overrides_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, qiip="routing:\n  max_attempts: 5\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__MAX_ATTEMPTS", "9")

        settings = Settings(_env_file=None)

        assert settings.routing.max_attempts == 9

    def test_yaml_overrides_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, qiip="etcd:\n  node_prefix: /yaml-nodes/\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))
        env_file = tmp_path / ".env"
        env_file.write_text("INFERENCE_PROXY_ETCD__NODE_PREFIX=/dotenv-nodes/\n")

        settings = Settings(_env_file=env_file)

        assert settings.etcd.node_prefix == "/yaml-nodes/"

    def test_yaml_null_leaves_dotenv_secret_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, auth="auth:\n  session_secret: null\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))
        env_file = tmp_path / ".env"
        env_file.write_text("INFERENCE_PROXY_AUTH__SESSION_SECRET=dotenv-secret\n")

        settings = Settings(_env_file=env_file)

        assert settings.auth.session_secret is not None
        assert settings.auth.session_secret.get_secret_value() == "dotenv-secret"

    def test_yaml_null_in_nested_config_leaves_dotenv_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(
            tmp_path,
            plugins="plugins:\n  config:\n    custom:\n      api_key: null\n",
        )
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))
        env_file = tmp_path / ".env"
        env_file.write_text(
            'INFERENCE_PROXY_PLUGINS__CONFIG={"custom":{"api_key":"test-secret"}}\n'
        )

        settings = Settings(_env_file=env_file)

        assert settings.plugins.config == {"custom": {"api_key": "test-secret"}}

    def test_null_section_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, qiip="quads:\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))

        settings = Settings(_env_file=None)

        assert settings.quads.base_url is None
        assert settings.quads.timeout == 10.0

    def test_missing_conf_dir_falls_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CONF_DIR_ENV, "/nonexistent-qiip-conf")

        settings = Settings(_env_file=None)

        assert settings.etcd.endpoints == ["http://localhost:2379"]
        assert settings.routing.max_attempts == 3

    def test_yaml_values_are_validated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(
            tmp_path,
            qiip="etcd:\n  node_lease_ttl: 50\nresilience:\n  health_check_interval: 30\n",
        )
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))

        with pytest.raises(ValidationError, match="node_lease_ttl"):
            Settings(_env_file=None)

    def test_unknown_top_level_key_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, qiip="bogus_top: 1\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))

        with pytest.raises(ValidationError, match="bogus_top"):
            Settings(_env_file=None)

    def test_non_mapping_yaml_file_names_the_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        conf = _conf_dir(tmp_path, qiip="- not\n- a mapping\n")
        monkeypatch.setenv(CONF_DIR_ENV, str(conf))

        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            Settings(_env_file=None)


def test_empty_conf_dir_env_does_not_scan_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from inference_proxy.config.settings import yaml_conf_files

    (tmp_path / "qiip.yml").write_text("etcd: {}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CONF_DIR_ENV, "")

    assert yaml_conf_files() == []


def _duplicate_keys(text: str) -> list[str]:
    """Return key names repeated within one mapping (safe_load last-wins)."""
    duplicates: list[str] = []

    def walk(node: yaml.Node) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, _value_node in node.value:
                key = key_node.value
                if key in seen and key not in duplicates:
                    duplicates.append(key)
                seen.add(key)
                walk(_value_node)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                walk(child)

    root = yaml.compose(text)
    if root is not None:
        walk(root)
    return duplicates


def _collect_yaml_pairs(conf_dir: Path) -> list[str]:
    pairs: list[str] = []
    for name in sorted(conf_dir.iterdir()):
        if not name.name.endswith((".yml.example", ".yaml.example")):
            continue
        text = name.read_text(encoding="utf-8")
        assert not _duplicate_keys(text), f"duplicate keys in {name}"
        data = yaml.safe_load(text)
        assert isinstance(data, dict), f"{name} must be a mapping"
        for section, values in data.items():
            assert isinstance(values, dict), f"{section} in {name} must be a mapping"
            pairs.extend(f"{section}.{field}" for field in values)
    return pairs


def test_yaml_examples_cover_every_application_setting_exactly_once() -> None:
    """Keep the shipped YAML inventory synchronized with Settings."""
    expected = _expected_settings()
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    pairs = _collect_yaml_pairs(conf_dir)

    assert len(pairs) == len(set(pairs)), "duplicate YAML setting across files"
    assert set(pairs) == expected
    example_names = {
        path.name
        for path in conf_dir.iterdir()
        if path.name.endswith((".yml.example", ".yaml.example"))
    }
    assert example_names == {f"{name}.example" for name in _CONF_FILES}
