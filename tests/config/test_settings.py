"""Unit tests for configuration settings loading and env var overrides."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic_settings import BaseSettings

from inference_proxy.config.settings import (
    DashboardSettings,
    EtcdSettings,
    GatewaySettings,
    HuggingFaceSettings,
    ProvisioningSettings,
    QUADSSettings,
    RedfishSettings,
    RoutingSettings,
    Settings,
    SSHSettings,
)


class TestDefaultGatewaySettings:
    def test_default_gateway_settings(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.gateway.host == "0.0.0.0"
        assert settings.gateway.port == 8080


class TestDefaultEtcdSettings:
    def test_default_etcd_settings(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.etcd.endpoints == ["http://localhost:2379"]
        assert settings.etcd.node_prefix == "/nodes/"


class TestDefaultRoutingSettings:
    def test_default_routing_settings(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.routing.strategy == "least_connections"
        assert settings.routing.health_check_interval == 30
        assert settings.routing.max_retries == 3
        assert settings.routing.timeout == 30
        assert settings.routing.allowed_endpoint_hosts == ["localhost"]
        assert settings.routing.allowed_endpoint_networks == [
            "127.0.0.0/8",
            "::1/128",
        ]
        assert settings.routing.allowed_endpoint_ports == [8000]

    def test_max_retries_rejects_zero(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RoutingSettings(max_retries=0)

        assert caught.value.errors()[0]["loc"] == ("max_retries",)

    def test_provisioning_port_must_be_allowed(self) -> None:
        with pytest.raises(
            ValidationError,
            match="provisioning.vllm_port must be included",
        ):
            Settings(
                _env_file=None,
                provisioning=ProvisioningSettings(vllm_port=9000),
            )


class TestEnvVarOverrideGatewayPort:
    def test_env_var_override_gateway_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_GATEWAY__PORT", "9090")
        settings = Settings(_env_file=None)
        assert settings.gateway.port == 9090


class TestEnvVarOverrideEtcdPrefix:
    def test_env_var_override_etcd_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ETCD__NODE_PREFIX", "/test-nodes/")
        settings = Settings(_env_file=None)
        assert settings.etcd.node_prefix == "/test-nodes/"


class TestEnvVarOverrideRoutingStrategy:
    def test_env_var_override_routing_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__STRATEGY", "round_robin")
        settings = Settings(_env_file=None)
        assert settings.routing.strategy == "round_robin"

    def test_endpoint_allowlist_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_HOSTS",
            '["*.lab.example.com"]',
        )
        monkeypatch.setenv(
            "INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_NETWORKS",
            '["10.0.1.0/24"]',
        )
        monkeypatch.setenv(
            "INFERENCE_PROXY_ROUTING__ALLOWED_ENDPOINT_PORTS",
            "[8000,8443]",
        )

        settings = Settings(_env_file=None)

        assert settings.routing.allowed_endpoint_hosts == ["*.lab.example.com"]
        assert settings.routing.allowed_endpoint_networks == ["10.0.1.0/24"]
        assert settings.routing.allowed_endpoint_ports == [8000, 8443]


class TestDefaultDashboardSettings:
    def test_default_dashboard_poll_interval(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.dashboard.poll_interval == 10


class TestEnvVarOverrideDashboardPollInterval:
    def test_env_var_override_dashboard_poll_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_DASHBOARD__POLL_INTERVAL", "30")
        settings = Settings(_env_file=None)
        assert settings.dashboard.poll_interval == 30


class TestSubModelsAreNotBaseSettings:
    def test_sub_models_are_not_base_settings(self) -> None:
        assert not issubclass(GatewaySettings, BaseSettings)
        assert not issubclass(EtcdSettings, BaseSettings)
        assert not issubclass(RoutingSettings, BaseSettings)
        assert not issubclass(DashboardSettings, BaseSettings)
        assert issubclass(GatewaySettings, BaseModel)
        assert issubclass(EtcdSettings, BaseModel)
        assert issubclass(RoutingSettings, BaseModel)
        assert issubclass(DashboardSettings, BaseModel)


class TestEtcdSettingsEmptyEndpointsRejected:
    """EtcdSettings rejects an empty endpoints list with a validation error."""

    def test_empty_endpoints_raises_validation_error(self) -> None:
        with pytest.raises(
            ValidationError, match="At least one etcd endpoint must be configured"
        ):
            EtcdSettings(endpoints=[])


class TestDefaultSSHSettings:
    """D-01, D-02, D-04: SSHSettings defaults."""

    def test_default_key_path(self) -> None:
        from pathlib import Path

        settings = Settings(_env_file=None)
        assert settings.ssh.key_path == Path("~/.ssh/id_rsa").expanduser()

    def test_default_username(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.ssh.username == "root"

    def test_default_connect_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.ssh.connect_timeout == 10


class TestDefaultProvisioningSettings:
    """D-09, D-17: ProvisioningSettings defaults."""

    def test_default_health_poll_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.provisioning.health_poll_timeout == 600

    def test_default_health_poll_interval(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.provisioning.health_poll_interval == 10

    def test_default_vllm_port(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.provisioning.vllm_port == 8000


class TestEnvVarOverrideSSHUsername:
    def test_env_var_override_ssh_username(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_SSH__USERNAME", "deploy")
        settings = Settings(_env_file=None)
        assert settings.ssh.username == "deploy"


class TestEnvVarOverrideProvisioningTimeout:
    def test_env_var_override_provisioning_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_PROVISIONING__HEALTH_POLL_TIMEOUT", "300")
        settings = Settings(_env_file=None)
        assert settings.provisioning.health_poll_timeout == 300


class TestSSHAndProvisioningAreNotBaseSettings:
    def test_ssh_settings_is_base_model_not_base_settings(self) -> None:
        assert not issubclass(SSHSettings, BaseSettings)
        assert issubclass(SSHSettings, BaseModel)

    def test_provisioning_settings_is_base_model_not_base_settings(self) -> None:
        assert not issubclass(ProvisioningSettings, BaseSettings)
        assert issubclass(ProvisioningSettings, BaseModel)


class TestDefaultQUADSSettings:
    def test_base_url_is_none(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.quads.base_url is None

    def test_timeout_is_10(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.quads.timeout == 10.0


class TestEnvVarOverrideQUADSBaseUrl:
    def test_env_var_override_quads_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_QUADS__BASE_URL", "http://quads.example.com"
        )
        settings = Settings(_env_file=None)
        assert settings.quads.base_url == "http://quads.example.com"


class TestEnvVarOverrideQUADSTimeout:
    def test_env_var_override_quads_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_QUADS__TIMEOUT", "5.0")
        settings = Settings(_env_file=None)
        assert settings.quads.timeout == 5.0


class TestQUADSSettingsIsNotBaseSettings:
    def test_quads_settings_is_base_model_not_base_settings(self) -> None:
        assert not issubclass(QUADSSettings, BaseSettings)
        assert issubclass(QUADSSettings, BaseModel)


class TestDefaultRedfishSettings:
    def test_bmc_username_is_none(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_username is None

    def test_bmc_password_is_none(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_password is None

    def test_bmc_host_template(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_host_template == "mgmt-{hostname}"

    def test_system_id(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.system_id == "1"

    def test_connect_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.connect_timeout == 10.0

    def test_read_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.read_timeout == 60.0

    def test_power_poll_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.power_poll_timeout == 60.0

    def test_power_poll_interval(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.power_poll_interval == 5.0

    def test_verify_ssl_is_false(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.redfish.verify_ssl is False


class TestEnvVarOverrideRedfishBmcUsername:
    def test_env_var_override_redfish_bmc_username(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_REDFISH__BMC_USERNAME", "admin")
        monkeypatch.setenv("INFERENCE_PROXY_REDFISH__BMC_PASSWORD", "secret123")
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_username == "admin"


class TestEnvVarOverrideRedfishBmcPassword:
    def test_env_var_override_redfish_bmc_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_REDFISH__BMC_USERNAME", "admin")
        monkeypatch.setenv("INFERENCE_PROXY_REDFISH__BMC_PASSWORD", "secret123")
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_password is not None
        assert settings.redfish.bmc_password.get_secret_value() == "secret123"


class TestEnvVarOverrideRedfishBmcHostTemplate:
    def test_env_var_override_redfish_bmc_host_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_REDFISH__BMC_HOST_TEMPLATE", "bmc-{hostname}.mgmt"
        )
        settings = Settings(_env_file=None)
        assert settings.redfish.bmc_host_template == "bmc-{hostname}.mgmt"


class TestRedfishSettingsSecretStr:
    def test_repr_does_not_contain_password(self) -> None:
        rs = RedfishSettings(
            bmc_username="admin",
            bmc_password=SecretStr("hunter2"),
        )
        assert "hunter2" not in repr(rs)

    def test_model_dump_masks_password(self) -> None:
        rs = RedfishSettings(
            bmc_username="admin",
            bmc_password=SecretStr("hunter2"),
        )
        dumped = rs.model_dump()
        assert dumped["bmc_password"] != "hunter2"


@pytest.mark.parametrize(
    ("values", "missing_field"),
    [
        ({"bmc_username": "admin"}, "bmc_password"),
        ({"bmc_password": SecretStr("secret")}, "bmc_username"),
    ],
)
def test_redfish_partial_config_rejected(
    values: dict[str, object],
    missing_field: str,
) -> None:
    with pytest.raises(ValidationError, match=missing_field):
        RedfishSettings.model_validate(values)


@pytest.mark.parametrize(
    "template",
    [
        "mgmt-static",
        "{hostname}-{hostname}",
        "mgmt-{other}",
        "mgmt-{hostname!r}",
        "mgmt-{hostname:>10}",
        "https://mgmt-{hostname}",
        "mgmt-{hostname}:443",
        "mgmt-{hostname}/redfish",
        "mgmt-{hostname}?target=other",
        "mgmt-{hostname}#fragment",
        "mgmt-{hostname",
    ],
)
def test_bmc_host_template_validation_matrix_rejects_unsafe(
    template: str,
) -> None:
    with pytest.raises(ValidationError, match="bmc_host_template"):
        RedfishSettings(bmc_host_template=template)


@pytest.mark.parametrize(
    "template",
    [
        "mgmt-{hostname}",
        "bmc-{hostname}.lab.example.com",
        "{hostname}.bmc.example.com",
    ],
)
def test_bmc_host_template_validation_matrix_accepts_plain_hostname(
    template: str,
) -> None:
    assert RedfishSettings(bmc_host_template=template).bmc_host_template == template


class TestRedfishSettingsIsNotBaseSettings:
    def test_redfish_settings_is_base_model_not_base_settings(self) -> None:
        assert not issubclass(RedfishSettings, BaseSettings)
        assert issubclass(RedfishSettings, BaseModel)


class TestSettingsIsBaseSettings:
    def test_settings_is_base_settings(self) -> None:
        assert issubclass(Settings, BaseSettings)


class TestHuggingFaceSettings:
    def test_cache_dir_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR", raising=False)
        with pytest.raises(ValidationError, match="huggingface"):
            Settings(_env_file=None)

    def test_api_token_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR", "/data/hf")
        monkeypatch.delenv("INFERENCE_PROXY_HUGGINGFACE__API_TOKEN", raising=False)
        settings = Settings(_env_file=None)
        assert settings.huggingface.api_token is None

    def test_api_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_HUGGINGFACE__CACHE_DIR", "/data/hf")
        monkeypatch.setenv("INFERENCE_PROXY_HUGGINGFACE__API_TOKEN", "hf_test123")
        settings = Settings(_env_file=None)
        assert settings.huggingface.api_token is not None
        assert settings.huggingface.api_token.get_secret_value() == "hf_test123"


class TestHuggingFaceSettingsIsNotBaseSettings:
    def test_huggingface_settings_is_base_model_not_base_settings(self) -> None:
        assert not issubclass(HuggingFaceSettings, BaseSettings)
        assert issubclass(HuggingFaceSettings, BaseModel)
