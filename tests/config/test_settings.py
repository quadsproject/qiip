"""Unit tests for configuration settings loading and env var overrides."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic_settings import BaseSettings

from inference_proxy.config.settings import (
    DEFAULT_LLAMACPP_SHA256,
    DEFAULT_LLAMACPP_SOURCE_URL,
    DEFAULT_LLAMACPP_VERSION,
    DEFAULT_LLMFIT_VERSION,
    AdminSettings,
    DashboardSettings,
    EtcdSettings,
    HuggingFaceSettings,
    LLMFitSettings,
    LoggingSettings,
    ProvisioningSettings,
    QUADSSettings,
    RedfishSettings,
    ResilienceSettings,
    RoutingSettings,
    Settings,
    SSHSettings,
)


class TestAdminSettings:
    def test_admin_credentials_required(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("INFERENCE_PROXY_ADMIN__USERNAME", raising=False)
        monkeypatch.delenv("INFERENCE_PROXY_ADMIN__PASSWORD", raising=False)

        with pytest.raises(ValidationError, match="admin"):
            Settings(_env_file=None)

    @pytest.mark.parametrize(
        ("present", "missing"),
        [("USERNAME", "password"), ("PASSWORD", "username")],
    )
    def test_admin_partial_credentials_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        present: str,
        missing: str,
    ) -> None:
        monkeypatch.delenv("INFERENCE_PROXY_ADMIN__USERNAME", raising=False)
        monkeypatch.delenv("INFERENCE_PROXY_ADMIN__PASSWORD", raising=False)
        monkeypatch.setenv(f"INFERENCE_PROXY_ADMIN__{present}", "configured")

        with pytest.raises(ValidationError, match=missing):
            Settings(_env_file=None)

    def test_admin_password_is_masked(self) -> None:
        settings = AdminSettings(
            username="operator",
            password=SecretStr("admin-secret"),
        )

        assert "admin-secret" not in repr(settings)
        assert settings.model_dump()["password"] != "admin-secret"

    def test_admin_credentials_load_from_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ADMIN__USERNAME", "operator")
        monkeypatch.setenv("INFERENCE_PROXY_ADMIN__PASSWORD", "admin-secret")

        settings = Settings(_env_file=None)

        assert settings.admin.username == "operator"
        assert settings.admin.password.get_secret_value() == "admin-secret"


class TestLoggingSettings:
    @pytest.mark.parametrize(
        ("configured", "canonical"),
        [
            ("debug", "DEBUG"),
            ("INFO", "INFO"),
            ("Warning", "WARNING"),
            ("ERROR", "ERROR"),
            ("critical", "CRITICAL"),
        ],
    )
    def test_log_level_validation_matrix(
        self,
        configured: str,
        canonical: str,
    ) -> None:
        assert LoggingSettings(level=configured).level == canonical

    @pytest.mark.parametrize("configured", ["TRACE", "VERBOSE", "inf", "", 10])
    def test_unknown_log_level_is_rejected(self, configured: object) -> None:
        with pytest.raises(ValidationError, match="logging.level"):
            LoggingSettings.model_validate({"level": configured})


class TestDefaultEtcdSettings:
    def test_default_etcd_settings(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.etcd.endpoints == ["http://localhost:2379"]
        assert settings.etcd.node_prefix == "/nodes/"
        assert settings.etcd.node_lease_ttl == 600

    def test_node_lease_ttl_exceeds_health_and_restart_budgets(self) -> None:
        settings = Settings(_env_file=None)

        assert settings.etcd.node_lease_ttl > 300
        assert (
            settings.etcd.node_lease_ttl > 3 * settings.resilience.health_check_interval
        )

        with pytest.raises(ValidationError, match="greater than 300"):
            EtcdSettings(node_lease_ttl=300)
        with pytest.raises(ValidationError, match="three times"):
            Settings(
                _env_file=None,
                etcd=EtcdSettings(node_lease_ttl=600),
                resilience=ResilienceSettings(health_check_interval=200),
            )


class TestDefaultRoutingSettings:
    def test_default_routing_settings(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.routing.max_attempts == 3
        assert settings.routing.timeout == 30
        assert settings.routing.allowed_endpoint_hosts == ["localhost"]
        assert settings.routing.allowed_endpoint_networks == [
            "127.0.0.0/8",
            "::1/128",
        ]
        assert settings.routing.allowed_endpoint_ports == [8000]

    def test_max_attempts_rejects_zero(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RoutingSettings(max_attempts=0)

        assert caught.value.errors()[0]["loc"] == ("max_attempts",)

    def test_max_attempts_loads_from_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__MAX_ATTEMPTS", "4")

        settings = Settings(_env_file=None)

        assert settings.routing.max_attempts == 4

    def test_max_retries_environment_name_is_not_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__MAX_RETRIES", "9")

        settings = Settings(_env_file=None)

        assert settings.routing.max_attempts == 3

    def test_provisioning_port_must_be_allowed(self) -> None:
        with pytest.raises(
            ValidationError,
            match="provisioning.vllm_port must be included",
        ):
            Settings(
                _env_file=None,
                provisioning=ProvisioningSettings(vllm_port=9000),
            )


class TestPreAdoptionSettingsCleanup:
    def test_removed_compatibility_fields_are_absent(self) -> None:
        """Guard only the retired fields, without freezing unrelated settings."""
        affected_models = (RoutingSettings, ProvisioningSettings)
        assert all(
            not field_name.startswith("retired_")
            for model in affected_models
            for field_name in model.model_fields
        )
        assert "gateway" not in Settings.model_fields
        assert "strategy" not in RoutingSettings.model_fields
        assert "health_check_interval" not in RoutingSettings.model_fields
        assert "llmfit_version" not in ProvisioningSettings.model_fields
        assert "nfs_server" not in ProvisioningSettings.model_fields
        assert "max_retries" not in RoutingSettings.model_fields
        assert "max_attempts" in RoutingSettings.model_fields

    def test_removed_environment_names_are_silently_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_GATEWAY__GRACEFUL_SHUTDOWN_TIMEOUT",
            "60",
        )
        monkeypatch.setenv("INFERENCE_PROXY_GATEWAY__HOST", "127.0.0.1")
        monkeypatch.setenv("INFERENCE_PROXY_GATEWAY__PORT", "9090")
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__STRATEGY", "round_robin")
        monkeypatch.setenv("INFERENCE_PROXY_ROUTING__HEALTH_CHECK_INTERVAL", "5")
        monkeypatch.setenv(
            "INFERENCE_PROXY_PROVISIONING__LLMFIT_VERSION",
            "9.9.9",
        )
        monkeypatch.setenv(
            "INFERENCE_PROXY_PROVISIONING__NFS_SERVER",
            "legacy.example:/old/export",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            settings = Settings(_env_file=None)

        assert settings.resilience.health_check_interval == 30
        assert settings.llmfit.version == DEFAULT_LLMFIT_VERSION
        assert settings.huggingface.nfs_export is None


class TestEnvVarOverrideEtcdPrefix:
    def test_env_var_override_etcd_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFERENCE_PROXY_ETCD__NODE_PREFIX", "/test-nodes/")
        settings = Settings(_env_file=None)
        assert settings.etcd.node_prefix == "/test-nodes/"


class TestEnvVarOverrideRoutingAllowlist:
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
        assert not issubclass(EtcdSettings, BaseSettings)
        assert not issubclass(RoutingSettings, BaseSettings)
        assert not issubclass(DashboardSettings, BaseSettings)
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
        settings = Settings(_env_file=None)
        assert settings.ssh.key_path == Path("~/.ssh/id_rsa").expanduser()

    def test_default_username(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.ssh.username == "root"

    def test_default_connect_timeout(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.ssh.connect_timeout == 10

    def test_default_streaming_deadlines(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.ssh.streaming_command_timeout == 3600.0
        assert settings.ssh.streaming_inactivity_timeout == 900.0

    def test_ssh_key_path_from_environment_is_expanded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(
            "INFERENCE_PROXY_SSH__KEY_PATH",
            "~/.ssh/provisioning-key",
        )

        settings = Settings(_env_file=None)

        assert settings.ssh.key_path == tmp_path / ".ssh/provisioning-key"


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


class TestArtifactDigestSettings:
    def test_default_version_digest_pairs_are_usable(self) -> None:
        provisioning = ProvisioningSettings()
        llmfit = LLMFitSettings()

        assert provisioning.nvidia_driver_version == "580.126.09"
        assert provisioning.nvidia_driver_sha256 == (
            "4cac53e48f8adff661d47c8788ed24059a248c9fd8098ceafd088a498986ec26"
        )
        assert provisioning.llamacpp_version == DEFAULT_LLAMACPP_VERSION
        assert provisioning.llamacpp_sha256 == DEFAULT_LLAMACPP_SHA256
        assert provisioning.llamacpp_source_url == DEFAULT_LLAMACPP_SOURCE_URL
        assert provisioning.llamacpp_source_download_url() == (
            "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10242.tar.gz"
        )
        assert llmfit.version == "1.1.6"
        assert llmfit.sha256 == (
            "1e09232a128455596a2d348ab5893741d04b94aa6d924f1253462dc13304f7c6"
        )

    @pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64, "sha256:" + "a" * 64])
    def test_nvidia_driver_digest_must_be_sha256(self, digest: str) -> None:
        with pytest.raises(ValidationError, match="64 hexadecimal characters"):
            ProvisioningSettings(nvidia_driver_sha256=digest)

    @pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64, "sha256:" + "a" * 64])
    def test_llmfit_digest_must_be_sha256(self, digest: str) -> None:
        with pytest.raises(ValidationError, match="64 hexadecimal characters"):
            LLMFitSettings(sha256=digest)

    @pytest.mark.parametrize("digest", ["", "a" * 63, "g" * 64, "sha256:" + "a" * 64])
    def test_llamacpp_digest_must_be_sha256(self, digest: str) -> None:
        with pytest.raises(ValidationError, match="64 hexadecimal characters"):
            ProvisioningSettings(llamacpp_sha256=digest)

    def test_custom_driver_version_requires_explicit_digest(self) -> None:
        with pytest.raises(ValidationError, match="nvidia_driver_sha256"):
            ProvisioningSettings(nvidia_driver_version="999.1")

        configured = ProvisioningSettings(
            nvidia_driver_version="999.1", nvidia_driver_sha256="b" * 64
        )
        assert configured.nvidia_driver_sha256 == "b" * 64

    def test_custom_llmfit_version_requires_explicit_digest(self) -> None:
        with pytest.raises(ValidationError, match="llmfit.sha256"):
            LLMFitSettings(version="2.0.0")

        configured = LLMFitSettings(version="2.0.0", sha256="b" * 64)
        assert configured.sha256 == "b" * 64

    def test_custom_llamacpp_version_requires_explicit_digest(self) -> None:
        with pytest.raises(ValidationError, match="llamacpp_sha256"):
            ProvisioningSettings(llamacpp_version="b12345")

        configured = ProvisioningSettings(
            llamacpp_version="b12345", llamacpp_sha256="b" * 64
        )
        assert configured.llamacpp_sha256 == "b" * 64

    @pytest.mark.parametrize("version", ["10242", "v10242", "b0", "b1.2", "latest"])
    def test_llamacpp_version_requires_build_tag(self, version: str) -> None:
        with pytest.raises(ValidationError, match=r"b<number>"):
            ProvisioningSettings(
                llamacpp_version=version,
                llamacpp_sha256="b" * 64,
            )

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://downloads.example/llama-{version}.tar.gz",
            "https://user:secret@downloads.example/llama-{version}.tar.gz",
            "https://downloads.example/llama.tar.gz",
            "https://downloads.example/{version}/{version}/llama.tar.gz",
            "https://downloads.example/llama-{version!r}.tar.gz",
            "https://downloads.example/llama-{other}.tar.gz",
            "https://downloads.example/llama-{version}.tar.gz#fragment",
            "https://downloads.example/llama-{version}.tar.gz\ncommand",
        ],
    )
    def test_llamacpp_source_url_rejects_unsafe_templates(self, url: str) -> None:
        with pytest.raises(ValidationError, match="llamacpp_source_url"):
            ProvisioningSettings(llamacpp_source_url=url)

    @pytest.mark.parametrize("scheme", ["https", "http"])
    def test_llamacpp_source_url_accepts_verified_mirrors(self, scheme: str) -> None:
        settings = ProvisioningSettings(
            llamacpp_source_url=(
                f"{scheme}://mirror.example/llama/{{version}}/source.tar.gz"
            )
        )
        assert settings.llamacpp_source_download_url() == (
            f"{scheme}://mirror.example/llama/b10242/source.tar.gz"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://downloads.example/llmfit-{version}.tar.gz",
            "https://user:secret@downloads.example/llmfit-{version}.tar.gz",
            "https://downloads.example/llmfit.tar.gz",
            "https://downloads.example/{version}/{version}/llmfit.tar.gz",
            "https://downloads.example/llmfit-{version!r}.tar.gz",
            "https://downloads.example/llmfit-{other}.tar.gz",
            "https://downloads.example/llmfit-{version}.tar.gz#fragment",
            "https://downloads.example/llmfit-{version}.tar.gz\ncommand",
        ],
    )
    def test_llmfit_install_url_rejects_unsafe_templates(self, url: str) -> None:
        with pytest.raises(ValidationError, match="llmfit.install_url"):
            LLMFitSettings(install_url=url)

    @pytest.mark.parametrize("scheme", ["https", "http"])
    def test_llmfit_install_url_accepts_verified_mirrors(self, scheme: str) -> None:
        settings = LLMFitSettings(
            install_url=f"{scheme}://mirror.example/releases/{{version}}/llmfit.tar.gz"
        )
        assert settings.install_url.startswith(f"{scheme}://")


class TestProvisioningResourceLimits:
    @pytest.mark.parametrize(
        "field",
        [
            "max_concurrent_provisions",
            "log_max_entries_per_host",
            "log_max_bytes_per_host",
            "log_max_entry_bytes",
            "log_max_completed_hosts",
        ],
    )
    def test_provisioning_resource_limits_reject_zero(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ProvisioningSettings(**{field: 0})

    def test_log_entry_limit_cannot_exceed_host_budget(self) -> None:
        with pytest.raises(
            ValidationError,
            match="log_max_entry_bytes must not exceed",
        ):
            ProvisioningSettings(
                log_max_bytes_per_host=100,
                log_max_entry_bytes=101,
            )


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

    def test_server_timezone_is_none_while_disabled(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.quads.server_timezone is None


class TestEnvVarOverrideQUADSBaseUrl:
    def test_env_var_override_quads_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_QUADS__BASE_URL", "http://quads.example.com"
        )
        monkeypatch.setenv("INFERENCE_PROXY_QUADS__SERVER_TIMEZONE", "America/New_York")
        settings = Settings(_env_file=None)
        assert settings.quads.base_url == "http://quads.example.com"
        assert settings.quads.server_timezone == "America/New_York"

    def test_enabled_quads_requires_server_timezone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "INFERENCE_PROXY_QUADS__BASE_URL", "http://quads.example.com"
        )
        monkeypatch.delenv("INFERENCE_PROXY_QUADS__SERVER_TIMEZONE", raising=False)

        with pytest.raises(ValidationError, match="server_timezone is required"):
            Settings(_env_file=None)

    def test_unknown_quads_server_timezone_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="known IANA timezone"):
            QUADSSettings(
                base_url="http://quads.example.com",
                server_timezone="Mars/Olympus_Mons",
            )


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

    def test_nfs_export_optional_for_proxy_only_deployment(self) -> None:
        settings = HuggingFaceSettings(cache_dir="/data/huggingface")

        assert settings.nfs_export is None

    def test_cache_paths_resolve_to_same_export(self) -> None:
        settings = Settings(
            admin=AdminSettings(
                username="operator", password=SecretStr("admin-secret")
            ),
            huggingface=HuggingFaceSettings(
                cache_dir="/data/huggingface",
                nfs_export="storage.example:/exports/huggingface",
            ),
            provisioning=ProvisioningSettings(nfs_mount_point="/srv/hf-cache"),
        )

        assert settings.huggingface.cache_dir == "/data/huggingface"
        assert settings.provisioning.nfs_mount_point == "/srv/hf-cache"
        assert settings.huggingface.nfs_export == (
            "storage.example:/exports/huggingface"
        )

    @pytest.mark.parametrize("value", ["", "   "])
    def test_explicit_empty_nfs_export_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError, match="nfs_export must not be empty"):
            HuggingFaceSettings(cache_dir="/data/huggingface", nfs_export=value)

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


def test_env_example_covers_every_application_setting_exactly_once() -> None:
    """Keep the shipped environment inventory synchronized with Settings."""
    expected: set[str] = set()
    for group_name, group_field in Settings.model_fields.items():
        model = group_field.annotation
        assert isinstance(model, type)
        assert issubclass(model, BaseModel)
        expected.update(
            f"INFERENCE_PROXY_{group_name.upper()}__{field_name.upper()}"
            for field_name in model.model_fields
        )

    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    occurrences = re.findall(
        r"(?m)^\s*#?\s*(INFERENCE_PROXY_[A-Z0-9_]+)=",
        env_example.read_text(),
    )

    assert len(occurrences) == len(set(occurrences)), "duplicate .env.example entry"
    assert set(occurrences) == expected
