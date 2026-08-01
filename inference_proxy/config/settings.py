"""Application settings via pydantic-settings.

Sub-models inherit from BaseModel (not BaseSettings) to ensure
nested env var resolution works correctly through the root Settings class.
Only the root Settings class inherits from BaseSettings.
"""

import re
import warnings
from pathlib import Path
from string import Formatter
from typing import Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference_proxy.models.endpoint import (
    EndpointPolicy,
    EndpointValidationError,
    parse_endpoint,
)

DEFAULT_NVIDIA_DRIVER_VERSION = "580.126.09"
DEFAULT_NVIDIA_DRIVER_SHA256 = (
    "4cac53e48f8adff661d47c8788ed24059a248c9fd8098ceafd088a498986ec26"
)
DEFAULT_LLMFIT_VERSION = "1.1.6"
DEFAULT_LLMFIT_SHA256 = (
    "1e09232a128455596a2d348ab5893741d04b94aa6d924f1253462dc13304f7c6"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validate_sha256(value: str, *, setting: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{setting} must be exactly 64 hexadecimal characters")
    return normalized


class GatewaySettings(BaseModel):
    """Gateway server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    graceful_shutdown_timeout: int = 30


class EtcdSettings(BaseModel):
    """etcd service discovery configuration."""

    endpoints: list[str] = ["http://localhost:2379"]
    node_prefix: str = "/nodes/"

    @field_validator("endpoints")
    @classmethod
    def endpoints_must_be_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one etcd endpoint is configured."""
        if not v:
            raise ValueError("At least one etcd endpoint must be configured")
        return v


class RoutingSettings(BaseModel):
    """Request routing and load balancing configuration."""

    strategy: str = "least_connections"
    health_check_interval: int = 30
    max_retries: int = Field(default=3, ge=1)
    timeout: int = 30
    allowed_endpoint_hosts: list[str] = Field(default_factory=lambda: ["localhost"])
    allowed_endpoint_networks: list[str] = Field(
        default_factory=lambda: ["127.0.0.0/8", "::1/128"]
    )
    allowed_endpoint_ports: list[int] = Field(default_factory=lambda: [8000])

    @model_validator(mode="after")
    def endpoint_allowlist_is_valid(self) -> Self:
        """Reject malformed endpoint trust rules during settings loading."""
        self.endpoint_policy()
        return self

    def endpoint_policy(self) -> EndpointPolicy:
        """Build the immutable endpoint policy used by discovery."""
        return EndpointPolicy.from_values(
            allowed_hosts=self.allowed_endpoint_hosts,
            allowed_networks=self.allowed_endpoint_networks,
            allowed_ports=self.allowed_endpoint_ports,
        )


class ProxySettings(BaseModel):
    """Proxy client configuration for httpx.AsyncClient.

    Timeouts are tuned for LLM inference: ``read_timeout`` defaults to
    120 s because first-token latency on large prompts can exceed 30 s.
    """

    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    write_timeout: float = 10.0
    pool_timeout: float = 10.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: int = 30


class ResilienceSettings(BaseModel):
    """Resilience configuration for circuit breakers and health checking.

    ``circuit_breaker_threshold``: consecutive failures before a circuit
    breaker trips to OPEN (per D-06, default 3).

    ``health_check_failure_threshold``: consecutive probe failures before
    marking a node UNHEALTHY (per D-03, default 3).

    ``health_check_interval``: seconds between health check probe cycles
    (default 30).  This is the canonical source; the legacy
    ``RoutingSettings.health_check_interval`` is retained for backward
    compatibility.
    """

    circuit_breaker_threshold: int = 3
    health_check_failure_threshold: int = 3
    health_check_interval: int = 30


class LoggingSettings(BaseModel):
    """Logging configuration."""

    json_output: bool = False
    level: str = "INFO"


class AdminSettings(BaseModel):
    """Required shared credentials for the administrative surface."""

    username: str = Field(min_length=1)
    password: SecretStr

    @field_validator("username")
    @classmethod
    def username_has_no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("admin.username must not contain surrounding whitespace")
        return value

    @field_validator("password")
    @classmethod
    def password_is_not_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("admin.password must not be empty")
        return value


class DashboardSettings(BaseModel):
    """Dashboard UI configuration."""

    poll_interval: int = Field(default=10, ge=1)


class SSHSettings(BaseModel):
    """SSH connection configuration (D-16).

    All hosts use the same key and username per D-01, D-02.
    """

    key_path: Path = Path("~/.ssh/id_rsa").expanduser()  # D-01
    username: str = "root"  # D-02
    connect_timeout: int = 10  # D-04
    streaming_command_timeout: float = Field(default=3600.0, gt=0)
    streaming_inactivity_timeout: float = Field(default=900.0, gt=0)

    @field_validator("key_path", mode="before")
    @classmethod
    def expand_key_path(cls, value: str | Path) -> Path:
        """Expand user-relative paths after environment settings are loaded."""
        return Path(value).expanduser()


class ProvisioningSettings(BaseModel):
    """Node provisioning configuration.

    Controls health polling after vLLM startup.
    """

    health_poll_timeout: int = 600  # 10 minutes for large model loading
    health_poll_interval: int = 10
    vllm_port: int = 8000
    min_disk_gb: int = 20
    drain_timeout: int = 30
    scripts_dir: Path = Path("auto-vllm")
    boot_wait_timeout: int = 300  # D-05: 5 minutes for cold boot
    boot_wait_interval: int = 10
    nfs_mount_point: str = "/srv/hf-cache"
    nvidia_driver_version: str = DEFAULT_NVIDIA_DRIVER_VERSION
    nvidia_driver_sha256: str = DEFAULT_NVIDIA_DRIVER_SHA256
    retired_nfs_server: str | None = Field(
        default=None,
        validation_alias="nfs_server",
        exclude=True,
        repr=False,
    )
    retired_llmfit_version: str | None = Field(
        default=None,
        validation_alias="llmfit_version",
        exclude=True,
        repr=False,
    )

    @model_validator(mode="after")
    def warn_about_retired_llmfit_version(self) -> Self:
        """Make the retired provisioning-scoped LLMFit knob fail visibly."""
        if self.retired_llmfit_version is not None:
            warnings.warn(
                "INFERENCE_PROXY_PROVISIONING__LLMFIT_VERSION is ignored; "
                "configure INFERENCE_PROXY_LLMFIT__VERSION instead",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def warn_about_retired_nfs_server(self) -> Self:
        """Make migration from the duplicate NFS export knob visible."""
        if self.retired_nfs_server is not None:
            warnings.warn(
                "INFERENCE_PROXY_PROVISIONING__NFS_SERVER is ignored; "
                "configure INFERENCE_PROXY_HUGGINGFACE__NFS_EXPORT instead",
                UserWarning,
                stacklevel=2,
            )
        return self

    @field_validator("nvidia_driver_sha256")
    @classmethod
    def nvidia_driver_digest_is_sha256(cls, value: str) -> str:
        """Require a complete digest before remote driver installation."""
        return _validate_sha256(value, setting="provisioning.nvidia_driver_sha256")

    @model_validator(mode="after")
    def custom_driver_version_has_explicit_digest(self) -> Self:
        """Keep the committed default version and digest as one upgrade unit."""
        if (
            self.nvidia_driver_version != DEFAULT_NVIDIA_DRIVER_VERSION
            and "nvidia_driver_sha256" not in self.model_fields_set
        ):
            raise ValueError(
                "provisioning.nvidia_driver_sha256 must be configured when "
                "nvidia_driver_version differs from the built-in default"
            )
        return self


class QUADSSettings(BaseModel):
    """QUADS API configuration.

    When ``base_url`` is ``None`` (the default), QUADS features are
    disabled (D-10).  Setting it via ``INFERENCE_PROXY_QUADS__BASE_URL``
    activates the QUADS integration.
    """

    base_url: str | None = None
    server_timezone: str | None = None
    timeout: float = 10.0
    poll_interval: int = 300
    verify_ssl: bool = True
    schedule_check_interval: int = 300
    schedule_lookahead_hours: int = 24

    @model_validator(mode="after")
    def server_timezone_matches_quads_clock(self) -> Self:
        """Require the IANA timezone used by QUADS' naive date parser."""
        if self.base_url is not None and self.server_timezone is None:
            raise ValueError(
                "quads.server_timezone is required when quads.base_url is set"
            )
        if self.server_timezone is not None:
            try:
                ZoneInfo(self.server_timezone)
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError(
                    f"quads.server_timezone is not a known IANA timezone: "
                    f"{self.server_timezone!r}"
                ) from exc
        return self


class LLMFitSettings(BaseModel):
    """LLMFit remote execution configuration."""

    binary_path: str = "/usr/local/bin/llmfit"
    timeout: float = 60.0
    allowed_providers: list[str] = []
    version: str = DEFAULT_LLMFIT_VERSION
    sha256: str = DEFAULT_LLMFIT_SHA256
    install_url: str = "https://github.com/AlexsJones/llmfit/releases/download/v{version}/llmfit-v{version}-x86_64-unknown-linux-musl.tar.gz"

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        """Require a complete digest before remote binary installation."""
        return _validate_sha256(value, setting="llmfit.sha256")

    @field_validator("install_url")
    @classmethod
    def install_url_is_safe_template(cls, value: str) -> str:
        """Allow one safely-rendered HTTP(S) release URL template."""
        if any(ord(character) < 32 for character in value):
            raise ValueError("llmfit.install_url must not contain control characters")
        try:
            parsed_fields = list(Formatter().parse(value))
        except ValueError as exc:
            raise ValueError("llmfit.install_url is malformed") from exc
        fields = [
            (field_name, format_spec, conversion)
            for _literal, field_name, format_spec, conversion in parsed_fields
            if field_name is not None
        ]
        if fields != [("version", "", None)]:
            raise ValueError(
                "llmfit.install_url must contain exactly one plain {version} field"
            )

        parsed = urlsplit(value.format(version="1.2.3"))
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("llmfit.install_url must be an HTTP(S) URL with a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("llmfit.install_url must not contain credentials")
        if parsed.fragment:
            raise ValueError("llmfit.install_url must not contain a fragment")
        return value

    @model_validator(mode="after")
    def custom_version_has_explicit_digest(self) -> Self:
        """Keep the committed default version and digest as one upgrade unit."""
        if (
            self.version != DEFAULT_LLMFIT_VERSION
            and "sha256" not in self.model_fields_set
        ):
            raise ValueError(
                "llmfit.sha256 must be configured when version differs from the "
                "built-in default"
            )
        return self


class HuggingFaceSettings(BaseModel):
    """HuggingFace cache and authentication configuration."""

    cache_dir: str  # Required -- gateway won't start without it
    nfs_export: str | None = None
    api_token: SecretStr | None = None

    @field_validator("nfs_export")
    @classmethod
    def nfs_export_is_not_empty(cls, value: str | None) -> str | None:
        """Reject an explicitly configured empty export."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("huggingface.nfs_export must not be empty")
        return normalized


class RedfishSettings(BaseModel):
    """Redfish BMC configuration.

    Redfish is disabled when both credentials are absent. Username and
    password must be configured together to activate the integration.
    """

    bmc_username: str | None = None
    bmc_password: SecretStr | None = None
    bmc_host_template: str = "mgmt-{hostname}"  # D-01, D-02
    system_id: str = "1"
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    power_poll_timeout: float = 60.0
    power_poll_interval: float = 5.0
    verify_ssl: bool = False  # D-05: always False for self-signed BMC certs

    @field_validator("bmc_host_template")
    @classmethod
    def bmc_template_is_plain_hostname(cls, value: str) -> str:
        """Require one safe hostname placeholder and no URL components."""
        try:
            parsed_fields = list(Formatter().parse(value))
        except ValueError as exc:
            raise ValueError("bmc_host_template is malformed") from exc

        fields = [
            (field_name, format_spec, conversion)
            for _literal, field_name, format_spec, conversion in parsed_fields
            if field_name is not None
        ]
        if fields != [("hostname", "", None)]:
            raise ValueError(
                "bmc_host_template must contain exactly one plain {hostname} field"
            )

        rendered = value.format(hostname="node.example")
        try:
            endpoint = parse_endpoint(f"https://{rendered}:443")
        except EndpointValidationError as exc:
            raise ValueError(
                "bmc_host_template must render a plain hostname without a "
                "scheme, port, path, query, or fragment"
            ) from exc
        if endpoint.host != rendered.lower().rstrip("."):
            raise ValueError("bmc_host_template must render a plain hostname")
        return value

    @model_validator(mode="after")
    def credentials_are_paired(self) -> Self:
        """Require both BMC credentials or neither of them."""
        if self.bmc_username is None and self.bmc_password is not None:
            raise ValueError(
                "redfish.bmc_username must be configured with bmc_password"
            )
        if self.bmc_username is not None and self.bmc_password is None:
            raise ValueError(
                "redfish.bmc_password must be configured with bmc_username"
            )
        return self


class Settings(BaseSettings):
    """Root application settings.

    Loads configuration from environment variables with the prefix
    ``INFERENCE_PROXY_`` and nested delimiter ``__``.

    Example env var: ``INFERENCE_PROXY_GATEWAY__PORT=9090``
    """

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_PROXY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    gateway: GatewaySettings = GatewaySettings()
    etcd: EtcdSettings = EtcdSettings()
    routing: RoutingSettings = RoutingSettings()
    proxy: ProxySettings = ProxySettings()
    resilience: ResilienceSettings = ResilienceSettings()
    logging: LoggingSettings = LoggingSettings()
    admin: AdminSettings
    dashboard: DashboardSettings = DashboardSettings()
    ssh: SSHSettings = SSHSettings()
    provisioning: ProvisioningSettings = ProvisioningSettings()
    quads: QUADSSettings = QUADSSettings()
    redfish: RedfishSettings = RedfishSettings()
    llmfit: LLMFitSettings = LLMFitSettings()
    huggingface: HuggingFaceSettings

    @model_validator(mode="after")
    def provisioned_port_is_allowed(self) -> Self:
        """Fail fast when provisioned nodes would be rejected by discovery."""
        if self.provisioning.vllm_port not in self.routing.allowed_endpoint_ports:
            raise ValueError(
                "provisioning.vllm_port must be included in "
                "routing.allowed_endpoint_ports"
            )
        return self
