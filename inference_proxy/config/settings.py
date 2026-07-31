"""Application settings via pydantic-settings.

Sub-models inherit from BaseModel (not BaseSettings) to ensure
nested env var resolution works correctly through the root Settings class.
Only the root Settings class inherits from BaseSettings.
"""

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    nfs_server: str = "storage.example.com:/mnt/SATA/scratch/grafuls/hf-cache"
    nfs_mount_point: str = "/srv/hf-cache"
    nvidia_driver_version: str = "580.126.09"
    llmfit_version: str = "1.1.6"


class QUADSSettings(BaseModel):
    """QUADS API configuration.

    When ``base_url`` is ``None`` (the default), QUADS features are
    disabled (D-10).  Setting it via ``INFERENCE_PROXY_QUADS__BASE_URL``
    activates the QUADS integration.
    """

    base_url: str | None = None
    timeout: float = 10.0
    poll_interval: int = 300
    verify_ssl: bool = True
    schedule_check_interval: int = 300
    schedule_lookahead_hours: int = 24


class LLMFitSettings(BaseModel):
    """LLMFit remote execution configuration."""

    binary_path: str = "/usr/local/bin/llmfit"
    timeout: float = 60.0
    allowed_providers: list[str] = []
    version: str = "1.1.6"
    install_url: str = "https://github.com/AlexsJones/llmfit/releases/download/v{version}/llmfit-v{version}-x86_64-unknown-linux-musl.tar.gz"


class HuggingFaceSettings(BaseModel):
    """HuggingFace cache and authentication configuration."""

    cache_dir: str  # Required -- gateway won't start without it
    api_token: SecretStr | None = None


class RedfishSettings(BaseModel):
    """Redfish BMC configuration.

    When ``bmc_username`` is ``None`` (the default), Redfish features
    are disabled.  Setting it via ``INFERENCE_PROXY_REDFISH__BMC_USERNAME``
    activates the Redfish integration.
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
    dashboard: DashboardSettings = DashboardSettings()
    ssh: SSHSettings = SSHSettings()
    provisioning: ProvisioningSettings = ProvisioningSettings()
    quads: QUADSSettings = QUADSSettings()
    redfish: RedfishSettings = RedfishSettings()
    llmfit: LLMFitSettings = LLMFitSettings()
    huggingface: HuggingFaceSettings
