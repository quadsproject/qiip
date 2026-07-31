"""Canonical parsing and trust policy for vLLM backend endpoints."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from ipaddress import (
    IPv4Address,
    IPv4Network,
    IPv6Address,
    IPv6Network,
    ip_address,
    ip_network,
)
from urllib.parse import urlsplit

_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


class EndpointValidationError(ValueError):
    """Raised when a backend endpoint or allowlist rule is unsafe."""


@dataclass(frozen=True, slots=True)
class ParsedEndpoint:
    """A normalized backend origin with no path or query components."""

    scheme: str
    host: str
    port: int

    @property
    def origin(self) -> str:
        """Return the normalized, scheme-bearing endpoint origin."""
        rendered_host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{rendered_host}:{self.port}"


def _normalize_dns_name(value: str, *, context: str) -> str:
    hostname = value.lower().rstrip(".")
    if not hostname or len(hostname) > 253:
        raise EndpointValidationError(f"invalid {context}: {value!r}")
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EndpointValidationError(f"invalid {context}: {value!r}") from exc
    if any(_HOST_LABEL.fullmatch(label) is None for label in hostname.split(".")):
        raise EndpointValidationError(f"invalid {context}: {value!r}")
    return hostname


def parse_endpoint(value: str) -> ParsedEndpoint:
    """Parse an endpoint and normalize schemeless values to HTTP.

    Backend endpoints are origins, not arbitrary URLs. They must have an
    explicit port and may not contain credentials, paths, queries, or
    fragments. A single trailing slash is accepted and normalized away.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise EndpointValidationError(f"invalid backend endpoint: {value!r}")

    candidate = value if "://" in value else f"http://{value}"
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise EndpointValidationError(f"invalid backend endpoint: {value!r}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise EndpointValidationError(
            f"unsupported backend endpoint scheme in {value!r}"
        )
    if not parsed.netloc or host is None:
        raise EndpointValidationError(f"backend endpoint is missing a host: {value!r}")
    if "@" in parsed.netloc:
        raise EndpointValidationError(
            f"backend endpoint credentials are not allowed: {value!r}"
        )
    if port is None:
        raise EndpointValidationError(f"backend endpoint is missing a port: {value!r}")
    if not 1 <= port <= 65535:
        raise EndpointValidationError(f"backend endpoint has invalid port: {value!r}")
    if parsed.path not in {"", "/"}:
        raise EndpointValidationError(
            f"backend endpoint paths are not allowed: {value!r}"
        )
    if parsed.query or parsed.fragment:
        raise EndpointValidationError(
            f"backend endpoint query and fragment are not allowed: {value!r}"
        )

    try:
        normalized_host = str(ip_address(host))
    except ValueError:
        normalized_host = _normalize_dns_name(host, context="backend hostname")
    return ParsedEndpoint(scheme=scheme, host=normalized_host, port=port)


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """Allow backend origins only on configured hosts, networks, and ports."""

    allowed_hosts: frozenset[str]
    allowed_host_suffixes: tuple[str, ...]
    allowed_networks: tuple[IPv4Network | IPv6Network, ...]
    allowed_ports: frozenset[int]

    @classmethod
    def from_values(
        cls,
        *,
        allowed_hosts: Iterable[str],
        allowed_networks: Iterable[str],
        allowed_ports: Iterable[int],
    ) -> EndpointPolicy:
        """Validate configuration values and construct an endpoint policy."""
        exact_hosts: set[str] = set()
        suffixes: set[str] = set()
        for rule in allowed_hosts:
            if rule.startswith("*."):
                suffixes.add(
                    "." + _normalize_dns_name(rule[2:], context="host allowlist rule")
                )
            else:
                exact_hosts.add(
                    _normalize_dns_name(rule, context="host allowlist rule")
                )

        networks: list[IPv4Network | IPv6Network] = []
        for rule in allowed_networks:
            try:
                networks.append(ip_network(rule, strict=False))
            except ValueError as exc:
                raise EndpointValidationError(
                    f"invalid endpoint network allowlist rule: {rule!r}"
                ) from exc

        ports: set[int] = set()
        for port in allowed_ports:
            if isinstance(port, bool) or not 1 <= port <= 65535:
                raise EndpointValidationError(
                    f"invalid endpoint port allowlist rule: {port!r}"
                )
            ports.add(port)

        return cls(
            allowed_hosts=frozenset(exact_hosts),
            allowed_host_suffixes=tuple(sorted(suffixes)),
            allowed_networks=tuple(networks),
            allowed_ports=frozenset(ports),
        )

    def normalize(self, value: str) -> str:
        """Validate *value* against the allowlist and return its origin."""
        endpoint = parse_endpoint(value)
        if endpoint.port not in self.allowed_ports:
            raise EndpointValidationError(
                f"backend endpoint port is not allowed: {value!r}"
            )
        if not self._host_allowed(endpoint.host):
            raise EndpointValidationError(
                f"backend endpoint host is not allowed: {value!r}"
            )
        return endpoint.origin

    def normalize_hostname(self, value: str) -> str:
        """Validate a DNS hostname against the host allowlist.

        Redfish BMC destinations are derived from node names rather than
        backend origins, so this entry point deliberately does not accept a
        port. IP literals are rejected even when a backend CIDR would allow
        them because the BMC template is a hostname pattern.
        """
        if not isinstance(value, str) or not value or value != value.strip():
            raise EndpointValidationError(f"invalid node hostname: {value!r}")

        candidate = value.lower().rstrip(".")
        try:
            ip_address(candidate)
        except ValueError:
            hostname = _normalize_dns_name(value, context="node hostname")
        else:
            raise EndpointValidationError(
                f"IP-literal node hostnames are not allowed: {value!r}"
            )

        if not self._host_allowed(hostname):
            raise EndpointValidationError(f"node hostname is not allowed: {value!r}")
        return hostname

    def _host_allowed(self, host: str) -> bool:
        try:
            address: IPv4Address | IPv6Address = ip_address(host)
        except ValueError:
            return host in self.allowed_hosts or any(
                host.endswith(suffix) and host != suffix[1:]
                for suffix in self.allowed_host_suffixes
            )
        return any(
            address.version == network.version and address in network
            for network in self.allowed_networks
        )


def build_backend_url(endpoint: str, path: str) -> str:
    """Build a backend URL from a validated endpoint and absolute path."""
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError(f"backend path must be absolute: {path!r}")
    return f"{parse_endpoint(endpoint).origin}{path}"
