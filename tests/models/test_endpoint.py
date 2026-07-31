"""Tests for canonical backend endpoint parsing and trust policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_proxy.config.settings import RoutingSettings
from inference_proxy.models.endpoint import (
    EndpointPolicy,
    EndpointValidationError,
    build_backend_url,
)


def _policy() -> EndpointPolicy:
    return EndpointPolicy.from_values(
        allowed_hosts=["gpu01", "*.example.com"],
        allowed_networks=["10.0.1.0/24", "::1/128", "2001:db8::/32"],
        allowed_ports=[8000, 8443],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gpu01:8000", "http://gpu01:8000"),
        ("HTTP://GPU01:8000/", "http://gpu01:8000"),
        ("https://worker.example.com:8443", "https://worker.example.com:8443"),
        ("10.0.1.100:8000", "http://10.0.1.100:8000"),
        ("[::1]:8000", "http://[::1]:8000"),
        ("https://[2001:db8::10]:8443", "https://[2001:db8::10]:8443"),
    ],
)
def test_endpoint_normalization_matrix(value: str, expected: str) -> None:
    assert _policy().normalize(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "gpu01",
        "ftp://gpu01:8000",
        "http://user:secret@gpu01:8000",
        "http://gpu01:8000/v1/completions",
        "http://gpu01:8000?debug=true",
        "http://gpu01:8000#fragment",
        "http://bad_host:8000",
        "http://gpu01:0",
        "http://gpu01:65536",
        "::1:8000",
        " gpu01:8000",
    ],
)
def test_endpoint_rejects_unsafe_url_components(value: str) -> None:
    with pytest.raises(EndpointValidationError):
        _policy().normalize(value)


@pytest.mark.parametrize(
    ("value", "allowed"),
    [
        ("gpu01:8000", True),
        ("worker.example.com:8000", True),
        ("example.com:8000", False),
        ("evil-example.com:8000", False),
        ("10.0.1.42:8000", True),
        ("10.0.2.42:8000", False),
        ("gpu01:9000", False),
    ],
)
def test_endpoint_allowlist_matrix(value: str, allowed: bool) -> None:
    if allowed:
        assert _policy().normalize(value).endswith(value)
    else:
        with pytest.raises(EndpointValidationError, match="not allowed"):
            _policy().normalize(value)


def test_empty_endpoint_host_allowlist_denies_all() -> None:
    policy = EndpointPolicy.from_values(
        allowed_hosts=[],
        allowed_networks=[],
        allowed_ports=[8000],
    )

    with pytest.raises(EndpointValidationError, match="host is not allowed"):
        policy.normalize("gpu01:8000")


def test_build_backend_url_preserves_scheme_and_ipv6_brackets() -> None:
    assert (
        build_backend_url("https://gpu01:8443", "/health")
        == "https://gpu01:8443/health"
    )
    assert (
        build_backend_url("http://[::1]:8000", "/v1/completions")
        == "http://[::1]:8000/v1/completions"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_endpoint_hosts", ["*"]),
        ("allowed_endpoint_hosts", ["bad_host"]),
        ("allowed_endpoint_networks", ["10.0.0.0/99"]),
        ("allowed_endpoint_ports", [0]),
        ("allowed_endpoint_ports", [65536]),
    ],
)
def test_endpoint_allowlist_settings_validation(
    field: str,
    value: list[str] | list[int],
) -> None:
    with pytest.raises(ValidationError):
        RoutingSettings.model_validate({field: value})
