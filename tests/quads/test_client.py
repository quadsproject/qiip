"""Unit tests for QUADSClient, canonical_hostname, and QUADSConnectionError."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from inference_proxy.quads.client import (
    QUADSClient,
    QUADSConnectionError,
    availability_window_end,
    canonical_hostname,
)

QUADS_URL = "https://quads.example.com"


class TestCanonicalHostname:
    def test_strips_whitespace(self) -> None:
        assert canonical_hostname("  host01  ") == "host01"

    def test_lowercases(self) -> None:
        assert canonical_hostname("HOST01") == "host01"

    def test_strips_trailing_dot(self) -> None:
        assert canonical_hostname("host01.") == "host01"

    def test_combined(self) -> None:
        assert canonical_hostname("  Host01. ") == "host01"


class TestAvailabilityWindow:
    def test_uses_configured_hours(self) -> None:
        now = datetime(2026, 7, 31, 16, 45, tzinfo=UTC)

        end = availability_window_end(24, now=now)

        assert end == datetime(2026, 8, 1, 16, 45, tzinfo=UTC)

    def test_rejects_naive_start(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            availability_window_end(24, now=datetime(2026, 7, 31, 16, 45))


def _gpu_host(
    name: str,
    *,
    broken: bool = False,
    retired: bool = False,
    gpus: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal QUADS host dict for testing."""
    if gpus is None:
        gpus = [{"processor_type": "GPU", "vendor": "NVIDIA", "product": "A100"}]
    return {
        "name": name,
        "broken": broken,
        "retired": retired,
        "processors": gpus,
    }


def _cpu_host(name: str) -> dict[str, object]:
    return {
        "name": name,
        "broken": False,
        "retired": False,
        "processors": [
            {"processor_type": "CPU", "vendor": "Intel", "product": "Xeon"},
        ],
    }


class TestGetHosts:
    async def test_filters_to_gpu_only(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[_gpu_host("gpu01"), _cpu_host("cpu01")],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert len(hosts) == 1
        assert hosts[0].hostname == "gpu01"

    async def test_excludes_broken(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[_gpu_host("ok"), _gpu_host("bad", broken=True)],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert len(hosts) == 1
        assert hosts[0].hostname == "ok"

    async def test_excludes_retired(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[_gpu_host("ok"), _gpu_host("old", retired=True)],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert len(hosts) == 1
        assert hosts[0].hostname == "ok"

    async def test_hostname_normalized(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[_gpu_host("  GPU-HOST01. ")],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert hosts[0].hostname == "gpu-host01"

    async def test_gpu_fields_extracted(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[
                _gpu_host(
                    "h1",
                    gpus=[
                        {
                            "processor_type": "GPU",
                            "vendor": "NVIDIA",
                            "product": "A100",
                        },
                    ],
                )
            ],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert hosts[0].gpu_vendor == "NVIDIA"
        assert hosts[0].gpu_model == "A100"
        assert hosts[0].gpu_count == 1

    async def test_multiple_gpu_processors_counted(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            json=[
                _gpu_host(
                    "h1",
                    gpus=[
                        {
                            "processor_type": "GPU",
                            "vendor": "NVIDIA",
                            "product": "A100",
                        },
                        {
                            "processor_type": "GPU",
                            "vendor": "NVIDIA",
                            "product": "A100",
                        },
                        {
                            "processor_type": "GPU",
                            "vendor": "NVIDIA",
                            "product": "A100",
                        },
                        {
                            "processor_type": "GPU",
                            "vendor": "NVIDIA",
                            "product": "A100",
                        },
                    ],
                )
            ],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            hosts = await quads.get_hosts()

        assert hosts[0].gpu_count == 4


class TestGetAvailable:
    async def test_returns_normalized_hostnames(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/available",
            json=["Host01", "HOST02", "host03."],
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            available = await quads.get_available()

        assert available == ["host01", "host02", "host03"]

    async def test_normalizes_end_to_server_timezone(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        expected_end = "2026-08-01T12:34"
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/available?end={expected_end}",
            json=["gpu01"],
        )
        end = datetime(2026, 8, 1, 16, 34, tzinfo=UTC)

        async with httpx.AsyncClient() as client:
            quads = QUADSClient(
                client,
                QUADS_URL,
                server_timezone="America/New_York",
            )
            available = await quads.get_available(end=end)

        request = httpx_mock.get_request()
        assert available == ["gpu01"]
        assert request is not None
        assert request.url.params["end"] == expected_end

    async def test_sends_exact_end_query_parameter(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        expected_end = "2026-08-01T16:34"
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/available?end={expected_end}",
            json=[],
        )

        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL, server_timezone="UTC")
            await quads.get_available(end=datetime(2026, 8, 1, 16, 34, 56, tzinfo=UTC))

        request = httpx_mock.get_request()
        assert request is not None
        assert set(request.url.params) == {"end"}
        assert request.url.params["end"] == expected_end

    async def test_rejects_naive_end(self) -> None:
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            with pytest.raises(ValueError, match="timezone-aware"):
                await quads.get_available(end=datetime(2026, 8, 1, 16, 34))


class TestConnectionError:
    async def test_get_hosts_raises_on_network_error(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url=f"{QUADS_URL}/api/v3/hosts",
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            with pytest.raises(QUADSConnectionError):
                await quads.get_hosts()

    async def test_get_available_raises_on_network_error(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url=f"{QUADS_URL}/api/v3/available",
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            with pytest.raises(QUADSConnectionError):
                await quads.get_available()


class TestResponseErrors:
    async def test_non_json_success_raises_connection_error(
        self,
        httpx_mock: HTTPXMock,
    ) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/hosts",
            text="<html>login required</html>",
            status_code=200,
        )

        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            with pytest.raises(QUADSConnectionError) as caught:
                await quads.get_hosts()

        assert caught.value.__cause__ is not None

    @pytest.mark.parametrize(
        ("path", "payload", "method_name"),
        [
            ("/api/v3/hosts", {"name": "gpu01"}, "get_hosts"),
            (
                "/api/v3/hosts",
                [
                    {
                        "processors": [
                            {
                                "processor_type": "GPU",
                                "vendor": "NVIDIA",
                                "product": "A100",
                            }
                        ]
                    }
                ],
                "get_hosts",
            ),
            ("/api/v3/available", [42], "get_available"),
        ],
    )
    async def test_invalid_success_shape_raises_connection_error(
        self,
        httpx_mock: HTTPXMock,
        path: str,
        payload: object,
        method_name: str,
    ) -> None:
        httpx_mock.add_response(url=f"{QUADS_URL}{path}", json=payload)

        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            method = getattr(quads, method_name)
            with pytest.raises(QUADSConnectionError, match="invalid QUADS response"):
                await method()

    @pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
    async def test_programming_error_is_not_wrapped(
        self,
        httpx_mock: HTTPXMock,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[Exception],
    ) -> None:
        httpx_mock.add_response(
            url=f"{QUADS_URL}/api/v3/available",
            json=["gpu01"],
        )

        def broken_normalizer(_hostname: str) -> str:
            raise error_type("programming defect")

        monkeypatch.setattr(
            "inference_proxy.quads.client.canonical_hostname",
            broken_normalizer,
        )
        async with httpx.AsyncClient() as client:
            quads = QUADSClient(client, QUADS_URL)
            with pytest.raises(error_type, match="programming defect"):
                await quads.get_available()
