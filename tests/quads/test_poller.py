"""Unit tests for QUADSPoller background polling and caching."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from inference_proxy.models.quads import QUADSHost
from inference_proxy.quads.client import QUADSClient, QUADSConnectionError
from inference_proxy.quads.poller import QUADSPoller

HOST_A = QUADSHost(hostname="gpu01", gpu_vendor="NVIDIA", gpu_model="A100", gpu_count=4)
HOST_B = QUADSHost(hostname="gpu02", gpu_vendor="AMD", gpu_model="MI300X", gpu_count=8)


def _mock_client(
    hosts: list[QUADSHost] | None = None,
    available: list[str] | None = None,
) -> AsyncMock:
    """Build a mock QUADSClient with configurable returns."""
    client = AsyncMock(spec=QUADSClient)
    client.get_hosts.return_value = hosts or []
    client.get_available.return_value = available or []
    return client


class TestFreshPoller:
    def test_empty_hosts(self) -> None:
        poller = QUADSPoller(_mock_client(), poll_interval=60)
        assert poller.hosts == []

    def test_empty_available(self) -> None:
        poller = QUADSPoller(_mock_client(), poll_interval=60)
        assert poller.available_hostnames == []

    def test_last_sync_is_none(self) -> None:
        poller = QUADSPoller(_mock_client(), poll_interval=60)
        assert poller.last_sync is None

    def test_consecutive_failures_zero(self) -> None:
        poller = QUADSPoller(_mock_client(), poll_interval=60)
        assert poller.consecutive_failures == 0


class TestPollOnce:
    async def test_populates_cache(self) -> None:
        client = _mock_client(hosts=[HOST_A, HOST_B], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)

        await poller._poll_once()

        assert poller.hosts == [HOST_A, HOST_B]
        assert poller.available_hostnames == ["gpu01"]

    async def test_success_sets_last_sync(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)
        before = datetime.now(tz=UTC)

        await poller._poll_once()

        assert poller.last_sync is not None
        assert poller.last_sync >= before

    async def test_success_resets_failures(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)
        # Simulate a prior failure
        poller._consecutive_failures = 3

        await poller._poll_once()

        assert poller.consecutive_failures == 0


class TestPollFailure:
    async def test_retains_cached_data(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)
        await poller._poll_once()

        # Now make it fail
        client.get_hosts.side_effect = QUADSConnectionError("down")
        await poller._poll_once()

        assert poller.hosts == [HOST_A]
        assert poller.available_hostnames == ["gpu01"]

    async def test_increments_consecutive_failures(self) -> None:
        client = _mock_client()
        client.get_hosts.side_effect = QUADSConnectionError("down")
        poller = QUADSPoller(client, poll_interval=60)

        await poller._poll_once()
        assert poller.consecutive_failures == 1

        await poller._poll_once()
        assert poller.consecutive_failures == 2

    async def test_last_sync_unchanged_on_failure(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)
        await poller._poll_once()
        sync_after_success = poller.last_sync

        client.get_hosts.side_effect = QUADSConnectionError("down")
        await poller._poll_once()

        assert poller.last_sync == sync_after_success


class TestLifecycle:
    async def test_start_creates_task(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)

        poller.start()
        try:
            assert poller._task is not None
            assert not poller._task.done()
        finally:
            await poller.stop()

    async def test_stop_cancels_cleanly(self) -> None:
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=60)

        poller.start()
        await poller.stop()

        assert poller._task is None

    async def test_initial_poll_before_sleep(self) -> None:
        """Cache is populated immediately after start, before interval sleep."""
        client = _mock_client(hosts=[HOST_A], available=["gpu01"])
        poller = QUADSPoller(client, poll_interval=300)

        poller.start()
        # Give the task a moment to run the initial _poll_once
        await asyncio.sleep(0.05)

        try:
            assert poller.hosts == [HOST_A]
            assert poller.available_hostnames == ["gpu01"]
        finally:
            await poller.stop()
