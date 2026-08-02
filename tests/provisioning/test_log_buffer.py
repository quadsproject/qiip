"""Unit tests for ProvisioningLogBuffer."""

from __future__ import annotations

import asyncio

import pytest

from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer


class TestBasicOperations:
    def test_create_and_has(self) -> None:
        buf = ProvisioningLogBuffer()
        assert not buf.has("host1")
        buf.create("host1")
        assert buf.has("host1")

    def test_append_and_get_entries(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "hello")
        buf.append("host1", "error", "boom", stream="stderr")
        entries = buf.get_entries("host1")
        assert len(entries) == 2
        assert entries[0]["level"] == "info"
        assert entries[0]["msg"] == "hello"
        assert entries[0]["stream"] is None
        assert entries[1]["stream"] == "stderr"

    def test_append_to_unknown_host_is_noop(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.append("ghost", "info", "nope")
        assert not buf.has("ghost")

    def test_get_entries_unknown_host_returns_empty(self) -> None:
        buf = ProvisioningLogBuffer()
        assert buf.get_entries("ghost") == []

    def test_create_starts_fresh_operation_log(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "old")
        buf.create("host1")
        assert buf.get_entries("host1") == []


class TestIterFrom:
    @pytest.mark.asyncio
    async def test_iter_completed_buffer(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "line1")
        buf.append("host1", "info", "line2")
        buf.mark_complete("host1")

        collected = []
        async for pos, entry in buf.iter_from("host1"):
            collected.append((pos, entry["msg"]))
        assert collected == [(0, "line1"), (1, "line2")]

    @pytest.mark.asyncio
    async def test_iter_unknown_host_yields_nothing(self) -> None:
        buf = ProvisioningLogBuffer()
        collected = []
        async for _pos, entry in buf.iter_from("ghost"):
            collected.append(entry)
        assert collected == []

    @pytest.mark.asyncio
    async def test_iter_from_offset(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "a")
        buf.append("host1", "info", "b")
        buf.append("host1", "info", "c")
        buf.mark_complete("host1")

        collected = []
        async for _pos, entry in buf.iter_from("host1", pos=1):
            collected.append(entry["msg"])
        assert collected == ["b", "c"]

    @pytest.mark.asyncio
    async def test_iter_blocks_until_new_entry(self) -> None:
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "first")

        collected: list[str] = []

        async def consumer() -> None:
            async for _pos, entry in buf.iter_from("host1"):
                collected.append(entry["msg"])

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.05)
        assert collected == ["first"]

        buf.append("host1", "info", "second")
        await asyncio.sleep(0.05)
        assert collected == ["first", "second"]

        buf.mark_complete("host1")
        await asyncio.sleep(0.05)
        assert task.done()

    @pytest.mark.asyncio
    async def test_replacing_unfinished_buffer_completes_existing_consumer(
        self,
    ) -> None:
        """C3: direct replacement cannot strand the displaced generation."""
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "old operation")
        first_seen = asyncio.Event()
        collected: list[str] = []

        async def consume_old_generation() -> None:
            async for _pos, entry in buf.iter_from("host1"):
                collected.append(entry["msg"])
                first_seen.set()

        consumer = asyncio.create_task(consume_old_generation())
        await asyncio.wait_for(first_seen.wait(), timeout=1)

        buf.create("host1")
        await asyncio.wait_for(consumer, timeout=1)

        assert collected == ["old operation"]
        assert buf.get_entries("host1") == []

    @pytest.mark.asyncio
    async def test_sequential_operation_swap_closes_old_and_keeps_new_fresh(
        self,
    ) -> None:
        """The lease-serialized production boundary retains separate logs."""
        buf = ProvisioningLogBuffer()
        buf.create("host1")
        buf.append("host1", "info", "provision complete")
        buf.mark_complete("host1")

        old_entries = []
        async for _pos, entry in buf.iter_from("host1"):
            old_entries.append(entry["msg"])

        buf.create("host1")
        buf.append("host1", "info", "teardown started")
        buf.mark_complete("host1")
        new_entries = []
        async for _pos, entry in buf.iter_from("host1"):
            new_entries.append(entry["msg"])

        assert old_entries == ["provision complete"]
        assert new_entries == ["teardown started"]


class TestRetention:
    @pytest.mark.asyncio
    async def test_oversized_line_is_truncated_with_visible_marker(self) -> None:
        buf = ProvisioningLogBuffer(
            max_entries_per_host=10,
            max_bytes_per_host=100,
            max_entry_bytes=20,
        )
        buf.create("host1")
        buf.append("host1", "info", "x" * 100)
        buf.mark_complete("host1")

        replay = []
        async for _pos, entry in buf.iter_from("host1"):
            replay.append(entry["msg"])

        assert replay == ["xxxx ... [truncated]"]
        assert len(replay[0].encode("utf-8")) == 20

    @pytest.mark.asyncio
    async def test_log_retains_latest_entries_with_absolute_positions(self) -> None:
        """S9: a discarded prefix is bounded and visible during replay."""
        buf = ProvisioningLogBuffer(
            max_entries_per_host=3,
            max_bytes_per_host=1_000,
            max_entry_bytes=100,
        )
        buf.create("host1")
        for number in range(5):
            buf.append("host1", "info", f"line {number}")
        buf.mark_complete("host1")

        replay = []
        async for pos, entry in buf.iter_from("host1"):
            replay.append((pos, entry["level"], entry["msg"]))

        assert replay == [
            (1, "warning", "2 earlier log entries evicted by retention limits"),
            (2, "info", "line 2"),
            (3, "info", "line 3"),
            (4, "info", "line 4"),
        ]

    @pytest.mark.asyncio
    async def test_live_consumer_reports_gap_and_continues_after_prefix_eviction(
        self,
    ) -> None:
        """Absolute cursors prevent a slow iterator hanging on shifted indexes."""
        buf = ProvisioningLogBuffer(
            max_entries_per_host=2,
            max_bytes_per_host=1_000,
            max_entry_bytes=100,
        )
        buf.create("host1")
        buf.append("host1", "info", "line 0")
        first_seen = asyncio.Event()
        resume = asyncio.Event()
        collected: list[str] = []

        async def consume() -> None:
            async for _pos, entry in buf.iter_from("host1"):
                collected.append(entry["msg"])
                if entry["msg"] == "line 0":
                    first_seen.set()
                    await resume.wait()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(first_seen.wait(), timeout=1)
        buf.append("host1", "info", "line 1")
        buf.append("host1", "info", "line 2")
        buf.append("host1", "info", "line 3")
        buf.mark_complete("host1")
        resume.set()
        await asyncio.wait_for(consumer, timeout=1)

        assert collected == [
            "line 0",
            "1 earlier log entry evicted by retention limits",
            "line 2",
            "line 3",
        ]

    @pytest.mark.asyncio
    async def test_entry_and_host_byte_limits_are_visible(self) -> None:
        buf = ProvisioningLogBuffer(
            max_entries_per_host=10,
            max_bytes_per_host=32,
            max_entry_bytes=16,
        )
        buf.create("host1")
        buf.append("host1", "info", "x" * 100)
        buf.append("host1", "info", "y" * 16)
        buf.append("host1", "info", "z" * 16)
        buf.mark_complete("host1")

        replay = []
        async for _pos, entry in buf.iter_from("host1"):
            replay.append(entry["msg"])

        assert replay == [
            "1 earlier log entry evicted by retention limits",
            "y" * 16,
            "z" * 16,
        ]

    @pytest.mark.asyncio
    async def test_completed_hosts_evict_oldest_and_attached_reader_terminates(
        self,
    ) -> None:
        """Completed-host eviction closes readers through the shared path."""
        buf = ProvisioningLogBuffer(max_completed_hosts=2)
        buf.create("host1")
        buf.append("host1", "info", "host1")
        reader_paused = asyncio.Event()
        reader_resume = asyncio.Event()

        async def consume_host1() -> None:
            async for _pos, _entry in buf.iter_from("host1"):
                reader_paused.set()
                await reader_resume.wait()

        consumer = asyncio.create_task(consume_host1())
        await asyncio.wait_for(reader_paused.wait(), timeout=1)
        buf.mark_complete("host1")

        for hostname in ("host2", "host3"):
            buf.create(hostname)
            buf.append(hostname, "info", hostname)
            buf.mark_complete(hostname)

        assert not buf.has("host1")
        assert buf.has("host2")
        assert buf.has("host3")

        reader_resume.set()
        await asyncio.wait_for(consumer, timeout=1)

    def test_active_host_is_never_evicted_for_completed_retention(self) -> None:
        buf = ProvisioningLogBuffer(max_completed_hosts=1)
        buf.create("active")
        buf.append("active", "info", "still running")
        for hostname in ("done1", "done2"):
            buf.create(hostname)
            buf.mark_complete(hostname)

        assert buf.has("active")
        assert not buf.has("done1")
        assert buf.has("done2")
