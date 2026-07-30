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

    def test_create_replaces_existing(self) -> None:
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
