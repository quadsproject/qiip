"""Subprocess probe for provisioning cancellation during app shutdown."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr

from inference_proxy.config.settings import (
    AdminSettings,
    GatewaySettings,
    HuggingFaceSettings,
    Settings,
)
from inference_proxy.main import create_app


def _health_worker(*args: object) -> None:
    stop_event = args[2]
    if not isinstance(stop_event, threading.Event):
        raise TypeError("health worker did not receive a threading.Event")
    stop_event.wait(timeout=1)


async def _run_inline[T](function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return function(*args, **kwargs)


async def _main(cache_dir: Path) -> None:
    order: list[str] = []
    etcd = MagicMock()
    etcd.get_prefix.return_value = []
    etcd.prefix = "/nodes/"
    etcd.put.side_effect = lambda *_args: order.append("state-write")
    etcd.close.side_effect = lambda: order.append("etcd-close")
    settings = Settings(
        gateway=GatewaySettings(host="127.0.0.1", port=9999),
        admin=AdminSettings(
            username="test-admin",
            password=SecretStr("test-password"),
        ),
        huggingface=HuggingFaceSettings(
            cache_dir=str(cache_dir),
            nfs_export="storage.example:/exports/huggingface",
        ),
    )
    if "graceful_shutdown_timeout" in type(settings.gateway).model_fields:
        gateway = settings.gateway.model_copy(update={"graceful_shutdown_timeout": 0})
        settings = settings.model_copy(update={"gateway": gateway})

    started = asyncio.Event()

    async def blocked_provision(
        hostname: str, *, managed: bool = True, model: str | None = None
    ) -> None:
        del hostname, managed, model
        started.set()
        await asyncio.Event().wait()

    task: asyncio.Task[None] | None = None
    try:
        with (
            patch("inference_proxy.main.EtcdClient", return_value=etcd),
            patch("inference_proxy.main.EtcdWatcher"),
            patch("inference_proxy.main.run_health_checker", new=_health_worker),
            patch(
                "inference_proxy.provisioning.provisioner.asyncio.to_thread",
                new=_run_inline,
            ),
        ):
            app = create_app(settings=settings)
            async with app.router.lifespan_context(app):
                provisioner = app.state.provisioner
                provisioner._provision = AsyncMock(side_effect=blocked_provision)
                provisioner._log_buffer.create("localhost")
                task = provisioner.fire_background(
                    provisioner.provision("localhost"),
                    provisioning_hostname="localhost",
                )
                await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert task is not None
    assert task.cancelled() is True
    assert provisioner._background_tasks == set()
    assert provisioner._provisioning_tasks == {}
    assert order.index("state-write") < order.index("etcd-close")
    state_payloads = [
        json.loads(call.args[1])
        for call in etcd.put.call_args_list
        if str(call.args[0]).startswith("/provisioning/")
    ]
    assert state_payloads[-1]["current_step"] == "failed"
    assert state_payloads[-1]["failed_step"] == "cancelled"


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1])))
