"""In-memory doubles for placement tests.

``FakeEtcd`` implements real compare-and-swap semantics (a revision per key,
zero for an absent key), so the claim tests exercise genuine races rather than
mocked return values.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from inference_proxy.discovery.etcd_client import EtcdRecord, EtcdSnapshot
from inference_proxy.discovery.registry import NodeRegistry
from inference_proxy.huggingface.artifacts import ArtifactScanResult, GGUFArtifact
from inference_proxy.models.node import (
    InferenceEngine,
    LlamaCppRuntimeRequest,
    Node,
    NodePlacement,
    NodeStatus,
)
from inference_proxy.models.quads import QUADSHost
from inference_proxy.placement.catalog import BUILTIN_PROFILES
from inference_proxy.provisioning.host_lifecycle import (
    HostLifecycleCoordinator,
    HostLifecycleLease,
)
from inference_proxy.provisioning.provisioner import (
    ProvisioningCapacityError,
    ProvisioningIdentity,
)


class FakeEtcd:
    def __init__(self) -> None:
        self.data: dict[str, tuple[bytes, int]] = {}
        self.revision = 0
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("etcd is unreachable")

    def get_snapshot(self, prefix: str | None = None) -> EtcdSnapshot:
        self._check()
        records = tuple(
            EtcdRecord(key=key.encode(), value=value, mod_revision=revision)
            for key, (value, revision) in sorted(self.data.items())
            if key.startswith(prefix or "")
        )
        return EtcdSnapshot(records, self.revision)

    def replace_if_revision(
        self,
        key: str,
        value: str | bytes,
        *,
        expected_mod_revision: int,
        lease_id: int,
    ) -> int | None:
        self._check()
        assert lease_id == 0, "claims must be persistent, never leased"
        current = self.data.get(key, (b"", 0))[1]
        if current != expected_mod_revision:
            return None
        self.revision += 1
        raw = value.encode() if isinstance(value, str) else value
        self.data[key] = (raw, self.revision)
        return self.revision

    def delete_if_revision(self, key: str, *, expected_mod_revision: int) -> bool:
        self._check()
        if self.data.get(key, (b"", 0))[1] != expected_mod_revision:
            return False
        del self.data[key]
        self.revision += 1
        return True


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeQuads:
    def __init__(self, available: list[str]) -> None:
        self.available = available
        self.fail = False

    async def get_available(self, end: datetime | None = None) -> list[str]:
        assert end is not None, "placement must ask for the whole lookahead window"
        if self.fail:
            raise ConnectionError("QUADS is down")
        return list(self.available)


class FakePoller:
    def __init__(self, hosts: list[QUADSHost]) -> None:
        self.hosts = hosts
        self.last_sync: datetime | None = datetime(2026, 9, 19, tzinfo=UTC)


class FakeArtifactIndex:
    def __init__(self, artifacts: list[GGUFArtifact]) -> None:
        self.artifacts = artifacts
        self.fail = False

    def scan(self) -> ArtifactScanResult:
        if self.fail:
            raise OSError("NFS is not mounted")
        return ArtifactScanResult(artifacts=tuple(self.artifacts))


def catalog_artifacts() -> list[GGUFArtifact]:
    artifacts: list[GGUFArtifact] = []
    for index, profile in enumerate(BUILTIN_PROFILES):
        # One hex digit per file: targets a, b, c, e; drafts d (Muse), f (Gemma).
        for ref, marker in (
            (profile.target, "abce"[index]),
            (profile.draft, "00df"[index]),
        ):
            if ref is None:
                continue
            artifacts.append(
                GGUFArtifact(
                    artifact_id=marker * 64,
                    repo_id=ref.repo_id,
                    resolved_revision=ref.revision,
                    files=(ref.filename,),
                    entrypoint=ref.filename,
                    model_alias=ref.repo_id,
                    file_sizes={ref.filename: ref.size_bytes},
                )
            )
    return artifacts


def l4_host(name: str, gpus: int = 1) -> QUADSHost:
    return QUADSHost(
        hostname=name, gpu_vendor="NVIDIA", gpu_model="AD104GL [L4]", gpu_count=gpus
    )


def a30_host(name: str) -> QUADSHost:
    return QUADSHost(
        hostname=name,
        gpu_vendor="NVIDIA",
        gpu_model="GA100GL [A30 PCIe]",
        gpu_count=1,
    )


class FakeProvisioner:
    """Behaves like NodeProvisioner at the seams placement touches."""

    def __init__(self, registry: NodeRegistry, *, capacity: int = 32) -> None:
        self.registry = registry
        self.lifecycle = HostLifecycleCoordinator()
        self.capacity = capacity
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.calls: list[dict[str, Any]] = []
        self.cleaned: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.failures: dict[str, list[str]] = {}
        self.hold = False
        self.remote_processes: dict[str, list[str]] = {}
        self.remote_check_error: Exception | None = None
        self.remote_check_errors: dict[str, Exception] = {}
        self.remote_checks: list[str] = []
        self.during_remote_check: Callable[[str], Awaitable[None]] | None = None
        self.fenced: list[str] = []

    async def try_reserve_host(self, hostname: str) -> HostLifecycleLease | None:
        return await self.lifecycle.try_acquire(hostname)

    def host_operation_in_progress(self, hostname: str) -> bool:
        return self.lifecycle.is_busy(hostname)

    async def remote_lifecycle_processes(self, hostname: str) -> list[str]:
        self.remote_checks.append(hostname)
        if self.during_remote_check is not None:
            await self.during_remote_check(hostname)
        if self.remote_check_error is not None:
            raise self.remote_check_error
        if hostname in self.remote_check_errors:
            raise self.remote_check_errors[hostname]
        return list(self.remote_processes.get(hostname, []))

    async def cancel_active_provision(self, hostname: str) -> None:
        """Explicit cancel: stops the task and, like the real one, the remote worker."""
        self.fenced.append(hostname)
        self.remote_processes.pop(hostname, None)
        task = self.tasks.get(hostname)
        if task is not None and not task.done():
            task.cancel()

    async def cleanup_stale_node(self, hostname: str) -> None:
        self.cleaned.append(hostname)
        self.registry.remove(hostname)

    def fire_background(
        self,
        coro: Coroutine[object, object, None],
        *,
        provisioning_hostname: str | None = None,
        provisioning_identity: ProvisioningIdentity | None = None,
    ) -> asyncio.Task[None]:
        assert provisioning_hostname is not None
        assert provisioning_identity is not None
        active = sum(1 for task in self.tasks.values() if not task.done())
        if active >= self.capacity:
            raise ProvisioningCapacityError(active=active, limit=self.capacity)
        task = asyncio.create_task(coro)
        self.tasks[provisioning_hostname] = task
        return task

    async def provision(
        self,
        hostname: str,
        *,
        managed: bool,
        engine: InferenceEngine,
        artifact_id: str,
        llamacpp_request: LlamaCppRuntimeRequest,
        lifecycle_lease: HostLifecycleLease,
        placement: NodePlacement,
    ) -> None:
        assert lifecycle_lease.belongs_to(self.lifecycle, hostname)
        self.calls.append(
            {
                "hostname": hostname,
                "artifact_id": artifact_id,
                "request": llamacpp_request,
                "placement": placement,
                "managed": managed,
                "engine": engine,
            }
        )
        try:
            if self.hold:
                await self.gates.setdefault(hostname, asyncio.Event()).wait()
            errors = self.failures.get(hostname) or []
            if errors:
                self.registry.add(
                    _node(hostname, NodeStatus.FAILED, placement=placement)
                )
                raise RuntimeError(errors.pop(0))
            self.registry.add(_node(hostname, NodeStatus.HEALTHY, placement=placement))
        finally:
            lifecycle_lease.release()

    async def drain(self) -> None:
        pending = [task for task in self.tasks.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _node(
    hostname: str,
    status: NodeStatus,
    *,
    placement: NodePlacement | None = None,
    managed: bool = True,
    owner: str = "",
    self_setup: bool = False,
) -> Node:
    return Node(
        node_id=hostname,
        endpoint=f"http://{hostname}:8000",
        status=status,
        engine=InferenceEngine.LLAMA_CPP,
        managed=managed and not self_setup,
        self_setup=self_setup,
        owner=owner,
        placement=placement,
    )


node = _node
