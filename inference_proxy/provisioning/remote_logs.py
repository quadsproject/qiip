"""Retrieve node-recorded attempts without rerunning an uncertain command."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from weakref import WeakValueDictionary

from inference_proxy.config.settings import ProvisioningSettings
from inference_proxy.provisioning.diagnostics import (
    JOURNAL_SOURCES,
    SOURCE_NAMES,
    journal_until,
)
from inference_proxy.provisioning.log_buffer import ProvisioningLogBuffer
from inference_proxy.provisioning.log_store import AttemptLogStore, timestamp
from inference_proxy.provisioning.ssh_client import (
    RemoteCommandError,
    SSHClient,
    SSHConnectionError,
)


class RemoteLogCollector:
    def __init__(
        self,
        ssh: SSHClient,
        store: AttemptLogStore,
        buffer: ProvisioningLogBuffer,
        settings: ProvisioningSettings,
    ) -> None:
        self.ssh = ssh
        self.store = store
        self.buffer = buffer
        self.settings = settings
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._diagnostic_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )

    def record_failure(self, attempt_id: str, stage: str, error: BaseException) -> None:
        """Save the original failure before making any diagnostic SSH requests."""
        attempt = self.store.get(attempt_id)
        if attempt.get("failure"):
            return
        cause: BaseException | None = error
        command_error: RemoteCommandError | None = None
        while cause is not None:
            if isinstance(cause, RemoteCommandError):
                command_error = cause
                break
            cause = cause.__cause__
        now = datetime.now(UTC)
        command = attempt.get("active_command")
        if command_error and (not command or command_error.command != command["stage"]):
            command = dict(
                stage=stage,
                phase_id=None,
                sha256=hashlib.sha256(command_error.command.encode()).hexdigest(),
            )
        if command:
            phase = attempt.get("remote_phases", {}).get(command["phase_id"], {})
            command = {
                **command,
                **{
                    key: phase[key]
                    for key in (
                        "started_at",
                        "finished_at",
                        "duration_seconds",
                        "exit_status",
                    )
                    if key in phase
                },
            }
        exit_status = command_error.exit_status if command_error else None
        failure = dict(
            failed_stage=stage,
            original_error=str(error)[:8192],
            error_type=type(error).__name__,
            started_at=attempt["started_at"],
            failed_at=now.isoformat(),
            duration_seconds=max(
                0, (now - datetime.fromisoformat(attempt["started_at"])).total_seconds()
            ),
            command=command,
            exit_code=exit_status
            if exit_status is not None and exit_status >= 0
            else None,
            signal=(command_error.exit_signal if command_error else None)
            or (-exit_status if exit_status is not None and exit_status < 0 else None),
        )
        self.store.update(
            attempt_id,
            failure=failure,
            failure_summary=f"{stage}: {error}",
            diagnostics=dict(
                sources={
                    name: dict(
                        status="unavailable", deferred=True, reason="Collection pending"
                    )
                    for name in SOURCE_NAMES
                }
            ),
        )

    async def diagnose(self, attempt_id: str) -> None:
        """Best-effort bounded collection; never substitute a collector error."""
        lock = self._diagnostic_locks.setdefault(attempt_id, asyncio.Lock())
        try:
            async with asyncio.timeout(self.settings.diagnostics_timeout):
                async with lock:
                    attempt = self.store.get(attempt_id)
                    if not attempt.get("failure"):
                        return
                    sources = attempt.get("diagnostics", {}).get("sources", {})
                    for name in JOURNAL_SOURCES:
                        source = sources.get(name)
                        if source is not None and source.get(
                            "window_until"
                        ) != journal_until(attempt["failure"]):
                            source["deferred"] = True
                    if sources:
                        self.store.update(
                            attempt_id,
                            diagnostics={**attempt["diagnostics"], "sources": sources},
                        )
                    if sources and not any(s.get("deferred") for s in sources.values()):
                        return
                    async with self._locks.setdefault(attempt_id, asyncio.Lock()):
                        page = await self._request(
                            attempt_id,
                            "diagnose",
                            failure=attempt["failure"],
                            mount_point=self.settings.nfs_mount_point,
                            diagnostics_source_timeout=self.settings.diagnostics_source_timeout,
                            # Reserve time for transfer/ingestion inside the gateway budget.
                            diagnostics_timeout=self.settings.diagnostics_timeout * 0.8,
                            diagnostics_source_max_bytes=self.settings.diagnostics_source_max_bytes,
                        )
                        await self._ingest(attempt_id, page)
                    if page["has_more"]:
                        await self.collect(attempt_id)
        except Exception as exc:
            # Avoid command strings (which can contain credentials) in failures.
            reason = (
                "Collection deadline exceeded"
                if isinstance(exc, TimeoutError)
                else f"Collection unavailable ({type(exc).__name__})"
            )
            attempt = self.store.get(attempt_id)
            diagnostics = attempt.get("diagnostics", {})
            for source in diagnostics.get("sources", {}).values():
                if source.get("deferred"):
                    source.update(
                        status="timed_out"
                        if isinstance(exc, TimeoutError)
                        else "unavailable",
                        reason=reason,
                    )
            self.store.update(attempt_id, diagnostics=diagnostics)

    def config(self, attempt_id: str) -> dict[str, Any]:
        attempt = self.store.get(attempt_id)
        return dict(
            attempt_id=attempt_id,
            hostname=attempt["hostname"],
            engine=attempt["engine"],
            model=attempt["model"],
            bundle_version=attempt["bundle_version"],
            root=self.settings.log_remote_root,
            max_bytes=self.settings.log_remote_max_bytes,
            attempt_max_bytes=self.settings.log_remote_attempt_max_bytes,
            max_attempts=self.settings.log_remote_max_attempts,
            retention_days=self.settings.log_remote_retention_days,
            max_record_bytes=self.settings.log_max_entry_bytes,
            health_timeout=self.settings.health_poll_timeout,
            inactivity_timeout=self.ssh.inactivity_timeout,
            after=attempt["remote_cursor"],
        )

    async def _run_recorder(
        self, attempt_id: str, action: str, **extra: Any
    ) -> dict[str, Any]:
        config = self.config(attempt_id)
        config.update(extra)
        return await self._run_recorder_with_config(config, action)

    async def _request(
        self, attempt_id: str, action: str, **extra: Any
    ) -> dict[str, Any]:
        page = await self._run_recorder(attempt_id, action, **extra)
        if page.get("unavailable"):
            raise SSHConnectionError(
                self.config(attempt_id)["hostname"],
                str(page["unavailable"]),
            )
        return page

    def _host_config(self, hostname: str) -> dict[str, Any]:
        """Recorder config for host-level actions that have no attempt row."""
        return dict(
            hostname=hostname,
            root=self.settings.log_remote_root,
            max_bytes=self.settings.log_remote_max_bytes,
            attempt_max_bytes=self.settings.log_remote_attempt_max_bytes,
            max_attempts=self.settings.log_remote_max_attempts,
            retention_days=self.settings.log_remote_retention_days,
            max_record_bytes=self.settings.log_max_entry_bytes,
            health_timeout=self.settings.health_poll_timeout,
            inactivity_timeout=self.ssh.inactivity_timeout,
            after=0,
        )

    async def _run_recorder_with_config(
        self, config: dict[str, Any], action: str
    ) -> dict[str, Any]:
        command = (
            "printf %s "
            + shlex.quote(json.dumps(config))
            + " | python3 common/provision-logs.py "
            + action
        )
        try:
            stdout, _stderr, _status = await self.ssh.run(
                config["hostname"],
                command,
                timeout=self.settings.diagnostics_timeout
                if action == "diagnose"
                else 30,
                log_label=f"provisioning log recorder ({action})",
            )
        except RemoteCommandError as exc:
            # The transport command can contain environment secrets. Do not
            # expose it in summaries, downloads, or structured application logs.
            raise SSHConnectionError(
                config["hostname"],
                f"Node log recorder exited with status {exc.exit_status}",
            ) from None
        try:
            page: dict[str, Any] = json.loads(stdout)
            if not isinstance(page, dict):
                raise ValueError("expected object")
            return page
        except ValueError as exc:
            raise SSHConnectionError(
                config["hostname"], f"Node logs unavailable: {exc}"
            ) from None

    async def host_active(self, hostname: str) -> dict[str, Any]:
        """Return whether the node has any live phase for *hostname*.

        The node is the authority on its own process groups: this probe
        re-checks recorded pids so an orphan left by a dead recorder is closed
        (and does not block the host forever) while a genuinely live or
        surviving phase is reported as active. The recorder ``active`` action
        opens the node store, which creates the log root/database and prunes
        expired attempts as a side effect, so this is not a no-write probe.
        """
        try:
            return await self._run_recorder_with_config(
                self._host_config(hostname), "active"
            )
        except (SSHConnectionError, TimeoutError) as exc:
            return {"unreachable": True, "error": str(exc)}

    async def _ingest(
        self, attempt_id: str, page: dict[str, Any]
    ) -> list[dict[str, Any]]:
        added = await asyncio.to_thread(self._ingest_page, attempt_id, page)
        for entry in added:
            host = entry["hostname"]
            if self.buffer.attempts.get(host) == attempt_id:
                self.buffer.append(
                    host,
                    entry["level"],
                    entry["msg"],
                    stream=entry["stream"],
                    persist=False,
                )
        return added

    def _ingest_page(
        self, attempt_id: str, page: dict[str, Any]
    ) -> list[dict[str, Any]]:
        added = self.store.append_many(
            attempt_id,
            [
                dict(
                    msg=record["msg"],
                    level=record["level"],
                    source=record["source"],
                    stage=record["stage"],
                    ts=record["ts"],
                    remote_seq=record["seq"],
                    stream=record["stream"],
                )
                for record in page["records"]
            ],
        )
        remote = page["attempt"]
        diagnostic_fields: dict[str, Any] = {}
        current = self.store.get(attempt_id)
        failure = current.get("failure")
        if failure and failure.get("command"):
            phase = remote.get("phases", {}).get(failure["command"].get("phase_id"), {})
            failure["command"].update(
                {
                    key: phase[key]
                    for key in (
                        "started_at",
                        "finished_at",
                        "duration_seconds",
                        "exit_status",
                    )
                    if key in phase
                }
            )
            diagnostic_fields["failure"] = failure
        if "diagnostics" in remote:
            previous = current.get("diagnostics", {}).get("sources", {})
            sources = {}
            for name, source in remote["diagnostics"]["sources"].items():
                detail = dict(source)
                first, last = (
                    detail.pop("first_seq", None),
                    detail.pop("last_seq", None),
                )
                if first is not None:
                    detail.update(remote_first_seq=first, remote_last_seq=last)
                    old = previous.get(name, {})
                    local = [
                        entry["seq"]
                        for entry in added
                        if entry["source"] == "diagnostics." + name
                        and first <= entry["remote_seq"] <= last
                    ]
                    if (
                        old.get("remote_first_seq") == first
                        and old.get("first_seq") is not None
                    ):
                        local.extend([old["first_seq"], old["last_seq"]])
                    if local:
                        detail.update(first_seq=min(local), last_seq=max(local))
                    if last >= current["remote_cursor"]:
                        detail.update(
                            status="unavailable",
                            deferred=True,
                            reason="Snapshot recorded on node; retrieval pending",
                        )
                    elif first < remote["dropped_records"]:
                        detail.update(
                            status="truncated",
                            truncated=True,
                            reason="Diagnostic records evicted by node retention",
                        )
                sources[name] = detail
            diagnostic_fields["diagnostics"] = {
                **remote["diagnostics"],
                "sources": sources,
            }
        issues = []
        for issue in remote["issues"]:
            message = "Node: " + issue
            if message not in issues:
                issues.append(message[:2048])
        if remote["dropped_records"]:
            message = "Node retention evicted records; consult remote_dropped_records"
            if message not in issues:
                issues.append(message)
        self.store.update(
            attempt_id,
            remote_status=remote["status"],
            remote_phases=remote.get("phases", {}),
            remote_sources=remote["sources"],
            remote_dropped_records=remote["dropped_records"],
            issues=issues[-32:],
            **diagnostic_fields,
        )
        # An empty retained suffix still needs an explicit gap and cursor advance.
        if (
            not page["has_more"]
            and remote["next_seq"] > self.store.get(attempt_id)["remote_cursor"]
        ):
            self.store.append(
                attempt_id,
                "Remote collection reached an unavailable suffix",
                source="collector",
                level="warning",
                remote_seq=remote["next_seq"] - 1,
            )
            self.store.issue(attempt_id, "Remote collection has an unavailable suffix")
        return added

    async def collect(
        self, attempt_id: str, *, finish: bool = False, cancel: bool = False
    ) -> list[dict[str, Any]]:
        lock = self._locks.setdefault(attempt_id, asyncio.Lock())
        async with lock:
            added = []
            action = "cancel" if cancel else "finish" if finish else "read"
            try:
                # Bounded pages and a finite collection budget even for a noisy host.
                for _ in range(128):
                    page = await self._request(attempt_id, action)
                    action = "read"
                    added.extend(await self._ingest(attempt_id, page))
                    if not page["has_more"]:
                        return added
                self.store.issue(
                    attempt_id, "Remote collection page limit reached; collect again"
                )
                return added
            except (SSHConnectionError, TimeoutError) as exc:
                self.store.issue(
                    attempt_id, "Remote logs unavailable: " + str(exc), source="remote"
                )
                raise

    async def follow(self, attempt_id: str) -> None:
        """Poll through one SSH connection, reconnecting only after a failure."""
        hostname = (await asyncio.to_thread(self.store.get, attempt_id))["hostname"]
        while True:
            try:
                async with self.ssh.connection(hostname):
                    while True:
                        await self.collect(attempt_id)
                        await asyncio.sleep(self.settings.log_poll_interval)
            except (SSHConnectionError, TimeoutError):
                await asyncio.sleep(self.settings.log_poll_interval)

    async def run(
        self,
        hostname: str,
        command: str,
        *,
        stage: str,
        timeout: float | None = None,
        engine_log: str | None = None,
    ) -> AsyncGenerator[tuple[str, str]]:
        attempt_id = self.buffer.attempts[hostname]
        phase = uuid.uuid4().hex
        parse_offset = self.store.get(attempt_id)["next_seq"]
        self.store.update(attempt_id, stage=stage)
        duration = timeout or self.ssh.command_timeout
        # Record launch intent before touching SSH. If launch acknowledgement is
        # lost, only retrieve this identity; never repeat setup or engine launch.
        attempt = self.store.get(attempt_id)
        self.store.update(
            attempt_id,
            phases=[*attempt.get("phases", []), phase],
            active_command=dict(phase_id=phase, stage=stage, started_at=timestamp()),
        )
        try:
            page = await self._request(
                attempt_id,
                "launch",
                phase=phase,
                stage=stage,
                command=command,
                timeout=duration,
                engine_log=engine_log,
            )
        except (SSHConnectionError, TimeoutError):
            page = None
        failures = 0
        failure_tail: deque[str] = deque(maxlen=20)
        deadline = asyncio.get_running_loop().time() + duration + 30
        while True:
            try:
                records = (
                    await self._ingest(attempt_id, page)
                    if page is not None
                    else await self.collect(attempt_id)
                )
                page = None
                failures = 0
                # API collection and the health follower may advance the remote
                # cursor concurrently. Parse committed rows using our own offset.
                parsed = await asyncio.to_thread(
                    self.store.read, attempt_id, after=parse_offset
                )
                parse_offset = parsed["next_offset"]
                records = parsed["records"]
                for entry in records:
                    if entry["source"] in {
                        stage + ".stdout",
                        stage + ".stderr",
                    }:
                        failure_tail.append(entry["msg"])
                        yield entry["stream"], entry["msg"]
                remote = (
                    self.store.get(attempt_id).get("remote_phases", {}).get(phase, {})
                )
                if remote.get("status") == "complete" and (
                    engine_log is not None or not remote.get("recording")
                ):
                    # Drain all pages before reporting command completion.
                    if records:
                        continue
                    exit_status = remote["exit_status"]
                    if exit_status:
                        raise RemoteCommandError(
                            hostname,
                            stage,
                            exit_status,
                            stderr="\n".join(failure_tail)
                            or "See persisted attempt logs",
                        )
                    return
            except (SSHConnectionError, TimeoutError):
                failures += 1
                if failures > self.settings.log_reconnect_attempts:
                    raise
            if asyncio.get_running_loop().time() >= deadline:
                self.store.issue(
                    attempt_id,
                    "Command completion unavailable after deadline",
                    source=stage,
                )
                raise TimeoutError(
                    f"Node {stage} completion unavailable; inspect attempt {attempt_id}"
                )
            await asyncio.sleep(self.settings.log_poll_interval)
