"""Verify that a freshly started vLLM server is this attempt's usable runtime.

A healthy ``/health`` response only proves that something answered on the
port. These helpers collect the evidence the gateway needs before a node is
allowed to route traffic:

* the running ``vllm serve`` process that owns the port, its age relative to
  the provisioning attempt, and (when available) the generation it runs from;
* the runtime version and model cards the server itself reports;
* one bounded inference request that exercises the model's task.

The inference probe deliberately does not check the generated text. It only
requires a well-formed completion, so a valid response for any model passes
while a server that answers health but cannot serve inference fails.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import httpx

MODELS_PATH = "/v1/models"
VERSION_PATH = "/version"
CHAT_PATH = "/v1/chat/completions"
COMPLETION_PATH = "/v1/completions"

# Statuses that mean this server does not implement the selected OpenAI
# task, so the other supported endpoint may still be valid.
_UNSUPPORTED_TASK_STATUSES = frozenset({400, 404, 405, 422})
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PROBE_PROMPT = "ping"


class VllmServingError(Exception):
    """Raised when the started server cannot prove usable inference."""


@dataclass(frozen=True)
class VllmModelCard:
    """One model entry from the server's ``/v1/models`` response."""

    model_id: str
    root: str | None = None
    max_model_len: int | None = None


@dataclass(frozen=True)
class VllmServerIdentity:
    """Version and model identity reported by the server itself."""

    version: str
    models: tuple[VllmModelCard, ...]


@dataclass(frozen=True)
class VllmProcess:
    """One live ``vllm serve`` process observed with ``ps``."""

    pid: int
    age_seconds: int
    model: str
    port: int
    executable: str
    arguments: tuple[str, ...]

    def runs_from(self, runtime_path: str) -> bool:
        """Whether any launch argument lives under the selected generation."""
        root = PurePosixPath(runtime_path)
        for argument in self.arguments:
            candidate = PurePosixPath(argument)
            if candidate == root or candidate.is_relative_to(root):
                return True
        return False


@dataclass(frozen=True)
class VllmInferenceProbe:
    """Bounded inference request result; the generated text is not asserted."""

    task: str
    endpoint: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


def normalize_model_identity(value: str) -> str:
    """Compare model ids without treating path formatting as a different model."""
    candidate = value.strip()
    if candidate.startswith("/"):
        return posixpath.normpath(candidate)
    return candidate.rstrip("/")


def model_identity_matches(observed: str, expected: str) -> bool:
    """Whether two model ids name the same model."""
    return normalize_model_identity(observed) == normalize_model_identity(expected)


def parse_vllm_processes(output: str) -> tuple[VllmProcess, ...]:
    """Parse ``ps -eo pid=,etimes=,args=`` output into vLLM serve processes."""
    processes: list[VllmProcess] = []
    for raw in output.splitlines():
        parts = raw.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, age_text, arguments_text = parts
        try:
            pid = int(pid_text)
            age = int(age_text)
        except ValueError:
            continue
        arguments = tuple(arguments_text.split())
        try:
            serve_index = arguments.index("serve")
        except ValueError:
            continue
        executable = _vllm_serve_executable(arguments, serve_index)
        if executable is None:
            continue
        model = arguments[serve_index + 1] if serve_index + 1 < len(arguments) else ""
        port = _port_argument(arguments)
        if port is None:
            continue
        processes.append(
            VllmProcess(
                pid=pid,
                age_seconds=age,
                model=model,
                port=port,
                executable=executable,
                arguments=arguments,
            )
        )
    return tuple(processes)


def select_process(
    processes: tuple[VllmProcess, ...],
    *,
    port: int,
    expected_model: str,
    runtime_path: str | None,
    attempt_elapsed_seconds: float,
    slack_seconds: float,
) -> VllmProcess:
    """Choose the one process that owns the port and belongs to this attempt.

    A server that predates the attempt is a stale listener: it may answer
    health and even inference, but it is not the runtime this attempt started,
    so it must not satisfy readiness.
    """
    candidates = tuple(process for process in processes if process.port == port)
    if not candidates:
        raise VllmServingError(
            f"no running vLLM process serves port {port}; the listener does "
            "not belong to this attempt"
        )
    if len(candidates) > 1:
        pids = ", ".join(str(process.pid) for process in candidates)
        raise VllmServingError(
            f"{len(candidates)} vLLM processes serve port {port} (pids {pids})"
        )
    process = candidates[0]
    if not model_identity_matches(process.model, expected_model):
        raise VllmServingError(
            f"the vLLM process serves model {process.model!r}, expected "
            f"{expected_model!r}"
        )
    if runtime_path is not None and not process.runs_from(runtime_path):
        raise VllmServingError(
            f"the vLLM process does not run from the selected generation at "
            f"{runtime_path}"
        )
    if process.age_seconds > attempt_elapsed_seconds + slack_seconds:
        raise VllmServingError(
            "the responding vLLM process predates this provisioning attempt "
            f"({process.age_seconds}s old, attempt is "
            f"{int(attempt_elapsed_seconds)}s old); refusing a stale listener"
        )
    return process


async def fetch_server_identity(
    client: httpx.AsyncClient, base_url: str, *, timeout: float
) -> VllmServerIdentity:
    """Read the runtime version and model cards from the server."""
    version = await _fetch_runtime_version(client, base_url, timeout=timeout)
    models = await _fetch_model_cards(client, base_url, timeout=timeout)
    return VllmServerIdentity(version=version, models=models)


def select_model(
    models: tuple[VllmModelCard, ...], expected_model: str
) -> VllmModelCard:
    """Return the model card that matches the requested model."""
    for card in models:
        if model_identity_matches(card.model_id, expected_model):
            return card
        if card.root is not None and model_identity_matches(card.root, expected_model):
            return card
    reported = ", ".join(repr(card.model_id) for card in models)
    raise VllmServingError(
        f"the server reports model(s) {reported}, expected {expected_model!r}"
    )


def choose_task(model: str) -> str:
    """Pick the OpenAI task that best matches the model's naming convention."""
    name = PurePosixPath(model).name.lower()
    if "base" in name or "completion" in name or name.startswith("gpt2"):
        return "completion"
    return "chat"


async def probe_inference(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    model: str,
    timeout: float,
    max_tokens: int = 1,
) -> VllmInferenceProbe:
    """Exercise one bounded inference request without asserting its text.

    Chat models are probed on ``/v1/chat/completions``; base models on
    ``/v1/completions``. If the server rejects the selected task (for example
    a chat endpoint without a chat template), the other supported endpoint is
    tried once.
    """
    primary = choose_task(model)
    tasks = (primary, "completion" if primary == "chat" else "chat")
    failures: list[str] = []
    for task in tasks:
        path = CHAT_PATH if task == "chat" else COMPLETION_PATH
        url = _join(base_url, path)
        body = _probe_body(task, model, max_tokens)
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout):
                response = await client.post(url, json=body, timeout=timeout)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise VllmServingError(
                f"inference probe at {path} timed out after {timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise VllmServingError(f"inference probe at {path} failed: {exc}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        if response.status_code in _UNSUPPORTED_TASK_STATUSES:
            failures.append(
                f"{path} returned HTTP {response.status_code}: {response.text[:200]}"
            )
            continue
        if response.status_code != 200:
            raise VllmServingError(
                f"inference probe at {path} returned HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise VllmServingError(
                f"inference probe at {path} returned invalid JSON"
            ) from exc
        return _probe_result(task, path, payload, latency_ms)
    raise VllmServingError(
        "the server does not support a usable inference task: " + "; ".join(failures)
    )


def revision_from_root(root: str | None) -> str | None:
    """Extract an immutable model commit from a Hugging Face snapshot path."""
    if root is None:
        return None
    parts = PurePosixPath(root).parts
    if len(parts) < 2 or parts[-2] != "snapshots":
        return None
    candidate = parts[-1]
    return candidate if _COMMIT_PATTERN.fullmatch(candidate) else None


def _vllm_serve_executable(arguments: tuple[str, ...], serve_index: int) -> str | None:
    """Return the vLLM executable only for the managed launch argv shapes.

    ``start-vllm.sh`` execs either ``<venv>/bin/vllm serve ...`` or
    ``<venv>/bin/python <venv>/bin/vllm serve ...``. Requiring ``vllm`` to be
    the directly preceding argument, with a direct binary or Python launcher,
    keeps ``grep``/``pgrep`` command lines that merely contain the words from
    looking like a server on the default port.
    """
    if serve_index == 1 and PurePosixPath(arguments[0]).name == "vllm":
        return arguments[0]
    if serve_index == 2:
        launcher = PurePosixPath(arguments[0]).name
        if launcher.startswith("python") and (
            PurePosixPath(arguments[1]).name == "vllm"
        ):
            return arguments[1]
    return None


def _port_argument(arguments: tuple[str, ...]) -> int | None:
    for index, argument in enumerate(arguments):
        if argument == "--port" and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith("--port="):
            value = argument.split("=", 1)[1]
        else:
            continue
        try:
            return int(value)
        except ValueError:
            return None
    # vLLM's default port when the launch omits the flag.
    return 8000


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


async def _fetch_runtime_version(
    client: httpx.AsyncClient, base_url: str, *, timeout: float
) -> str:
    url = _join(base_url, VERSION_PATH)
    try:
        async with asyncio.timeout(timeout):
            response = await client.get(url, timeout=timeout)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise VllmServingError(
            f"runtime version probe timed out after {timeout:g}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise VllmServingError(f"runtime version probe failed: {exc}") from exc
    if response.status_code != 200:
        raise VllmServingError(
            f"runtime version probe returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
    try:
        payload: object = response.json()
    except ValueError as exc:
        raise VllmServingError("runtime version response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise VllmServingError("runtime version response was not an object")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise VllmServingError("runtime version response did not report a version")
    return version.strip()


async def _fetch_model_cards(
    client: httpx.AsyncClient, base_url: str, *, timeout: float
) -> tuple[VllmModelCard, ...]:
    url = _join(base_url, MODELS_PATH)
    try:
        async with asyncio.timeout(timeout):
            response = await client.get(url, timeout=timeout)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise VllmServingError(f"model discovery timed out after {timeout:g}s") from exc
    except httpx.HTTPError as exc:
        raise VllmServingError(f"model discovery failed: {exc}") from exc
    if response.status_code != 200:
        raise VllmServingError(
            f"model discovery returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
    try:
        payload: object = response.json()
    except ValueError as exc:
        raise VllmServingError("model discovery response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise VllmServingError("model discovery response was not an object")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise VllmServingError("the server reported no models")
    cards: list[VllmModelCard] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        root_value = item.get("root")
        root = root_value if isinstance(root_value, str) and root_value else None
        max_model_len_value = item.get("max_model_len")
        max_model_len = (
            max_model_len_value
            if isinstance(max_model_len_value, int)
            and not isinstance(max_model_len_value, bool)
            and max_model_len_value > 0
            else None
        )
        cards.append(
            VllmModelCard(
                model_id=model_id.strip(),
                root=root,
                max_model_len=max_model_len,
            )
        )
    if not cards:
        raise VllmServingError("the model list contained no usable model ids")
    return tuple(cards)


def _probe_body(task: str, model: str, max_tokens: int) -> dict[str, object]:
    common: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if task == "chat":
        return {
            **common,
            "messages": [{"role": "user", "content": _PROBE_PROMPT}],
        }
    return {**common, "prompt": _PROBE_PROMPT}


def _probe_result(
    task: str, path: str, payload: object, latency_ms: float
) -> VllmInferenceProbe:
    if not isinstance(payload, dict):
        raise VllmServingError(f"inference probe at {path} was not a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VllmServingError(f"inference probe at {path} returned no choices")
    choice = choices[0]
    if task == "chat":
        if not isinstance(choice.get("message"), dict):
            raise VllmServingError(f"inference probe at {path} returned no message")
    elif not isinstance(choice.get("text"), str):
        raise VllmServingError(f"inference probe at {path} returned no text")
    finish_reason_value = choice.get("finish_reason")
    finish_reason = (
        finish_reason_value if isinstance(finish_reason_value, str) else None
    )
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return VllmInferenceProbe(
        task=task,
        endpoint=path,
        latency_ms=latency_ms,
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        finish_reason=finish_reason,
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
