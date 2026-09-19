"""Coding-harness catalog and per-harness config rendering.

Each harness describes where its config file lives, how many models it can
be configured with, and how its config is merged into an existing file by
the generated setup script (see ``script.py``). Rendering is pure: it takes
the public base URL, the raw API token, and the chosen models and returns
the config text, so it is trivially testable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

MergeMode = Literal["json", "codex-toml", "replace"]

# Markers that fence the lines QIIP manages inside a user's config.toml.
CODEX_BLOCK_START = "# >>> qiip >>>"
CODEX_BLOCK_END = "# <<< qiip <<<"
# Separates the top-level keys from the provider table in the codex payload;
# TOML requires top-level keys to precede every table.
CODEX_SPLIT = "# --- qiip tables ---"


@dataclass(frozen=True)
class Harness:
    """One supported coding harness."""

    id: str
    label: str
    tagline: str
    command: str
    config_path: str  # relative to $HOME
    multi_model: bool
    merge: MergeMode
    render: Callable[[str, str, list[str]], str]
    # /v1 route the harness speaks; it is offered only when QIIP serves it.
    requires_route: str = "/v1/chat/completions"


def _json(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2) + "\n"


def _render_opencode(base_url: str, token: str, models: list[str]) -> str:
    return _json(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "qiip": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "qiip inference proxy",
                    "options": {"baseURL": f"{base_url}/v1", "apiKey": token},
                    "models": {model: {"name": model} for model in models},
                }
            },
            "model": f"qiip/{models[0]}",
        }
    )


def _render_pi(base_url: str, token: str, models: list[str]) -> str:
    return _json(
        {
            "providers": {
                "qiip": {
                    "baseUrl": f"{base_url}/v1",
                    "api": "openai-completions",
                    "apiKey": token,
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                    },
                    "models": [{"id": model} for model in models],
                }
            }
        }
    )


def _yaml_scalar(value: str) -> str:
    """Double-quote a YAML scalar (JSON strings are valid YAML scalars)."""
    return json.dumps(value)


def _render_omp(base_url: str, token: str, models: list[str]) -> str:
    lines = [
        "providers:",
        "  qiip:",
        f"    baseUrl: {_yaml_scalar(base_url + '/v1')}",
        "    auth: apiKey",
        f"    apiKey: {_yaml_scalar(token)}",
        "    api: openai-completions",
        "    models:",
    ]
    for model in models:
        lines.append(f"      - id: {_yaml_scalar(model)}")
        lines.append(f"        name: {_yaml_scalar(model + ' (qiip)')}")
    return "\n".join(lines) + "\n"


def _render_codex(base_url: str, token: str, models: list[str]) -> str:
    # json.dumps output is a valid TOML basic string for these values.
    return "\n".join(
        [
            f"model = {json.dumps(models[0])}",
            'model_provider = "qiip"',
            CODEX_SPLIT,
            "[model_providers.qiip]",
            'name = "qiip inference proxy"',
            f"base_url = {json.dumps(base_url + '/v1')}",
            'wire_api = "responses"',
            f"experimental_bearer_token = {json.dumps(token)}",
            "",
        ]
    )


def _render_claude(base_url: str, token: str, models: list[str]) -> str:
    model = models[0]
    return _json(
        {
            "env": {
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_AUTH_TOKEN": token,
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
    )


HARNESSES: tuple[Harness, ...] = (
    Harness(
        id="opencode",
        label="OpenCode",
        tagline="Open source terminal agent",
        command="opencode",
        config_path=".config/opencode/opencode.json",
        multi_model=True,
        merge="json",
        render=_render_opencode,
    ),
    Harness(
        id="pi",
        label="Pi",
        tagline="Minimal, fast coding agent",
        command="pi",
        config_path=".pi/agent/models.json",
        multi_model=True,
        merge="json",
        render=_render_pi,
    ),
    Harness(
        id="omp",
        label="Oh My Pi",
        tagline="Pi with batteries included",
        command="omp",
        config_path=".omp/agent/models.yml",
        multi_model=True,
        merge="replace",
        render=_render_omp,
    ),
    Harness(
        id="claude",
        label="Claude Code",
        tagline="Anthropic's coding agent",
        command="claude",
        config_path=".claude/settings.json",
        multi_model=False,
        merge="json",
        render=_render_claude,
        requires_route="/v1/messages",
    ),
    Harness(
        id="codex",
        label="Codex",
        tagline="OpenAI's coding agent",
        command="codex",
        config_path=".codex/config.toml",
        multi_model=False,
        merge="codex-toml",
        render=_render_codex,
        requires_route="/v1/responses",
    ),
)

_BY_ID = {harness.id: harness for harness in HARNESSES}


def get_harness(harness_id: str) -> Harness | None:
    """Return the harness registered under *harness_id*, or None."""
    return _BY_ID.get(harness_id)
