"""Render the bash setup script a user pipes from curl.

The script only writes the harness config file (Linux, bash). It is wrapped
in ``main`` so a truncated download never executes a partial script, backs up
any existing config before touching it, merges into existing files where the
format allows, and keeps the file private (it contains the API token).
"""

from __future__ import annotations

import shlex

from inference_proxy.onboarding.harness import (
    CODEX_BLOCK_END,
    CODEX_BLOCK_START,
    CODEX_SPLIT,
    Harness,
)

HEREDOC_DELIMITER = "QIIP_CONFIG_EOF"

_PRELUDE = r"""#!/usr/bin/env bash
# QIIP setup script. Writes one config file, nothing else.
set -euo pipefail

main() {
  local bold="" dim="" green="" yellow="" reset=""
  if [ -t 1 ]; then
    bold=$'\033[1m'; dim=$'\033[2m'; green=$'\033[32m'
    yellow=$'\033[33m'; reset=$'\033[0m'
  fi
  say()  { printf '%s\n' "$*"; }
  ok()   { printf '%s\n' "  ${green}✓${reset} $*"; }
  note() { printf '%s\n' "  ${yellow}!${reset} $*"; }

  if [ -z "${HOME:-}" ]; then
    say "HOME is not set, so I don't know where to put the config." >&2
    exit 1
  fi
"""

_JSON_MERGE = r"""    if command -v python3 >/dev/null 2>&1; then
      if python3 - "$target" "$fresh" <<'QIIP_PY_EOF'
import json, sys

target, fresh = sys.argv[1:3]


def merge(old, new):
    for key, value in new.items():
        if key != "qiip" and isinstance(value, dict) and isinstance(old.get(key), dict):
            merge(old[key], value)
        else:
            old[key] = value
    return old


try:
    with open(target) as handle:
        text = handle.read()
    current = json.loads(text) if text.strip() else {}
except (OSError, ValueError):
    sys.exit(3)
if not isinstance(current, dict):
    sys.exit(3)
with open(fresh) as handle:
    merged = merge(current, json.load(handle))
with open(fresh, "w") as handle:
    json.dump(merged, handle, indent=2)
    handle.write("\n")
QIIP_PY_EOF
      then
        ok "Kept your other settings"
      else
        # Comments or trailing commas (JSONC) cannot be merged safely.
        say "Your existing $target is not plain JSON, so it was left unchanged." >&2
        say "Add the qiip settings by hand, or move it aside and run this again." >&2
        exit 1
      fi
    else
      note "python3 not found, so the file was replaced (backup kept)"
    fi
"""


def _codex_merge() -> str:
    start = shlex.quote(CODEX_BLOCK_START)
    end = shlex.quote(CODEX_BLOCK_END)
    split = shlex.quote(CODEX_SPLIT)
    return rf"""  # config.toml: top-level keys must precede every table, so the managed
  # lines live in two fenced blocks (top and bottom) around the user's file.
  local rest="$fresh.rest" built="$fresh.built"
  : > "$rest"
  if [ -f "$target" ]; then
    awk -v start={start} -v end={end} '
      index($0, start) == 1 {{ skip = 1; next }}
      index($0, end) == 1 {{ skip = 0; next }}
      skip {{ next }}
      /^[[:space:]]*\[/ {{
        intable = 1
        # A qiip provider table written by hand (the manual setup in the
        # README) is replaced by the managed one; two would not parse.
        ours = $0 ~ /^[[:space:]]*\[[[:space:]]*model_providers[[:space:]]*\.[[:space:]]*"?qiip"?[[:space:]]*[].]/
      }}
      ours {{ next }}
      !intable && /^[[:space:]]*(model|model_provider)[[:space:]]*=/ {{ next }}
      /^[[:space:]]*$/ {{ blank = 1; next }}
      {{ if (seen && blank) print ""; blank = 0; seen = 1; print }}
    ' "$target" > "$rest"
  fi
  {{
    printf '%s\n' {start}
    awk -v sep={split} '$0 == sep {{ exit }} {{ print }}' "$fresh"
    printf '%s\n\n' {end}
    cat "$rest"
    printf '\n%s\n' {start}
    awk -v sep={split} 'found {{ print }} $0 == sep {{ found = 1 }}' "$fresh"
    printf '%s\n' {end}
  }} > "$built"
  mv "$built" "$fresh"
  rm -f "$rest"
  if command -v python3 >/dev/null 2>&1 && ! python3 - "$fresh" <<'QIIP_TOML_EOF'
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11: nothing to check with
    sys.exit(0)
with open(sys.argv[1], "rb") as handle:
    tomllib.load(handle)
QIIP_TOML_EOF
  then
    say "Your existing $target could not be combined into valid TOML, so it was left unchanged." >&2
    say "Add the qiip settings by hand, or move it aside and run this again." >&2
    exit 1
  fi
"""


def render_setup_script(
    harness: Harness,
    *,
    base_url: str,
    token: str,
    models: list[str],
) -> str:
    """Return the bash script that configures *harness* for QIIP."""
    config = harness.render(base_url, token, models)
    if HEREDOC_DELIMITER in config:
        raise ValueError("config collides with the heredoc delimiter")
    target = shlex.quote(harness.config_path)
    label = shlex.quote(harness.label)
    command = shlex.quote(harness.command)
    parts = [
        _PRELUDE,
        f"  local label={label} command={command}\n",
        f'  local target="$HOME/"{target}\n',
        r"""  fresh=""  # global on purpose: the EXIT trap runs after main returns
  say ""
  say "${bold}Setting up ${label} for qiip${reset}"
  say ""
  mkdir -p "$(dirname "$target")"
  umask 077
  fresh="$(mktemp "$target.qiip.XXXXXX")"
  trap '[ -n "$fresh" ] && rm -f "$fresh" "$fresh.rest" "$fresh.built"' EXIT
""",
        f"  cat > \"$fresh\" <<'{HEREDOC_DELIMITER}'\n{config}{HEREDOC_DELIMITER}\n",
    ]
    if harness.merge == "codex-toml":
        parts.append(
            r"""  if [ -f "$target" ]; then
    # Keep the first backup: a rerun must not replace the pre-qiip original.
    if [ ! -e "$target.bak" ]; then
      cp -p "$target" "$target.bak"
      ok "Backed up your existing config to ${dim}$target.bak${reset}"
    fi
  fi
"""
        )
        parts.append(_codex_merge())
    else:
        parts.append(
            r"""  if [ -f "$target" ]; then
    # Keep the first backup: a rerun must not replace the pre-qiip original.
    if [ ! -e "$target.bak" ]; then
      cp -p "$target" "$target.bak"
      ok "Backed up your existing config to ${dim}$target.bak${reset}"
    fi
"""
        )
        if harness.merge == "json":
            parts.append(_JSON_MERGE)
        parts.append("  fi\n")
    parts.append(
        r"""  chmod 600 "$fresh"
  mv "$fresh" "$target"
  ok "Wrote ${dim}$target${reset}"
  say ""
  if command -v "$command" >/dev/null 2>&1; then
    say "${bold}All done.${reset} Start it by typing: ${green}${command}${reset}"
  else
    say "${bold}Config is ready.${reset} ${label} isn't installed on this machine yet."
    say "Once it is, start it by typing: ${green}${command}${reset}"
  fi
  say ""
}

main "$@"
"""
    )
    return "".join(parts)


def render_expired_script() -> str:
    """Script served for unknown or expired links: explain, change nothing."""
    return (
        "#!/usr/bin/env bash\n"
        "echo 'This qiip setup link has expired or was already replaced.' >&2\n"
        "echo 'Open qiip in your browser and create a new one.' >&2\n"
        "exit 1\n"
    )
