#!/bin/bash
set -euo pipefail

PID_FILE="${AUTOLLAMACPP_PID_FILE:-/var/run/llamacpp.pid}"
PROC_ROOT="${AUTOLLAMACPP_PROC_ROOT:-/proc}"
STOP_TIMEOUT="${AUTOLLAMACPP_STOP_TIMEOUT:-30}"
STOP_INTERVAL="${AUTOLLAMACPP_STOP_INTERVAL:-1}"
COMMAND_PATTERN="${AUTOLLAMACPP_COMMAND_PATTERN:-llama-server}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=auto-llamacpp/llamacpp-process.sh
source "${SCRIPT_DIR}/llamacpp-process.sh"

signal="TERM"
if [ "${1:-}" = "--force" ]; then
    signal="KILL"
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--force]" >&2
    exit 2
fi

declare -a targets=()
declare -A seen=()

add_target() {
    local pid="$1"
    if [ -z "${seen[$pid]:-}" ]; then
        targets+=("$pid")
        seen["$pid"]=1
    fi
}

if [ -f "$PID_FILE" ]; then
    pid=$(<"$PID_FILE")
    if is_llamacpp_pid "$pid"; then
        add_target "$pid"
    else
        echo "Ignoring stale llama-server PID file for PID '${pid}'" >&2
    fi
fi

for cmdline_file in "${PROC_ROOT}"/[0-9]*/cmdline; do
    pid_dir=${cmdline_file%/cmdline}
    candidate=${pid_dir##*/}
    if is_llamacpp_pid "$candidate"; then
        add_target "$candidate"
    fi
done

if [ "${#targets[@]}" -eq 0 ]; then
    rm -f "$PID_FILE"
    echo "No running llama-server process found"
    exit 0
fi

for pid in "${targets[@]}"; do
    if ! kill -s "$signal" "$pid"; then
        echo "Failed to send SIG${signal} to llama-server PID ${pid}" >&2
        exit 1
    fi
done

deadline=$((SECONDS + STOP_TIMEOUT))
while true; do
    declare -a remaining=()
    for pid in "${targets[@]}"; do
        if is_llamacpp_pid "$pid"; then
            remaining+=("$pid")
        fi
    done

    if [ "${#remaining[@]}" -eq 0 ]; then
        rm -f "$PID_FILE"
        echo "Stopped llama-server process(es): ${targets[*]}"
        exit 0
    fi

    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "Timed out waiting for llama-server PID(s): ${remaining[*]}" >&2
        exit 1
    fi
    sleep "$STOP_INTERVAL"
done
