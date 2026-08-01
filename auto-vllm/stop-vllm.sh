#!/bin/bash
set -euo pipefail

PID_FILE="${AUTOVLLM_PID_FILE:-/var/run/vllm.pid}"
PROC_ROOT="${AUTOVLLM_PROC_ROOT:-/proc}"
STOP_TIMEOUT="${AUTOVLLM_STOP_TIMEOUT:-30}"
STOP_INTERVAL="${AUTOVLLM_STOP_INTERVAL:-1}"
VLLM_BIN="${AUTOVLLM_BIN:-/opt/vllm-venv/bin/vllm}"
COMMAND_PATTERN="${AUTOVLLM_COMMAND_PATTERN:-${VLLM_BIN} serve}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=auto-vllm/vllm-process.sh
source "${SCRIPT_DIR}/vllm-process.sh"

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
    if is_vllm_pid "$pid"; then
        add_target "$pid"
    else
        echo "Ignoring stale vLLM PID file for PID '${pid}'" >&2
    fi
fi

# A failed relaunch can overwrite the PID file while the older vLLM remains
# alive. Scan the expected command line so teardown also finds that orphan.
for cmdline_file in "${PROC_ROOT}"/[0-9]*/cmdline; do
    pid_dir=${cmdline_file%/cmdline}
    candidate=${pid_dir##*/}
    if is_vllm_pid "$candidate"; then
        add_target "$candidate"
    fi
done

if [ "${#targets[@]}" -eq 0 ]; then
    rm -f "$PID_FILE"
    echo "No running vLLM process found"
    exit 0
fi

for pid in "${targets[@]}"; do
    if ! kill -s "$signal" "$pid"; then
        echo "Failed to send SIG${signal} to vLLM PID ${pid}" >&2
        exit 1
    fi
done

deadline=$((SECONDS + STOP_TIMEOUT))
while true; do
    declare -a remaining=()
    for pid in "${targets[@]}"; do
        if is_vllm_pid "$pid"; then
            remaining+=("$pid")
        fi
    done

    if [ "${#remaining[@]}" -eq 0 ]; then
        rm -f "$PID_FILE"
        echo "Stopped vLLM process(es): ${targets[*]}"
        exit 0
    fi

    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "Timed out waiting for vLLM PID(s): ${remaining[*]}" >&2
        exit 1
    fi
    sleep "$STOP_INTERVAL"
done
