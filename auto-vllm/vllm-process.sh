#!/bin/bash

# Shared process-identity check for start-vllm.sh and stop-vllm.sh.
# Callers define PROC_ROOT and COMMAND_PATTERN before invoking this function.
is_vllm_pid() {
    local pid="$1"
    local cmdline argument previous_argument="" runtime_root

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [ -r "${PROC_ROOT}/${pid}/cmdline" ] || return 1
    cmdline=$(tr '\0' ' ' < "${PROC_ROOT}/${pid}/cmdline")
    [[ "$cmdline" == *"$COMMAND_PATTERN"* ]] && return 0
    # A newly activated generation must still identify the prior server for
    # stop/start. Match exact executable and serve argv tokens in managed paths.
    runtime_root="${AUTOVLLM_RUNTIME_ROOT:-/opt/vllm-venv-generations}"
    while IFS= read -r -d '' argument; do
        if [ "$argument" = "serve" ]; then
            case "$previous_argument" in
                "$runtime_root"/*/bin/vllm|/opt/vllm-venv/bin/vllm) return 0 ;;
            esac
        fi
        previous_argument="$argument"
    done < "${PROC_ROOT}/${pid}/cmdline"
    return 1
}
