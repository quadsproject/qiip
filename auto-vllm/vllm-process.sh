#!/bin/bash

# Shared process-identity check for start-vllm.sh and stop-vllm.sh.
# Callers define PROC_ROOT and COMMAND_PATTERN before invoking this function.
is_vllm_pid() {
    local pid="$1"
    local cmdline

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [ -r "${PROC_ROOT}/${pid}/cmdline" ] || return 1
    cmdline=$(tr '\0' ' ' < "${PROC_ROOT}/${pid}/cmdline")
    [[ "$cmdline" == *"$COMMAND_PATTERN"* ]]
}
