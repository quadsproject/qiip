#!/bin/bash

# Process-identity check for llama-server, mirroring vllm-process.sh.
is_llamacpp_pid() {
    local pid="$1"
    local cmdline

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [ -r "${PROC_ROOT}/${pid}/cmdline" ] || return 1
    cmdline=$(tr '\0' ' ' < "${PROC_ROOT}/${pid}/cmdline")
    [[ "$cmdline" == *"$COMMAND_PATTERN"* ]]
}
