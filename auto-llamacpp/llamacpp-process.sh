#!/bin/bash

# Exact process-identity check shared by start and stop.
is_llamacpp_pid() {
    local pid="$1"
    local actual_executable expected_executable managed_install_root argument
    local has_model_argument=1

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [ -r "${PROC_ROOT}/${pid}/cmdline" ] || return 1
    [ -L "${PROC_ROOT}/${pid}/exe" ] || return 1
    actual_executable=$(readlink -f "${PROC_ROOT}/${pid}/exe") || return 1
    if expected_executable=$(readlink -f "$LLAMACPP_BIN" 2>/dev/null) \
        && [ "$actual_executable" = "$expected_executable" ]; then
        :
    else
        # A version upgrade repoints LLAMACPP_BIN before the old process is
        # stopped. Accept only exact managed llama-server locations so both
        # the PID-file and orphan /proc scans can still identify that process.
        managed_install_root=$(readlink -f "${LLAMACPP_INSTALL_ROOT:-/opt/llama.cpp}") \
            || return 1
        case "$actual_executable" in
            "$managed_install_root"/*/bin/llama-server) ;;
            *) return 1 ;;
        esac
    fi

    # A second llama-server process may be running a maintenance command.
    # Only model-serving invocations carry one of these exact argv tokens.
    while IFS= read -r -d '' argument; do
        if [ "$argument" = "--model" ] || [ "$argument" = "-m" ]; then
            has_model_argument=0
            break
        fi
    done < "${PROC_ROOT}/${pid}/cmdline"
    return "$has_model_argument"
}
