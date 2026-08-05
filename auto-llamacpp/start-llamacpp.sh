#!/bin/bash
set -euo pipefail

API_PORT="${AUTOLLAMACPP_PORT:-8000}"
NFS_MOUNT_POINT="${AUTOLLAMACPP_NFS_MOUNT_POINT:-/srv/hf-cache}"
GGUF_RELATIVE_PATH="${AUTOLLAMACPP_GGUF_PATH:-}"
MODEL_ALIAS="${AUTOLLAMACPP_MODEL_ALIAS:-}"
GPU_LAYERS_OVERRIDE="${AUTOLLAMACPP_GPU_LAYERS:-}"
CTX_SIZE_OVERRIDE="${AUTOLLAMACPP_CTX_SIZE:-}"
PARALLEL_OVERRIDE="${AUTOLLAMACPP_PARALLEL:-}"
BATCH_SIZE_OVERRIDE="${AUTOLLAMACPP_BATCH_SIZE:-}"
EXTRA_ARGS_OVERRIDE="${AUTOLLAMACPP_EXTRA_ARGS:-}"
SCRIPT_DIR="${AUTOLLAMACPP_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LLAMACPP_BIN="${AUTOLLAMACPP_BIN:-/usr/local/bin/llama-server}"
LLAMACPP_INSTALL_ROOT="${AUTOLLAMACPP_INSTALL_ROOT:-/opt/llama.cpp}"
PID_FILE="${AUTOLLAMACPP_PID_FILE:-/var/run/llamacpp.pid}"
LLAMACPP_LOG_FILE="${AUTOLLAMACPP_LOG_FILE:-/var/log/llamacpp-serve.log}"
PROC_ROOT="${AUTOLLAMACPP_PROC_ROOT:-/proc}"
STARTUP_GRACE_PERIOD="${AUTOLLAMACPP_STARTUP_GRACE_PERIOD:-2}"
STARTUP_LOG_LINES="${AUTOLLAMACPP_STARTUP_LOG_LINES:-40}"
REQUIRE_CUDA="${AUTOLLAMACPP_REQUIRE_CUDA:-1}"

# shellcheck source=auto-llamacpp/llamacpp-process.sh
source "${SCRIPT_DIR}/llamacpp-process.sh"

detect_gpu_info() {
    GPU_COUNT=0
    GPU_MODEL="cpu-only"
    GPU_VRAM_MB=0
    GPU_VRAM_GB=0
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
        GPU_VRAM_GB=$(( (GPU_VRAM_MB + 512) / 1024 ))
    fi
    if [ "$REQUIRE_CUDA" = "1" ] && [ "$GPU_COUNT" -eq 0 ]; then
        echo "FATAL: managed llama.cpp requires a working NVIDIA GPU" >&2
        return 1
    fi
}

resolve_gguf_artifact() {
    if [ -z "$GGUF_RELATIVE_PATH" ]; then
        echo "FATAL: AUTOLLAMACPP_GGUF_PATH must name an exact cache-relative .gguf file" >&2
        return 1
    fi
    if [ -z "$MODEL_ALIAS" ]; then
        echo "FATAL: AUTOLLAMACPP_MODEL_ALIAS must name the selected artifact" >&2
        return 1
    fi
    if [[ "$GGUF_RELATIVE_PATH" == /* ]]; then
        echo "FATAL: AUTOLLAMACPP_GGUF_PATH must be relative to the NFS mount" >&2
        return 1
    fi
    case "$GGUF_RELATIVE_PATH" in
        ../*|*/../*|.|./*|*/./*|*//*|*\\*)
            echo "FATAL: AUTOLLAMACPP_GGUF_PATH must be a canonical relative POSIX path" >&2
            return 1
            ;;
    esac
    case "$GGUF_RELATIVE_PATH" in
        *.gguf) ;;
        *)
            echo "FATAL: AUTOLLAMACPP_GGUF_PATH must end in .gguf" >&2
            return 1
            ;;
    esac
    local candidate
    local mount_root
    local resolved
    mount_root=$(readlink -f -- "$NFS_MOUNT_POINT") || {
        echo "FATAL: NFS mount is unavailable: ${NFS_MOUNT_POINT}" >&2
        return 1
    }
    candidate="${mount_root}/${GGUF_RELATIVE_PATH}"
    resolved=$(readlink -f -- "$candidate") || {
        echo "FATAL: selected GGUF artifact is unavailable: ${GGUF_RELATIVE_PATH}" >&2
        return 1
    }
    case "$resolved" in
        "${mount_root}"/*) ;;
        *)
            echo "FATAL: selected GGUF artifact escapes the NFS mount" >&2
            return 1
            ;;
    esac
    if [ ! -f "$resolved" ]; then
        echo "FATAL: selected GGUF artifact is not a regular file: ${GGUF_RELATIVE_PATH}" >&2
        return 1
    fi
    # llama.cpp derives sibling split paths from the entrypoint filename. Keep
    # the validated symlink path so the -00001-of-0000N.gguf suffix survives.
    GGUF_PATH="$candidate"
}

configure_llamacpp_params() {
    N_GPU_LAYERS="auto"
    CTX_SIZE=4096
    PARALLEL=4
    BATCH_SIZE=2048
    EXTRA_ARGS=""

    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "No GPU detected: running CPU-only inference"
        N_GPU_LAYERS=0
        PARALLEL=2
        CTX_SIZE=2048
    else
        case "$GPU_MODEL" in
            *"H100"*|*"A100"*)
                echo "High-end GPU detected: full GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=32768
                PARALLEL=8
                ;;
            *"T4"*)
                echo "Tesla T4 detected: partial GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=8192
                PARALLEL=4
                ;;
            *"RTX"*|*"GeForce"*)
                echo "Consumer GPU detected: conservative GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=8192
                PARALLEL=4
                ;;
            *)
                echo "Unknown GPU: using conservative defaults"
                N_GPU_LAYERS=99
                CTX_SIZE=4096
                PARALLEL=4
                ;;
        esac
    fi

    N_GPU_LAYERS="${GPU_LAYERS_OVERRIDE:-$N_GPU_LAYERS}"
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-$CTX_SIZE}"
    PARALLEL="${PARALLEL_OVERRIDE:-$PARALLEL}"
    BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
    EXTRA_ARGS="${EXTRA_ARGS_OVERRIDE:-$EXTRA_ARGS}"
}

clear_script_environment() {
    unset AUTOLLAMACPP_PORT AUTOLLAMACPP_NFS_MOUNT_POINT
    unset AUTOLLAMACPP_GGUF_PATH AUTOLLAMACPP_MODEL_ALIAS
    unset AUTOLLAMACPP_GPU_LAYERS AUTOLLAMACPP_CTX_SIZE AUTOLLAMACPP_PARALLEL
    unset AUTOLLAMACPP_BATCH_SIZE AUTOLLAMACPP_EXTRA_ARGS
    unset AUTOLLAMACPP_SCRIPT_DIR AUTOLLAMACPP_BIN AUTOLLAMACPP_INSTALL_ROOT
    unset AUTOLLAMACPP_PID_FILE
    unset AUTOLLAMACPP_LOG_FILE AUTOLLAMACPP_PROC_ROOT
    unset AUTOLLAMACPP_STARTUP_GRACE_PERIOD AUTOLLAMACPP_STARTUP_LOG_LINES
    unset AUTOLLAMACPP_STOP_TIMEOUT AUTOLLAMACPP_STOP_INTERVAL
    unset AUTOLLAMACPP_REQUIRE_CUDA

    # llama-server owns LLAMA_ARG_* as an alternate option namespace. QIIP
    # supplies the complete managed command line, so ambient values must not
    # alter the process after provisioning validated its configuration.
    local name
    while IFS= read -r name; do
        unset "$name"
    done < <(compgen -A variable LLAMA_ARG_)
}

verify_llamacpp_started() {
    local pid="$1"
    sleep "$STARTUP_GRACE_PERIOD"
    if is_llamacpp_pid "$pid"; then
        return 0
    fi
    rm -f "$PID_FILE"
    echo "FATAL: llama-server process ${pid} exited during startup; last ${STARTUP_LOG_LINES} log lines:" >&2
    if [ -f "$LLAMACPP_LOG_FILE" ]; then
        tail -n "$STARTUP_LOG_LINES" "$LLAMACPP_LOG_FILE" >&2
    else
        echo "(llama-server log file ${LLAMACPP_LOG_FILE} was not created)" >&2
    fi
    return 1
}

run_llamacpp() {
    resolve_gguf_artifact

    bash "${SCRIPT_DIR}/stop-llamacpp.sh"
    clear_script_environment

    cat <<EOF

# llama.cpp Configuration
# ================================================
# GPU:                ${GPU_COUNT} x ${GPU_MODEL} (${GPU_VRAM_GB} GB)
# Model:              ${MODEL_ALIAS}
# GGUF:               ${GGUF_PATH}
# GPU Layers:         ${N_GPU_LAYERS}
# Context Size:       ${CTX_SIZE} tokens
# Parallel Slots:     ${PARALLEL}
# ================================================

EOF

    set -f
    # shellcheck disable=SC2086
    "$LLAMACPP_BIN" \
        --model "$GGUF_PATH" \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --alias "$MODEL_ALIAS" \
        -ngl "$N_GPU_LAYERS" \
        -c "$CTX_SIZE" \
        --parallel "$PARALLEL" \
        -b "$BATCH_SIZE" \
        --flash-attn auto \
        --cont-batching \
        --metrics \
        ${EXTRA_ARGS:-} \
        > "$LLAMACPP_LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    verify_llamacpp_started "$pid"
    echo "llama-server started (PID ${pid})"
}

main() {
    detect_gpu_info
    configure_llamacpp_params
    run_llamacpp
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
