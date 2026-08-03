#!/bin/bash
set -euo pipefail

API_PORT="${AUTOLLAMACPP_PORT:-8000}"
NFS_MOUNT_POINT="${AUTOLLAMACPP_NFS_MOUNT_POINT:-/srv/hf-cache}"
MODEL_OVERRIDE="${AUTOLLAMACPP_MODEL:-}"
GPU_LAYERS_OVERRIDE="${AUTOLLAMACPP_GPU_LAYERS:-}"
CTX_SIZE_OVERRIDE="${AUTOLLAMACPP_CTX_SIZE:-}"
PARALLEL_OVERRIDE="${AUTOLLAMACPP_PARALLEL:-}"
BATCH_SIZE_OVERRIDE="${AUTOLLAMACPP_BATCH_SIZE:-}"
QUANTIZATION="${AUTOLLAMACPP_QUANTIZATION:-Q4_K_M}"
EXTRA_ARGS_OVERRIDE="${AUTOLLAMACPP_EXTRA_ARGS:-}"
SCRIPT_DIR="${AUTOLLAMACPP_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LLAMACPP_BIN="${AUTOLLAMACPP_BIN:-/usr/local/bin/llama-server}"
PID_FILE="${AUTOLLAMACPP_PID_FILE:-/var/run/llamacpp.pid}"
LLAMACPP_LOG_FILE="${AUTOLLAMACPP_LOG_FILE:-/var/log/llamacpp-serve.log}"
PROC_ROOT="${AUTOLLAMACPP_PROC_ROOT:-/proc}"
COMMAND_PATTERN="${AUTOLLAMACPP_COMMAND_PATTERN:-llama-server}"
STARTUP_GRACE_PERIOD="${AUTOLLAMACPP_STARTUP_GRACE_PERIOD:-2}"
STARTUP_LOG_LINES="${AUTOLLAMACPP_STARTUP_LOG_LINES:-40}"

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
}

find_gguf_model() {
    local model_name="$1"
    local quant_lower
    quant_lower=$(echo "$QUANTIZATION" | tr '[:upper:]' '[:lower:]')
    local gguf_dir="${NFS_MOUNT_POINT}/gguf"

    local -a candidates=()
    if [ -d "$gguf_dir" ]; then
        if [ -n "$model_name" ]; then
            local model_pattern
            model_pattern="${model_name//\//--}"
            mapfile -t candidates < <(
                find "$gguf_dir" -path "*${model_pattern}*" -name "*${quant_lower}*" -name "*.gguf" -type f 2>/dev/null
            )
        fi
        if [ "${#candidates[@]}" -eq 0 ]; then
            mapfile -t candidates < <(
                find "$gguf_dir" -name "*${quant_lower}*" -name "*.gguf" -type f 2>/dev/null
            )
        fi
    fi

    if [ "${#candidates[@]}" -gt 0 ]; then
        GGUF_PATH="${candidates[0]}"
        return 0
    fi

    echo "No GGUF file found for quantization ${QUANTIZATION} in ${gguf_dir}" >&2
    return 1
}

derive_model_alias() {
    local gguf_path="$1"
    local dir_name
    dir_name=$(basename "$(dirname "$gguf_path")")
    # Strip -GGUF suffix and convert -- back to /
    MODEL_ALIAS="${dir_name/--//}"
    MODEL_ALIAS="${MODEL_ALIAS%-GGUF}"
    MODEL_ALIAS="${MODEL_ALIAS%-gguf}"
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
    unset AUTOLLAMACPP_PORT AUTOLLAMACPP_NFS_MOUNT_POINT AUTOLLAMACPP_MODEL
    unset AUTOLLAMACPP_GPU_LAYERS AUTOLLAMACPP_CTX_SIZE AUTOLLAMACPP_PARALLEL
    unset AUTOLLAMACPP_BATCH_SIZE AUTOLLAMACPP_QUANTIZATION AUTOLLAMACPP_EXTRA_ARGS
    unset AUTOLLAMACPP_SCRIPT_DIR AUTOLLAMACPP_BIN AUTOLLAMACPP_PID_FILE
    unset AUTOLLAMACPP_LOG_FILE AUTOLLAMACPP_PROC_ROOT AUTOLLAMACPP_COMMAND_PATTERN
    unset AUTOLLAMACPP_STARTUP_GRACE_PERIOD AUTOLLAMACPP_STARTUP_LOG_LINES
    unset AUTOLLAMACPP_STOP_TIMEOUT AUTOLLAMACPP_STOP_INTERVAL
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
    if [ -n "$MODEL_OVERRIDE" ]; then
        if ! find_gguf_model "$MODEL_OVERRIDE"; then
            echo "FATAL: could not locate GGUF model for ${MODEL_OVERRIDE}" >&2
            exit 1
        fi
    else
        local gguf_dir="${NFS_MOUNT_POINT}/gguf"
        local quant_lower
        quant_lower=$(echo "$QUANTIZATION" | tr '[:upper:]' '[:lower:]')
        local -a all_ggufs=()
        if [ -d "$gguf_dir" ]; then
            mapfile -t all_ggufs < <(
                find "$gguf_dir" -name "*${quant_lower}*" -name "*.gguf" -type f 2>/dev/null
            )
        fi
        if [ "${#all_ggufs[@]}" -eq 0 ]; then
            echo "FATAL: no GGUF models found in ${gguf_dir}" >&2
            exit 1
        fi
        GGUF_PATH="${all_ggufs[0]}"
    fi

    derive_model_alias "$GGUF_PATH"

    bash "${SCRIPT_DIR}/stop-llamacpp.sh"
    clear_script_environment

    cat <<EOF

# llama.cpp Configuration
# ================================================
# GPU:                ${GPU_COUNT} x ${GPU_MODEL} (${GPU_VRAM_GB} GB)
# Model:              ${MODEL_ALIAS}
# GGUF:               ${GGUF_PATH}
# Quantization:       ${QUANTIZATION}
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
        -fa \
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
