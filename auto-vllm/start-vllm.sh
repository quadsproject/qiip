#!/bin/bash
set -euo pipefail

API_PORT="${AUTOVLLM_API_PORT:-8000}"
NFS_MOUNT_POINT="${AUTOVLLM_NFS_MOUNT_POINT:-/srv/hf-cache}"
MODEL_OVERRIDE="${AUTOVLLM_MODEL:-}"
TENSOR_PARALLEL_OVERRIDE="${AUTOVLLM_TENSOR_PARALLEL:-}"
GPU_MEM_UTIL_OVERRIDE="${AUTOVLLM_GPU_MEM_UTIL:-}"
MAX_MODEL_LEN_OVERRIDE="${AUTOVLLM_MAX_MODEL_LEN:-}"
MAX_BATCHED_TOKENS_OVERRIDE="${AUTOVLLM_MAX_BATCHED_TOKENS:-}"
EXTRA_ARGS_OVERRIDE="${AUTOVLLM_EXTRA_ARGS:-}"
SCRIPT_DIR="${AUTOVLLM_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
VLLM_BIN="${AUTOVLLM_BIN:-/opt/vllm-venv/bin/vllm}"
PID_FILE="${AUTOVLLM_PID_FILE:-/var/run/vllm.pid}"
HF_CACHE_LINK="${AUTOVLLM_HF_CACHE_LINK:-/root/.cache/huggingface}"
VLLM_LOG_FILE="${AUTOVLLM_LOG_FILE:-/var/log/vllm-serve.log}"
VLLM_PYTHON="${AUTOVLLM_PYTHON:-$(dirname "$VLLM_BIN")/python}"
PROC_ROOT="${AUTOVLLM_PROC_ROOT:-/proc}"
COMMAND_PATTERN="${AUTOVLLM_COMMAND_PATTERN:-${VLLM_BIN} serve}"
STARTUP_GRACE_PERIOD="${AUTOVLLM_STARTUP_GRACE_PERIOD:-2}"
STARTUP_LOG_LINES="${AUTOVLLM_STARTUP_LOG_LINES:-40}"

warn_retired_override() {
    local legacy_name="$1"
    local replacement="$2"
    if [[ -v $legacy_name ]]; then
        echo "WARNING: ${legacy_name} is ignored; use ${replacement} instead" >&2
    fi
}

# These five names were previously script-level tuning overrides. Warn only
# for that retired set: vLLM legitimately owns other VLLM_* environment names.
warn_retired_override VLLM_TENSOR_PARALLEL AUTOVLLM_TENSOR_PARALLEL
warn_retired_override VLLM_GPU_MEM_UTIL AUTOVLLM_GPU_MEM_UTIL
warn_retired_override VLLM_MAX_MODEL_LEN AUTOVLLM_MAX_MODEL_LEN
warn_retired_override VLLM_MAX_BATCHED_TOKENS AUTOVLLM_MAX_BATCHED_TOKENS
warn_retired_override VLLM_EXTRA_ARGS AUTOVLLM_EXTRA_ARGS

# Ignore legacy script inputs instead of leaking them into vLLM's reserved
# environment namespace. VLLM_MODEL was an internal gateway handoff;
# VLLM_PORT is the upstream collision that motivated the namespace change.
unset VLLM_MODEL VLLM_PORT VLLM_TENSOR_PARALLEL VLLM_GPU_MEM_UTIL
unset VLLM_MAX_MODEL_LEN VLLM_MAX_BATCHED_TOKENS VLLM_EXTRA_ARGS

# shellcheck source=auto-vllm/vllm-process.sh
source "${SCRIPT_DIR}/vllm-process.sh"

detect_gpu_info() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "FATAL: nvidia-smi not found. Run setup.sh first or install NVIDIA drivers." >&2
        exit 1
    fi
    if ! nvidia-smi &>/dev/null; then
        echo "FATAL: nvidia-smi failed. NVIDIA driver may not be loaded." >&2
        exit 1
    fi
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
    GPU_VRAM_GB=$(( (GPU_VRAM_MB + 512) / 1024 ))
}

configure_vllm_params() {
    local total_vram=$((GPU_COUNT * GPU_VRAM_GB))

    TENSOR_PARALLEL=$GPU_COUNT
    GPU_MEM_UTIL=0.90
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=32768
    EXTRA_ARGS=""

    case "$GPU_MODEL" in
        *"H100"*|*"A100"*)
            echo "High-end GPU detected: optimizing for throughput"
            # BF16 weights need roughly two bytes per parameter. These
            # thresholds leave additional memory for KV cache and runtime
            # overhead at the configured GPU memory utilization.
            if [ $total_vram -ge 240 ]; then
                MODEL="Qwen/Qwen2.5-72B-Instruct"
                MAX_MODEL_LEN=32768
            elif [ $total_vram -ge 80 ]; then
                MODEL="Qwen/Qwen2.5-32B-Instruct"
                MAX_MODEL_LEN=32768
            else
                MODEL="Qwen/Qwen2.5-14B-Instruct"
                MAX_MODEL_LEN=32768
            fi
            GPU_MEM_UTIL=0.90
            ;;

        *"T4"*)
            echo "Tesla T4 detected: optimizing for memory efficiency"
            TENSOR_PARALLEL=1
            MAX_MODEL_LEN=2048
            MAX_BATCHED_TOKENS=2048
            EXTRA_ARGS="--dtype float16"

            if [ $GPU_VRAM_GB -le 16 ]; then
                MODEL="Qwen/Qwen3-14B-AWQ"
                MAX_MODEL_LEN=8192
                MAX_BATCHED_TOKENS=8192
            else
                MODEL="Qwen/Qwen2.5-7B-Instruct"
            fi
            ;;

        *"V100"*)
            echo "Tesla V100 detected: balanced configuration"
            TENSOR_PARALLEL=$GPU_COUNT
            GPU_MEM_UTIL=0.85
            MAX_MODEL_LEN=8192
            EXTRA_ARGS="--dtype float16"

            if [ $total_vram -ge 96 ]; then
                MODEL="Qwen/Qwen2.5-32B-Instruct"
            else
                MODEL="Qwen/Qwen2.5-14B-Instruct"
            fi
            ;;

        *"RTX"*|*"GeForce"*)
            echo "Consumer GPU detected: conservative settings"
            TENSOR_PARALLEL=1
            GPU_MEM_UTIL=0.80
            MAX_MODEL_LEN=4096

            if [ $GPU_VRAM_GB -ge 48 ]; then
                MODEL="Qwen/Qwen2.5-14B-Instruct"
            else
                MODEL="Qwen/Qwen2.5-7B-Instruct"
            fi
            EXTRA_ARGS="--enforce-eager"
            ;;

        *)
            echo "Unknown GPU: using conservative defaults"
            TENSOR_PARALLEL=1
            GPU_MEM_UTIL=0.75
            MAX_MODEL_LEN=4096
            MODEL="Qwen/Qwen2.5-7B-Instruct"
            EXTRA_ARGS="--enforce-eager"
            ;;
    esac

    MODEL="${MODEL_OVERRIDE:-$MODEL}"
    TENSOR_PARALLEL="${TENSOR_PARALLEL_OVERRIDE:-$TENSOR_PARALLEL}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL_OVERRIDE:-$GPU_MEM_UTIL}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-$MAX_MODEL_LEN}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS_OVERRIDE:-$MAX_BATCHED_TOKENS}"
    EXTRA_ARGS="${EXTRA_ARGS_OVERRIDE:-$EXTRA_ARGS}"
}

clear_script_environment() {
    # stop-vllm.sh consumes the shared process-control inputs first. Clear all
    # start/stop script parameters only after that child exits, immediately
    # before the long-lived vLLM process is launched.
    unset AUTOVLLM_API_PORT AUTOVLLM_NFS_MOUNT_POINT AUTOVLLM_MODEL
    unset AUTOVLLM_TENSOR_PARALLEL AUTOVLLM_GPU_MEM_UTIL
    unset AUTOVLLM_MAX_MODEL_LEN AUTOVLLM_MAX_BATCHED_TOKENS AUTOVLLM_EXTRA_ARGS
    unset AUTOVLLM_SCRIPT_DIR AUTOVLLM_BIN AUTOVLLM_PID_FILE
    unset AUTOVLLM_HF_CACHE_LINK AUTOVLLM_LOG_FILE AUTOVLLM_PYTHON
    unset AUTOVLLM_PROC_ROOT AUTOVLLM_COMMAND_PATTERN
    unset AUTOVLLM_STARTUP_GRACE_PERIOD AUTOVLLM_STARTUP_LOG_LINES
    unset AUTOVLLM_STOP_TIMEOUT AUTOVLLM_STOP_INTERVAL
}

verify_flashinfer_aot() {
    if ! "$VLLM_PYTHON" -c \
        'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public' \
        &>/dev/null; then
        echo "FATAL: matching FlashInfer AOT kernels are unavailable; run setup.sh to install the matching flashinfer-cubin package" >&2
        return 1
    fi
    export FLASHINFER_DISABLE_JIT=1
}

prepare_hf_cache() {
    mkdir -p "$(dirname "$HF_CACHE_LINK")"
    if [ -d "$HF_CACHE_LINK" ] && [ ! -L "$HF_CACHE_LINK" ]; then
        echo "FATAL: Hugging Face cache target ${HF_CACHE_LINK} is a real directory; move it aside before linking the NFS cache" >&2
        return 1
    fi
    ln -sfnT "${NFS_MOUNT_POINT}" "$HF_CACHE_LINK"
}

verify_vllm_started() {
    local pid="$1"

    sleep "$STARTUP_GRACE_PERIOD"
    if is_vllm_pid "$pid"; then
        return 0
    fi

    rm -f "$PID_FILE"
    echo "FATAL: vLLM process ${pid} exited during startup; last ${STARTUP_LOG_LINES} log lines:" >&2
    if [ -f "$VLLM_LOG_FILE" ]; then
        tail -n "$STARTUP_LOG_LINES" "$VLLM_LOG_FILE" >&2
    else
        echo "(vLLM log file ${VLLM_LOG_FILE} was not created)" >&2
    fi
    return 1
}

run_vllm() {
    verify_flashinfer_aot
    prepare_hf_cache

    # Never launch over an older or orphaned server. A failed verified stop
    # aborts this script under set -e rather than registering the wrong model.
    bash "${SCRIPT_DIR}/stop-vllm.sh"
    clear_script_environment

    cat <<EOF

# vLLM Configuration
# ================================================
# GPU:                $GPU_COUNT x $GPU_MODEL ($GPU_VRAM_GB GB)
# Model:              $MODEL
# Tensor Parallel:    $TENSOR_PARALLEL
# Memory Util:        ${GPU_MEM_UTIL}
# Max Context:        $MAX_MODEL_LEN tokens
# Max Batched Tokens: $MAX_BATCHED_TOKENS tokens
# ================================================

EOF

    set -f
    # EXTRA_ARGS is an intentional word-split shell override.
    # shellcheck disable=SC2086
    "$VLLM_BIN" serve "$MODEL" \
        --host 0.0.0.0 \
        --port "${API_PORT}" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        ${EXTRA_ARGS:-} \
        > "$VLLM_LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    verify_vllm_started "$pid"
    echo "vLLM started (PID ${pid})"
}

main() {
    detect_gpu_info
    configure_vllm_params
    run_vllm
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
