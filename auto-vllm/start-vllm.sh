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
ATTENTION_BACKEND_OVERRIDE="${AUTOVLLM_ATTENTION_BACKEND:-}"
FLASHINFER_CACHE="${AUTOVLLM_FLASHINFER_CACHE_DIR:-/var/cache/flashinfer}"
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

# Ignore legacy script inputs instead of leaking them into vLLM's reserved
# environment namespace. VLLM_MODEL was an internal gateway handoff;
# VLLM_PORT is the upstream collision that motivated the namespace change.
unset VLLM_MODEL VLLM_PORT VLLM_TENSOR_PARALLEL VLLM_GPU_MEM_UTIL
unset VLLM_MAX_MODEL_LEN VLLM_MAX_BATCHED_TOKENS VLLM_EXTRA_ARGS
unset VLLM_ATTENTION_BACKEND FLASHINFER_DISABLE_JIT FLASHINFER_CACHE_DIR

# shellcheck source=auto-vllm/vllm-process.sh
source "${SCRIPT_DIR}/vllm-process.sh"
# shellcheck source=auto-vllm/preflight.sh
source "${SCRIPT_DIR}/preflight.sh"

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
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | xargs)
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
    unset AUTOVLLM_ATTENTION_BACKEND AUTOVLLM_FLASHINFER_CACHE_DIR
}

configure_attention_backend() {
    local major="${GPU_COMPUTE_CAP%%.*}"
    local minor="${GPU_COMPUTE_CAP#*.}"
    local sm=$(( major * 10 + minor ))

    # SM90+ (Hopper/Blackwell): FlashInfer wins.
    # SM80-89 (Ampere/Ada): FLASH_ATTN ships prebuilt wheels, no compiler needed.
    # Below SM80: let vLLM pick (xformers, eager, etc.).
    local backend=""
    if [ "$sm" -ge 90 ]; then
        backend="FLASHINFER"
    elif [ "$sm" -ge 80 ]; then
        backend="FLASH_ATTN"
    fi

    if [ -n "$ATTENTION_BACKEND_OVERRIDE" ]; then
        if [ "$ATTENTION_BACKEND_OVERRIDE" = "FLASHINFER" ] && [ "$sm" -lt 90 ]; then
            echo "WARNING: FlashInfer on SM${sm} may require JIT compilation; FLASH_ATTN ships prebuilt for this architecture" >&2
        fi
        backend="$ATTENTION_BACKEND_OVERRIDE"
    fi

    if [ -z "$backend" ]; then
        echo "SM${sm}: no attention backend forced; vLLM will select its default"
        return 0
    fi

    export VLLM_ATTENTION_BACKEND="$backend"
    echo "Attention backend: ${backend} (SM${sm})"

    if [ "$backend" = "FLASHINFER" ]; then
        configure_flashinfer
    fi
}

configure_flashinfer() {
    if "$VLLM_PYTHON" -c \
        'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public' \
        &>/dev/null; then
        export FLASHINFER_DISABLE_JIT=1
        echo "FlashInfer: AOT kernels matched; JIT disabled"
        return 0
    fi

    echo "FlashInfer: AOT kernels unavailable; verifying JIT toolchain"
    verify_jit_toolchain

    unset FLASHINFER_DISABLE_JIT
    mkdir -p "$FLASHINFER_CACHE"
    export FLASHINFER_CACHE_DIR="$FLASHINFER_CACHE"
    echo "FlashInfer: JIT enabled; cache at ${FLASHINFER_CACHE}"
}

verify_jit_toolchain() {
    local venv_bin missing=()
    venv_bin="$(dirname "$VLLM_BIN")"

    if ! "$venv_bin/ninja" --version &>/dev/null; then
        missing+=("ninja (dnf install ninja-build / apt install ninja-build)")
    fi

    if ! command -v nvcc &>/dev/null; then
        missing+=("nvcc (install cuda-toolkit)")
    else
        local nvcc_ver torch_cuda_ver
        nvcc_ver=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' | head -1) || true
        torch_cuda_ver=$("$VLLM_PYTHON" -c 'import torch; print(".".join(torch.version.cuda.split(".")[:2]))' 2>/dev/null) || true
        if [ -n "$nvcc_ver" ] && [ -n "$torch_cuda_ver" ] && [ "$nvcc_ver" != "$torch_cuda_ver" ]; then
            missing+=("nvcc ${torch_cuda_ver} (installed: ${nvcc_ver}; must match torch.version.cuda)")
        fi
    fi

    if ! command -v gcc &>/dev/null; then
        missing+=("gcc")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "FATAL: FlashInfer JIT requires:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        return 1
    fi
}

prepare_hf_cache() {
    mkdir -p "$(dirname "$HF_CACHE_LINK")"
    if [ -d "$HF_CACHE_LINK" ] && [ ! -L "$HF_CACHE_LINK" ]; then
        echo "FATAL: Hugging Face cache target ${HF_CACHE_LINK} is a real directory; move it aside before linking the NFS cache" >&2
        return 1
    fi
    ln -sfnT "${NFS_MOUNT_POINT}" "$HF_CACHE_LINK"
    echo "HF cache: ${HF_CACHE_LINK} → ${NFS_MOUNT_POINT} (symlink into NFS; HF_HOME resolves through this)"
}

prestage_model_weights() {
    local model="$1"

    if [ -d "$model" ]; then
        return 0
    fi

    local avail_bytes model_bytes
    avail_bytes=$(df --output=avail -B1 "$NFS_MOUNT_POINT" 2>/dev/null | tail -1 | xargs) || true
    model_bytes=$("$VLLM_PYTHON" -c "
import sys; from huggingface_hub import model_info
print(sum(s.size or 0 for s in model_info(sys.argv[1]).siblings))
" "$model" 2>/dev/null) || true

    if [ -n "$model_bytes" ] && [ -n "$avail_bytes" ] \
        && [ "$model_bytes" -gt "$avail_bytes" ]; then
        local avail_gb=$((avail_bytes / 1073741824))
        local need_gb=$((model_bytes / 1073741824))
        echo "FATAL: ${NFS_MOUNT_POINT} has ${avail_gb}GB free but ${model} needs ${need_gb}GB" >&2
        return 1
    fi

    echo "Pre-staging model weights in single process: ${model}"
    HF_HOME="$NFS_MOUNT_POINT" "$VLLM_PYTHON" -c "
import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
" "$model"
    echo "Model weights staged; vLLM will launch with HF_HUB_OFFLINE=1"
    export HF_HUB_OFFLINE=1
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
    configure_attention_backend

    if [ -z "${INVOCATION_ID:-}" ]; then
        # Never launch over an older or orphaned server. A failed verified stop
        # aborts this script under set -e rather than registering the wrong model.
        bash "${SCRIPT_DIR}/stop-vllm.sh"
    fi
    clear_script_environment

    # Running the venv binary directly doesn't activate the venv, so tools
    # like ninja (needed by FlashInfer JIT) aren't on PATH.
    local venv_bin_dir
    venv_bin_dir="$(dirname "$VLLM_BIN")"
    export PATH="${venv_bin_dir}:$PATH"

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
    # Under systemd (INVOCATION_ID set), exec into vLLM so systemd owns the
    # process directly. Journal captures stdout/stderr.
    if [ -n "${INVOCATION_ID:-}" ]; then
        # shellcheck disable=SC2086
        exec "$VLLM_BIN" serve "$MODEL" \
            --host 0.0.0.0 \
            --port "${API_PORT}" \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            ${EXTRA_ARGS:-}
    fi

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

    EXPECTED_GPU_COUNT="$GPU_COUNT"
    EXPECTED_TENSOR_PARALLEL="$TENSOR_PARALLEL"
    MODEL_PATH="$MODEL"
    export EXPECTED_GPU_COUNT EXPECTED_TENSOR_PARALLEL MODEL_PATH
    prepare_hf_cache
    run_preflight
    prestage_model_weights "$MODEL"
    run_vllm
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
