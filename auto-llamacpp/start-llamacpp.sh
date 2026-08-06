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
MANAGED="${AUTOLLAMACPP_MANAGED:-0}"
FIT_TARGET_MIB="${AUTOLLAMACPP_FIT_TARGET_MIB:-512}"
MANAGED_SIZING="${AUTOLLAMACPP_MANAGED_SIZING:-auto}"
MANAGED_REQUESTED_CONTEXT="${AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT:-}"
MANAGED_REQUESTED_PARALLEL="${AUTOLLAMACPP_MANAGED_PARALLEL:-}"
MANAGED_REQUESTED_CACHE_TYPE="${AUTOLLAMACPP_MANAGED_CACHE_TYPE:-}"
SCRIPT_DIR="${AUTOLLAMACPP_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LLAMACPP_BIN="${AUTOLLAMACPP_BIN:-/usr/local/bin/llama-server}"
LLAMACPP_FIT_BIN="${AUTOLLAMACPP_FIT_BIN:-/usr/local/bin/llama-fit-params}"
LLAMACPP_INSTALL_ROOT="${AUTOLLAMACPP_INSTALL_ROOT:-/opt/llama.cpp}"
PID_FILE="${AUTOLLAMACPP_PID_FILE:-/var/run/llamacpp.pid}"
LLAMACPP_LOG_FILE="${AUTOLLAMACPP_LOG_FILE:-/var/log/llamacpp-serve.log}"
PROC_ROOT="${AUTOLLAMACPP_PROC_ROOT:-/proc}"
STARTUP_GRACE_PERIOD="${AUTOLLAMACPP_STARTUP_GRACE_PERIOD:-2}"
STARTUP_LOG_LINES="${AUTOLLAMACPP_STARTUP_LOG_LINES:-40}"
REQUIRE_CUDA="${AUTOLLAMACPP_REQUIRE_CUDA:-1}"
LLAMACPP_MAX_SEQUENCES=256
LLAMACPP_CONTEXT_ALIGNMENT=256
LLAMACPP_MAX_AGGREGATE_CONTEXT=4294967040
LLAMACPP_ESTIMATE_ROUNDING_MIB=4
LLAMACPP_PRIMARY_CACHE_TYPE=f16
LLAMACPP_FALLBACK_CACHE_TYPE=q8_0
MANAGED_CONTEXT_PER_SLOT=0
MANAGED_PARALLEL=0
MANAGED_AGGREGATE_CONTEXT=0
MANAGED_TRAIN_CONTEXT=0
MANAGED_CACHE_TYPE_K="$LLAMACPP_PRIMARY_CACHE_TYPE"
MANAGED_CACHE_TYPE_V="$LLAMACPP_PRIMARY_CACHE_TYPE"
MANAGED_FLASH_ATTN=auto
MANAGED_GPU_FREE_MIB=()

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
    if [ "$MANAGED" = "1" ]; then
        local override
        for override in \
            AUTOLLAMACPP_GPU_LAYERS \
            AUTOLLAMACPP_CTX_SIZE \
            AUTOLLAMACPP_PARALLEL \
            AUTOLLAMACPP_BATCH_SIZE \
            AUTOLLAMACPP_EXTRA_ARGS; do
            if [ -n "${!override:-}" ]; then
                echo "FATAL: ${override} is not supported for managed llama.cpp; VRAM fitting owns sizing" >&2
                return 1
            fi
        done
        if [[ ! "$FIT_TARGET_MIB" =~ ^[1-9][0-9]*$ ]]; then
            echo "FATAL: AUTOLLAMACPP_FIT_TARGET_MIB must be a positive integer MiB value" >&2
            return 1
        fi
        case "$MANAGED_SIZING" in
            auto)
                if [ -n "$MANAGED_REQUESTED_CONTEXT" ] \
                    || [ -n "$MANAGED_REQUESTED_PARALLEL" ] \
                    || [ -n "$MANAGED_REQUESTED_CACHE_TYPE" ]; then
                    echo "FATAL: automatic managed sizing does not accept custom values" >&2
                    return 1
                fi
                ;;
            custom)
                if [[ ! "$MANAGED_REQUESTED_CONTEXT" =~ ^[1-9][0-9]*$ ]]; then
                    echo "FATAL: custom context_per_slot must be a positive 256-token increment" >&2
                    return 1
                fi
                if [ "${#MANAGED_REQUESTED_CONTEXT}" -gt "${#LLAMACPP_MAX_AGGREGATE_CONTEXT}" ] \
                    || { [ "${#MANAGED_REQUESTED_CONTEXT}" -eq "${#LLAMACPP_MAX_AGGREGATE_CONTEXT}" ] \
                        && [ "$MANAGED_REQUESTED_CONTEXT" -gt "$LLAMACPP_MAX_AGGREGATE_CONTEXT" ]; }; then
                    echo "FATAL: custom aggregate context exceeds ${LLAMACPP_MAX_AGGREGATE_CONTEXT}" >&2
                    return 1
                fi
                if [ "$MANAGED_REQUESTED_CONTEXT" -lt "$LLAMACPP_CONTEXT_ALIGNMENT" ] \
                    || [ $((MANAGED_REQUESTED_CONTEXT % LLAMACPP_CONTEXT_ALIGNMENT)) -ne 0 ]; then
                    echo "FATAL: custom context_per_slot must be a positive 256-token increment" >&2
                    return 1
                fi
                if [[ ! "$MANAGED_REQUESTED_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
                    echo "FATAL: custom parallel slots must be between 1 and ${LLAMACPP_MAX_SEQUENCES}" >&2
                    return 1
                fi
                if [ "${#MANAGED_REQUESTED_PARALLEL}" -gt "${#LLAMACPP_MAX_SEQUENCES}" ] \
                    || { [ "${#MANAGED_REQUESTED_PARALLEL}" -eq "${#LLAMACPP_MAX_SEQUENCES}" ] \
                        && [ "$MANAGED_REQUESTED_PARALLEL" -gt "$LLAMACPP_MAX_SEQUENCES" ]; }; then
                    echo "FATAL: custom parallel slots must be between 1 and ${LLAMACPP_MAX_SEQUENCES}" >&2
                    return 1
                fi
                case "$MANAGED_REQUESTED_CACHE_TYPE" in
                    "$LLAMACPP_PRIMARY_CACHE_TYPE"|"$LLAMACPP_FALLBACK_CACHE_TYPE") ;;
                    *)
                        echo "FATAL: custom KV cache type must be f16 or q8_0" >&2
                        return 1
                        ;;
                esac
                if [ $((MANAGED_REQUESTED_CONTEXT * MANAGED_REQUESTED_PARALLEL)) \
                    -gt "$LLAMACPP_MAX_AGGREGATE_CONTEXT" ]; then
                    echo "FATAL: custom aggregate context exceeds ${LLAMACPP_MAX_AGGREGATE_CONTEXT}" >&2
                    return 1
                fi
                ;;
            *)
                echo "FATAL: managed sizing must be auto or custom" >&2
                return 1
                ;;
        esac
        if [ ! -x "$LLAMACPP_FIT_BIN" ]; then
            echo "FATAL: managed llama.cpp requires the VRAM planner at ${LLAMACPP_FIT_BIN}" >&2
            return 1
        fi
        return 0
    fi

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
    unset AUTOLLAMACPP_MANAGED AUTOLLAMACPP_FIT_TARGET_MIB
    unset AUTOLLAMACPP_MANAGED_SIZING AUTOLLAMACPP_MANAGED_CONTEXT_PER_SLOT
    unset AUTOLLAMACPP_MANAGED_PARALLEL AUTOLLAMACPP_MANAGED_CACHE_TYPE
    unset AUTOLLAMACPP_SCRIPT_DIR AUTOLLAMACPP_BIN AUTOLLAMACPP_FIT_BIN
    unset AUTOLLAMACPP_INSTALL_ROOT
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

managed_aggregate_context() {
    local context_per_slot="$1"
    local slots="$2"
    local requested=$((context_per_slot * slots))
    printf '%d\n' "$((
        (requested + LLAMACPP_CONTEXT_ALIGNMENT - 1)
        / LLAMACPP_CONTEXT_ALIGNMENT
        * LLAMACPP_CONTEXT_ALIGNMENT
    ))"
}

read_managed_gpu_free_memory() {
    local free_output row
    if ! free_output=$(nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits); then
        echo "FATAL: could not query free GPU memory for llama.cpp planning" >&2
        return 1
    fi

    MANAGED_GPU_FREE_MIB=()
    while IFS= read -r row; do
        row="${row//[[:space:]]/}"
        if [ -z "$row" ]; then
            continue
        fi
        if [[ ! "$row" =~ ^[0-9]+$ ]]; then
            echo "FATAL: nvidia-smi returned invalid free-memory telemetry: ${row}" >&2
            return 1
        fi
        MANAGED_GPU_FREE_MIB+=("$row")
    done <<< "$free_output"

    if [ "${#MANAGED_GPU_FREE_MIB[@]}" -ne "$GPU_COUNT" ]; then
        echo "FATAL: llama.cpp planner saw ${#MANAGED_GPU_FREE_MIB[@]} memory rows for ${GPU_COUNT} GPUs" >&2
        return 1
    fi
}

read_model_train_context() {
    local output status train_context
    status=0
    output=$("$LLAMACPP_FIT_BIN" \
        --model "$GGUF_PATH" \
        --parallel 1 \
        --kv-unified \
        --gpu-layers all \
        --verbosity 5 2>&1) || status=$?
    train_context=$(printf '%s\n' "$output" | sed -nE \
        's/.*n_ctx_train[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' | tail -n 1)
    if [[ ! "$train_context" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: llama.cpp planner could not determine the model training context" >&2
        if [ "$status" -ne 0 ]; then
            printf 'llama-fit-params exited with status %s\n' "$status" >&2
        fi
        printf '%s\n' "$output" >&2
        return 1
    fi
    printf '%s\n' "$train_context"
}

managed_candidate_fits() {
    local context_per_slot="$1"
    local slots="$2"
    local aggregate_context output row device model_mib context_mib compute_mib extra
    local index required_mib
    local -a estimated_mib=()

    aggregate_context=$(managed_aggregate_context "$context_per_slot" "$slots")
    if [ "$aggregate_context" -gt "$LLAMACPP_MAX_AGGREGATE_CONTEXT" ]; then
        return 1
    fi
    if ! output=$("$LLAMACPP_FIT_BIN" \
        --model "$GGUF_PATH" \
        --ctx-size "$aggregate_context" \
        --parallel "$slots" \
        --kv-unified \
        --gpu-layers all \
        --cache-type-k "$MANAGED_CACHE_TYPE_K" \
        --cache-type-v "$MANAGED_CACHE_TYPE_V" \
        --flash-attn "$MANAGED_FLASH_ATTN" \
        --fit-print on \
        --verbosity 0 2>&1); then
        echo "FATAL: llama.cpp failed to estimate VRAM for ${slots} slots" >&2
        printf '%s\n' "$output" >&2
        return 2
    fi

    while IFS= read -r row; do
        read -r device model_mib context_mib compute_mib extra <<< "$row"
        if [[ ! "$device" =~ ^CUDA[0-9]+$ ]]; then
            continue
        fi
        if [[ ! "$model_mib" =~ ^[0-9]+$ ]] \
            || [[ ! "$context_mib" =~ ^[0-9]+$ ]] \
            || [[ ! "$compute_mib" =~ ^[0-9]+$ ]] \
            || [ -n "${extra:-}" ]; then
            echo "FATAL: llama.cpp returned an invalid CUDA memory estimate: ${row}" >&2
            return 2
        fi
        estimated_mib+=("$((model_mib + context_mib + compute_mib))")
    done <<< "$output"

    if [ "${#estimated_mib[@]}" -eq 0 ]; then
        echo "FATAL: llama.cpp returned no CUDA device memory estimates" >&2
        return 2
    fi
    if [ "${#estimated_mib[@]}" -ne "${#MANAGED_GPU_FREE_MIB[@]}" ]; then
        echo "FATAL: llama.cpp returned ${#estimated_mib[@]} CUDA estimates for ${#MANAGED_GPU_FREE_MIB[@]} GPUs" >&2
        return 2
    fi
    for index in "${!estimated_mib[@]}"; do
        # Each of llama-fit-params' three MiB components is rounded down, so
        # their sum can hide almost 3 MiB. Four adds the next whole MiB plus a
        # one-MiB guard before applying the operator's free-VRAM target.
        required_mib=$((
            estimated_mib[index]
            + FIT_TARGET_MIB
            + LLAMACPP_ESTIMATE_ROUNDING_MIB
        ))
        if [ "$required_mib" -gt "${MANAGED_GPU_FREE_MIB[index]}" ]; then
            return 1
        fi
    done
    return 0
}

select_managed_cache_policy() {
    case "$1" in
        "$LLAMACPP_PRIMARY_CACHE_TYPE")
            MANAGED_CACHE_TYPE_K="$LLAMACPP_PRIMARY_CACHE_TYPE"
            MANAGED_CACHE_TYPE_V="$LLAMACPP_PRIMARY_CACHE_TYPE"
            MANAGED_FLASH_ATTN=auto
            ;;
        "$LLAMACPP_FALLBACK_CACHE_TYPE")
            MANAGED_CACHE_TYPE_K="$LLAMACPP_FALLBACK_CACHE_TYPE"
            MANAGED_CACHE_TYPE_V="$LLAMACPP_FALLBACK_CACHE_TYPE"
            # b10242 requires Flash Attention for a quantized V cache. Make
            # that dependency explicit so estimation and serving cannot
            # resolve AUTO differently.
            MANAGED_FLASH_ATTN=on
            ;;
        *)
            echo "FATAL: unsupported managed KV cache type: $1" >&2
            return 2
            ;;
    esac
}

plan_managed_cache_policy() {
    local train_context="$1"
    local min_context="$2"
    local low high mid best_context
    local best_slots probe upper status

    status=0
    managed_candidate_fits "$train_context" 1 || status=$?
    if [ "$status" -eq 0 ]; then
        best_context="$train_context"
    elif [ "$status" -eq 2 ]; then
        return 2
    else
        status=0
        managed_candidate_fits "$min_context" 1 || status=$?
        if [ "$status" -eq 2 ]; then
            return 2
        fi
        if [ "$status" -ne 0 ]; then
            return 1
        fi

        low=$((min_context / LLAMACPP_CONTEXT_ALIGNMENT))
        high=$((train_context / LLAMACPP_CONTEXT_ALIGNMENT))
        best_context="$min_context"
        while [ "$low" -le "$high" ]; do
            mid=$(((low + high) / 2))
            status=0
            managed_candidate_fits "$((mid * LLAMACPP_CONTEXT_ALIGNMENT))" 1 \
                || status=$?
            if [ "$status" -eq 2 ]; then
                return 2
            elif [ "$status" -eq 0 ]; then
                best_context=$((mid * LLAMACPP_CONTEXT_ALIGNMENT))
                low=$((mid + 1))
            else
                high=$((mid - 1))
            fi
        done
    fi

    best_slots=1
    probe=2
    upper=0
    while [ "$probe" -le "$LLAMACPP_MAX_SEQUENCES" ]; do
        status=0
        managed_candidate_fits "$best_context" "$probe" || status=$?
        if [ "$status" -eq 2 ]; then
            return 2
        elif [ "$status" -eq 0 ]; then
            best_slots="$probe"
            if [ "$probe" -eq "$LLAMACPP_MAX_SEQUENCES" ]; then
                break
            fi
            probe=$((probe * 2))
            if [ "$probe" -gt "$LLAMACPP_MAX_SEQUENCES" ]; then
                probe="$LLAMACPP_MAX_SEQUENCES"
            fi
        else
            upper=$((probe - 1))
            break
        fi
    done

    if [ "$upper" -gt "$best_slots" ]; then
        low=$((best_slots + 1))
        high="$upper"
        while [ "$low" -le "$high" ]; do
            mid=$(((low + high) / 2))
            status=0
            managed_candidate_fits "$best_context" "$mid" || status=$?
            if [ "$status" -eq 2 ]; then
                return 2
            elif [ "$status" -eq 0 ]; then
                best_slots="$mid"
                low=$((mid + 1))
            else
                high=$((mid - 1))
            fi
        done
    fi

    MANAGED_CONTEXT_PER_SLOT="$best_context"
    MANAGED_PARALLEL="$best_slots"
    MANAGED_AGGREGATE_CONTEXT=$(managed_aggregate_context \
        "$MANAGED_CONTEXT_PER_SLOT" "$MANAGED_PARALLEL")
    return 0
}

plan_managed_configuration() {
    local train_context min_context status

    echo "Planning llama.cpp context and concurrency from free VRAM"
    read_managed_gpu_free_memory
    train_context=$(read_model_train_context)
    MANAGED_TRAIN_CONTEXT="$train_context"

    if [ "$MANAGED_SIZING" = "custom" ]; then
        if [ "$MANAGED_REQUESTED_CONTEXT" -gt "$train_context" ]; then
            echo "FATAL: custom context_per_slot ${MANAGED_REQUESTED_CONTEXT} exceeds model training context ${train_context}" >&2
            return 1
        fi
        select_managed_cache_policy "$MANAGED_REQUESTED_CACHE_TYPE"
        status=0
        managed_candidate_fits \
            "$MANAGED_REQUESTED_CONTEXT" "$MANAGED_REQUESTED_PARALLEL" \
            || status=$?
        if [ "$status" -eq 2 ]; then
            return 1
        fi
        if [ "$status" -ne 0 ]; then
            echo "FATAL: custom llama.cpp sizing cannot fully offload while preserving ${FIT_TARGET_MIB} MiB free per GPU" >&2
            return 1
        fi
        MANAGED_CONTEXT_PER_SLOT="$MANAGED_REQUESTED_CONTEXT"
        MANAGED_PARALLEL="$MANAGED_REQUESTED_PARALLEL"
        MANAGED_AGGREGATE_CONTEXT=$(managed_aggregate_context \
            "$MANAGED_CONTEXT_PER_SLOT" "$MANAGED_PARALLEL")
        echo "Selected exact custom configuration: ${MANAGED_PARALLEL} slots x ${MANAGED_CONTEXT_PER_SLOT} tokens (${MANAGED_AGGREGATE_CONTEXT} aggregate) with ${MANAGED_CACHE_TYPE_K}/${MANAGED_CACHE_TYPE_V} KV cache"
        return 0
    fi

    min_context=4096
    if [ "$train_context" -lt "$min_context" ]; then
        min_context="$train_context"
    fi

    select_managed_cache_policy "$LLAMACPP_PRIMARY_CACHE_TYPE"
    status=0
    plan_managed_cache_policy "$train_context" "$min_context" || status=$?
    if [ "$status" -eq 2 ]; then
        return 1
    fi
    if [ "$status" -ne 0 ]; then
        echo "F16 KV cache cannot meet the ${FIT_TARGET_MIB} MiB free-VRAM target at ${min_context} context tokens; retrying with Q8_0 KV cache"
        select_managed_cache_policy "$LLAMACPP_FALLBACK_CACHE_TYPE"
        status=0
        plan_managed_cache_policy "$train_context" "$min_context" || status=$?
        if [ "$status" -eq 2 ]; then
            return 1
        fi
        if [ "$status" -ne 0 ]; then
            echo "FATAL: the model cannot fully offload with ${min_context} context tokens and ${FIT_TARGET_MIB} MiB free per GPU, even with Q8_0 KV cache" >&2
            return 1
        fi
    fi

    echo "Selected ${MANAGED_PARALLEL} slots x ${MANAGED_CONTEXT_PER_SLOT} tokens (${MANAGED_AGGREGATE_CONTEXT} aggregate) with ${MANAGED_CACHE_TYPE_K}/${MANAGED_CACHE_TYPE_V} KV cache"
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

    if [ "$MANAGED" = "1" ]; then
        plan_managed_configuration
    fi

    if [ "$MANAGED" = "1" ]; then
        cat <<EOF

# llama.cpp Managed Configuration
# ================================================
# GPU:                ${GPU_COUNT} x ${GPU_MODEL} (${GPU_VRAM_GB} GB)
# Model:              ${MODEL_ALIAS}
# GGUF:               ${GGUF_PATH}
# VRAM Fit Target:    ${FIT_TARGET_MIB} MiB free per GPU
# Training Context:   ${MANAGED_TRAIN_CONTEXT} tokens
# Context Per Slot:   ${MANAGED_CONTEXT_PER_SLOT} tokens
# Parallel Slots:     ${MANAGED_PARALLEL}
# Aggregate Context:  ${MANAGED_AGGREGATE_CONTEXT} tokens
# KV Cache:           K=${MANAGED_CACHE_TYPE_K}, V=${MANAGED_CACHE_TYPE_V}
# Flash Attention:    ${MANAGED_FLASH_ATTN}
# ================================================

EOF
    else
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
    fi

    if [ "$MANAGED" = "1" ]; then
        printf 'qiip_fit_plan: sizing=%s train_context=%s context_per_slot=%s slots=%s aggregate_context=%s fit_target_mib=%s cache_type_k=%s cache_type_v=%s flash_attn=%s\n' \
            "$MANAGED_SIZING" \
            "$MANAGED_TRAIN_CONTEXT" \
            "$MANAGED_CONTEXT_PER_SLOT" \
            "$MANAGED_PARALLEL" \
            "$MANAGED_AGGREGATE_CONTEXT" \
            "$FIT_TARGET_MIB" \
            "$MANAGED_CACHE_TYPE_K" \
            "$MANAGED_CACHE_TYPE_V" \
            "$MANAGED_FLASH_ATTN" > "$LLAMACPP_LOG_FILE"
        # llama.cpp's auto parallel value is a fixed four, not a VRAM fit. The
        # planner uses llama-fit-params to select the largest full-context slot
        # count that preserves the requested free-memory margin. A unified KV
        # buffer sized to slots * context lets every slot reach model context.
        # LLAMA_LOG_INFO sizing records require trace verbosity 4; --verbose
        # would instead enable debug-level probe noise.
        "$LLAMACPP_BIN" \
            --model "$GGUF_PATH" \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --alias "$MODEL_ALIAS" \
            --ctx-size "$MANAGED_AGGREGATE_CONTEXT" \
            --parallel "$MANAGED_PARALLEL" \
            --kv-unified \
            --gpu-layers all \
            --cache-type-k "$MANAGED_CACHE_TYPE_K" \
            --cache-type-v "$MANAGED_CACHE_TYPE_V" \
            --flash-attn "$MANAGED_FLASH_ATTN" \
            --fit off \
            --verbosity 4 \
            --metrics \
            >> "$LLAMACPP_LOG_FILE" 2>&1 &
    else
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
            --metrics \
            ${EXTRA_ARGS:-} \
            > "$LLAMACPP_LOG_FILE" 2>&1 &
    fi

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
