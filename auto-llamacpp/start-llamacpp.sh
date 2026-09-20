#!/bin/bash
set -euo pipefail

API_PORT="${AUTOLLAMACPP_PORT:-8000}"
NFS_MOUNT_POINT="${AUTOLLAMACPP_NFS_MOUNT_POINT:-/srv/hf-cache}"
# Shared name (setup.sh and the gateway use AUTOVLLM_NFS_EXPORT for both engines).
NFS_EXPORT="${AUTOVLLM_NFS_EXPORT:-}"
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
MANAGED_ALLOW_ESTIMATOR_OVERRUN="${AUTOLLAMACPP_MANAGED_ALLOW_ESTIMATOR_OVERRUN:-0}"
PROFILE_ID="${AUTOLLAMACPP_PROFILE_ID:-}"
PROFILE_VERSION="${AUTOLLAMACPP_PROFILE_VERSION:-}"
PROFILE_UBATCH="${AUTOLLAMACPP_PROFILE_UBATCH:-}"
PROFILE_REQUIRED_FREE_MIB="${AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB:-}"
PROFILE_GPU_NAME="${AUTOLLAMACPP_PROFILE_GPU_NAME:-}"
PROFILE_GPU_MIN_TOTAL_MIB="${AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB:-}"
PROFILE_SPEC_TYPE="${AUTOLLAMACPP_PROFILE_SPEC_TYPE:-}"
PROFILE_SPEC_DRAFT_N_MAX="${AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX:-}"
PROFILE_DRAFT_CACHE_TYPE="${AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE:-}"
PROFILE_DRAFT_GGUF_RELATIVE_PATH="${AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH:-}"
PROFILE_TEMPERATURE="${AUTOLLAMACPP_PROFILE_TEMPERATURE:-}"
PROFILE_TOP_P="${AUTOLLAMACPP_PROFILE_TOP_P:-}"
PROFILE_TOP_K="${AUTOLLAMACPP_PROFILE_TOP_K:-}"
PROFILE_MIN_P="${AUTOLLAMACPP_PROFILE_MIN_P:-}"
PROFILE_PRESENCE_PENALTY="${AUTOLLAMACPP_PROFILE_PRESENCE_PENALTY:-}"
PROFILE_DISABLE_CUDA_GRAPHS="${AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS:-0}"
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
LLAMACPP_PROFILE_ONLY_CACHE_TYPE=q4_0
PROFILE_DRAFT_GGUF_PATH=""
PROFILE_GPU_UUID=""
MANAGED_CONTEXT_PER_SLOT=0
MANAGED_PARALLEL=0
MANAGED_AGGREGATE_CONTEXT=0
MANAGED_TRAIN_CONTEXT=0
MANAGED_CACHE_TYPE_K="$LLAMACPP_PRIMARY_CACHE_TYPE"
MANAGED_CACHE_TYPE_V="$LLAMACPP_PRIMARY_CACHE_TYPE"
MANAGED_FLASH_ATTN=auto
MANAGED_ESTIMATOR_OVERRUN_USED=false
MANAGED_GPU_FREE_MIB=()

# shellcheck source=auto-llamacpp/llamacpp-process.sh
source "${SCRIPT_DIR}/llamacpp-process.sh"
# Shared storage primitives (source/fstype/option verification).
# shellcheck disable=SC1091 source=../common/setup-base.sh
source "$(cd -- "${SCRIPT_DIR}/.." && pwd)/common/setup-base.sh"

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

# Resolve one cache-relative GGUF path under the NFS mount and print the
# validated path. $1 is the relative path, $2 the variable named in errors and
# $3 a short description of the file.
resolve_shared_gguf() {
    local relative="$1"
    local variable="$2"
    local description="$3"
    if [[ "$relative" == /* ]]; then
        echo "FATAL: ${variable} must be relative to the NFS mount" >&2
        return 1
    fi
    case "$relative" in
        ../*|*/../*|.|./*|*/./*|*//*|*\\*)
            echo "FATAL: ${variable} must be a canonical relative POSIX path" >&2
            return 1
            ;;
    esac
    case "$relative" in
        *.gguf) ;;
        *)
            echo "FATAL: ${variable} must end in .gguf" >&2
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
    candidate="${mount_root}/${relative}"
    resolved=$(readlink -f -- "$candidate") || {
        echo "FATAL: ${description} is unavailable: ${relative}" >&2
        return 1
    }
    case "$resolved" in
        "${mount_root}"/*) ;;
        *)
            echo "FATAL: ${description} escapes the NFS mount" >&2
            return 1
            ;;
    esac
    if [ ! -f "$resolved" ]; then
        echo "FATAL: ${description} is not a regular file: ${relative}" >&2
        return 1
    fi
    # llama.cpp derives sibling split paths from the entrypoint filename. Keep
    # the validated symlink path so the -00001-of-0000N.gguf suffix survives.
    # Verify every declared shard of a split-family GGUF: llama.cpp derives
    # sibling paths from the entrypoint filename, so a missing shard surfaces
    # later only as a failed load. The gateway owns the exact file set/sizes
    # (artifacts.py); the node verifies count + readability of the family.
    local base
    base=$(basename -- "$candidate")
    if [[ "$base" =~ ^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        local prefix family_total dir missing total_shards
        prefix="${BASH_REMATCH[1]}"
        family_total="${BASH_REMATCH[3]}"
        dir=$(dirname -- "$candidate")
        missing=0
        # The padded total keeps leading zeroes (used verbatim in sibling
        # filenames); Bash arithmetic would read 00010 as octal, so force
        # base 10 for the loop bound.
        total_shards=$((10#$family_total))
        local i shard_path
        for (( i = 1; i <= total_shards; i++ )); do
            shard_path="$(printf '%s/%s-%05d-of-%s.gguf' "$dir" "$prefix" "$i" "$family_total")"
            if [ ! -f "$shard_path" ] || [ ! -r "$shard_path" ] || [ ! -s "$shard_path" ]; then
                echo "FATAL: split GGUF shard missing, unreadable, or empty: ${shard_path}" >&2
                missing=1
            fi
        done
        if [ "$missing" -ne 0 ]; then
            return 1
        fi
    elif [ ! -r "$resolved" ]; then
        echo "FATAL: ${description} is not readable: ${relative}" >&2
        return 1
    fi
    printf '%s\n' "$candidate"
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
    GGUF_PATH=$(resolve_shared_gguf \
        "$GGUF_RELATIVE_PATH" AUTOLLAMACPP_GGUF_PATH "selected GGUF artifact")
    if [ -n "$PROFILE_DRAFT_GGUF_RELATIVE_PATH" ]; then
        PROFILE_DRAFT_GGUF_PATH=$(resolve_shared_gguf \
            "$PROFILE_DRAFT_GGUF_RELATIVE_PATH" \
            AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH "selected draft GGUF artifact")
        if [ "$PROFILE_DRAFT_GGUF_PATH" = "$GGUF_PATH" ]; then
            echo "FATAL: the draft GGUF artifact must differ from the target" >&2
            return 1
        fi
    fi
}

# Catalog profiles replace the VRAM planner with one measured configuration.
# Every input is an enumeration or a bounded number: none of them can carry
# additional llama-server argv.
validate_profile_inputs() {
    local decimal='^(0|[1-9][0-9]{0,2})(\.[0-9]{1,4})?$'
    if [ "$MANAGED_ALLOW_ESTIMATOR_OVERRUN" != "0" ]; then
        echo "FATAL: profile sizing does not accept estimator overrun" >&2
        return 1
    fi
    if [[ ! "$PROFILE_ID" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_ID must be a catalog profile id" >&2
        return 1
    fi
    if [[ ! "$PROFILE_VERSION" =~ ^[1-9][0-9]{0,8}$ ]]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_VERSION must be a positive integer" >&2
        return 1
    fi
    if [[ ! "$MANAGED_REQUESTED_CONTEXT" =~ ^[1-9][0-9]{0,9}$ ]] \
        || [ "$MANAGED_REQUESTED_CONTEXT" -gt "$LLAMACPP_MAX_AGGREGATE_CONTEXT" ] \
        || [ $((MANAGED_REQUESTED_CONTEXT % LLAMACPP_CONTEXT_ALIGNMENT)) -ne 0 ]; then
        echo "FATAL: profile context_per_slot must be a positive 256-token increment" >&2
        return 1
    fi
    if [ "$MANAGED_REQUESTED_PARALLEL" != "1" ]; then
        echo "FATAL: profile sizing serves exactly one slot" >&2
        return 1
    fi
    case "$MANAGED_REQUESTED_CACHE_TYPE" in
        "$LLAMACPP_PRIMARY_CACHE_TYPE"|"$LLAMACPP_FALLBACK_CACHE_TYPE"|"$LLAMACPP_PROFILE_ONLY_CACHE_TYPE") ;;
        *)
            echo "FATAL: profile KV cache type must be f16, q8_0 or q4_0" >&2
            return 1
            ;;
    esac
    if [[ ! "$PROFILE_UBATCH" =~ ^[1-9][0-9]{1,3}$ ]] \
        || [ "$PROFILE_UBATCH" -lt 32 ] || [ "$PROFILE_UBATCH" -gt 4096 ]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_UBATCH must be between 32 and 4096" >&2
        return 1
    fi
    if [[ ! "$PROFILE_REQUIRED_FREE_MIB" =~ ^[1-9][0-9]{0,6}$ ]]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB must be a positive integer MiB value" >&2
        return 1
    fi
    if [[ ! "$PROFILE_GPU_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9\ ._-]{1,127}$ ]]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_GPU_NAME must be an nvidia-smi product name" >&2
        return 1
    fi
    if [[ ! "$PROFILE_GPU_MIN_TOTAL_MIB" =~ ^[1-9][0-9]{0,6}$ ]]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB must be a positive integer MiB value" >&2
        return 1
    fi
    if [[ ! "$PROFILE_SPEC_DRAFT_N_MAX" =~ ^[1-9][0-9]?$ ]] \
        || [ "$PROFILE_SPEC_DRAFT_N_MAX" -gt 16 ]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX must be between 1 and 16" >&2
        return 1
    fi
    case "$PROFILE_SPEC_TYPE" in
        draft-mtp)
            if [ -n "$PROFILE_DRAFT_GGUF_RELATIVE_PATH" ]; then
                echo "FATAL: MTP drafts from the target file and takes no draft GGUF" >&2
                return 1
            fi
            case "$PROFILE_DRAFT_CACHE_TYPE" in
                "$LLAMACPP_PRIMARY_CACHE_TYPE"|"$LLAMACPP_FALLBACK_CACHE_TYPE"|"$LLAMACPP_PROFILE_ONLY_CACHE_TYPE") ;;
                *)
                    echo "FATAL: MTP profiles must state an f16, q8_0 or q4_0 draft cache type" >&2
                    return 1
                    ;;
            esac
            ;;
        draft-mtp-assistant)
            # An MTP head shipped as its own GGUF (Gemma 4). It has no KV
            # cache of its own, so a draft cache type would be meaningless.
            if [ -z "$PROFILE_DRAFT_GGUF_RELATIVE_PATH" ]; then
                echo "FATAL: MTP assistant profiles require AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH" >&2
                return 1
            fi
            if [ -n "$PROFILE_DRAFT_CACHE_TYPE" ]; then
                echo "FATAL: an MTP assistant shares the target KV cache and takes no draft cache type" >&2
                return 1
            fi
            ;;
        draft-dflash)
            if [ -z "$PROFILE_DRAFT_GGUF_RELATIVE_PATH" ]; then
                echo "FATAL: DFlash profiles require AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH" >&2
                return 1
            fi
            if [ -n "$PROFILE_DRAFT_CACHE_TYPE" ]; then
                echo "FATAL: DFlash profiles use the default draft cache type" >&2
                return 1
            fi
            ;;
        *)
            echo "FATAL: AUTOLLAMACPP_PROFILE_SPEC_TYPE must be draft-mtp, draft-mtp-assistant or draft-dflash" >&2
            return 1
            ;;
    esac
    if [[ ! "$PROFILE_TEMPERATURE" =~ $decimal ]] \
        || [[ ! "$PROFILE_TOP_P" =~ $decimal ]]; then
        echo "FATAL: profile temperature and top_p must be plain decimals" >&2
        return 1
    fi
    if [[ ! "$PROFILE_TOP_K" =~ ^(0|[1-9][0-9]{0,3})$ ]]; then
        echo "FATAL: profile top_k must be an integer" >&2
        return 1
    fi
    if [ -n "$PROFILE_MIN_P" ] && [[ ! "$PROFILE_MIN_P" =~ $decimal ]]; then
        echo "FATAL: profile min_p must be a plain decimal" >&2
        return 1
    fi
    if [ -n "$PROFILE_PRESENCE_PENALTY" ] \
        && [[ ! "${PROFILE_PRESENCE_PENALTY#-}" =~ $decimal ]]; then
        echo "FATAL: profile presence_penalty must be a plain decimal" >&2
        return 1
    fi
    if [ "$PROFILE_DISABLE_CUDA_GRAPHS" != "0" ] \
        && [ "$PROFILE_DISABLE_CUDA_GRAPHS" != "1" ]; then
        echo "FATAL: AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS must be 0 or 1" >&2
        return 1
    fi
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
        if [ "$MANAGED_ALLOW_ESTIMATOR_OVERRUN" != "0" ] \
            && [ "$MANAGED_ALLOW_ESTIMATOR_OVERRUN" != "1" ]; then
            echo "FATAL: AUTOLLAMACPP_MANAGED_ALLOW_ESTIMATOR_OVERRUN must be 0 or 1" >&2
            return 1
        fi
        if [ "$MANAGED_SIZING" != "profile" ]; then
            local profile_input
            for profile_input in \
                AUTOLLAMACPP_PROFILE_ID \
                AUTOLLAMACPP_PROFILE_VERSION \
                AUTOLLAMACPP_PROFILE_UBATCH \
                AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB \
                AUTOLLAMACPP_PROFILE_GPU_NAME \
                AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB \
                AUTOLLAMACPP_PROFILE_SPEC_TYPE \
                AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX \
                AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE \
                AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH \
                AUTOLLAMACPP_PROFILE_TEMPERATURE \
                AUTOLLAMACPP_PROFILE_TOP_P \
                AUTOLLAMACPP_PROFILE_TOP_K \
                AUTOLLAMACPP_PROFILE_MIN_P \
                AUTOLLAMACPP_PROFILE_PRESENCE_PENALTY; do
                if [ -n "${!profile_input:-}" ]; then
                    echo "FATAL: ${profile_input} is only valid with profile sizing" >&2
                    return 1
                fi
            done
            if [ "$PROFILE_DISABLE_CUDA_GRAPHS" != "0" ]; then
                echo "FATAL: AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS is only valid with profile sizing" >&2
                return 1
            fi
        fi
        case "$MANAGED_SIZING" in
            profile)
                validate_profile_inputs
                ;;
            auto)
                if [ -n "$MANAGED_REQUESTED_CONTEXT" ] \
                    || [ -n "$MANAGED_REQUESTED_PARALLEL" ] \
                    || [ -n "$MANAGED_REQUESTED_CACHE_TYPE" ]; then
                    echo "FATAL: automatic managed sizing does not accept custom values" >&2
                    return 1
                fi
                if [ "$MANAGED_ALLOW_ESTIMATOR_OVERRUN" != "0" ]; then
                    echo "FATAL: automatic managed sizing does not accept estimator overrun" >&2
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
                echo "FATAL: managed sizing must be auto, custom or profile" >&2
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
        case "$PROFILE_BUCKET" in
            hopper|ampere-a100)
                echo "High-end GPU detected: full GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=32768
                PARALLEL=8
                ;;
            turing)
                echo "Tesla T4 detected: partial GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=8192
                PARALLEL=4
                ;;
            consumer-ada|consumer-ampere|consumer-turing)
                echo "Consumer GPU detected: conservative GPU offload"
                N_GPU_LAYERS=99
                CTX_SIZE=8192
                PARALLEL=4
                ;;
            ampere-a30|ga102-dc|volta)
                echo "Data-center GPU detected: conservative offload"
                N_GPU_LAYERS=99
                CTX_SIZE=4096
                PARALLEL=4
                ;;
            *)
                echo "FATAL: no runtime profile selected for this hardware; run setup.sh or check select_runtime_profile" >&2
                return 1
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
    unset AUTOLLAMACPP_MANAGED_ALLOW_ESTIMATOR_OVERRUN
    unset AUTOVLLM_NFS_EXPORT
    unset AUTOLLAMACPP_PROFILE_ID AUTOLLAMACPP_PROFILE_VERSION
    unset AUTOLLAMACPP_PROFILE_UBATCH AUTOLLAMACPP_PROFILE_REQUIRED_FREE_MIB
    unset AUTOLLAMACPP_PROFILE_GPU_NAME AUTOLLAMACPP_PROFILE_GPU_MIN_TOTAL_MIB
    unset AUTOLLAMACPP_PROFILE_SPEC_TYPE AUTOLLAMACPP_PROFILE_SPEC_DRAFT_N_MAX
    unset AUTOLLAMACPP_PROFILE_DRAFT_CACHE_TYPE AUTOLLAMACPP_PROFILE_DRAFT_GGUF_PATH
    unset AUTOLLAMACPP_PROFILE_TEMPERATURE AUTOLLAMACPP_PROFILE_TOP_P
    unset AUTOLLAMACPP_PROFILE_TOP_K AUTOLLAMACPP_PROFILE_MIN_P
    unset AUTOLLAMACPP_PROFILE_PRESENCE_PENALTY
    unset AUTOLLAMACPP_PROFILE_DISABLE_CUDA_GRAPHS
    unset GGML_CUDA_DISABLE_GRAPHS
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
            # llama.cpp requires Flash Attention for a quantized V cache. Make
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

# A catalog profile is one measured configuration for one GPU. llama-fit-params
# cannot estimate a speculative draft, so the launch gate is the profile's
# measured requirement instead: free memory as nvidia-smi reports it before
# the launch must cover that requirement plus the operator's reserve.
plan_profile_configuration() {
    local train_context="$1"
    local free_mib needed_mib uuid_output
    if [ "$GPU_COUNT" -ne 1 ]; then
        echo "FATAL: catalog profiles support exactly one GPU per host; found ${GPU_COUNT}" >&2
        return 1
    fi
    # The profile was measured on one GPU product. A host whose inventory
    # string is stale must not run it on something else.
    if [ "$GPU_MODEL" != "$PROFILE_GPU_NAME" ]; then
        echo "FATAL: profile ${PROFILE_ID} was planned for '${PROFILE_GPU_NAME}', but this host has '${GPU_MODEL}'" >&2
        return 1
    fi
    if [ "$GPU_VRAM_MB" -lt "$PROFILE_GPU_MIN_TOTAL_MIB" ]; then
        echo "FATAL: profile ${PROFILE_ID} needs a GPU with at least ${PROFILE_GPU_MIN_TOTAL_MIB} MiB, but this one reports ${GPU_VRAM_MB} MiB" >&2
        return 1
    fi
    if [ "$MANAGED_REQUESTED_CONTEXT" -gt "$train_context" ]; then
        echo "FATAL: profile context_per_slot ${MANAGED_REQUESTED_CONTEXT} exceeds model training context ${train_context}" >&2
        return 1
    fi
    if ! uuid_output=$(nvidia-smi --query-gpu=uuid --format=csv,noheader); then
        echo "FATAL: could not query the GPU UUID for the profile launch" >&2
        return 1
    fi
    PROFILE_GPU_UUID="${uuid_output//[[:space:]]/}"
    if [[ ! "$PROFILE_GPU_UUID" =~ ^GPU-[0-9a-fA-F-]{8,64}$ ]]; then
        echo "FATAL: nvidia-smi returned an invalid GPU UUID: ${PROFILE_GPU_UUID}" >&2
        return 1
    fi
    free_mib="${MANAGED_GPU_FREE_MIB[0]}"
    needed_mib=$((PROFILE_REQUIRED_FREE_MIB + FIT_TARGET_MIB))
    if [ "$free_mib" -lt "$needed_mib" ]; then
        echo "FATAL: profile ${PROFILE_ID} needs ${PROFILE_REQUIRED_FREE_MIB} MiB plus a ${FIT_TARGET_MIB} MiB reserve, but the GPU has ${free_mib} MiB free" >&2
        return 1
    fi
    MANAGED_CACHE_TYPE_K="$MANAGED_REQUESTED_CACHE_TYPE"
    MANAGED_CACHE_TYPE_V="$MANAGED_REQUESTED_CACHE_TYPE"
    MANAGED_FLASH_ATTN=on
    MANAGED_CONTEXT_PER_SLOT="$MANAGED_REQUESTED_CONTEXT"
    MANAGED_PARALLEL=1
    MANAGED_AGGREGATE_CONTEXT=$(managed_aggregate_context \
        "$MANAGED_CONTEXT_PER_SLOT" "$MANAGED_PARALLEL")
    echo "Selected catalog profile ${PROFILE_ID} v${PROFILE_VERSION}: ${MANAGED_CONTEXT_PER_SLOT} tokens, ${MANAGED_CACHE_TYPE_K}/${MANAGED_CACHE_TYPE_V} KV cache, ${PROFILE_SPEC_TYPE} n_max ${PROFILE_SPEC_DRAFT_N_MAX}; ${free_mib} MiB free covers ${needed_mib} MiB"
}

plan_managed_configuration() {
    local train_context min_context status

    echo "Planning llama.cpp context and concurrency from free VRAM"
    read_managed_gpu_free_memory
    train_context=$(read_model_train_context)
    MANAGED_TRAIN_CONTEXT="$train_context"

    if [ "$MANAGED_SIZING" = "profile" ]; then
        plan_profile_configuration "$train_context"
        return
    fi

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
            if [ "$MANAGED_ALLOW_ESTIMATOR_OVERRUN" != "1" ]; then
                echo "FATAL: custom llama.cpp sizing cannot fully offload while preserving ${FIT_TARGET_MIB} MiB free per GPU" >&2
                return 1
            fi
            MANAGED_ESTIMATOR_OVERRUN_USED=true
            echo "WARNING: llama.cpp estimator predicts less than ${FIT_TARGET_MIB} MiB free per GPU; attempting the exact custom configuration and verifying actual post-load VRAM"
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
# Profile:            ${PROFILE_BUCKET:-unknown} (${PROFILE_REASON:-})
# Model:              ${MODEL_ALIAS}
# GGUF:               ${GGUF_PATH}
# VRAM Fit Target:    ${FIT_TARGET_MIB} MiB free per GPU
# Training Context:   ${MANAGED_TRAIN_CONTEXT} tokens
# Context Per Slot:   ${MANAGED_CONTEXT_PER_SLOT} tokens
# Parallel Slots:     ${MANAGED_PARALLEL}
# Aggregate Context:  ${MANAGED_AGGREGATE_CONTEXT} tokens
# KV Cache:           K=${MANAGED_CACHE_TYPE_K}, V=${MANAGED_CACHE_TYPE_V}
# Flash Attention:    ${MANAGED_FLASH_ATTN}
# Estimator Overrun:  ${MANAGED_ESTIMATOR_OVERRUN_USED}
# ================================================

EOF
        if [ "$MANAGED_SIZING" = "profile" ]; then
            cat <<EOF
# Catalog Profile:    ${PROFILE_ID} v${PROFILE_VERSION}
# Micro-batch:        ${PROFILE_UBATCH}
# Speculation:        ${PROFILE_SPEC_TYPE}, n_max ${PROFILE_SPEC_DRAFT_N_MAX}
# Draft GGUF:         ${PROFILE_DRAFT_GGUF_PATH:-(target file)}
# Required Free VRAM: ${PROFILE_REQUIRED_FREE_MIB} MiB before launch
# GPU UUID:           ${PROFILE_GPU_UUID}
# ================================================

EOF
        fi
    else
        cat <<EOF

# llama.cpp Configuration
# ================================================
# GPU:                ${GPU_COUNT} x ${GPU_MODEL} (${GPU_VRAM_GB} GB)
# Profile:            ${PROFILE_BUCKET:-unknown} (${PROFILE_REASON:-})
# Model:              ${MODEL_ALIAS}
# GGUF:               ${GGUF_PATH}
# GPU Layers:         ${N_GPU_LAYERS}
# Context Size:       ${CTX_SIZE} tokens
# Parallel Slots:     ${PARALLEL}
# ================================================

EOF
    fi

    # The recorder unlocks at the serving-process handoff so the long-lived
    # engine and its log sink cannot hold the host mutation fence after the
    # launch step completes.
    if [ -n "${QIIP_LOCK_FD:-}" ] && [[ "${QIIP_LOCK_FD}" =~ ^[0-9]+$ ]] && [ "${QIIP_LOCK_FD}" -ge 3 ]; then
        eval "exec ${QIIP_LOCK_FD}>&-"
    fi

    local engine_log_fd
    if [ -n "${QIIP_LOG_CONFIG:-}" ]; then
        # "exec" matters: bash 5.1 (RHEL 9) otherwise keeps a wrapper shell alive
        # for the process substitution, and that shell still holds this script's
        # stdout and stderr. The gateway's command worker reads those until EOF,
        # so it would never see the start command finish.
        exec {engine_log_fd}> >(exec python3 "${SCRIPT_DIR}/../common/provision-logs.py" engine >/dev/null 2>&1)
    else
        exec {engine_log_fd}> "$LLAMACPP_LOG_FILE"
    fi

    if [ "$MANAGED" = "1" ]; then
        printf 'qiip_fit_plan: sizing=%s train_context=%s context_per_slot=%s slots=%s aggregate_context=%s fit_target_mib=%s cache_type_k=%s cache_type_v=%s flash_attn=%s estimator_overrun_used=%s\n' \
            "$MANAGED_SIZING" \
            "$MANAGED_TRAIN_CONTEXT" \
            "$MANAGED_CONTEXT_PER_SLOT" \
            "$MANAGED_PARALLEL" \
            "$MANAGED_AGGREGATE_CONTEXT" \
            "$FIT_TARGET_MIB" \
            "$MANAGED_CACHE_TYPE_K" \
            "$MANAGED_CACHE_TYPE_V" \
            "$MANAGED_FLASH_ATTN" \
            "$MANAGED_ESTIMATOR_OVERRUN_USED" >&"$engine_log_fd"
    fi

    if [ "$MANAGED" = "1" ] && [ "$MANAGED_SIZING" = "profile" ]; then
        printf 'qiip_profile_plan: profile_id=%s profile_version=%s ubatch=%s spec_type=%s spec_draft_n_max=%s draft_cache_type=%s draft_gguf=%s required_free_mib=%s gpu_free_mib=%s gpu_uuid=%s cuda_graphs=%s\n' \
            "$PROFILE_ID" \
            "$PROFILE_VERSION" \
            "$PROFILE_UBATCH" \
            "$PROFILE_SPEC_TYPE" \
            "$PROFILE_SPEC_DRAFT_N_MAX" \
            "${PROFILE_DRAFT_CACHE_TYPE:-default}" \
            "${PROFILE_DRAFT_GGUF_RELATIVE_PATH:-none}" \
            "$PROFILE_REQUIRED_FREE_MIB" \
            "${MANAGED_GPU_FREE_MIB[0]}" \
            "$PROFILE_GPU_UUID" \
            "$([ "$PROFILE_DISABLE_CUDA_GRAPHS" = "1" ] && echo off || echo on)" \
            >&"$engine_log_fd"
        # Same fixed managed invariants as the planner launch below: one
        # unified KV buffer, full GPU offload, no runtime re-fitting. The
        # profile adds only typed, validated options.
        # llama-server has one MTP implementation; an assistant differs only
        # in drafting from its own file.
        local server_spec_type="$PROFILE_SPEC_TYPE"
        if [ "$server_spec_type" = "draft-mtp-assistant" ]; then
            server_spec_type="draft-mtp"
        fi
        local -a profile_args=(
            --ubatch-size "$PROFILE_UBATCH"
            --spec-type "$server_spec_type"
            --spec-draft-n-max "$PROFILE_SPEC_DRAFT_N_MAX"
        )
        if [ -n "$PROFILE_DRAFT_GGUF_PATH" ]; then
            profile_args+=(--model-draft "$PROFILE_DRAFT_GGUF_PATH")
        fi
        if [ -n "$PROFILE_DRAFT_CACHE_TYPE" ]; then
            profile_args+=(
                --cache-type-k-draft "$PROFILE_DRAFT_CACHE_TYPE"
                --cache-type-v-draft "$PROFILE_DRAFT_CACHE_TYPE"
            )
        fi
        profile_args+=(
            --no-mmproj
            --jinja
            --temp "$PROFILE_TEMPERATURE"
            --top-p "$PROFILE_TOP_P"
            --top-k "$PROFILE_TOP_K"
        )
        if [ -n "$PROFILE_MIN_P" ]; then
            profile_args+=(--min-p "$PROFILE_MIN_P")
        fi
        if [ -n "$PROFILE_PRESENCE_PENALTY" ]; then
            profile_args+=(--presence-penalty "$PROFILE_PRESENCE_PENALTY")
        fi
        if [ "$PROFILE_DISABLE_CUDA_GRAPHS" = "1" ]; then
            export GGML_CUDA_DISABLE_GRAPHS=1
        fi
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
            "${profile_args[@]}" \
            --verbosity 4 \
            --metrics \
            >&"$engine_log_fd" 2>&1 &
    elif [ "$MANAGED" = "1" ]; then
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
            >&"$engine_log_fd" 2>&1 &
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
            >&"$engine_log_fd" 2>&1 &
    fi

    local pid=$!
    exec {engine_log_fd}>&-
    echo "$pid" > "$PID_FILE"
    verify_llamacpp_started "$pid"
    echo "llama-server started (PID ${pid})"
}

run_storage_preflight() {
    # Before engine load the mount must be the intended NFSv3 export with the
    # exact required options; a wrong-export mount would silently serve a
    # different cache. Artifact completeness is verified in run_llamacpp
    # (resolve_gguf_artifact) right before the server loads it.
    if [ -n "$NFS_EXPORT" ]; then
        if ! verify_nfs_storage; then
            echo "FATAL: NFS storage verification failed at ${NFS_MOUNT_POINT}; refusing to start" >&2
            return 1
        fi
        echo "NFS storage verified: source=${NFS_EXPORT}, NFSv3, required options"
    else
        echo "WARNING: AUTOVLLM_NFS_EXPORT not provided; storage source not verified" >&2
    fi
}

main() {
    detect_gpu_info
    if [ "$GPU_COUNT" -gt 0 ]; then
        select_runtime_profile llamacpp || exit $?
        wait_nvswitch_fabric
        fabric_ready
    fi
    configure_llamacpp_params
    run_storage_preflight
    run_llamacpp
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
