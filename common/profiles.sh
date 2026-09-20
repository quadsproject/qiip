#!/bin/bash
# Runtime profile catalog for managed inference nodes. Sourced by
# common/setup-base.sh so both engines share one measurement-driven
# selection. Profiles are keyed on measured hardware (compute capability,
# VRAM, OS/ABI), never GPU marketing names.

# Profile-relevant constants (single source of truth)
PROFILE_CUDA_TOOLKIT_VERSION="13.0"  # matches torch 2.11.0 CUDA-13.0 wheels in auto-vllm/uv.lock
# CUDA 13.0 removed Maxwell/Pascal/Volta (offline compilation and libraries);
# 12.x is the last series that can target Volta SM70, so Volta pins 12.9.
PROFILE_VOLTA_CUDA_TOOLKIT_VERSION="12.9"
PROFILE_OS_ID="rhel"
PROFILE_OS_MAJOR_MIN=9
PROFILE_ARCH="x86_64"
PROFILE_GLIBC_MIN="2.34"  # vLLM wheel ABI (manylinux_2_34, auto-vllm/setup.sh:26)

detect_profile_hardware() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "FATAL: nvidia-smi not found; run setup.sh first or install NVIDIA drivers." >&2
        return 1
    fi
    if ! nvidia-smi &>/dev/null; then
        echo "FATAL: nvidia-smi failed; NVIDIA driver may not be loaded." >&2
        return 1
    fi
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    # Profile data comes from the first SELECTED device (mirrors the gateway's
    # AUTOVLLM_GPU_DEVICES semantics): an explicit subset is sized against its
    # own cards. Physical inventory is the full list; nvidia-smi ignores
    # CUDA_VISIBLE_DEVICES, so queries stay physical.
    local query_index=0
    if [ -n "${GPU_DEVICES_OVERRIDE:-}" ]; then
        local -a _profile_devices=()
        IFS=',' read -r -a _profile_devices <<< "$GPU_DEVICES_OVERRIDE"
        query_index="${_profile_devices[0]}"
    fi
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$query_index" | xargs)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$query_index")
    GPU_VRAM_GB=$(( (GPU_VRAM_MB + 512) / 1024 ))
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$query_index" | xargs)
    GPU_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
        | sed '/^[[:space:]]*$/d' | head -1 | xargs)
    NVSWITCH_COUNT=$(lspci 2>/dev/null | grep -ci nvswitch || true)
    OS_ID="unknown"
    OS_VERSION_ID="0"
    local os_release="${PROFILE_OS_RELEASE:-/etc/os-release}"
    if [ -r "$os_release" ]; then
# shellcheck disable=SC1091,SC1090
        . "$os_release"
        OS_ID="${ID:-unknown}"
        OS_VERSION_ID="${VERSION_ID:-0}"
    fi
    OS_ARCH=$(uname -m)
    GLIBC_VERSION=$(ldd --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | head -1)
    [ -n "${GLIBC_VERSION:-}" ] || GLIBC_VERSION="0"
    return 0
}

check_os_abi() {
    # engine arg: "vllm" additionally requires the wheel ABI glibc floor
    local engine="$1"
    if [ "$OS_ID" != "$PROFILE_OS_ID" ]; then
        echo "OS ${OS_ID:-unknown} is not supported (requires ${PROFILE_OS_ID}${PROFILE_OS_MAJOR_MIN}+)" >&2
        return 1
    fi
    if ! [[ "$OS_VERSION_ID" =~ ^[0-9]+ ]] \
        || [ "${OS_VERSION_ID%%.*}" -lt "$PROFILE_OS_MAJOR_MIN" ]; then
        echo "OS version ${OS_VERSION_ID:-invalid} is not supported (requires ${PROFILE_OS_ID}${PROFILE_OS_MAJOR_MIN}+)" >&2
        return 1
    fi
    if [ "$OS_ARCH" != "$PROFILE_ARCH" ]; then
        echo "architecture ${OS_ARCH} is not supported (requires ${PROFILE_ARCH})" >&2
        return 1
    fi
    if [ "$engine" = "vllm" ]; then
        if [ "$(printf '%s\n' "$GLIBC_VERSION" "$PROFILE_GLIBC_MIN" | sort -V | head -1)" != "$PROFILE_GLIBC_MIN" ]; then
            echo "glibc ${GLIBC_VERSION} is too old (requires ${PROFILE_GLIBC_MIN}+ for the vLLM wheel ABI)" >&2
            return 1
        fi
    fi
    return 0
}

# Selects the tested runtime profile for the measured hardware of <engine>
# (vllm|llamacpp). Exports PROFILE_* and prints:
#   [PROFILE:select:<name> (<reason>)]          selected
#   [REJECT:unsupported_hardware:<explanation>] unsupported combination
# Exit: 0 selected, 1 probe/transient failure, 3 unsupported.
select_runtime_profile() {
    local engine="$1"
    detect_profile_hardware || return 1

    local reason
    reason="model=${GPU_MODEL} sm=${GPU_COMPUTE_CAP} vram=${GPU_VRAM_GB}GB count=${GPU_COUNT} driver=${GPU_DRIVER_VERSION} nvswitch=${NVSWITCH_COUNT} os=${OS_ID}${OS_VERSION_ID} ${OS_ARCH} glibc=${GLIBC_VERSION}"
    local signature="${reason}"
    reason="${reason} engine=${engine}"

    if ! check_os_abi "$engine"; then
        echo "[REJECT:unsupported_hardware:OS/ABI check failed; measured: ${signature}]" >&2
        return 3
    fi

    local major="${GPU_COMPUTE_CAP%%.*}"
    local minor="${GPU_COMPUTE_CAP#*.}"
    if ! [[ "$major" =~ ^[0-9]+$ ]] || ! [[ "$minor" =~ ^[0-9]+$ ]]; then
        echo "FATAL: cannot parse GPU compute capability '${GPU_COMPUTE_CAP}' from nvidia-smi" >&2
        return 1
    fi
    local sm=$(( major * 10 + minor ))

    local bucket=""
    if [ "$sm" -eq 90 ] && [ "$GPU_VRAM_GB" -ge 80 ]; then
        bucket="hopper"
    elif [ "$sm" -eq 80 ] && [ "$GPU_VRAM_GB" -ge 40 ]; then
        bucket="ampere-a100"
    elif [ "$sm" -eq 80 ] && [ "$GPU_VRAM_GB" -ge 24 ]; then
        bucket="ampere-a30"
    elif [ "$sm" -eq 86 ] && [ "$GPU_VRAM_GB" -ge 48 ]; then
        bucket="ga102-dc"
    elif [ "$sm" -eq 86 ]; then
        bucket="consumer-ampere"
    elif [ "$sm" -eq 89 ]; then
        bucket="consumer-ada"
    # A 16 GB T4 reports 15,079 MiB usable (nvidia-smi), which rounds to 15 GB
    # but is data-center class; decide on reported MiB (14 GiB floor) so the
    # T4 stays on the tuned turing arm instead of consumer-turing.
    elif [ "$sm" -eq 75 ] && [ "$GPU_VRAM_MB" -ge 14336 ]; then
        bucket="turing"
    elif [ "$sm" -eq 75 ]; then
        bucket="consumer-turing"
    elif [ "$sm" -eq 70 ] && [ "$GPU_VRAM_GB" -ge 16 ]; then
        bucket="volta"
    fi

    if [ -z "$bucket" ]; then
        echo "[REJECT:unsupported_hardware:no tested profile for ${signature}; supported: sm90 80GB+, sm80 24GB+, sm86 (48GB+ dc / below consumer), sm89, sm75 (14GiB+ reported dc / below consumer), sm70 16GB+]" >&2
        return 3
    fi

    # The frozen vLLM stack is torch 2.11.0 CUDA-13.0, which cannot target
    # Volta (CUDA 13.0 removed pre-Turing support); reject instead of
    # accepting and failing later. llama.cpp is source-built, so it pins the
    # last CUDA 12.x series that still supports SM70.
    if [ "$bucket" = "volta" ] && [ "$engine" = "vllm" ]; then
        echo "[REJECT:unsupported_hardware:profile volta needs a CUDA 12 toolkit to target SM70, but the pinned vLLM stack is CUDA 13.0 (dropped Volta); ${signature}]" >&2
        return 3
    fi
    if [ "$bucket" = "volta" ]; then
        PROFILE_CUDA_TOOLKIT_VERSION="$PROFILE_VOLTA_CUDA_TOOLKIT_VERSION"
    fi

    PROFILE_NAME="${engine}-${bucket}"
    PROFILE_BUCKET="$bucket"
    PROFILE_REASON="$reason"
    export PROFILE_NAME PROFILE_BUCKET PROFILE_REASON
    export PROFILE_CUDA_TOOLKIT_VERSION
    echo "[PROFILE:select:${PROFILE_NAME} (${PROFILE_REASON})]"
    return 0
}
