#!/bin/bash
# shellcheck disable=SC2034
set -euo pipefail

# --- Configurable defaults (shared vars use AUTOVLLM_ prefix for compat) ---
SCRIPT_DIR="${AUTOLLAMACPP_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
NFS_EXPORT="${AUTOVLLM_NFS_EXPORT:-}"
NFS_MOUNT_POINT="${AUTOVLLM_NFS_MOUNT_POINT:-/srv/hf-cache}"
DEFAULT_DRIVER_VERSION="580.126.09"
DEFAULT_DRIVER_SHA256="4cac53e48f8adff661d47c8788ed24059a248c9fd8098ceafd088a498986ec26"
DRIVER_VERSION="${AUTOVLLM_NVIDIA_DRIVER_VERSION-$DEFAULT_DRIVER_VERSION}"
if [[ -v AUTOVLLM_NVIDIA_DRIVER_SHA256 ]]; then
    DRIVER_SHA256="$AUTOVLLM_NVIDIA_DRIVER_SHA256"
elif [ "$DRIVER_VERSION" = "$DEFAULT_DRIVER_VERSION" ]; then
    DRIVER_SHA256="$DEFAULT_DRIVER_SHA256"
else
    DRIVER_SHA256=""
fi
NVIDIA_DRIVER_URL="${NVIDIA_DRIVER_URL:-https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run}"
API_PORT="${AUTOVLLM_API_PORT:-8000}"
DEFAULT_LLMFIT_RELEASE="1.1.6"
DEFAULT_LLMFIT_SHA256="1e09232a128455596a2d348ab5893741d04b94aa6d924f1253462dc13304f7c6"
LLMFIT_RELEASE="${AUTOVLLM_LLMFIT_VERSION-$DEFAULT_LLMFIT_RELEASE}"
if [[ -v AUTOVLLM_LLMFIT_SHA256 ]]; then
    LLMFIT_SHA256="$AUTOVLLM_LLMFIT_SHA256"
elif [ "$LLMFIT_RELEASE" = "$DEFAULT_LLMFIT_RELEASE" ]; then
    LLMFIT_SHA256="$DEFAULT_LLMFIT_SHA256"
else
    LLMFIT_SHA256=""
fi
LLMFIT_URL="${LLMFIT_URL:-https://github.com/AlexsJones/llmfit/releases/download/v${LLMFIT_RELEASE}/llmfit-v${LLMFIT_RELEASE}-x86_64-unknown-linux-musl.tar.gz}"
LLMFIT_BIN="${AUTOVLLM_LLMFIT_BIN:-/usr/local/bin/llmfit}"
INSTALL_TMP_DIR="${AUTOVLLM_TMP_DIR:-/tmp}"

# llama.cpp-specific
LLAMACPP_VERSION="${AUTOLLAMACPP_VERSION:-b10242}"
LLAMACPP_SHA256="${AUTOLLAMACPP_SHA256:-}"

unset AUTOVLLM_NFS_EXPORT AUTOVLLM_NFS_MOUNT_POINT
unset AUTOVLLM_NVIDIA_DRIVER_VERSION AUTOVLLM_NVIDIA_DRIVER_SHA256 AUTOVLLM_API_PORT
unset AUTOVLLM_LLMFIT_VERSION AUTOVLLM_LLMFIT_SHA256
unset AUTOVLLM_LLMFIT_BIN AUTOVLLM_TMP_DIR AUTOLLAMACPP_SCRIPT_DIR
unset AUTOLLAMACPP_VERSION AUTOLLAMACPP_SHA256

# Source shared setup functions
# shellcheck disable=SC1091 source=../common/setup-base.sh
source "$(cd -- "${SCRIPT_DIR}/.." && pwd)/common/setup-base.sh"

# --- llama.cpp-specific functions ---

install_llamacpp() {
    if [ -x /usr/local/bin/llama-server ]; then
        local installed_version
        installed_version=$(/usr/local/bin/llama-server --version 2>/dev/null \
            | grep -Eo 'b[0-9]+' | head -n 1) || true
        if [ "$installed_version" = "$LLAMACPP_VERSION" ]; then
            echo "llama-server ${LLAMACPP_VERSION} already installed, skipping"
            return 0
        fi
    fi

    # To bump llama.cpp version:
    # 1. Update AUTOLLAMACPP_VERSION in provisioning settings (or env var)
    # 2. Download the release tarball and compute: sha256sum llama-b{TAG}-*.tar.gz
    # 3. Update AUTOLLAMACPP_SHA256 in provisioning settings (or env var)
    # 4. Test provisioning on a single node before fleet rollout
    local cuda_suffix="cu12.2"
    local url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_VERSION}/llama-${LLAMACPP_VERSION}-bin-ubuntu-x64-cuda-${cuda_suffix}.tar.gz"

    if [ -n "$LLAMACPP_SHA256" ]; then
        require_sha256 "llama.cpp ${LLAMACPP_VERSION}" "$LLAMACPP_SHA256" \
            "AUTOLLAMACPP_SHA256"
    fi

    local work_dir status
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-llamacpp.XXXXXX")
    if wget -q "$url" -O "${work_dir}/llamacpp.tar.gz"; then
        :
    else
        status=$?
        rm -rf "$work_dir"
        return "$status"
    fi

    if [ -n "$LLAMACPP_SHA256" ]; then
        if ! verify_sha256 "${work_dir}/llamacpp.tar.gz" "$LLAMACPP_SHA256" \
            "llama.cpp ${LLAMACPP_VERSION}"; then
            rm -rf "$work_dir"
            return 1
        fi
    fi

    tar xzf "${work_dir}/llamacpp.tar.gz" -C "$work_dir"
    local server_bin
    server_bin=$(find "$work_dir" -name llama-server -type f | head -n 1)
    local quantize_bin
    quantize_bin=$(find "$work_dir" -name llama-quantize -type f | head -n 1)

    if [ -z "$server_bin" ]; then
        echo "FATAL: llama-server binary not found in release archive" >&2
        rm -rf "$work_dir"
        return 1
    fi
    sudo install -m 755 "$server_bin" /usr/local/bin/llama-server
    if [ -n "$quantize_bin" ]; then
        sudo install -m 755 "$quantize_bin" /usr/local/bin/llama-quantize
    fi
    rm -rf "$work_dir"
}

# --- Main ---
main() {
    if [ -z "$NFS_EXPORT" ]; then
        echo "FATAL: AUTOVLLM_NFS_EXPORT is required for node provisioning" >&2
        return 2
    fi
    require_sha256 "NVIDIA driver ${DRIVER_VERSION}" "$DRIVER_SHA256" \
        "AUTOVLLM_NVIDIA_DRIVER_SHA256"
    require_sha256 "llmfit ${LLMFIT_RELEASE}" "$LLMFIT_SHA256" \
        "AUTOVLLM_LLMFIT_SHA256"
    step system_update run_system_update
    step nvidia_driver install_nvidia_driver
    step cuda_toolkit install_cuda_toolkit
    step llamacpp_install install_llamacpp
    step nfs_mount mount_nfs_cache
    step firewall configure_firewall
    soft_step llmfit_install install_llmfit

    echo "Setup complete"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
