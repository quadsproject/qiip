#!/bin/bash
# shellcheck disable=SC2034
set -euo pipefail

# --- Configurable defaults ---
SCRIPT_DIR="${AUTOVLLM_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
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
VLLM_VENV="${AUTOVLLM_VENV:-/opt/vllm-venv}"
LLMFIT_BIN="${AUTOVLLM_LLMFIT_BIN:-/usr/local/bin/llmfit}"
INSTALL_TMP_DIR="${AUTOVLLM_TMP_DIR:-/tmp}"
UV_BIN="${AUTOVLLM_UV_BIN:-/usr/local/bin/uv}"
UV_PROJECT="${AUTOVLLM_UV_PROJECT:-$SCRIPT_DIR}"
UV_TARGET="x86_64-unknown-linux-gnu"
UV_PYTHON_PLATFORM="x86_64-manylinux_2_34"
UV_VERSION="$(<"${SCRIPT_DIR}/.uv-version")"
UV_ARCHIVE="uv-${UV_TARGET}.tar.gz"
UV_CHECKSUM_FILE="${SCRIPT_DIR}/${UV_ARCHIVE}.sha256"
FLASHINFER_INDEX_OVERRIDE_SET=0
if [[ -v FLASHINFER_INDEX_URL ]]; then
    FLASHINFER_INDEX_OVERRIDE_SET=1
fi

unset AUTOVLLM_NFS_EXPORT AUTOVLLM_NFS_MOUNT_POINT
unset AUTOVLLM_NVIDIA_DRIVER_VERSION AUTOVLLM_NVIDIA_DRIVER_SHA256 AUTOVLLM_API_PORT
unset AUTOVLLM_LLMFIT_VERSION AUTOVLLM_LLMFIT_SHA256
unset AUTOVLLM_VENV AUTOVLLM_LLMFIT_BIN AUTOVLLM_TMP_DIR
unset AUTOVLLM_UV_BIN AUTOVLLM_UV_PROJECT AUTOVLLM_SCRIPT_DIR FLASHINFER_INDEX_URL

# Source shared setup functions
# shellcheck disable=SC1091 source=../common/setup-base.sh
source "$(cd -- "${SCRIPT_DIR}/.." && pwd)/common/setup-base.sh"

# --- vLLM-specific functions ---

reject_retired_flashinfer_index() {
    if [ "$FLASHINFER_INDEX_OVERRIDE_SET" -eq 1 ]; then
        echo "FATAL: FLASHINFER_INDEX_URL is retired; the frozen auto-vllm/uv.lock owns package sources. Ship a regenerated auto-vllm bundle for a custom mirror." >&2
        return 2
    fi
}

install_uv() {
    local installed_version=""
    if [ -x "$UV_BIN" ]; then
        installed_version=$("$UV_BIN" --version 2>/dev/null | awk '{print $2}') || true
    fi
    if [ "$installed_version" = "$UV_VERSION" ]; then
        echo "uv ${UV_VERSION} already installed, skipping bootstrap"
        return 0
    fi

    if [ ! -s "$UV_CHECKSUM_FILE" ]; then
        echo "FATAL: pinned uv checksum file is missing or empty: ${UV_CHECKSUM_FILE}" >&2
        return 2
    fi

    local work_dir archive extracted status
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-vllm-uv.XXXXXX")
    archive="${work_dir}/${UV_ARCHIVE}"
    extracted="${work_dir}/uv-${UV_TARGET}/uv"
    if wget -q \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_ARCHIVE}" \
        -O "$archive"; then
        :
    else
        status=$?
        rm -rf "$work_dir"
        return "$status"
    fi
    if ! (cd "$work_dir" && sha256sum -c "$UV_CHECKSUM_FILE" >/dev/null); then
        echo "FATAL: uv ${UV_VERSION} SHA-256 verification failed" >&2
        rm -rf "$work_dir"
        return 1
    fi
    tar -xzf "$archive" -C "$work_dir"
    if sudo install -m 755 "$extracted" "$UV_BIN"; then
        status=0
    else
        status=$?
    fi
    rm -rf "$work_dir"
    return "$status"
}

install_vllm() {
    reject_retired_flashinfer_index
    install_uv

    UV_PROJECT_ENVIRONMENT="$VLLM_VENV" "$UV_BIN" sync \
        --project "$UV_PROJECT" \
        --frozen \
        --no-dev \
        --no-install-project \
        --no-build \
        --python /usr/bin/python3.12 \
        --python-platform "$UV_PYTHON_PLATFORM"

    "${VLLM_VENV}/bin/python" -c \
        'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public'
}

# --- Main ---
main() {
    if [ -z "$NFS_EXPORT" ]; then
        echo "FATAL: AUTOVLLM_NFS_EXPORT is required for node provisioning" >&2
        return 2
    fi
    reject_retired_flashinfer_index
    require_sha256 "NVIDIA driver ${DRIVER_VERSION}" "$DRIVER_SHA256" \
        "AUTOVLLM_NVIDIA_DRIVER_SHA256"
    require_sha256 "llmfit ${LLMFIT_RELEASE}" "$LLMFIT_SHA256" \
        "AUTOVLLM_LLMFIT_SHA256"
    step system_update run_system_update
    step nvidia_driver install_nvidia_driver
    step cuda_toolkit install_cuda_toolkit
    step vllm_install install_vllm
    step nfs_mount mount_nfs_cache
    step firewall configure_firewall
    soft_step llmfit_install install_llmfit

    echo "Setup complete"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
