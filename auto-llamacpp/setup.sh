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

# llama.cpp-specific. GitHub does not publish a Linux CUDA archive for this
# release, so managed nodes compile the verified source for their attached GPU.
DEFAULT_LLAMACPP_VERSION="b10242"
DEFAULT_LLAMACPP_SHA256="b5c2b0d09d2af9988e47570f7f96e8473b4e07fad2c99f6e2e0745e5b3935fe3"
LLAMACPP_VERSION="${AUTOLLAMACPP_VERSION-$DEFAULT_LLAMACPP_VERSION}"
if [[ -v AUTOLLAMACPP_SHA256 ]]; then
    LLAMACPP_SHA256="$AUTOLLAMACPP_SHA256"
elif [ "$LLAMACPP_VERSION" = "$DEFAULT_LLAMACPP_VERSION" ]; then
    LLAMACPP_SHA256="$DEFAULT_LLAMACPP_SHA256"
else
    LLAMACPP_SHA256=""
fi
LLAMACPP_SOURCE_URL="${AUTOLLAMACPP_SOURCE_URL:-https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMACPP_VERSION}.tar.gz}"
LLAMACPP_INSTALL_ROOT="${AUTOLLAMACPP_INSTALL_ROOT:-/opt/llama.cpp}"
LLAMACPP_LINK_DIR="${AUTOLLAMACPP_LINK_DIR:-/usr/local/bin}"
LLAMACPP_CUDA_ARCHITECTURES="${AUTOLLAMACPP_CUDA_ARCHITECTURES:-native}"
LLAMACPP_BUILD_PROFILE="cuda-portable-cpu-v2-fit-concurrency"
LLAMACPP_FIT_PATCH_FROM=').set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));'
LLAMACPP_FIT_PATCH_TO=').set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL, LLAMA_EXAMPLE_FIT_PARAMS}));'
LLAMACPP_FIT_PATCH_SHA256="58917efc78ca760a2a1dd162d84e6cf1930c5b62a8dd9710bb4579ca4f2d69dc"
CUDA_NVCC="${AUTOLLAMACPP_NVCC:-/usr/local/cuda/bin/nvcc}"

unset AUTOVLLM_NFS_EXPORT AUTOVLLM_NFS_MOUNT_POINT
unset AUTOVLLM_NVIDIA_DRIVER_VERSION AUTOVLLM_NVIDIA_DRIVER_SHA256 AUTOVLLM_API_PORT
unset AUTOVLLM_LLMFIT_VERSION AUTOVLLM_LLMFIT_SHA256
unset AUTOVLLM_LLMFIT_BIN AUTOVLLM_TMP_DIR AUTOLLAMACPP_SCRIPT_DIR
unset AUTOLLAMACPP_VERSION AUTOLLAMACPP_SHA256 AUTOLLAMACPP_SOURCE_URL
unset AUTOLLAMACPP_INSTALL_ROOT AUTOLLAMACPP_LINK_DIR
unset AUTOLLAMACPP_CUDA_ARCHITECTURES AUTOLLAMACPP_NVCC

# Source shared setup functions
# shellcheck disable=SC1091 source=../common/setup-base.sh
source "$(cd -- "${SCRIPT_DIR}/.." && pwd)/common/setup-base.sh"

# --- llama.cpp-specific functions ---

installed_llamacpp_version() {
    local binary="$1"
    "$binary" --version 2>&1 \
        | sed -nE 's/^version:[[:space:]]*([0-9]+).*/b\1/p' \
        | head -n 1
}

cuda_compute_capabilities() {
    local capabilities
    if ! capabilities=$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
            | sed '/^[[:space:]]*$/d' \
            | sort -Vu
    ); then
        echo "FATAL: nvidia-smi could not query CUDA compute capabilities" >&2
        return 1
    fi
    if [ -z "$capabilities" ] \
        || grep -Evq '^[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*$' <<<"$capabilities"; then
        echo "FATAL: no valid NVIDIA CUDA compute capability was detected" >&2
        return 1
    fi
    printf '%s\n' "$capabilities"
}

atomic_link() {
    local target="$1"
    local link="$2"
    local temporary_link="${link}.qiip.$$"
    sudo ln -sfn "$target" "$temporary_link"
    sudo mv -Tf "$temporary_link" "$link"
}

verify_fit_params_patch_identity() {
    local actual
    actual=$(printf '%s\0%s\0' \
        "$LLAMACPP_FIT_PATCH_FROM" \
        "$LLAMACPP_FIT_PATCH_TO" \
        | sha256sum | cut -d ' ' -f 1)
    if [ "$actual" != "$LLAMACPP_FIT_PATCH_SHA256" ]; then
        echo "FATAL: llama.cpp fit-params source transformation digest mismatch" >&2
        return 1
    fi
}

enable_fit_params_unified_kv() {
    local arg_source="$1/common/arg.cpp"
    if grep -Fq "$LLAMACPP_FIT_PATCH_TO" "$arg_source"; then
        return 0
    fi
    if [ "$(grep -Fo "$LLAMACPP_FIT_PATCH_FROM" "$arg_source" | wc -l)" -ne 1 ]; then
        echo "FATAL: pinned llama.cpp source no longer has the expected --kv-unified example list" >&2
        return 1
    fi
    sed -i \
        '/LLAMA_ARG_KV_UNIFIED/s/LLAMA_EXAMPLE_PARALLEL}));$/LLAMA_EXAMPLE_PARALLEL, LLAMA_EXAMPLE_FIT_PARAMS}));/' \
        "$arg_source"
    if ! grep -Fq "$LLAMACPP_FIT_PATCH_TO" "$arg_source"; then
        echo "FATAL: could not expose unified-KV estimation in llama-fit-params" >&2
        return 1
    fi
}

verify_fit_params_cli() {
    local binary="$1"
    if ! "$binary" \
        --ctx-size 4096 \
        --parallel 2 \
        --kv-unified \
        --gpu-layers all \
        --fit-print on \
        --verbosity 4 \
        --version >/dev/null 2>&1; then
        echo "FATAL: built llama-fit-params does not accept the managed planner CLI" >&2
        return 1
    fi
}

install_llamacpp() {
    require_sha256 "llama.cpp ${LLAMACPP_VERSION}" "$LLAMACPP_SHA256" \
        "AUTOLLAMACPP_SHA256"
    if [[ ! "$LLAMACPP_VERSION" =~ ^b[1-9][0-9]*$ ]]; then
        echo "FATAL: AUTOLLAMACPP_VERSION must use the b<number> build-tag format" >&2
        return 2
    fi
    if [[ ! "$LLAMACPP_SOURCE_URL" =~ ^https?:// ]]; then
        echo "FATAL: AUTOLLAMACPP_SOURCE_URL must be an HTTP(S) URL" >&2
        return 2
    fi
    if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
        echo "FATAL: cmake and ninja are required to build llama.cpp" >&2
        return 1
    fi
    if [ ! -x "$CUDA_NVCC" ]; then
        echo "FATAL: CUDA nvcc is required to build managed llama.cpp" >&2
        return 1
    fi

    verify_fit_params_patch_identity

    local compute_capabilities build_identity install_dir marker
    compute_capabilities=$(cuda_compute_capabilities) || return
    marker=$(printf 'version=%s\nsource_sha256=%s\nbuild_profile=%s\nfit_cli_patch_sha256=%s\ncompute_capabilities=%s\ncmake_cuda_architectures=%s\n' \
        "$LLAMACPP_VERSION" \
        "$LLAMACPP_SHA256" \
        "$LLAMACPP_BUILD_PROFILE" \
        "$LLAMACPP_FIT_PATCH_SHA256" \
        "${compute_capabilities//$'\n'/,}" \
        "$LLAMACPP_CUDA_ARCHITECTURES")
    build_identity=$(printf '%s' "$marker" | sha256sum | cut -c1-16)
    install_dir="${LLAMACPP_INSTALL_ROOT%/}/${LLAMACPP_VERSION}-${build_identity}"

    if [ -x "${install_dir}/bin/llama-server" ] \
        && [ -x "${install_dir}/bin/llama-fit-params" ] \
        && [ -f "${install_dir}/BUILD-INFO" ] \
        && [ "$(<"${install_dir}/BUILD-INFO")" = "$marker" ] \
        && [ "$(installed_llamacpp_version "${install_dir}/bin/llama-server")" = "$LLAMACPP_VERSION" ]; then
        sudo mkdir -p "$LLAMACPP_LINK_DIR"
        atomic_link "${install_dir}/bin/llama-server" "${LLAMACPP_LINK_DIR%/}/llama-server"
        atomic_link "${install_dir}/bin/llama-fit-params" "${LLAMACPP_LINK_DIR%/}/llama-fit-params"
        atomic_link "${install_dir}/bin/llama-quantize" "${LLAMACPP_LINK_DIR%/}/llama-quantize"
        echo "llama-server ${LLAMACPP_VERSION} already installed for CUDA capabilities ${compute_capabilities//$'\n'/,}, skipping"
        return 0
    fi

    # run_with_errexit invokes steps inside its own set -e subshell, where a
    # function RETURN trap is not reliable. Keep the entire build and publish
    # sequence in a dedicated subshell whose EXIT trap owns all cleanup.
    (
        set -e
        local work_dir archive source_dir build_dir server_bin fit_bin quantize_bin build_status
        work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-llamacpp.XXXXXX")
        # shellcheck disable=SC2317,SC2329  # Invoked indirectly by the EXIT trap.
        cleanup_llamacpp_build() {
            build_status=$?
            trap - EXIT
            rm -rf "$work_dir"
            exit "$build_status"
        }
        trap cleanup_llamacpp_build EXIT
        archive="${work_dir}/llamacpp.tar.gz"
        source_dir="${work_dir}/source"
        build_dir="${work_dir}/build"
        mkdir -p "$source_dir"

        wget -q "$LLAMACPP_SOURCE_URL" -O "$archive"
        verify_sha256 "$archive" "$LLAMACPP_SHA256" \
            "llama.cpp ${LLAMACPP_VERSION} source"
        tar xzf "$archive" -C "$source_dir" --strip-components=1
        # b10242's memory estimator supports unified KV internally, but its CLI
        # allowlist omits llama-fit-params. Expose the existing option so the
        # planner estimates the exact KV mode used by llama-server.
        enable_fit_params_unified_kv "$source_dir"

        # Source tarballs have no .git, so cmake/build-info.cmake logs two harmless
        # "fatal: not a git repository" lines and falls back to BUILD_NUMBER=0.
        # LLAMA_BUILD_NUMBER/COMMIT below override that fallback and are what
        # installed_llamacpp_version() matches against -- they are not decorative.
        cmake -S "$source_dir" -B "$build_dir" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
            -DCMAKE_CUDA_ARCHITECTURES="$LLAMACPP_CUDA_ARCHITECTURES" \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON \
            -DGGML_NATIVE=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_SERVER=ON \
            -DLLAMA_BUILD_APP=OFF \
            -DLLAMA_BUILD_UI=OFF \
            -DLLAMA_BUILD_MTMD=OFF \
            -DLLAMA_OPENSSL=OFF \
            -DLLAMA_BUILD_NUMBER="${LLAMACPP_VERSION#b}" \
            -DLLAMA_BUILD_COMMIT="$LLAMACPP_VERSION"
        cmake --build "$build_dir" --target llama-server llama-fit-params llama-quantize \
            --parallel "$(nproc)"

        server_bin="${build_dir}/bin/llama-server"
        fit_bin="${build_dir}/bin/llama-fit-params"
        quantize_bin="${build_dir}/bin/llama-quantize"
        if [ ! -x "$server_bin" ] || [ ! -x "$fit_bin" ] || [ ! -x "$quantize_bin" ]; then
            echo "FATAL: llama.cpp build did not produce the required binaries" >&2
            exit 1
        fi
        if [ "$(installed_llamacpp_version "$server_bin")" != "$LLAMACPP_VERSION" ]; then
            echo "FATAL: built llama-server did not report ${LLAMACPP_VERSION}" >&2
            exit 1
        fi
        verify_fit_params_cli "$fit_bin"

        printf '%s\n' "$marker" > "${work_dir}/BUILD-INFO"
        sudo mkdir -p "${install_dir}/bin" "$LLAMACPP_LINK_DIR"
        sudo install -m 755 "$server_bin" "${install_dir}/bin/llama-server"
        sudo install -m 755 "$fit_bin" "${install_dir}/bin/llama-fit-params"
        sudo install -m 755 "$quantize_bin" "${install_dir}/bin/llama-quantize"
        sudo install -m 644 "${work_dir}/BUILD-INFO" "${install_dir}/BUILD-INFO"
        atomic_link "${install_dir}/bin/llama-server" "${LLAMACPP_LINK_DIR%/}/llama-server"
        atomic_link "${install_dir}/bin/llama-fit-params" "${LLAMACPP_LINK_DIR%/}/llama-fit-params"
        atomic_link "${install_dir}/bin/llama-quantize" "${LLAMACPP_LINK_DIR%/}/llama-quantize"
    )
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
    require_sha256 "llama.cpp ${LLAMACPP_VERSION}" "$LLAMACPP_SHA256" \
        "AUTOLLAMACPP_SHA256"
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
