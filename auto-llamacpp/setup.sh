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

# llama.cpp-specific. A pinned artifact catalog can avoid on-node compilation.
DEFAULT_LLAMACPP_VERSION="v0.4.1"
DEFAULT_LLAMACPP_SHA256="ef3d5b1907a391500ae11b5e61a8e2022e0deaac9790899cad9c4e02f03bfb9a"
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
QIIP_GENERATION_ROOT="${QIIP_GENERATION_ROOT:-/opt/qiip/llama_cpp}"
LLAMACPP_LINK_DIR="${AUTOLLAMACPP_LINK_DIR:-/usr/local/bin}"
LLAMACPP_CUDA_ARCHITECTURES="${AUTOLLAMACPP_CUDA_ARCHITECTURES:-native}"
LLAMACPP_BUILD_PROFILE="cuda-portable-cpu-v3-artifact"
LLAMACPP_FIT_PATCH_FROM=').set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));'
LLAMACPP_FIT_PATCH_TO=').set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL, LLAMA_EXAMPLE_FIT_PARAMS}));'
LLAMACPP_FIT_PATCH_SHA256="58917efc78ca760a2a1dd162d84e6cf1930c5b62a8dd9710bb4579ca4f2d69dc"
CUDA_NVCC="${AUTOLLAMACPP_NVCC:-/usr/local/cuda/bin/nvcc}"
LLAMACPP_ARTIFACT_CATALOG_URL="${AUTOLLAMACPP_ARTIFACT_CATALOG_URL:-}"
LLAMACPP_ARTIFACT_CATALOG_SHA256="${AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256:-}"
LLAMACPP_ALLOW_SOURCE_BUILD="${AUTOLLAMACPP_ALLOW_SOURCE_BUILD:-1}"

unset AUTOVLLM_NFS_EXPORT AUTOVLLM_NFS_MOUNT_POINT
unset AUTOVLLM_NVIDIA_DRIVER_VERSION AUTOVLLM_NVIDIA_DRIVER_SHA256 AUTOVLLM_API_PORT
unset AUTOVLLM_LLMFIT_VERSION AUTOVLLM_LLMFIT_SHA256
unset AUTOVLLM_LLMFIT_BIN AUTOVLLM_TMP_DIR AUTOLLAMACPP_SCRIPT_DIR
unset AUTOLLAMACPP_VERSION AUTOLLAMACPP_SHA256 AUTOLLAMACPP_SOURCE_URL
unset AUTOLLAMACPP_INSTALL_ROOT AUTOLLAMACPP_LINK_DIR
unset AUTOLLAMACPP_CUDA_ARCHITECTURES AUTOLLAMACPP_NVCC
unset AUTOLLAMACPP_ARTIFACT_CATALOG_URL AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256
unset AUTOLLAMACPP_ALLOW_SOURCE_BUILD

# Source shared setup functions
# shellcheck disable=SC1091 source=../common/setup-base.sh
source "$(cd -- "${SCRIPT_DIR}/.." && pwd)/common/setup-base.sh"

# --- llama.cpp-specific functions ---

# Release tags (v<major>.<minor>.<patch>) report "version: 0.4.1 (build N, commit
# C)". Nightly b<number> tags report "version: 0.4.1-dev (build 11052, commit C)",
# or "version: 10242 (C)" before upstream adopted release versions.
installed_llamacpp_version() {
    local binary="$1"
    "$binary" --version 2>&1 \
        | sed -nE \
            -e 's/^version:[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+)[[:space:]]+\(build .*/v\1/p' \
            -e 's/^version:[[:space:]]*[0-9]+\.[0-9]+\.[0-9]+-dev[[:space:]]+\(build[[:space:]]+([1-9][0-9]*),.*/b\1/p' \
            -e 's/^version:[[:space:]]*([1-9][0-9]*)[[:space:]].*/b\1/p' \
        | head -n 1
}

cuda_compute_capabilities() {
    local capabilities devices
    devices=$(profile_gpu_devices)
    local -a selection=()
    [ -z "$devices" ] || selection=(-i "$devices")
    if ! capabilities=$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits "${selection[@]}" 2>/dev/null \
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
        --parallel 1 \
        --kv-unified \
        --gpu-layers all \
        --verbosity 5 \
        --version >/dev/null 2>&1; then
        echo "FATAL: built llama-fit-params does not accept the managed metadata CLI" >&2
        return 1
    fi
    if ! "$binary" \
        --ctx-size 4096 \
        --parallel 2 \
        --kv-unified \
        --gpu-layers all \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --flash-attn on \
        --fit-print on \
        --verbosity 0 \
        --version >/dev/null 2>&1; then
        echo "FATAL: built llama-fit-params does not accept the managed estimation CLI" >&2
        return 1
    fi
}

verify_managed_server_cli() {
    local binary="$1"
    if ! "$binary" \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --flash-attn on \
        --version >/dev/null 2>&1; then
        echo "FATAL: built llama-server does not accept the managed Q8 KV CLI" >&2
        return 1
    fi
    # Catalog profiles launch with these options. Reject a pin that lacks any
    # of them at setup time, not after a model has been loaded.
    if ! "$binary" \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --flash-attn on \
        --ubatch-size 256 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        --cache-type-k-draft f16 \
        --cache-type-v-draft f16 \
        --no-mmproj \
        --jinja \
        --version >/dev/null 2>&1 \
        || ! "$binary" \
            --spec-type draft-dflash \
            --spec-draft-n-max 7 \
            --version >/dev/null 2>&1; then
        echo "FATAL: built llama-server does not accept the catalog profile CLI" >&2
        return 1
    fi
}

llamacpp_artifact_tool() {
    python3 "${SCRIPT_DIR}/../common/llamacpp_artifacts.py" "$@"
}

llamacpp_host_identity() {
    [ -n "${OS_ID:-}" ] || detect_profile_os
    local capabilities nvcc installed_toolkit=""
    capabilities=$(cuda_compute_capabilities)
    nvcc=$(find_nvcc) || nvcc=""
    if [ -n "$nvcc" ]; then
        installed_toolkit=$("$nvcc" --version | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p') || installed_toolkit=""
    fi
    python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
        version "$LLAMACPP_VERSION" source_sha256 "${LLAMACPP_SHA256,,}" \
        build_profile "$LLAMACPP_BUILD_PROFILE" fit_cli_patch_sha256 "$LLAMACPP_FIT_PATCH_SHA256" \
        compute_capabilities "${capabilities//$'\n'/,}" \
        cmake_cuda_architectures "$LLAMACPP_CUDA_ARCHITECTURES" \
        cuda_toolkit "$PROFILE_CUDA_TOOLKIT_VERSION" profile "${PROFILE_NAME:-unselected}" \
        os_id "$OS_ID" os_major "${OS_VERSION_ID%%.*}" arch "$OS_ARCH" glibc "$GLIBC_VERSION" \
        installed_cuda_toolkit "$installed_toolkit"
}

verify_llamacpp_binaries() {
    local runtime="$1" binary quantize_help quantize_status=0
    export LD_LIBRARY_PATH="${runtime}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    for binary in llama-server llama-fit-params; do
        if [ "$(installed_llamacpp_version "${runtime}/bin/${binary}")" != "$LLAMACPP_VERSION" ]; then
            echo "FATAL: built ${binary} did not report ${LLAMACPP_VERSION}" >&2
            return 1
        fi
    done
    verify_managed_server_cli "${runtime}/bin/llama-server"
    verify_fit_params_cli "${runtime}/bin/llama-fit-params"
    # The pinned quantizer has no --version and returns 1 for --help.
    quantize_help=$("${runtime}/bin/llama-quantize" --help 2>&1) || quantize_status=$?
    if [ "$quantize_status" -gt 1 ] || ! grep -q '^usage:' <<< "$quantize_help"; then
        echo "FATAL: llama-quantize did not report its CLI usage" >&2
        return 1
    fi
}

verify_llamacpp_cuda() {
    local runtime="$1" devices listed
    devices=$(profile_gpu_devices)
    local -a probe_env=("LD_LIBRARY_PATH=${runtime}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    [ -z "$devices" ] || probe_env+=("CUDA_VISIBLE_DEVICES=$devices")
    echo "[STEP:cuda_proof:START]"
    if ! listed=$(env "${probe_env[@]}" "${runtime}/bin/llama-server" --list-devices 2>&1) \
        || ! grep -Eq 'CUDA[0-9]+' <<< "$listed"; then
        echo "FATAL: llama-server does not expose the required CUDA backend: ${listed}" >&2
        return 10
    fi
    if ! env "${probe_env[@]}" "${runtime}/bin/cuda-probe"; then
        echo "[STEP:cuda_proof:FAIL]"
        echo "FATAL: artifact CUDA execution failed; inspect driver/device/fabric health" >&2
        return 10
    fi
    echo "[STEP:cuda_proof:OK]"
}

reuse_llamacpp_artifact() (
    set -e
    local host="$1" runtime work_dir entry marker identity url archive_bytes unpacked_bytes
    # A local completed package needs no catalog download or toolchain.
    if runtime=$(llamacpp_artifact_tool local "$LLAMACPP_INSTALL_ROOT" "$host"); then
        verify_llamacpp_binaries "$runtime"
        verify_llamacpp_cuda "$runtime"
        select_llamacpp_runtime "$runtime"
        echo "[ARTIFACT:hit:local] ${runtime}"
        echo "llama-server ${LLAMACPP_VERSION} already installed for CUDA capabilities $(cuda_compute_capabilities | paste -sd, -), skipping"
        exit 0
    fi
    if [ -z "$LLAMACPP_ARTIFACT_CATALOG_URL" ]; then
        echo "[ARTIFACT:miss:catalog_not_configured]"
        exit 4
    fi
    require_sha256 "llama.cpp artifact catalog" "$LLAMACPP_ARTIFACT_CATALOG_SHA256" AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/llamacpp-artifact.XXXXXX")
    trap 'rm -rf "$work_dir"' EXIT
    wget -q --timeout=30 --tries=2 "$LLAMACPP_ARTIFACT_CATALOG_URL" -O "${work_dir}/catalog.json"
    verify_sha256 "${work_dir}/catalog.json" "$LLAMACPP_ARTIFACT_CATALOG_SHA256" "llama.cpp artifact catalog"
    if entry=$(llamacpp_artifact_tool select "${work_dir}/catalog.json" "$host"); then :; else
        local status=$?
        echo "[ARTIFACT:miss:catalog_selection_failed:${status}]"
        exit "$status"
    fi
    marker=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["identity"])' "$entry")
    url=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["url"])' "$entry")
    archive_bytes=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["archive_bytes"])' "$entry")
    unpacked_bytes=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["unpacked_bytes"])' "$entry")
    identity=$(printf '%s' "$marker" | sha256sum | cut -c1-16)
    runtime="${LLAMACPP_INSTALL_ROOT%/}/${LLAMACPP_VERSION}-${identity}"
    qiip_require_inactive_runtime "$runtime" "$QIIP_GENERATION_ROOT" "$LLAMACPP_INSTALL_ROOT"
    llamacpp_artifact_tool capacity "$work_dir" "$((archive_bytes + unpacked_bytes))" \
        "$LLAMACPP_INSTALL_ROOT" "$unpacked_bytes"
    echo "[ARTIFACT:download:START] ${url}"
    wget -q --timeout=30 --tries=2 "$url" -O "${work_dir}/runtime.tar.gz"
    llamacpp_artifact_tool unpack "${work_dir}/runtime.tar.gz" "${work_dir}/runtime" "$entry"
    verify_llamacpp_binaries "${work_dir}/runtime"
    verify_llamacpp_cuda "${work_dir}/runtime"
    sudo python3 "${SCRIPT_DIR}/../common/llamacpp_artifacts.py" install "${work_dir}/runtime" "$runtime"
    verify_llamacpp_binaries "$runtime"
    verify_llamacpp_cuda "$runtime"
    select_llamacpp_runtime "$runtime"
    echo "[ARTIFACT:hit:catalog] ${runtime}"
)

install_llamacpp() {
    require_sha256 "llama.cpp ${LLAMACPP_VERSION}" "$LLAMACPP_SHA256" \
        "AUTOLLAMACPP_SHA256"
    if [[ ! "$LLAMACPP_VERSION" =~ ^(v[0-9]+\.[0-9]+\.[0-9]+|b[1-9][0-9]*)$ ]]; then
        echo "FATAL: AUTOLLAMACPP_VERSION must use the v<major>.<minor>.<patch> release-tag or b<number> build-tag format" >&2
        return 2
    fi
    if [[ ! "$LLAMACPP_SOURCE_URL" =~ ^https?:// ]]; then
        echo "FATAL: AUTOLLAMACPP_SOURCE_URL must be an HTTP(S) URL" >&2
        return 2
    fi
    if [[ ! "$LLAMACPP_ALLOW_SOURCE_BUILD" =~ ^[01]$ ]]; then
        echo "FATAL: AUTOLLAMACPP_ALLOW_SOURCE_BUILD must be 0 or 1" >&2
        return 2
    fi
    if [ -n "$LLAMACPP_ARTIFACT_CATALOG_URL" ]; then
        if [[ ! "$LLAMACPP_ARTIFACT_CATALOG_URL" =~ ^https?:// ]]; then
            echo "FATAL: AUTOLLAMACPP_ARTIFACT_CATALOG_URL must be an HTTP(S) URL" >&2
            return 2
        fi
        require_sha256 "llama.cpp artifact catalog" "$LLAMACPP_ARTIFACT_CATALOG_SHA256" AUTOLLAMACPP_ARTIFACT_CATALOG_SHA256
    elif [ -n "$LLAMACPP_ARTIFACT_CATALOG_SHA256" ]; then
        echo "FATAL: artifact catalog digest requires AUTOLLAMACPP_ARTIFACT_CATALOG_URL" >&2
        return 2
    fi
    verify_fit_params_patch_identity
    local started="$SECONDS" host fallback_reason
    [ -n "${OS_ID:-}" ] || detect_profile_os
    host=$(llamacpp_host_identity)
    echo "[ARTIFACT:lookup:START] ${host}"
    run_with_errexit reuse_llamacpp_artifact "$host"
    echo "[TIMING:llamacpp:lookup_seconds=$((SECONDS - started))]"
    if [ "$STEP_STATUS" -eq 0 ]; then
        echo "[TIMING:llamacpp:artifact_seconds=$((SECONDS - started))]"
        return 0
    fi
    # A failed CUDA execution proof is a stack failure, not a cache miss.
    if [ "$STEP_STATUS" -eq 10 ]; then return 10; fi
    fallback_reason="no_compatible_artifact"
    [ "$STEP_STATUS" -eq 4 ] || fallback_reason="artifact_verification_or_download_failed"
    echo "[ARTIFACT:miss:${fallback_reason}]"
    if [ "$LLAMACPP_ALLOW_SOURCE_BUILD" = "0" ]; then
        echo "FATAL: no verified compatible llama.cpp artifact; source fallback is disabled" >&2
        return 1
    fi
    echo "[BUILD:fallback:${fallback_reason}]"
    local plan jobs build_tmp
    plan=$(llamacpp_artifact_tool resources "$INSTALL_TMP_DIR" "$LLAMACPP_INSTALL_ROOT")
    jobs=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["jobs"])' "$plan")
    build_tmp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["temporary"])' "$plan")
    echo "[BUILD:resources] ${plan}"
    step llamacpp_toolchain install_runtime_prerequisites llamacpp
    step cuda_toolkit install_cuda_toolkit
    # Installing the toolkit can consume RAM/disk on these same filesystems.
    # Refresh the plan before any source download or compiler work begins.
    plan=$(llamacpp_artifact_tool resources "$INSTALL_TMP_DIR" "$LLAMACPP_INSTALL_ROOT")
    jobs=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["jobs"])' "$plan")
    build_tmp=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["temporary"])' "$plan")
    echo "[BUILD:resources:after_toolchain] ${plan}"
    if ! command -v cmake >/dev/null || ! command -v make >/dev/null; then
        echo "FATAL: cmake and make are required to build llama.cpp" >&2
        return 1
    fi
    # The toolkit step ran in its own subshell (step -> run_with_errexit), so
    # resolve nvcc here, after select_runtime_profile pinned the toolkit. The
    # profile dnf layout installs under /usr/local/cuda-<version>/bin and the
    # toolkit step also symlinks /usr/local/cuda, so find_nvcc covers both.
    CUDA_NVCC="$(find_nvcc)" || CUDA_NVCC=""
    if [ -z "$CUDA_NVCC" ]; then
        echo "FATAL: CUDA nvcc is required to build managed llama.cpp" >&2
        return 1
    fi

    verify_fit_params_patch_identity

    local compute_capabilities build_identity install_dir marker cuda_toolkit nvcc_version compiler
    if ! nvcc_version=$("$CUDA_NVCC" --version); then
        echo "FATAL: could not determine CUDA toolkit version from ${CUDA_NVCC}" >&2
        return 1
    fi
    cuda_toolkit=$(printf '%s\n' "$nvcc_version" | sed -n 's/.*release [0-9.]*, V\([0-9][0-9.]*\).*/\1/p')
    if [[ ! "$cuda_toolkit" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "FATAL: could not determine CUDA toolkit version from ${CUDA_NVCC}" >&2
        return 1
    fi
    compute_capabilities=$(cuda_compute_capabilities) || return
    compiler=$(g++ -dumpfullversion -dumpversion)
    marker=$(printf 'publication_schema=2\nversion=%s\nsource_sha256=%s\nbuild_profile=%s\nfit_cli_patch_sha256=%s\ncompute_capabilities=%s\ncmake_cuda_architectures=%s\ncuda_toolkit=%s\nprofile=%s\ncompiler=GNU-%s\nos_id=%s\nos_major=%s\narch=%s\nglibc=%s\n' \
        "$LLAMACPP_VERSION" \
        "${LLAMACPP_SHA256,,}" \
        "$LLAMACPP_BUILD_PROFILE" \
        "$LLAMACPP_FIT_PATCH_SHA256" \
        "${compute_capabilities//$'\n'/,}" \
        "$LLAMACPP_CUDA_ARCHITECTURES" \
        "$cuda_toolkit" "${PROFILE_NAME:-unselected}" "$compiler" \
        "$OS_ID" "${OS_VERSION_ID%%.*}" "$OS_ARCH" "$GLIBC_VERSION")
    build_identity=$(printf '%s' "$marker" | sha256sum | cut -c1-16)
    install_dir="${LLAMACPP_INSTALL_ROOT%/}/${LLAMACPP_VERSION}-${build_identity}"

    if [ -f "${install_dir}/RUNTIME.json" ]; then
        qiip_generation_tool verify-runtime "$install_dir" "$marker"
        verify_llamacpp_binaries "$install_dir"
        verify_llamacpp_cuda "$install_dir"
        select_llamacpp_runtime "$install_dir"
        echo "llama-server ${LLAMACPP_VERSION} already installed for CUDA capabilities ${compute_capabilities//$'\n'/,}, skipping"
        return 0
    fi

    qiip_require_inactive_runtime "$install_dir" "$QIIP_GENERATION_ROOT" "$LLAMACPP_INSTALL_ROOT"

    # run_with_errexit invokes steps inside its own set -e subshell, where a
    # function RETURN trap is not reliable. Keep the entire build and publish
    # sequence in a dedicated subshell whose EXIT trap owns all cleanup.
    (
        set -e
        local work_dir archive source_dir build_dir server_bin fit_bin quantize_bin build_status
        work_dir=$(mktemp -d "${build_tmp%/}/auto-llamacpp.XXXXXX")
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
        local staged="${work_dir}/runtime" devices
        mkdir -p "$source_dir" "${staged}/bin"
        # Prove the prepared driver before downloading or compiling the engine,
        # and retain that same probe in the completed package.
        verify_cuda_execution "${staged}/bin/cuda-probe"
        devices=$(profile_gpu_devices)
        [ -z "$devices" ] || export CUDA_VISIBLE_DEVICES="$devices"

        wget -q "$LLAMACPP_SOURCE_URL" -O "$archive"
        verify_sha256 "$archive" "$LLAMACPP_SHA256" \
            "llama.cpp ${LLAMACPP_VERSION} source"
        tar xzf "$archive" -C "$source_dir" --strip-components=1
        # The pinned memory estimator supports unified KV internally, but its CLI
        # allowlist omits llama-fit-params. Expose the existing option so the
        # planner estimates the exact KV mode used by llama-server.
        enable_fit_params_unified_kv "$source_dir"

        # Source tarballs have no .git, so cmake/build-info.cmake logs two harmless
        # "fatal: not a git repository" lines and falls back to BUILD_NUMBER=0.
        # The version flags below are what installed_llamacpp_version() matches
        # against -- they are not decorative. A release tag must clear the
        # default "-dev" suffix; a build tag must override the fallback number.
        local -a version_flags
        if [[ "$LLAMACPP_VERSION" == v* ]]; then
            version_flags=(-DLLAMA_BUILD_IS_DEV=OFF)
        else
            version_flags=(-DLLAMA_BUILD_NUMBER="${LLAMACPP_VERSION#b}")
        fi
        echo "[BUILD:configure:START]"
        cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="$(command -v gcc)" \
            -DCMAKE_CXX_COMPILER="$(command -v g++)" \
            -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
            -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
            -DCMAKE_CUDA_ARCHITECTURES="$LLAMACPP_CUDA_ARCHITECTURES" \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON \
            -DGGML_NATIVE=OFF \
            -DGGML_AVX=OFF \
            -DGGML_AVX2=OFF \
            -DGGML_AVX512=OFF \
            -DGGML_FMA=OFF \
            -DGGML_F16C=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_SERVER=ON \
            -DLLAMA_BUILD_APP=OFF \
            -DLLAMA_BUILD_UI=OFF \
            -DLLAMA_BUILD_MTMD=OFF \
            -DLLAMA_OPENSSL=OFF \
            "${version_flags[@]}" \
            -DLLAMA_BUILD_COMMIT="$LLAMACPP_VERSION"
        echo "[BUILD:compile:START:jobs=${jobs}]"
        cmake --build "$build_dir" --target llama-server llama-fit-params llama-quantize \
            --parallel "$jobs"
        echo "[BUILD:compile:OK:elapsed_seconds=$((SECONDS - started))]"

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
        verify_managed_server_cli "$server_bin"
        verify_fit_params_cli "$fit_bin"

        install -m 755 "$server_bin" "$fit_bin" "$quantize_bin" "${staged}/bin/"
        llamacpp_artifact_tool libraries "${staged}/bin" "${staged}/lib"
        printf '%s\n' "$marker" > "${staged}/BUILD-INFO"
        verify_llamacpp_binaries "$staged"
        verify_llamacpp_cuda "$staged"
        llamacpp_artifact_tool seal "$staged" "$marker"
        echo "[BUILD:publish:START]"
        sudo python3 "${SCRIPT_DIR}/../common/llamacpp_artifacts.py" install "$staged" "$install_dir"
        # Validate the installed copies too; failed copying or disk exhaustion
        # must never produce a completion manifest or switch the active set.
        verify_llamacpp_binaries "$install_dir"
        verify_llamacpp_cuda "$install_dir"
        select_llamacpp_runtime "$install_dir"
        echo "[TIMING:llamacpp:source_seconds=$((SECONDS - started))]"
    )
}

select_llamacpp_runtime() {
    local install_dir="$1" binary
    if [ -n "${QIIP_RUNTIME_SELECTION:-}" ]; then
        printf '%s\n' "$install_dir" > "$QIIP_RUNTIME_SELECTION"
    else
        # Standalone install_llamacpp callers use one pointer for all tools.
        # Stable compatibility links are installed once, before the commit.
        sudo mkdir -p "$LLAMACPP_LINK_DIR"
        if [ ! -L "${LLAMACPP_INSTALL_ROOT%/}/current" ] \
            && [ -L "${LLAMACPP_LINK_DIR%/}/llama-server" ] \
            && [ -e "${LLAMACPP_LINK_DIR%/}/llama-server" ]; then
            local old_binary
            old_binary=$(readlink -f "${LLAMACPP_LINK_DIR%/}/llama-server")
            atomic_link "$(dirname "$(dirname "$old_binary")")" \
                "${LLAMACPP_INSTALL_ROOT%/}/current"
        fi
        for binary in llama-server llama-fit-params llama-quantize; do
            atomic_link "${LLAMACPP_INSTALL_ROOT%/}/current/bin/${binary}" \
                "${LLAMACPP_LINK_DIR%/}/${binary}"
        done
        if [ -L "${LLAMACPP_INSTALL_ROOT%/}/current" ] \
            && [ "$(readlink -f "${LLAMACPP_INSTALL_ROOT%/}/current")" != "$install_dir" ]; then
            atomic_link "$(readlink -f "${LLAMACPP_INSTALL_ROOT%/}/current")" \
                "${LLAMACPP_INSTALL_ROOT%/}/previous"
        fi
        atomic_link "$install_dir" "${LLAMACPP_INSTALL_ROOT%/}/current"
    fi
}

install_llamacpp_driver_checked() {
    echo "[STEP:llamacpp_install:START]"
    run_with_errexit install_llamacpp
    local status="$STEP_STATUS"
    if [ "$status" -ne 0 ]; then
        echo "[STEP:llamacpp_install:FAIL]"
        if [ "$status" -eq 10 ]; then
            if [ "${QIIP_DRIVER_REPAIRED:-0}" -eq 1 ]; then
                require_driver_reboot "CUDA execution failed after driver repair"
                return 20
            fi
            resume_required maintenance_required "CUDA execution failed despite a compatible driver version; inspect device/fabric/runtime health, then retry setup"
            return 21
        fi
        return "$status"
    fi
    if [ -f "${DRIVER_STATE_DIR}/driver-reboot" ]; then
        sudo rm -f "${DRIVER_STATE_DIR}/driver-reboot"
    fi
    echo "[STEP:llamacpp_install:OK]"
}

# --- Main ---
main() {
    if [ -z "$NFS_EXPORT" ]; then
        echo "FATAL: AUTOVLLM_NFS_EXPORT is required for node provisioning" >&2
        return 2
    fi
    require_sha256 "llmfit ${LLMFIT_RELEASE}" "$LLMFIT_SHA256" \
        "AUTOVLLM_LLMFIT_SHA256"
    require_sha256 "llama.cpp ${LLAMACPP_VERSION}" "$LLAMACPP_SHA256" \
        "AUTOLLAMACPP_SHA256"
    begin_engine_generation
    prepare_runtime llamacpp 1
    install_llamacpp_driver_checked
    step nfs_mount mount_nfs_cache
    step firewall configure_firewall
    soft_step llmfit_install install_llmfit
    step generation_activate activate_engine_generation llama_cpp

    echo "Setup complete"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
