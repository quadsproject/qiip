#!/bin/bash
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

# Script configuration is captured above. Do not leak this namespace into
# installers or other child processes spawned by setup.sh.
unset AUTOVLLM_NFS_EXPORT AUTOVLLM_NFS_MOUNT_POINT
unset AUTOVLLM_NVIDIA_DRIVER_VERSION AUTOVLLM_NVIDIA_DRIVER_SHA256 AUTOVLLM_API_PORT
unset AUTOVLLM_LLMFIT_VERSION AUTOVLLM_LLMFIT_SHA256
unset AUTOVLLM_VENV AUTOVLLM_LLMFIT_BIN AUTOVLLM_TMP_DIR
unset AUTOVLLM_UV_BIN AUTOVLLM_UV_PROJECT AUTOVLLM_SCRIPT_DIR FLASHINFER_INDEX_URL

require_sha256() {
    local label="$1"
    local digest="$2"
    local setting="$3"
    if [[ ! "$digest" =~ ^[[:xdigit:]]{64}$ ]]; then
        echo "FATAL: ${label} requires a 64-character SHA-256 in ${setting}" >&2
        return 2
    fi
}

verify_sha256() {
    local file="$1"
    local digest="$2"
    local label="$3"
    if ! printf '%s  %s\n' "$digest" "$file" | sha256sum -c - >/dev/null; then
        echo "FATAL: ${label} SHA-256 verification failed" >&2
        return 1
    fi
}

reject_retired_flashinfer_index() {
    if [ "$FLASHINFER_INDEX_OVERRIDE_SET" -eq 1 ]; then
        echo "FATAL: FLASHINFER_INDEX_URL is retired; the frozen auto-vllm/uv.lock owns package sources. Ship a regenerated auto-vllm bundle for a custom mirror." >&2
        return 2
    fi
}

run_with_errexit() {
    # A function invoked directly as an if-condition inherits Bash's ignored
    # errexit state. Run it as an ordinary subshell command instead, then
    # restore the caller's hard-fail mode before interpreting its status.
    # Step functions therefore cannot export state back into the parent shell.
    set +e
    (set -e; "$@")
    STEP_STATUS=$?
    set -e
}

# --- Step wrapper ---
step() {
    local name="$1"; shift
    echo "[STEP:${name}:START]"
    run_with_errexit "$@"
    if [ "$STEP_STATUS" -eq 0 ]; then
        echo "[STEP:${name}:OK]"
    else
        echo "[STEP:${name}:FAIL]"
        exit 1
    fi
}

# --- Soft step wrapper (non-fatal) ---
soft_step() {
    local name="$1"; shift
    echo "[STEP:${name}:START]"
    run_with_errexit "$@"
    if [ "$STEP_STATUS" -eq 0 ]; then
        echo "[STEP:${name}:OK]"
    else
        echo "[STEP:${name}:WARN] (non-fatal, continuing)"
    fi
}

# --- Step functions (idempotent) ---

run_system_update() {
    local running_kernel
    running_kernel=$(uname -r)

    # Build the driver for the kernel that is running and keep that kernel as
    # the next boot target. Staging a newer kernel first can leave DKMS built
    # for the old kernel or make its matching kernel-devel disappear.
    sudo dnf -y install kernel-devel-"${running_kernel}" kernel-headers-"${running_kernel}" \
        gcc make wget nfs-utils elfutils-libelf-devel python3.12 python3.12-devel
    sudo dnf -y update '--exclude=kernel*'
}

install_nvidia_driver() {
    require_sha256 "NVIDIA driver ${DRIVER_VERSION}" "$DRIVER_SHA256" \
        "AUTOVLLM_NVIDIA_DRIVER_SHA256"
    if nvidia-smi &>/dev/null; then
        local installed_versions
        installed_versions=$(
            nvidia-smi --query-gpu=driver_version --format=csv,noheader \
                | sed '/^[[:space:]]*$/d' | sort -u
        )
        if [ "$installed_versions" = "$DRIVER_VERSION" ]; then
            echo "NVIDIA driver ${DRIVER_VERSION} already installed, skipping"
            return 0
        fi
        installed_versions=${installed_versions//$'\n'/, }
        echo "FATAL: installed NVIDIA driver ${installed_versions:-unknown} does not match requested ${DRIVER_VERSION}; upgrade the driver explicitly before provisioning" >&2
        return 1
    fi
    if modinfo nvidia &>/dev/null; then
        echo "NVIDIA kernel module found but not loaded, loading"
        sudo modprobe nvidia
    fi
    # RPM driver whose kmod isn't built for the running kernel.
    if ls /usr/lib64/libnvidia-ml.so.* &>/dev/null; then
        echo "RPM-installed NVIDIA driver found, kernel module missing for $(uname -r)"
        echo "Rebuilding kernel module"
        if (sudo dkms autoinstall 2>/dev/null || sudo akmods --force 2>/dev/null) \
            && sudo modprobe nvidia && nvidia-smi; then
            return 0
        fi
        echo "Kernel module rebuild failed, removing broken RPM driver"
        sudo dnf -y remove '*nvidia*driver*' 2>/dev/null || true
        sudo rm -f /etc/modprobe.d/blacklist-nouveau.conf
    fi
    local work_dir installer status
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-vllm-driver.XXXXXX")
    installer="${work_dir}/NVIDIA-driver.run"
    if wget -q "${NVIDIA_DRIVER_URL}" -O "$installer"; then
        :
    else
        status=$?
        rm -rf "$work_dir"
        return "$status"
    fi
    if ! verify_sha256 "$installer" "$DRIVER_SHA256" \
        "NVIDIA driver ${DRIVER_VERSION}"; then
        rm -rf "$work_dir"
        return 1
    fi
    chmod +x "$installer"
    echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
    sudo dracut --force
    sudo modprobe -r nouveau 2>/dev/null || true
    if sudo sh "$installer" --dkms --no-x-check --no-nouveau-check --ui=none --no-questions; then
        status=0
    else
        status=$?
    fi
    rm -rf "$work_dir"
    return "$status"
}

install_cuda_toolkit() {
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "CUDA toolkit already installed, skipping"
        return 0
    fi
    sudo dnf -y install dnf-plugins-core ninja-build
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    sudo dnf -y install cuda-toolkit
    ln -sfn /usr/bin/ninja-build /usr/local/bin/ninja
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

mount_nfs_cache() {
    if mountpoint -q "${NFS_MOUNT_POINT}"; then
        echo "NFS already mounted at ${NFS_MOUNT_POINT}, skipping"
        return 0
    fi
    sudo mkdir -p "${NFS_MOUNT_POINT}"
    sudo timeout --kill-after=5 30 \
        mount -t nfs -o vers=3,soft,timeo=100,retrans=2 "${NFS_EXPORT}" "${NFS_MOUNT_POINT}"
}

configure_firewall() {
    if command -v firewall-cmd &>/dev/null && systemctl is-active --quiet firewalld; then
        if sudo firewall-cmd --query-port="${API_PORT}/tcp" &>/dev/null; then
            echo "Firewall rule already exists for port ${API_PORT}, skipping"
            return 0
        fi
        sudo firewall-cmd --add-port="${API_PORT}/tcp" --permanent
        sudo firewall-cmd --reload
    elif command -v iptables &>/dev/null \
        && systemctl list-unit-files --type=service --no-legend iptables.service 2>/dev/null \
            | grep -q '^iptables\.service'; then
        if sudo iptables -C INPUT -p tcp --dport "${API_PORT}" -j ACCEPT 2>/dev/null; then
            echo "Firewall rule already exists for port ${API_PORT}, skipping"
            return 0
        fi
        sudo iptables -I INPUT -p tcp --dport "${API_PORT}" -j ACCEPT
        sudo iptables-save | sudo tee /etc/sysconfig/iptables > /dev/null
        sudo systemctl restart iptables
    else
        echo "No active firewalld or installed iptables service; no firewall rule required"
    fi
}

install_llmfit() {
    require_sha256 "llmfit ${LLMFIT_RELEASE}" "$LLMFIT_SHA256" \
        "AUTOVLLM_LLMFIT_SHA256"
    if [ -x "$LLMFIT_BIN" ]; then
        local installed_version
        installed_version=$(
            "$LLMFIT_BIN" --version 2>/dev/null \
                | grep -Eo '[0-9]+([.][0-9]+)+' | head -n 1
        ) || true
        if [ "$installed_version" = "$LLMFIT_RELEASE" ]; then
            echo "llmfit ${LLMFIT_RELEASE} already installed, skipping"
            return 0
        fi
        echo "Replacing llmfit ${installed_version:-unknown} with requested ${LLMFIT_RELEASE}"
    fi
    local work_dir archive status
    local -a llmfit_binaries=()
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-vllm-llmfit.XXXXXX")
    archive="${work_dir}/llmfit.tar.gz"
    if wget -q "${LLMFIT_URL}" -O "$archive"; then
        :
    else
        status=$?
        rm -rf "$work_dir"
        return "$status"
    fi
    if ! verify_sha256 "$archive" "$LLMFIT_SHA256" "llmfit ${LLMFIT_RELEASE}"; then
        rm -rf "$work_dir"
        return 1
    fi
    tar -xzf "$archive" -C "$work_dir"
    mapfile -t llmfit_binaries < <(find "$work_dir" -name llmfit -type f -print)
    if [ "${#llmfit_binaries[@]}" -ne 1 ]; then
        echo "FATAL: verified llmfit archive must contain exactly one llmfit binary" >&2
        rm -rf "$work_dir"
        return 1
    fi
    if sudo install -m 755 "${llmfit_binaries[0]}" "$LLMFIT_BIN"; then
        status=0
    else
        status=$?
    fi
    rm -rf "$work_dir"
    return "$status"
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
