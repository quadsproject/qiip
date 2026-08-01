#!/bin/bash
set -euo pipefail

# --- Configurable defaults ---
NFS_SERVER="${NFS_SERVER:-storage.example.com:/mnt/SATA/scratch/grafuls/hf-cache}"
NFS_MOUNT_POINT="${NFS_MOUNT_POINT:-/srv/hf-cache}"
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-580.126.09}"
NVIDIA_DRIVER_URL="${NVIDIA_DRIVER_URL:-https://us.download.nvidia.com/tesla/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run}"
VLLM_PORT="${VLLM_PORT:-8000}"
LLMFIT_VERSION="${LLMFIT_VERSION:-1.1.6}"
LLMFIT_URL="${LLMFIT_URL:-https://github.com/AlexsJones/llmfit/releases/download/v${LLMFIT_VERSION}/llmfit-v${LLMFIT_VERSION}-x86_64-unknown-linux-musl.tar.gz}"
VLLM_VENV="${AUTOVLLM_VENV:-/opt/vllm-venv}"
LLMFIT_BIN="${AUTOVLLM_LLMFIT_BIN:-/usr/local/bin/llmfit}"
INSTALL_TMP_DIR="${AUTOVLLM_TMP_DIR:-/tmp}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl}"

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
    if nvidia-smi &>/dev/null; then
        local installed_versions
        installed_versions=$(
            nvidia-smi --query-gpu=driver_version --format=csv,noheader \
                | sed '/^[[:space:]]*$/d' | sort -u
        )
        if [ "$installed_versions" = "$NVIDIA_DRIVER_VERSION" ]; then
            echo "NVIDIA driver ${NVIDIA_DRIVER_VERSION} already installed, skipping"
            return 0
        fi
        installed_versions=${installed_versions//$'\n'/, }
        echo "FATAL: installed NVIDIA driver ${installed_versions:-unknown} does not match requested ${NVIDIA_DRIVER_VERSION}; upgrade the driver explicitly before provisioning" >&2
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
    echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
    sudo dracut --force
    sudo modprobe -r nouveau 2>/dev/null || true
    wget -q "${NVIDIA_DRIVER_URL}" -O /tmp/NVIDIA-driver.run
    chmod +x /tmp/NVIDIA-driver.run
    sudo sh /tmp/NVIDIA-driver.run --dkms --no-x-check --no-nouveau-check --ui=none --no-questions
    rm -f /tmp/NVIDIA-driver.run
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

install_vllm() {
    if [ -x "${VLLM_VENV}/bin/vllm" ]; then
        echo "vLLM already installed in ${VLLM_VENV}, skipping core install"
    else
        python3.12 -m venv "${VLLM_VENV}"
        "${VLLM_VENV}/bin/pip" install --upgrade pip
        "${VLLM_VENV}/bin/pip" install vllm
    fi

    install_flashinfer_aot
}

install_flashinfer_aot() {
    local flashinfer_version
    flashinfer_version=$(
        "${VLLM_VENV}/bin/python" -c \
            'from importlib.metadata import version; from packaging.version import Version; print(Version(version("flashinfer-python")).public)'
    )

    if "${VLLM_VENV}/bin/python" -c \
        'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public' \
        &>/dev/null; then
        echo "FlashInfer AOT kernels ${flashinfer_version} already installed, skipping"
        return 0
    fi

    "${VLLM_VENV}/bin/pip" install --no-deps \
        --index-url "$FLASHINFER_INDEX_URL" \
        "flashinfer-cubin==${flashinfer_version}"
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
        mount -t nfs -o vers=3,soft,timeo=100,retrans=2 "${NFS_SERVER}" "${NFS_MOUNT_POINT}"
}

configure_firewall() {
    if command -v firewall-cmd &>/dev/null && systemctl is-active --quiet firewalld; then
        if sudo firewall-cmd --query-port="${VLLM_PORT}/tcp" &>/dev/null; then
            echo "Firewall rule already exists for port ${VLLM_PORT}, skipping"
            return 0
        fi
        sudo firewall-cmd --add-port="${VLLM_PORT}/tcp" --permanent
        sudo firewall-cmd --reload
    elif command -v iptables &>/dev/null \
        && systemctl list-unit-files --type=service --no-legend iptables.service 2>/dev/null \
            | grep -q '^iptables\.service'; then
        if sudo iptables -C INPUT -p tcp --dport "${VLLM_PORT}" -j ACCEPT 2>/dev/null; then
            echo "Firewall rule already exists for port ${VLLM_PORT}, skipping"
            return 0
        fi
        sudo iptables -I INPUT -p tcp --dport "${VLLM_PORT}" -j ACCEPT
        sudo iptables-save | sudo tee /etc/sysconfig/iptables > /dev/null
        sudo systemctl restart iptables
    else
        echo "No active firewalld or installed iptables service; no firewall rule required"
    fi
}

install_llmfit() {
    if [ -x "$LLMFIT_BIN" ]; then
        local installed_version
        installed_version=$(
            "$LLMFIT_BIN" --version 2>/dev/null \
                | grep -Eo '[0-9]+([.][0-9]+)+' | head -n 1
        ) || true
        if [ "$installed_version" = "$LLMFIT_VERSION" ]; then
            echo "llmfit ${LLMFIT_VERSION} already installed, skipping"
            return 0
        fi
        echo "Replacing llmfit ${installed_version:-unknown} with requested ${LLMFIT_VERSION}"
    fi
    local archive="${INSTALL_TMP_DIR}/llmfit.tar.gz"
    wget -q "${LLMFIT_URL}" -O "$archive"
    tar -xzf "$archive" -C "$INSTALL_TMP_DIR/"
    sudo install -m 755 \
        "$(find "$INSTALL_TMP_DIR/" -name llmfit -type f -print -quit)" \
        "$LLMFIT_BIN"
    rm -rf "$archive" "${INSTALL_TMP_DIR}"/llmfit-*
}

# --- Main ---
main() {
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
