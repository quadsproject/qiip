#!/bin/bash
# Shared setup functions sourced by engine-specific setup.sh scripts.
# Env vars use the AUTOVLLM_ prefix for backward compatibility.

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

run_with_errexit() {
    set +e
    (set -e; "$@")
    STEP_STATUS=$?
    set -e
}

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

run_system_update() {
    local running_kernel
    running_kernel=$(uname -r)
    sudo dnf -y install kernel-devel-"${running_kernel}" kernel-headers-"${running_kernel}" \
        cmake gcc gcc-c++ make wget nfs-utils elfutils-libelf-devel \
        python3.12 python3.12-devel
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
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-setup-driver.XXXXXX")
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
    sudo dnf -y install dnf-plugins-core ninja-build
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "CUDA toolkit already installed, skipping"
    else
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        sudo dnf -y install cuda-toolkit
    fi
    sudo ln -sfn /usr/bin/ninja-build /usr/local/bin/ninja
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
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-setup-llmfit.XXXXXX")
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
