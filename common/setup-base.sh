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
        echo "Installed NVIDIA driver ${installed_versions:-unknown} does not match requested ${DRIVER_VERSION}; uninstalling"
        if [ -x /usr/bin/nvidia-uninstall ]; then
            sudo /usr/bin/nvidia-uninstall --silent
        else
            sudo dnf -y remove '*nvidia*driver*' 2>/dev/null || true
        fi
        sudo rm -f /etc/modprobe.d/blacklist-nouveau.conf
        sudo modprobe -r nvidia 2>/dev/null || true
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
    sudo dnf -y install dnf-plugins-core
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "CUDA toolkit already installed, skipping"
    else
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        sudo dnf -y install cuda-toolkit
    fi
}

install_fabricmanager_rpm() {
    local driver_version="$1"
    local driver_major="${driver_version%%.*}"
    local pkg="nvidia-fabricmanager-${driver_version}-1"

    # Strategy 1: dnf module stream (RPM-installed drivers only)
    local module_license stream_suffix
    module_license=$(modinfo nvidia 2>/dev/null \
        | sed -n 's/^license:[[:space:]]*//Ip' | xargs) || true
    if [[ "$module_license" == *"MIT/GPL"* ]]; then
        stream_suffix="-open-dkms"
    else
        stream_suffix="-dkms"
    fi
    echo "Kernel module license: ${module_license:-unknown} (stream suffix: ${stream_suffix})"

    sudo dnf module reset -y nvidia-driver 2>/dev/null || true
    if sudo dnf module enable -y "nvidia-driver:${driver_major}${stream_suffix}" 2>/dev/null; then
        sudo dnf clean metadata
        if sudo dnf install -y "$pkg" 2>/dev/null; then
            return 0
        fi
    fi

    # Strategy 2: direct install from whatever repos are configured
    echo "Module stream unavailable or version not found; trying direct install"
    sudo dnf clean metadata
    if sudo dnf install -y --disableexcludes=all "$pkg" 2>/dev/null; then
        return 0
    fi

    # Strategy 3: NVIDIA redistributable archive. The standard CUDA repo
    # only carries fabricmanager for the latest driver branch. For .run-
    # installed drivers on a different branch, pull directly from NVIDIA.
    echo "RPM not in repos; trying NVIDIA redistributable archive for ${driver_version}"
    install_fabricmanager_from_redist "$driver_version"
}

install_fabricmanager_from_redist() {
    local driver_version="$1"
    local base_url="https://developer.download.nvidia.com/compute/nvidia-driver/redist/fabricmanager/linux-x86_64"
    local archive_name="fabricmanager-linux-x86_64-${driver_version}-archive.tar.xz"
    local url="${base_url}/${archive_name}"

    local work_dir archive status
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-setup-fm.XXXXXX")
    archive="${work_dir}/${archive_name}"

    echo "Downloading ${url}"
    if ! wget -q "$url" -O "$archive"; then
        rm -rf "$work_dir"
        echo "FATAL: fabricmanager ${driver_version} not found at NVIDIA redist archive" >&2
        echo "URL tried: ${url}" >&2
        echo "Either:" >&2
        echo "  1. Set AUTOVLLM_NVIDIA_DRIVER_VERSION to a version NVIDIA publishes fabricmanager for" >&2
        echo "  2. Provide the fabricmanager RPM or archive manually" >&2
        return 1
    fi

    tar -xf "$archive" -C "$work_dir"
    local extracted="${work_dir}/fabricmanager-linux-x86_64-${driver_version}-archive"

    if [ ! -d "$extracted" ]; then
        # Handle slight naming variations in the archive
        extracted=$(find "$work_dir" -maxdepth 1 -type d -name 'fabricmanager-*' | head -1)
    fi
    if [ -z "$extracted" ] || [ ! -d "$extracted" ]; then
        rm -rf "$work_dir"
        echo "FATAL: unexpected archive layout in ${archive_name}" >&2
        return 1
    fi

    # The redist archive contains: bin/nv-fabricmanager, lib/, systemd/, etc.
    if [ -x "${extracted}/bin/nv-fabricmanager" ]; then
        sudo install -m 755 "${extracted}/bin/nv-fabricmanager" /usr/bin/nv-fabricmanager
    else
        rm -rf "$work_dir"
        echo "FATAL: nv-fabricmanager binary not found in archive" >&2
        return 1
    fi

    # The redist archive's bundled unit and config may reference options
    # unsupported by this driver version (e.g. PARTITION_RAIL_POLICY).
    # Use a minimal unit that lets nv-fabricmanager pick its own defaults.
    cat <<'UNIT' | sudo tee /etc/systemd/system/nvidia-fabricmanager.service > /dev/null
[Unit]
Description=NVIDIA Fabric Manager
After=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nv-fabricmanager
LimitCORE=infinity

[Install]
WantedBy=multi-user.target
UNIT

    # Write a minimal config — the redist archive's bundled config may
    # contain directives unsupported by this driver version. The binary
    # requires the file to exist but works with just the mode setting.
    sudo mkdir -p /usr/share/nvidia/nvswitch
    cat <<'CFG' | sudo tee /usr/share/nvidia/nvswitch/fabricmanager.cfg > /dev/null
FABRIC_MODE=1
FABRIC_MODE_RESTART=0
CFG
    echo "Wrote minimal fabricmanager.cfg"

    sudo systemctl daemon-reload
    rm -rf "$work_dir"
    echo "Fabric Manager ${driver_version} installed from NVIDIA redistributable archive"
}

ensure_fabric_manager() {
    local nvswitch_count
    nvswitch_count=$(lspci 2>/dev/null | grep -ci nvswitch || true)
    if [ "$nvswitch_count" -eq 0 ]; then
        echo "No NVSwitch devices found, skipping Fabric Manager"
        return 0
    fi
    echo "Found ${nvswitch_count} NVSwitch device(s), Fabric Manager required"

    local driver_version=""
    if command -v nvidia-smi &>/dev/null; then
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
            | sed '/^[[:space:]]*$/d' | head -1 | xargs)
    fi
    if [ -z "$driver_version" ] && [ -f /proc/driver/nvidia/version ]; then
        driver_version=$(sed -n 's/.*Kernel Module[[:space:]]*\([0-9.]*\).*/\1/p' \
            /proc/driver/nvidia/version)
    fi
    if [ -z "$driver_version" ]; then
        echo "FATAL: cannot detect NVIDIA driver version for Fabric Manager install" >&2
        return 1
    fi
    echo "Detected NVIDIA driver version: ${driver_version}"

    local installed_fm_version=""
    if rpm -q nvidia-fabricmanager &>/dev/null; then
        installed_fm_version=$(rpm -q --qf '%{VERSION}' nvidia-fabricmanager)
    elif [ -x /usr/bin/nv-fabricmanager ]; then
        installed_fm_version=$(nv-fabricmanager --version 2>/dev/null \
            | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true
    fi

    if [ "$installed_fm_version" = "$driver_version" ]; then
        echo "nvidia-fabricmanager ${driver_version} already installed"
    else
        install_fabricmanager_rpm "$driver_version"
    fi

    # Redist-installed binary needs a config file and a correct systemd
    # unit. Ensure both exist even when install was skipped (version matched).
    if [ -x /usr/bin/nv-fabricmanager ] && ! rpm -q nvidia-fabricmanager &>/dev/null; then
        if [ ! -f /usr/share/nvidia/nvswitch/fabricmanager.cfg ]; then
            sudo mkdir -p /usr/share/nvidia/nvswitch
            cat <<'CFG' | sudo tee /usr/share/nvidia/nvswitch/fabricmanager.cfg > /dev/null
FABRIC_MODE=1
FABRIC_MODE_RESTART=0
CFG
            echo "Wrote minimal fabricmanager.cfg"
        fi

        # The unit must not use -D (unsupported in some builds).
        # Overwrite unconditionally — a stale unit from a prior install is
        # the most common cause of "exited during fabric training".
        cat <<'UNIT' | sudo tee /etc/systemd/system/nvidia-fabricmanager.service > /dev/null
[Unit]
Description=NVIDIA Fabric Manager
After=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nv-fabricmanager
LimitCORE=infinity

[Install]
WantedBy=multi-user.target
UNIT
        sudo systemctl daemon-reload
    fi

    # Versionlock only applies when fabricmanager came from RPM
    if rpm -q nvidia-fabricmanager &>/dev/null; then
        if ! rpm -q python3-dnf-plugin-versionlock &>/dev/null; then
            sudo dnf install -y python3-dnf-plugin-versionlock
        fi
        sudo dnf versionlock delete 'nvidia-fabricmanager*' 2>/dev/null || true
        sudo dnf versionlock delete '*nvidia-driver*' 2>/dev/null || true
        sudo dnf versionlock add nvidia-fabricmanager
        local pkg
        for pkg in $(rpm -qa 'nvidia-driver*' --qf '%{NAME}\n' | sort -u); do
            sudo dnf versionlock add "$pkg" 2>/dev/null || true
        done
    fi

    # Kill orphan nv-fabricmanager processes not managed by systemd (e.g.
    # leftover from a manual run) — they hold a PID lock that blocks restart.
    local orphan_pids
    orphan_pids=$(pgrep -x nv-fabricmanager || true)
    if [ -n "$orphan_pids" ]; then
        local svc_pid=""
        svc_pid=$(systemctl show -p MainPID --value nvidia-fabricmanager 2>/dev/null) || true
        local pid
        for pid in $orphan_pids; do
            if [ "$pid" != "$svc_pid" ]; then
                echo "Killing orphan nv-fabricmanager PID ${pid}"
                sudo kill "$pid" 2>/dev/null || true
            fi
        done
        sleep 1
    fi

    sudo systemctl enable nvidia-fabricmanager
    sudo systemctl restart nvidia-fabricmanager

    local timeout="${AUTOVLLM_FM_TIMEOUT:-120}" elapsed=0
    echo "Waiting for NVSwitch fabric training (timeout: ${timeout}s)..."
    while [ "$elapsed" -lt "$timeout" ]; do
        # For oneshot units: "active" means the process exited 0 (training done).
        # For long-running builds: check nvidia-smi fabric state.
        local svc_state fabric_state
        svc_state=$(systemctl show -p ActiveState --value nvidia-fabricmanager 2>/dev/null) || true
        if [ "$svc_state" = "failed" ]; then
            echo "FATAL: nvidia-fabricmanager failed during fabric training" >&2
            journalctl -u nvidia-fabricmanager --no-pager -n 20 >&2 2>/dev/null || true
            return 1
        fi

        fabric_state=$(nvidia-smi -q 2>/dev/null \
            | grep -A2 'Fabric' | grep 'State' | head -1 \
            | awk -F: '{print $2}' | xargs) || true

        if [ "$fabric_state" = "Completed" ]; then
            echo "NVSwitch fabric training completed (nvidia-smi)"
            return 0
        fi

        # Oneshot service that exited 0: training succeeded even if
        # nvidia-smi reports N/A (580.x driver doesn't populate the field).
        if [ "$svc_state" = "active" ] \
            && [ "$(systemctl show -p Type --value nvidia-fabricmanager 2>/dev/null)" = "oneshot" ]; then
            echo "NVSwitch fabric training completed (service exited successfully)"
            return 0
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "FATAL: NVSwitch fabric training did not complete within ${timeout}s" >&2
    echo "Service state: ${svc_state:-unknown}, fabric state: '${fabric_state:-unknown}'" >&2
    journalctl -u nvidia-fabricmanager --no-pager -n 20 >&2 2>/dev/null || true
    return 1
}

mount_nfs_cache() {
    # ponytail: hard mount blocks processes in D-state when the NFS server is
    # unreachable — df/ls on the path will hang. That is correct for bulk model
    # I/O (retries instead of EIO/corruption), but changes the failure mode
    # from "job dies with IOError" to "job stalls silently". Any external
    # monitoring or startup timeout around vLLM must account for this.
    local nfs_opts="vers=3,hard,proto=tcp,timeo=600,retrans=3"

    if mountpoint -q "${NFS_MOUNT_POINT}"; then
        local current_opts
        current_opts=$(awk -v mp="${NFS_MOUNT_POINT}" '$2 == mp {print $4}' /proc/mounts)
        if [[ "$current_opts" == *",hard,"* || "$current_opts" == "hard,"* ]] \
            && [[ "$current_opts" == *"timeo=600"* ]]; then
            echo "NFS already mounted at ${NFS_MOUNT_POINT} with correct options"
            ensure_nfs_persistence "$nfs_opts"
            return 0
        fi
        echo "NFS at ${NFS_MOUNT_POINT} has stale options: ${current_opts}"
        echo "hard/soft cannot be changed via remount; performing umount/mount cycle"
        if fuser -m "${NFS_MOUNT_POINT}" &>/dev/null; then
            echo "FATAL: ${NFS_MOUNT_POINT} is busy; processes holding it open:" >&2
            fuser -vm "${NFS_MOUNT_POINT}" >&2 || true
            echo "Stop the above processes, then re-run setup" >&2
            return 1
        fi
        sudo umount "${NFS_MOUNT_POINT}"
    fi

    sudo mkdir -p "${NFS_MOUNT_POINT}"
    sudo timeout --kill-after=5 60 \
        mount -t nfs -o "$nfs_opts" "${NFS_EXPORT}" "${NFS_MOUNT_POINT}"

    verify_nfs_mount_opts
    ensure_nfs_persistence "$nfs_opts"
}

verify_nfs_mount_opts() {
    local actual_opts
    actual_opts=$(awk -v mp="${NFS_MOUNT_POINT}" '$2 == mp {print $4}' /proc/mounts)
    if [ -z "$actual_opts" ]; then
        echo "FATAL: ${NFS_MOUNT_POINT} not in /proc/mounts after mount returned success" >&2
        return 1
    fi
    local opt
    for opt in hard timeo=600 retrans=3; do
        if [[ "$actual_opts" != *"$opt"* ]]; then
            echo "FATAL: /proc/mounts shows '${actual_opts}' — missing '${opt}'" >&2
            return 1
        fi
    done
    echo "NFS mount verified via /proc/mounts: ${actual_opts}"
}

ensure_nfs_persistence() {
    local nfs_opts="$1"

    local autofs_map=""
    autofs_map=$(grep -rl "${NFS_EXPORT}" /etc/auto.* 2>/dev/null | head -1) || true

    if [ -n "$autofs_map" ]; then
        echo "Mount managed by autofs (${autofs_map}); updating options in map"
        sudo sed -i "s|-fstype=nfs,[^[:space:]]*|-fstype=nfs,${nfs_opts}|" "$autofs_map"
        sudo systemctl reload autofs 2>/dev/null || true
        return 0
    fi

    local fstab_opts="${nfs_opts},_netdev,nofail"
    if grep -q "${NFS_MOUNT_POINT}" /etc/fstab; then
        sudo sed -i "\|${NFS_MOUNT_POINT}|d" /etc/fstab
    fi
    echo "${NFS_EXPORT} ${NFS_MOUNT_POINT} nfs ${fstab_opts} 0 0" \
        | sudo tee -a /etc/fstab > /dev/null
    echo "fstab entry for ${NFS_MOUNT_POINT} (nofail — storage outage will not block boot)"
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
