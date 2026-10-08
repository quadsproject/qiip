#!/bin/bash
# Shared setup functions sourced by engine-specific setup.sh scripts.
# Env vars use the AUTOVLLM_ prefix for backward compatibility.

# Runtime profile catalog (measurement-driven engine selection)
_qiip_profiles="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/profiles.sh"
[ -f "$_qiip_profiles" ] || _qiip_profiles=/usr/local/bin/qiip-profiles.sh
# shellcheck disable=SC1091,SC1090
source "$_qiip_profiles"
unset _qiip_profiles

qiip_generation_tool() {
    case "$1" in
        seal-runtime|setup-bundle|activate|rollback)
            sudo python3 "${SCRIPT_DIR}/../common/generations.py" "$@" ;;
        *) python3 "${SCRIPT_DIR}/../common/generations.py" "$@" ;;
    esac
}

qiip_require_inactive_runtime() {
    local runtime="$1" root selected resolved
    shift
    if [ -L "$runtime" ]; then
        echo "FATAL: versioned runtime path must not be a symlink: ${runtime}" >&2
        return 1
    fi
    [ -e "$runtime" ] || return 0
    resolved=$(readlink -f "$runtime")
    for root in "$@"; do
        for selected in "$root/current" "$root/previous" \
            "$root/current/runtime" "$root/previous/runtime" \
            "$root"/generations/*/runtime; do
            if [ -e "$selected" ] && [ "$(readlink -f "$selected")" = "$resolved" ]; then
                echo "FATAL: refusing to modify a published runtime: ${runtime}" >&2
                return 1
            fi
        done
    done
}

begin_engine_generation() {
    QIIP_RUNTIME_SELECTION=$(mktemp "${INSTALL_TMP_DIR%/}/qiip-runtime.XXXXXX")
    # Steps execute in subshells; this per-attempt file carries the verified
    # installation path back to main without touching a shared active path.
    trap 'rm -f "$QIIP_RUNTIME_SELECTION"' EXIT
    qiip_generation_tool selected "$QIIP_GENERATION_ROOT"
}

activate_engine_generation() {
    local engine="$1" runtime bundle config
    runtime=$(<"$QIIP_RUNTIME_SELECTION")
    [ -n "$runtime" ] || { echo "FATAL: no verified engine runtime" >&2; return 1; }
    bundle=$(qiip_generation_tool setup-bundle "$QIIP_GENERATION_ROOT" "$SCRIPT_DIR")
    config=$(python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
        QIIP_ENGINE "$engine" \
        AUTOVLLM_NFS_EXPORT "$NFS_EXPORT" \
        AUTOVLLM_NFS_MOUNT_POINT "$NFS_MOUNT_POINT" \
        AUTOVLLM_API_PORT "$API_PORT" \
        AUTOVLLM_MIN_FREE_GB "${AUTOVLLM_MIN_FREE_GB:-20}" \
        AUTOLLAMACPP_NFS_MOUNT_POINT "$NFS_MOUNT_POINT" \
        AUTOLLAMACPP_PORT "$API_PORT")
    qiip_generation_tool activate "$QIIP_GENERATION_ROOT" "$bundle" \
        "$(basename "$SCRIPT_DIR")" "$runtime" "$config"
}

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

# Install only absent packages: an ordinary retry must not upgrade the OS.
install_missing_packages() {
    local package
    local -a missing_packages=()
    for package in "$@"; do
        rpm -q "$package" &>/dev/null || missing_packages+=("$package")
    done
    if [ "${#missing_packages[@]}" -gt 0 ]; then
        sudo dnf -y install "${missing_packages[@]}"
    fi
}

install_runtime_prerequisites() {
    local engine="$1"
    # gcc-c++ is nvcc's host compiler for the CUDA proof in both profiles.
    install_missing_packages wget nfs-utils pciutils gcc-c++
    if [ "$engine" = "vllm" ]; then
        # Triton compiles its Python CUDA helper when vLLM first loads a model.
        install_missing_packages python3.12 python3.12-devel
    else
        install_missing_packages cmake make gcc
    fi
}

install_kernel_build_dependencies() {
    local running_kernel
    running_kernel=$(uname -r)
    if ! install_missing_packages \
        "kernel-devel-${running_kernel}" "kernel-headers-${running_kernel}" \
        gcc make elfutils-libelf-devel; then
        resume_required maintenance_required \
            "Matching kernel build dependencies for ${running_kernel} are unavailable; enable their repository or boot a kernel with matching headers, then retry setup"
        return 21
    fi
}

driver_version_compatible() {
    local version="$1" minimum="${2:-$PROFILE_DRIVER_MIN}" maximum="${3-$PROFILE_DRIVER_MAX_BRANCH}"
    [[ "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || return 1
    [ "$(printf '%s\n' "$minimum" "$version" | sort -V | head -1)" = "$minimum" ] || return 1
    [ -z "$maximum" ] || [ "${version%%.*}" -le "$maximum" ]
}

installed_driver_compatible() {
    local versions version minimum="${1:-$PROFILE_DRIVER_MIN}" maximum="${2-$PROFILE_DRIVER_MAX_BRANCH}"
    versions=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader) || return 2
    [ -n "$versions" ] || return 2
    while IFS= read -r version; do
        version=$(xargs <<< "$version")
        if [[ ! "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
            echo "FATAL: cannot establish driver compatibility from '${version}'" >&2
            return 2
        fi
        if ! driver_version_compatible "$version" "$minimum" "$maximum"; then
            echo "Driver ${version} cannot satisfy minimum ${minimum} / maximum branch ${maximum:-none}" >&2
            return 1
        fi
    done <<< "$versions"
    echo "Driver compatibility: installed=${versions//$'\n'/,} profile=${PROFILE_NAME:-bootstrap} minimum=${minimum} max_branch=${maximum:-none}; CUDA execution still required"
}

# These markers are retained in the gateway's failed task as resume_state.
# Retrying setup probes everything again; it never automatically reboots.
resume_required() {
    echo "[RESUME:$1:$2]"
}

DRIVER_STATE_DIR="${AUTOVLLM_DRIVER_STATE_DIR:-/var/lib/qiip/setup}"
BOOT_ID_FILE="${BOOT_ID_FILE:-/proc/sys/kernel/random/boot_id}"
GPU_DEVICE_ROOT="${GPU_DEVICE_ROOT:-/dev}"
GPU_MODULE_ROOT="${GPU_MODULE_ROOT:-/sys/module}"
GPU_DRM_SYS_ROOT="${GPU_DRM_SYS_ROOT:-/sys/class/drm}"

require_driver_reboot() {
    local reason="$1"
    sudo mkdir -p "$DRIVER_STATE_DIR"
    sudo cp "$BOOT_ID_FILE" "${DRIVER_STATE_DIR}/driver-reboot"
    resume_required reboot_required "${reason}; reboot the node, then retry setup"
}

check_driver_resume() {
    if [ -f "${DRIVER_STATE_DIR}/driver-reboot" ]; then
        if [ "$(cat "${DRIVER_STATE_DIR}/driver-reboot")" = "$(cat "$BOOT_ID_FILE")" ]; then
            resume_required reboot_required "Driver maintenance is waiting for a new boot; reboot the node, then retry setup"
            return 20
        fi
        echo "New boot detected after driver maintenance; revalidating the driver and CUDA runtime"
    fi
}

check_gpu_idle() {
    local clients status
    if nvidia-smi &>/dev/null; then
        if ! clients=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
            resume_required maintenance_required "Cannot prove GPU compute idleness; inspect GPU users, then retry setup"
            return 21
        fi
        if [ -n "${clients//[[:space:]]/}" ]; then
            resume_required maintenance_required "GPU compute processes are active; drain and stop GPU users, then retry setup"
            return 21
        fi
    fi
    # fuser also catches graphics clients, persistence and Fabric Manager,
    # including when NVML is broken. Never kill those users automatically.
    local -a devices=()
    local device vendor_file vendor
    for device in "$GPU_DEVICE_ROOT"/nvidia* "$GPU_DEVICE_ROOT"/nvidia-caps/*; do
        [ -e "$device" ] && [ ! -d "$device" ] && devices+=("$device")
    done
    for device in "$GPU_DEVICE_ROOT"/dri/*; do
        if [ ! -e "$device" ] || [ -d "$device" ]; then continue; fi
        vendor_file="${GPU_DRM_SYS_ROOT}/${device##*/}/device/vendor"
        [ -r "$vendor_file" ] || continue
        vendor=$(<"$vendor_file")
        # BMC consoles and other vendors' DRM clients do not use this driver.
        [ "${vendor,,}" = "0x10de" ] && devices+=("$device")
    done
    if [ "${#devices[@]}" -gt 0 ]; then
        if sudo fuser "${devices[@]}"; then status=0; else status=$?; fi
        if [ "$status" -ne 1 ]; then
            resume_required maintenance_required "GPU device users are active or cannot be inspected; stop GPU services and clients, then retry setup"
            return 21
        fi
    fi
}

nvidia_driver_rpms() {
    # Query each family separately and combine the matches into one union of
    # exact installed names/versions, including overlapping package families.
    local pattern
    for pattern in '*nvidia*driver*' '*kmod-nvidia*' '*xorg-x11-drv-nvidia*'; do
        rpm -qa --qf "$1" "$pattern" || return $?
    done | sort -u
}

installed_nvidia_build_version() {
    local version
    version=$(nvidia_driver_rpms '%{VERSION}\n' | sort -u) || version=""
    if [[ "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        printf '%s\n' "$version"
        return 0
    fi
    # Runfile --dkms installs retain the version even when the new kernel has
    # no module yet. Do not infer compatibility from a library's mere presence.
    version=$(dkms status -m nvidia 2>/dev/null \
        | sed -nE 's@^nvidia/([0-9]+\.[0-9]+(\.[0-9]+)?)(,|:).*@\1@p' | sort -u) || version=""
    [[ "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || return 1
    printf '%s\n' "$version"
}

# Called only after missing/incompatible driver evidence, never just because
# the installed version differs from the configured replacement artifact.
install_nvidia_driver() (
    set -e
    local recovery_minimum="${RECOVERY_DRIVER_MIN:-$PROFILE_DRIVER_MIN}"
    local recovery_maximum="${RECOVERY_DRIVER_MAX_BRANCH-$PROFILE_DRIVER_MAX_BRANCH}"
    # shellcheck disable=SC2153  # DRIVER_VERSION is set by the engine script
    require_sha256 "NVIDIA driver ${DRIVER_VERSION}" "$DRIVER_SHA256" \
        "AUTOVLLM_NVIDIA_DRIVER_SHA256"
    if ! driver_version_compatible "$DRIVER_VERSION" "$recovery_minimum" "$recovery_maximum"; then
        echo "FATAL: replacement driver ${DRIVER_VERSION} cannot satisfy profile minimum ${recovery_minimum} / maximum branch ${recovery_maximum:-none}" >&2
        return 2
    fi
    local work_dir installer installed_version
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR%/}/auto-setup-driver.XXXXXX")
    trap 'rm -rf "$work_dir"' EXIT
    installer="${work_dir}/NVIDIA-driver.run"
    install_missing_packages wget
    wget -q "$NVIDIA_DRIVER_URL" -O "$installer"
    verify_sha256 "$installer" "$DRIVER_SHA256" "NVIDIA driver ${DRIVER_VERSION}"
    # Verify the fallback artifact before any rebuild, removal or replacement.
    install_missing_packages psmisc
    check_gpu_idle
    install_kernel_build_dependencies
    # Recheck immediately before driver mutation after dependency installation.
    check_gpu_idle

    installed_version=$(installed_nvidia_build_version) || installed_version=""
    if ! nvidia-smi &>/dev/null && driver_version_compatible "$installed_version" "$recovery_minimum" "$recovery_maximum"; then
        echo "Compatible installed driver ${installed_version} has no working module; rebuilding for $(uname -r)"
        if (sudo dkms autoinstall || sudo akmods --force) \
            && sudo modprobe nvidia && nvidia-smi &>/dev/null; then
            if installed_driver_compatible "$recovery_minimum" "$recovery_maximum"; then
                echo "Existing NVIDIA module rebuilt; CUDA execution still required"
                return 0
            fi
        fi
        echo "Existing module rebuild failed; using verified replacement"
    fi
    if ! install_missing_packages dkms; then
        resume_required maintenance_required "DKMS is required for the verified runfile installer; enable its repository, then retry setup"
        return 21
    fi
    check_gpu_idle

    # Unload leaf modules first. Do not uninstall while the loaded driver is
    # still held by a client or the kernel; expose the maintenance boundary.
    local module holder holders
    for module in nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
        if [ -d "${GPU_MODULE_ROOT}/${module}" ]; then
            if ! sudo modprobe -r "$module"; then
                holders=""
                for holder in "${GPU_MODULE_ROOT}/${module}/holders/"*; do
                    [ -e "$holder" ] && holders+="${holder##*/} "
                done
                resume_required maintenance_required \
                    "Cannot unload ${module}; module holders: ${holders:-none detected}; stop clients or unload dependent modules, then retry setup"
                return 21
            fi
        fi
    done
    if [ -x /usr/bin/nvidia-uninstall ]; then
        sudo /usr/bin/nvidia-uninstall --silent
    else
        local packages
        local -a driver_packages=()
        packages=$(nvidia_driver_rpms '%{NAME}\n')
        if [ -n "$packages" ]; then
            mapfile -t driver_packages <<< "$packages"
            sudo dnf -y remove "${driver_packages[@]}"
        fi
    fi
    echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
    sudo dracut --force
    if [ -d "${GPU_MODULE_ROOT}/nouveau" ] && ! sudo modprobe -r nouveau; then
        require_driver_reboot "Nouveau remains loaded after blacklisting"
        return 20
    fi
    if ! sudo sh "$installer" --dkms --no-x-check --no-nouveau-check --ui=none --no-questions; then
        echo "FATAL: NVIDIA driver installation failed; inspect installer logs before retrying setup" >&2
        return 1
    fi
    if ! sudo modprobe nvidia; then
        resume_required maintenance_required "Replacement module could not load; inspect kernel/module signing and build diagnostics, then retry setup"
        return 21
    fi
    if ! nvidia-smi &>/dev/null \
        || [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | xargs)" != "$DRIVER_VERSION" ]; then
        require_driver_reboot "Replacement driver is installed but not active"
        return 20
    fi
)

# Select from the existing stack first. A broken/missing module may prevent
# hardware measurements: restore it with the pinned bootstrap driver, then
# select the measured profile before toolkit or engine installation.
prepare_runtime() {
    local engine="$1" deferred_cuda="${2:-0}" repaired=0 compatibility_status
    # Before NVML measurements, preserve any stack that could support this
    # engine. llama.cpp's Volta profile can use a CUDA 12.9 driver. After
    # recovery the measured profile, not this provisional floor, is enforced.
    local RECOVERY_DRIVER_MIN="$PROFILE_DEFAULT_DRIVER_MIN" RECOVERY_DRIVER_MAX_BRANCH=""
    if [ "$engine" = "llamacpp" ]; then
        RECOVERY_DRIVER_MIN="$PROFILE_VOLTA_DRIVER_MIN"
        RECOVERY_DRIVER_MAX_BRANCH="$PROFILE_VOLTA_DRIVER_MAX_BRANCH"
    fi
    check_driver_resume || return $?
    detect_profile_os
    if ! check_os_abi "$engine"; then
        echo "[REJECT:unsupported_hardware:OS/ABI check failed before driver preparation]" >&2
        return 3
    fi
    # Deferred llama.cpp checks exact artifact sizes or its build/install
    # budgets, including alternate scratch filesystems, in the installer.
    if [ "$engine" != "llamacpp" ] || [ "$deferred_cuda" != "1" ]; then
        step check_install_capacity check_install_capacity_or_warn
    fi
    # Required both for missing-driver PCI evidence and profile measurement.
    step system_prerequisites install_missing_packages pciutils
    if ! nvidia-smi &>/dev/null; then
        if modinfo nvidia &>/dev/null; then
            sudo modprobe nvidia || true
        fi
        if ! nvidia-smi &>/dev/null; then
            echo "Driver unavailable; GPU profile measurements require module repair first"
            local pci_inventory
            pci_inventory=$(lspci -Dn) || return 1
            if ! grep -Eq ' 03[0-9a-f]{2}: 10de:' <<< "$pci_inventory"; then
                echo "[REJECT:unsupported_hardware:no NVIDIA GPU PCI device found for driver repair]" >&2
                return 3
            fi
            step nvidia_driver install_nvidia_driver
            repaired=1
        fi
    fi
    select_runtime_profile "$engine" 1 || return $?
    RECOVERY_DRIVER_MIN="$PROFILE_DRIVER_MIN"
    RECOVERY_DRIVER_MAX_BRANCH="$PROFILE_DRIVER_MAX_BRANCH"
    if [ "$engine" = "llamacpp" ] && [ "$deferred_cuda" = "1" ]; then
        step system_prerequisites install_missing_packages wget nfs-utils python3
    else
        step system_prerequisites install_runtime_prerequisites "$engine"
    fi
    if installed_driver_compatible; then compatibility_status=0; else compatibility_status=$?; fi
    if [ "$compatibility_status" -ne 0 ]; then
        if [ "$compatibility_status" -eq 2 ]; then
            resume_required maintenance_required "Cannot establish the installed driver version; inspect NVML/module health, then retry setup"
            return 21
        fi
        echo "Installed driver cannot satisfy ${PROFILE_NAME}; preparing verified replacement"
        step nvidia_driver install_nvidia_driver
        repaired=1
        select_runtime_profile "$engine" 1 || return $?
        step nvidia_driver installed_driver_compatible
    fi
    # llama.cpp can prove the prepared driver with a verified prebuilt probe.
    # Its installer must complete that proof before selecting any runtime.
    if [ "$engine" = "llamacpp" ] && [ "$deferred_cuda" = "1" ]; then
        step fabric_manager ensure_fabric_manager
        # Read by the engine installer after its prebuilt or source CUDA proof.
        # shellcheck disable=SC2034
        QIIP_DRIVER_REPAIRED="$repaired"
        return 0
    fi
    step cuda_toolkit install_cuda_toolkit
    step fabric_manager ensure_fabric_manager
    echo "[STEP:cuda_proof:START]"
    run_with_errexit verify_cuda_execution
    if [ "$STEP_STATUS" -ne 0 ]; then
        echo "[STEP:cuda_proof:FAIL]"
        if [ "$STEP_STATUS" -eq 10 ]; then
            if [ "$repaired" -eq 1 ]; then
                require_driver_reboot "CUDA execution failed after driver repair"
                return 20
            fi
            resume_required maintenance_required "CUDA execution failed despite a compatible driver version; inspect device/fabric/runtime health, then retry setup"
            return 21
        fi
        return "$STEP_STATUS"
    fi
    echo "[STEP:cuda_proof:OK]"
    if [ -f "${DRIVER_STATE_DIR}/driver-reboot" ]; then
        sudo rm -f "${DRIVER_STATE_DIR}/driver-reboot"
    fi
    echo "GPU stack satisfies ${PROFILE_NAME}; continuing without further driver changes"
}

nvcc_toolkit_version() {
    "$1" --version 2>/dev/null | grep -oP 'V\K[0-9]+\.[0-9]+' | head -1
}

# Locates nvcc for the profile toolkit. The NVIDIA RHEL9 repo installs under
# /usr/local/cuda-<version>/bin (no /usr/local/cuda symlink). Prefer a matching
# version over an old default symlink or PATH compiler; retain a mismatched
# candidate only to diagnose it in the install/proof gates.
find_nvcc() {
    local required="${1:-${PROFILE_CUDA_TOOLKIT_VERSION:-}}"
    local candidate installed fallback=""
    for candidate in \
        "${CUDA_NVCC:-}" \
        "/usr/local/cuda/bin/nvcc" \
        "/usr/local/cuda-${required}/bin/nvcc" \
        "$(command -v nvcc 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            installed=$(nvcc_toolkit_version "$candidate") || installed=""
            if [ -z "$required" ] || [ "$installed" = "$required" ]; then
                echo "$candidate"
                return 0
            fi
            [ -n "$fallback" ] || fallback="$candidate"
        fi
    done
    if [ -n "$fallback" ]; then
        echo "$fallback"
        return 0
    fi
    return 1
}

install_cuda_toolkit() {
    local required="${PROFILE_CUDA_TOOLKIT_VERSION}"
    local nvcc
    nvcc="$(find_nvcc "$required")" || nvcc=""
    if [ -n "$nvcc" ] && [ -x "$nvcc" ]; then
        local installed
        installed=$(nvcc_toolkit_version "$nvcc") || installed=""
        if [ "$installed" = "$required" ]; then
            echo "CUDA toolkit ${required} already installed, skipping"
            return 0
        fi
        echo "CUDA toolkit ${installed:-unknown} installed; installing exact ${required}"
    fi
    install_missing_packages dnf-plugins-core
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    local pkg="cuda-toolkit-${required//./-}"
    if ! sudo dnf -y install "$pkg"; then
        echo "FATAL: could not install ${pkg}; check the NVIDIA CUDA repository" >&2
        return 1
    fi
    # The RHEL9 dnf packages install nvcc under /usr/local/cuda-<version>/bin
    # without a /usr/local/cuda symlink, so keep the conventional path valid
    # for engine scripts and parity with runfile installs. verify_cuda_execution
    # (step cuda_proof) is the hard gate for actual toolkit usability.
    sudo ln -sfn "cuda-${required}" /usr/local/cuda
    echo "CUDA toolkit ${required} installed (${pkg})"
}

# Proves real CUDA execution (driver + toolkit + device) with a tiny
# headless kernel. Keeps compiler diagnostics on failure; no X/GL needed.
verify_cuda_execution() {
    local nvcc
    nvcc="$(find_nvcc)" || nvcc=""
    if [ -z "$nvcc" ]; then
        echo "FATAL: nvcc not found; install the profile CUDA toolkit first" >&2
        return 1
    fi
    local toolkit_version
    toolkit_version=$(nvcc_toolkit_version "$nvcc") || toolkit_version=""
    if [ "$toolkit_version" != "$PROFILE_CUDA_TOOLKIT_VERSION" ]; then
        echo "FATAL: CUDA proof requires toolkit ${PROFILE_CUDA_TOOLKIT_VERSION}; ${nvcc} reports ${toolkit_version:-unknown}" >&2
        return 1
    fi
    local work_dir
    work_dir=$(mktemp -d "${INSTALL_TMP_DIR:-/tmp}/cuda-probe.XXXXXX")
    cat > "${work_dir}/cuda_probe.cu" <<'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void k(int *x) { *x = 42; }
int main() {
    int count = 0, driver = 0, runtime = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("cudaGetDeviceCount failed or no CUDA devices\n"); return 1;
    }
    if (cudaDriverGetVersion(&driver) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess) return 1;
    printf("CUDA driver API=%d runtime=%d devices=%d\n", driver, runtime, count);
    for (int i = 0; i < count; ++i) {
        int h = 0, *d = NULL;
        if (cudaSetDevice(i) != cudaSuccess || cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
            printf("cudaSetDevice/cudaMalloc failed on device %d\n", i); return 1;
        }
        k<<<1, 1>>>(d);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || h != 42) {
            printf("CUDA kernel/copy failed on device %d\n", i); return 1;
        }
        if (cudaFree(d) != cudaSuccess) return 1;
        printf("CUDA execution verified on device %d\n", i);
    }
    return 0;
}
EOF
    # Build for the selected profile, including SM70 on CUDA 12.9. Retain PTX
    # for newer selected cards, without compiling for unrelated physical GPUs.
    local capability="${GPU_COMPUTE_CAP:-}" devices query_index sm architecture targets="${2:-}"
    local -a arch_flags=()
    devices=$(profile_gpu_devices)
    if [ -z "$targets" ] && [ -z "$capability" ]; then
        query_index="${devices%%,*}"
        capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${query_index:-0}" | xargs) || capability=""
    fi
    if [ -z "$targets" ] && ! [[ "$capability" =~ ^[0-9]+\.[0-9]+$ ]]; then
        rm -rf "$work_dir"
        echo "FATAL: cannot establish the selected profile's CUDA architecture" >&2
        return 1
    fi
    targets="${targets:-${capability//./}}"
    if ! [[ "$targets" =~ ^[1-9][0-9]*(-real|-virtual)?(\;[1-9][0-9]*(-real|-virtual)?)*$ ]]; then
        rm -rf "$work_dir"
        echo "FATAL: CUDA proof requires explicit numeric compile targets" >&2
        return 1
    fi
    local -a target_list=()
    IFS=';' read -r -a target_list <<< "$targets"
    for architecture in "${target_list[@]}"; do
        sm="${architecture%%-*}"
        case "$architecture" in
            *-real) arch_flags+=(-gencode "arch=compute_${sm},code=sm_${sm}") ;;
            *-virtual) arch_flags+=(-gencode "arch=compute_${sm},code=compute_${sm}") ;;
            *) arch_flags+=(-gencode "arch=compute_${sm},code=[sm_${sm},compute_${sm}]") ;;
        esac
    done
    if ! "$nvcc" -o "${work_dir}/cuda_probe" "${work_dir}/cuda_probe.cu" "${arch_flags[@]}"; then
        rm -rf "$work_dir"
        echo "FATAL: nvcc failed to compile the CUDA execution probe" >&2
        return 1
    fi
    local -a probe_env=()
    [ -z "$devices" ] || probe_env+=("CUDA_VISIBLE_DEVICES=$devices")
    if ! env "${probe_env[@]}" "${work_dir}/cuda_probe"; then
        rm -rf "$work_dir"
        echo "FATAL: CUDA execution probe failed on the device; driver/toolkit/device unusable" >&2
        return 10
    fi
    # Artifact producers retain the exact kernel that passed on this node.
    if [ -n "${1:-}" ]; then
        if ! cp "${work_dir}/cuda_probe" "$1"; then
            rm -rf "$work_dir"
            return 1
        fi
    fi
    rm -rf "$work_dir"
    echo "CUDA execution verified: probe kernel compiled and ran"
    return 0
}

# Static Fabric Manager checks: installed, version matches driver, service
# active. Prints a FATAL reason on failure. rc-based, no _mark/_bail.
fabric_static_ok() {
    local driver_version fm_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
        | sed '/^[[:space:]]*$/d' | head -1 | xargs) || {
        echo "FATAL: cannot query NVIDIA driver version (nvidia-smi failed)" >&2
        return 1
    }
    # Prefer the binary version: redist installs bypass RPM.
    fm_version=""
    if [ -x /usr/bin/nv-fabricmanager ]; then
        fm_version=$(nv-fabricmanager --version 2>/dev/null \
            | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true
    fi
    if [ -z "$fm_version" ] && rpm -q nvidia-fabricmanager &>/dev/null; then
        fm_version=$(rpm -q --qf '%{VERSION}' nvidia-fabricmanager)
    fi
    if [ -z "$fm_version" ]; then
        echo "FATAL: NVSwitch present but nvidia-fabricmanager not installed (run setup.sh)" >&2
        return 1
    fi
    if [ "$fm_version" != "$driver_version" ]; then
        echo "FATAL: Fabric Manager ${fm_version} != driver ${driver_version} — version mismatch causes CUDA error 802" >&2
        return 1
    fi
    if ! systemctl is-active --quiet nvidia-fabricmanager; then
        echo "FATAL: nvidia-fabricmanager.service not active — systemctl start nvidia-fabricmanager" >&2
        return 1
    fi
    return 0
}

# Returns 0 once fabric training is complete: nvidia-smi State Completed, or
# the oneshot 580.x service exited successfully. rc-based.
fabric_trained() {
    local fabric_state
    fabric_state=$(nvidia-smi -q 2>/dev/null \
        | grep -A2 'Fabric' | grep 'State' | head -1 \
        | awk -F: '{print $2}' | xargs) || true
    if [ "$fabric_state" = "Completed" ]; then
        return 0
    fi
    if [ "$(systemctl show -p Type --value nvidia-fabricmanager 2>/dev/null)" = "oneshot" ]; then
        return 0
    fi
    return 1
}

# Waits for NVSwitch fabric training to complete. No-op without NVSwitches;
# fails fast on static Fabric Manager problems instead of waiting blind.
wait_nvswitch_fabric() {
    local timeout="${AUTOVLLM_FM_TIMEOUT:-120}"
    local nvswitch_count elapsed
    nvswitch_count=$(lspci 2>/dev/null | grep -ci nvswitch || true)
    if [ "$nvswitch_count" -eq 0 ]; then
        return 0
    fi
    if ! fabric_static_ok; then
        return 1
    fi
    elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        if fabric_trained; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "FATAL: NVSwitch fabric training did not complete within ${timeout}s; check /var/log/fabricmanager.log" >&2
    return 1
}

# Returns 0 when Fabric Manager is correctly installed and trained (no-op
# without NVSwitches). rc-based: no preflight _mark/_bail dependency.
fabric_ready() {
    local nvswitch_count
    nvswitch_count=$(lspci 2>/dev/null | grep -ci nvswitch || true)
    if [ "$nvswitch_count" -eq 0 ]; then
        return 0
    fi
    if ! fabric_static_ok; then
        return 1
    fi
    if ! fabric_trained; then
        echo "FATAL: Fabric State not Completed — check /var/log/fabricmanager.log" >&2
        return 1
    fi
    return 0
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
    if fabric_ready >/dev/null 2>&1; then
        echo "Matching Fabric Manager is active and trained, reusing without restart"
        return 0
    fi

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
    # from "job dies with IOError" to "job stalls silently". The control-plane
    # probe below fails fast when the server is down before the mount and every
    # data-plane probe is bounded by AUTOVLLM_PROBE_TIMEOUT with --kill-after,
    # but a D-state process cannot be killed at all: a server that drops
    # mid-I/O can still hang one probe until the kernel retransmission timeout.
    # External startup timeouts around vLLM must account for that. NFSv3 is a
    # fleet constraint; soft/intr escapes stay rejected (soft turns a blip into
    # EIO, intr is a no-op on modern kernels).
    # The exact option set (hard,timeo=600,retrans=3 on NFSv3) is deliberate:
    # it is the fleet's ~120s hard-mount window, so verification enforces these
    # options exactly and rejects soft/noac that would change that behavior.
    local nfs_opts="vers=3,hard,proto=tcp,timeo=600,retrans=3"

    if mountpoint -q "${NFS_MOUNT_POINT}"; then
        local verify_rc=0
        verify_nfs_storage || verify_rc=$?
        if [ "$verify_rc" -eq 0 ]; then
            echo "NFS already mounted at ${NFS_MOUNT_POINT} with the expected source and options"
            ensure_nfs_persistence "$nfs_opts"
            check_install_capacity || [ "$?" -eq 2 ] || return 1
            return 0
        fi
        if [ "$verify_rc" -eq 2 ]; then
            echo "FATAL: refusing to remount ${NFS_MOUNT_POINT}; its source is not ${NFS_EXPORT}" >&2
            echo "Unmount the stale mount manually once it is safe, then re-run setup" >&2
            return 1
        fi
        # Probe the server BEFORE any umount/fuser: on a hard NFS mount with a
        # dropped server those can block in D-state with no bound at all.
        nfs_server_reachable
        echo "NFS at ${NFS_MOUNT_POINT} failed verification; performing umount/mount cycle"
        local probe_timeout
        probe_timeout="${AUTOVLLM_PROBE_TIMEOUT:-10}"
        if timeout --kill-after=2 "$probe_timeout" fuser -m "${NFS_MOUNT_POINT}" &>/dev/null; then
            echo "FATAL: ${NFS_MOUNT_POINT} is busy; processes holding it open:" >&2
            timeout --kill-after=2 "$probe_timeout" fuser -vm "${NFS_MOUNT_POINT}" >&2 || true
            echo "Stop the above processes, then re-run setup" >&2
            return 1
        fi
        sudo umount "${NFS_MOUNT_POINT}"
    fi

    nfs_server_reachable

    sudo mkdir -p "${NFS_MOUNT_POINT}"
    sudo timeout --kill-after=5 60 \
        mount -t nfs -o "$nfs_opts" "${NFS_EXPORT}" "${NFS_MOUNT_POINT}"

    verify_nfs_storage
    ensure_nfs_persistence "$nfs_opts"
    check_install_capacity || [ "$?" -eq 2 ] || return 1
}

nfs_server_reachable() {
    # Control-plane probe: connecting needs no data-plane RPC, so a dead server
    # fails here in seconds with an actionable message instead of a D-state
    # hang inside df/stat/ls.
    local probe_timeout server display probe_bash
    probe_timeout="${AUTOVLLM_PROBE_TIMEOUT:-10}"
    probe_bash="${AUTOVLLM_NFS_PROBE_BASH:-bash}"
    display="${NFS_EXPORT%:*}"
    server="${display#\[}"
    server="${server%\]}"
    if timeout --kill-after=2 "$probe_timeout" "$probe_bash" -c "exec 3<>/dev/tcp/${server}/2049" 2>/dev/null; then
        echo "NFS server reachable at ${display}:2049"
        return 0
    fi
    echo "FATAL: NFS server ${display} not reachable on tcp/2049" >&2
    echo "Verify the server, export, and network path, then re-run setup" >&2
    return 1
}

_normalize_nfs_source() {
    # Split on the LAST colon: bracketed IPv6 ([addr]:/path) and the plain
    # host:path form both carry exactly one separator into the path.
    local src="$1" host path
    host="${src%:*}"
    path="${src#"${host}":}"
    host="${host#\[}"
    host="${host%\]}"
    host=$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')
    if [[ "$host" == *:* ]]; then
        host="[$host]"
    fi
    if [[ "$path" != "/" ]]; then
        path="${path%/}"
    fi
    printf '%s:%s' "$host" "$path"
}

_mount_opt() {
    case ",$1," in
        *",$2,"*) return 0 ;;
        *) return 1 ;;
    esac
}

verify_nfs_storage() {
    # Control-plane verification of the NFSv3 mount: source export, filesystem
    # type, and the exact option set QIIP requires. Return codes: 0 verified;
    # 2 wrong source (caller must not touch the mount); 3 wrong fstype/options.
    local mounts_file line real_mp
    mounts_file="${AUTOVLLM_MOUNTS_FILE:-/proc/mounts}"
    # /proc/mounts is ordered by mount ID; a mount stacked over this path
    # appears later, so the LAST match is the visible filesystem.
    real_mp="$(readlink -f -- "${NFS_MOUNT_POINT}" 2>/dev/null)" || real_mp=""
    line=$(awk -v mp="${NFS_MOUNT_POINT}" -v rmp="$real_mp" '
        function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
        d($2) == mp || (rmp != "" && d($2) == rmp) {got = $0}
        END {if (got != "") print got}
    ' "$mounts_file")
    if [ -z "$line" ]; then
        echo "FATAL: ${NFS_MOUNT_POINT} not found in ${mounts_file}; NFS storage is not mounted" >&2
        return 1
    fi
    local actual_source actual_fstype actual_opts
    actual_source=$(printf '%s\n' "$line" | awk '
        function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
        {print d($1)}
    ')
    actual_fstype=$(printf '%s\n' "$line" | awk '{print $3}')
    actual_opts=$(printf '%s\n' "$line" | awk '{print $4}')

    # An autofs-managed mount point carries an `autofs` placeholder in
    # /proc/mounts while the export is idled out (past the autofs timeout).
    # Nothing touches the path on an idle node, so the placeholder is the last
    # match and the source check below would reject a healthy mount. Trigger
    # the automount and re-read before judging the source/type/options.
    if [ "$actual_fstype" = "autofs" ]; then
        echo "${NFS_MOUNT_POINT} is an autofs placeholder; triggering the automount"
        timeout 5 stat "${NFS_MOUNT_POINT}/." >/dev/null 2>&1 || true
        line=$(awk -v mp="${NFS_MOUNT_POINT}" -v rmp="$real_mp" '
            function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
            d($2) == mp || (rmp != "" && d($2) == rmp) {got = $0}
            END {if (got != "") print got}
        ' "$mounts_file")
        if [ -z "$line" ]; then
            echo "FATAL: ${NFS_MOUNT_POINT} not found in ${mounts_file}; NFS storage is not mounted" >&2
            return 1
        fi
        actual_source=$(printf '%s\n' "$line" | awk '
            function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
            {print d($1)}
        ')
        actual_fstype=$(printf '%s\n' "$line" | awk '{print $3}')
        actual_opts=$(printf '%s\n' "$line" | awk '{print $4}')
    fi

    local expected
    expected=$(_normalize_nfs_source "$NFS_EXPORT")
    if [ "$(_normalize_nfs_source "$actual_source")" != "$expected" ]; then
        echo "FATAL: mount source '${actual_source}' is not the expected export '${NFS_EXPORT}'" >&2
        return 2
    fi
    if [ "$actual_fstype" != "nfs" ]; then
        echo "FATAL: ${NFS_MOUNT_POINT} filesystem type is '${actual_fstype}'; QIIP provisions NFSv3 only, NFSv4 mounts are unsupported (the provisioned v3 fstab/autofs entry wins on reboot anyway)" >&2
        return 3
    fi

    local missing=""
    if ! _mount_opt "$actual_opts" "vers=3" && ! _mount_opt "$actual_opts" "nfsvers=3"; then
        missing+="vers=3 "
    fi
    if ! _mount_opt "$actual_opts" "hard"; then
        missing+="hard "
    fi
    if ! _mount_opt "$actual_opts" "proto=tcp"; then
        missing+="proto=tcp "
    fi
    if ! _mount_opt "$actual_opts" "retrans=3"; then
        missing+="retrans=3 "
    fi
    local timeo
    timeo=$(printf '%s\n' "$actual_opts" | tr ',' '\n' | sed -n 's/^timeo=//p')
    if [ "$timeo" != "600" ]; then
        missing+="timeo=${timeo:-unset} (required timeo=600) "
    fi
    local opt sec
    for opt in soft noac; do
        if _mount_opt "$actual_opts" "$opt"; then
            missing+="${opt} (rejected) "
        fi
    done
    sec=$(printf '%s\n' "$actual_opts" | tr ',' '\n' | sed -n 's/^sec=//p')
    if [ -n "$sec" ] && [ "$sec" != "sys" ]; then
        missing+="sec=${sec} (QIIP requires sec=sys) "
    fi

    if [ -n "$missing" ]; then
        echo "FATAL: ${NFS_MOUNT_POINT} mount options are not the required set:" >&2
        echo "  ${mounts_file}: ${actual_opts}" >&2
        echo "  problems: ${missing}" >&2
        return 3
    fi
    echo "NFS mount verified via ${mounts_file}: source=${actual_source} fstype=${actual_fstype}"
}

check_storage_capacity() {
    # Probe each filesystem once (df target dedupe). NFSv3 hard mounts make df
    # a data-plane round trip; the probe is bounded by PROBE_TIMEOUT with
    # --kill-after, but a D-state process cannot be killed and waits for the
    # hard-mount reconnect cycle.
    local min_gb probe_timeout min_bytes need_gb
    min_gb="${AUTOVLLM_MIN_FREE_GB:-20}"
    probe_timeout="${AUTOVLLM_PROBE_TIMEOUT:-10}"
    min_bytes=$((min_gb * 1024 * 1024 * 1024))
    need_gb=$(((min_bytes + 1073741823) / 1073741824))
    local -a seen=()
    local path avail_bytes target
    for path in "$@"; do
        if ! timeout --kill-after=2 "$probe_timeout" test -e "$path"; then
            continue
        fi
        avail_bytes=$(timeout --kill-after=2 "$probe_timeout" df --output=avail -B1 "$path" 2>/dev/null | tail -1 | xargs) || {
            echo "WARNING: cannot determine free space on ${path} (probe failed or timed out)" >&2
            return 2
        }
        target=$(timeout --kill-after=2 "$probe_timeout" df --output=target "$path" 2>/dev/null | tail -1 | xargs) || true
        if printf '%s\n' "${seen[@]}" 2>/dev/null | grep -qxF "$target"; then
            continue
        fi
        seen+=("$target")
        if [ -z "$avail_bytes" ] || [ "$avail_bytes" -lt "$min_bytes" ]; then
            local avail_gb
            avail_gb=$(( (${avail_bytes:-0} + 1073741823) / 1073741824 ))
            echo "FATAL: ${path} (${target}) has ${avail_gb}GB free but ${need_gb}GB is required" >&2
            return 1
        fi
    done
    echo "Storage capacity verified (at least ${need_gb}GB free per filesystem)"
}

check_install_capacity() {
    # The engine setup.sh defines INSTALL_TMP_DIR, VLLM_VENV or
    # LLAMACPP_INSTALL_ROOT, and LLMFIT_BIN before sourcing this file; the
    # defaults mirror both engines.
    local vllm_root llmfit_root
    vllm_root="$(dirname "${VLLM_VENV:-/opt/vllm-venv}")"
    llmfit_root="$(dirname "${LLMFIT_BIN:-/usr/local/bin/llmfit}")"
    check_storage_capacity \
        "$NFS_MOUNT_POINT" \
        "${INSTALL_TMP_DIR:-/tmp}" \
        "$vllm_root" \
        "${LLAMACPP_INSTALL_ROOT:-/opt/llama.cpp}" \
        "$llmfit_root"
}

# Wraps check_install_capacity for a setup step: a proven shortage (rc 1) is
# fatal; a probe that cannot size a filesystem (rc 2) is a warning, matching
# the pre-existing tolerance in mount_nfs_cache.
check_install_capacity_or_warn() {
    check_install_capacity || {
        local rc=$?
        if [ "$rc" -eq 2 ]; then
            echo "WARNING: install capacity could not be fully verified (continuing)" >&2
            return 0
        fi
        return "$rc"
    }
}

ensure_nfs_persistence() {
    local nfs_opts="$1"

    # One persistence mechanism only: when autofs manages this export, its map
    # entry is updated (on-demand behavior preserved) and fstab is left alone;
    # otherwise the marked fstab entry is written.
    local autofs_map
    autofs_map=$(find_autofs_map) || true
    if [ -n "$autofs_map" ]; then
        ensure_autofs_entry "$nfs_opts" "$autofs_map"
        sudo systemctl reload autofs 2>/dev/null || true
        return 0
    fi

    ensure_fstab_entry "$nfs_opts" "${nfs_opts},_netdev,nofail"
}

find_autofs_map() {
    # Locate the map file whose entry names OUR mount point for the exact
    # export; an unrelated map that merely shares the export (a second mount
    # of the same export under another key) must stay untouched.
    local dir f base
    dir="${AUTOVLLM_AUTO_MAP_DIR:-/etc}"
    for f in "$dir"/auto.* "$dir"/auto.master.d/*; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        [ "$base" = "auto.master" ] && continue
        if awk -v mp="${NFS_MOUNT_POINT}" -v expval="${NFS_EXPORT}" '
            function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
            d($1) == mp && d($NF) == expval {found=1} END {exit found ? 0 : 1}' "$f"; then
            printf '%s\n' "$f"
            return 0
        fi
    done
    return 1
}

_escape_fstab_field() {
    # fstab/autofs maps are whitespace-delimited; spaces, tabs, and backslashes
    # must be octal-escaped (\\040/\\011/\\134) exactly as /proc/mounts and the
    # read side decode them.
    local s="$1"
    s=${s//\\/\\\\}
    s=${s// /\\040}
    s=${s//$'\t'/\\011}
    printf '%s' "$s"
}

ensure_fstab_entry() {
    local nfs_opts="$1" fstab_opts="$2"
    local fstab marker managed_entry line line_source
    fstab="${AUTOVLLM_FSTAB_FILE:-/etc/fstab}"
    marker="# qiip-managed"
    managed_entry="$(_escape_fstab_field "$NFS_EXPORT") $(_escape_fstab_field "$NFS_MOUNT_POINT") nfs ${fstab_opts} 0 0 ${marker}"
    line=$(awk -v mp="${NFS_MOUNT_POINT}" '
        function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
        d($2) == mp {print; exit}
    ' "$fstab") || true

    if [ -n "$line" ]; then
        line_source=$(printf '%s\n' "$line" | awk '
            function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
            {print d($1)}
        ')
        if [[ "$line" == *"${marker}"* ]]; then
            :
        elif [ "$line_source" = "$NFS_EXPORT" ]; then
            echo "QIIP fstab entry for ${NFS_MOUNT_POINT} exists without marker; adopting it"
        else
            echo "FATAL: ${fstab} already has an entry for ${NFS_MOUNT_POINT} from source '${line_source}':" >&2
            echo "  ${line}" >&2
            echo "QIIP will not overwrite it; remove or correct the entry first" >&2
            return 1
        fi
        if ! {
            # awk -v decodes backslash escapes before printing, which would
            # turn the \\040/\\011 field escapes back into literal spaces/tabs
            # (their meaning in /proc/mounts and fstab); double the backslashes
            # so -v restores the exact escaped string.
            sudo awk -v entry="${managed_entry//\\/\\\\}" -v mp="${NFS_MOUNT_POINT}" '
                function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
                d($2) == mp {print entry; next} {print}' \
                "$fstab" | sudo tee "${fstab}.qiip.tmp" > /dev/null \
                && sudo mv "${fstab}.qiip.tmp" "$fstab"
        }; then
            echo "FATAL: could not update ${fstab} (write failed); entries unchanged" >&2
            return 1
        fi
        echo "Updated QIIP fstab entry for ${NFS_MOUNT_POINT}"
        return 0
    fi

    printf '%s\n' "$managed_entry" | sudo tee -a "$fstab" > /dev/null \
        || { echo "FATAL: could not append to ${fstab} (write failed)" >&2; return 1; }
    echo "fstab entry for ${NFS_MOUNT_POINT} (${marker}, nofail — storage outage will not block boot)"
}

ensure_autofs_entry() {
    local nfs_opts="$1" map="$2"
    local line key marker_line
    # Match the entry by key AND export: the same export may be mounted under
    # another key elsewhere, and that unrelated entry must stay untouched.
    line=$(awk -v mp="${NFS_MOUNT_POINT}" -v expval="${NFS_EXPORT}" '
        function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
        d($1) == mp && d($NF) == expval {got = $0; exit}
        END {if (got != "") print got}
    ' "$map")
    if [ -z "$line" ]; then
        echo "FATAL: ${map} has no entry for ${NFS_MOUNT_POINT} (export ${NFS_EXPORT}); cannot manage autofs" >&2
        return 1
    fi
    key=$(printf '%s\n' "$line" | awk '
        function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
        {print d($1)}
    ')
    marker_line="# qiip-managed ${key}"
    if ! grep -qF "$marker_line" "$map"; then
        echo "QIIP autofs entry for ${key} exists without marker; adopting it"
    fi
    entry="$(_escape_fstab_field "$key") -fstype=nfs,${nfs_opts} $(_escape_fstab_field "$NFS_EXPORT")"
    if ! {
        # Same -v escape caveat as ensure_fstab_entry: double the backslashes
        # so the awk replacement keeps \\040/\\011 field escapes intact.
        sudo awk -v ml="${marker_line//\\/\\\\}" -v entry="${entry//\\/\\\\}" \
            -v mp="${NFS_MOUNT_POINT}" -v expval="${NFS_EXPORT}" '
            function d(s){ gsub(/\\040/," ",s); gsub(/\\011/,"\t",s); gsub(/\\134/,"\\",s); return s }
            $0 == ml {next}
            d($1) == mp && d($NF) == expval {print ml; print entry; next}
            {print}' \
            "$map" | sudo tee "${map}.qiip.tmp" > /dev/null \
            && sudo mv "${map}.qiip.tmp" "$map"
    }; then
        echo "FATAL: could not update ${map} (write failed); entries unchanged" >&2
        return 1
    fi
    echo "Updated QIIP autofs entry for ${key} in ${map}"
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
