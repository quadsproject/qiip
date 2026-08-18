#!/bin/bash
set -euo pipefail

# Pre-launch validation for vLLM. Runs each check in dependency order and
# stops at the first failure with an actionable message. Exits 0 only if
# every check passes.
#
# Usage:
#   ./preflight.sh --check-only          # operator dry-run
#   source preflight.sh; run_preflight   # called by start-vllm.sh

VLLM_BIN="${AUTOVLLM_BIN:-/opt/vllm-venv/bin/vllm}"
VLLM_PYTHON="${AUTOVLLM_PYTHON:-$(dirname "$VLLM_BIN")/python}"
NFS_MOUNT_POINT="${AUTOVLLM_NFS_MOUNT_POINT:-/srv/hf-cache}"
HF_CACHE_LINK="${AUTOVLLM_HF_CACHE_LINK:-/root/.cache/huggingface}"
FLASHINFER_CACHE="${AUTOVLLM_FLASHINFER_CACHE_DIR:-/var/cache/flashinfer}"
ATTENTION_BACKEND_OVERRIDE="${AUTOVLLM_ATTENTION_BACKEND:-}"

# Callers can pre-set these (start-vllm.sh does after detect_gpu_info /
# configure_vllm_params); otherwise preflight detects them itself.
EXPECTED_GPU_COUNT="${AUTOVLLM_EXPECTED_GPU_COUNT:-}"
EXPECTED_TENSOR_PARALLEL="${AUTOVLLM_TENSOR_PARALLEL:-}"
MODEL_PATH="${AUTOVLLM_MODEL:-}"

_pass=0
_fail=0
_warn=0
_rc=0

_mark() {
    local status="$1"; shift
    case "$status" in
        PASS) (( ++_pass )); printf '  [\e[32mPASS\e[0m] %s\n' "$*" ;;
        FAIL) (( ++_fail )); printf '  [\e[31mFAIL\e[0m] %s\n' "$*"; _rc=1 ;;
        WARN) (( ++_warn )); printf '  [\e[33mWARN\e[0m] %s\n' "$*" ;;
    esac
}

_bail() {
    _mark FAIL "$@"
    _summary
    exit 1
}

_summary() {
    echo
    printf 'Preflight: %d passed, %d warnings, %d failed\n' "$_pass" "$_warn" "$_fail"
}

# ── 1. Driver & GPU count ───────────────────────────────────────────────

check_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        _bail "nvidia-smi not found — install NVIDIA drivers or run setup.sh"
    fi
    if ! nvidia-smi &>/dev/null; then
        _bail "nvidia-smi failed — driver not loaded (modprobe nvidia, or reboot after install)"
    fi

    local count
    count=$(nvidia-smi --list-gpus | wc -l)
    if [ -n "$EXPECTED_GPU_COUNT" ] && [ "$count" -ne "$EXPECTED_GPU_COUNT" ]; then
        _bail "GPU count: expected ${EXPECTED_GPU_COUNT}, found ${count}"
    fi
    _mark PASS "Driver loaded, ${count} GPU(s) visible"

    GPU_COUNT="$count"
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | xargs)
}

# ── 2. NVSwitch / Fabric Manager ────────────────────────────────────────

check_fabric() {
    local nvswitch_count
    nvswitch_count=$(lspci 2>/dev/null | grep -ci nvswitch || true)
    if [ "$nvswitch_count" -eq 0 ]; then
        _mark PASS "No NVSwitches — Fabric Manager not required"
        return 0
    fi

    local driver_version fm_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
        | sed '/^[[:space:]]*$/d' | head -1 | xargs)

    # Prefer the binary version — redist installs bypass RPM, so a stale
    # RPM from a previous driver can report the wrong version.
    if [ -x /usr/bin/nv-fabricmanager ]; then
        fm_version=$(nv-fabricmanager --version 2>/dev/null \
            | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true
    fi
    if [ -z "$fm_version" ] && rpm -q nvidia-fabricmanager &>/dev/null; then
        fm_version=$(rpm -q --qf '%{VERSION}' nvidia-fabricmanager)
    fi
    if [ -z "$fm_version" ]; then
        _bail "NVSwitch present but nvidia-fabricmanager not installed (setup.sh ensure_fabric_manager)"
    fi
    if [ "$fm_version" != "$driver_version" ]; then
        _bail "Fabric Manager ${fm_version} != driver ${driver_version} — version mismatch causes CUDA error 802"
    fi

    if ! systemctl is-active --quiet nvidia-fabricmanager; then
        _bail "nvidia-fabricmanager.service not active — systemctl start nvidia-fabricmanager"
    fi

    local fabric_state
    fabric_state=$(nvidia-smi -q 2>/dev/null \
        | grep -A2 'Fabric' | grep 'State' | head -1 \
        | awk -F: '{print $2}' | xargs) || true
    if [ "$fabric_state" = "Completed" ]; then
        _mark PASS "Fabric Manager ${fm_version}, training completed (nvidia-smi)"
    elif [ "$(systemctl show -p Type --value nvidia-fabricmanager 2>/dev/null)" = "oneshot" ]; then
        _mark PASS "Fabric Manager ${fm_version}, training completed (service exited successfully)"
    else
        _bail "Fabric State: '${fabric_state:-unknown}' (expected Completed) — check /var/log/fabricmanager.log"
    fi
}

# ── 3. CUDA device count vs tensor-parallel ─────────────────────────────

check_cuda_devices() {
    local tp="${EXPECTED_TENSOR_PARALLEL:-$GPU_COUNT}"
    local cuda_count
    cuda_count=$("$VLLM_PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null) || true

    if [ -z "$cuda_count" ]; then
        _bail "torch.cuda.device_count() failed — check PyTorch/CUDA install in ${VLLM_PYTHON}"
    fi
    if [ "$cuda_count" -ne "$tp" ]; then
        _bail "torch sees ${cuda_count} CUDA device(s) but tensor-parallel needs ${tp} — check CUDA_VISIBLE_DEVICES or driver"
    fi
    _mark PASS "torch.cuda.device_count() = ${cuda_count} (matches TP=${tp})"
}

# ── 4. Model path / space ───────────────────────────────────────────────

check_model() {
    if [ -z "$MODEL_PATH" ]; then
        _mark WARN "No model specified (AUTOVLLM_MODEL) — skipping model-path check"
        return 0
    fi

    # Local directory — already downloaded
    if [ -d "$MODEL_PATH" ]; then
        if [ ! -r "$MODEL_PATH" ]; then
            _bail "Model directory ${MODEL_PATH} exists but is not readable"
        fi
        _mark PASS "Model ${MODEL_PATH} present locally"
        return 0
    fi

    # HF hub model — check cache, then free space
    local cache_dir="${NFS_MOUNT_POINT}"
    if [ -L "$HF_CACHE_LINK" ]; then
        cache_dir=$(readlink -f "$HF_CACHE_LINK")
    fi

    local slug="${MODEL_PATH//\//'--'}"
    if [ -d "${cache_dir}/hub/models--${slug}" ]; then
        _mark PASS "Model ${MODEL_PATH} cached at ${cache_dir}"
        return 0
    fi

    if [ ! -d "$cache_dir" ] || [ ! -w "$cache_dir" ]; then
        _bail "Cache ${cache_dir} missing or not writable — model ${MODEL_PATH} needs download"
    fi

    local avail_bytes model_bytes
    avail_bytes=$(df --output=avail -B1 "$cache_dir" 2>/dev/null | tail -1 | xargs) || true
    model_bytes=$("$VLLM_PYTHON" -c "
import sys; from huggingface_hub import model_info
print(sum(s.size or 0 for s in model_info(sys.argv[1]).siblings))
" "$MODEL_PATH" 2>/dev/null) || true

    if [ -n "$model_bytes" ] && [ -n "$avail_bytes" ]; then
        if [ "$model_bytes" -gt "$avail_bytes" ]; then
            local need_gb=$((model_bytes / 1073741824))
            local avail_gb=$((avail_bytes / 1073741824))
            _bail "Model ${MODEL_PATH} needs ~${need_gb}GB but ${cache_dir} has ${avail_gb}GB free"
        fi
        _mark PASS "Model ${MODEL_PATH} not cached; ${cache_dir} has space for download"
    else
        _mark WARN "Cannot verify model size or free space — download may fail"
    fi
}

# ── 5. NFS mount health ────────────────────────────────────────────────

check_nfs_mounts() {
    local paths=("$NFS_MOUNT_POINT" "$FLASHINFER_CACHE")
    if [ -L "$HF_CACHE_LINK" ]; then
        paths+=("$(readlink -f "$HF_CACHE_LINK")")
    fi

    local checked=()
    for path in "${paths[@]}"; do
        [ -d "$path" ] || continue
        local mountpoint
        mountpoint=$(df --output=target "$path" 2>/dev/null | tail -1 | xargs) || continue
        # deduplicate
        printf '%s\n' "${checked[@]}" 2>/dev/null | grep -qxF "$mountpoint" && continue
        checked+=("$mountpoint")

        local fstype opts
        fstype=$(awk -v mp="$mountpoint" '$2 == mp {print $3}' /proc/mounts)
        [ "$fstype" = "nfs" ] || [ "$fstype" = "nfs4" ] || continue

        opts=$(awk -v mp="$mountpoint" '$2 == mp {print $4}' /proc/mounts)
        if [[ "$opts" == *"soft"* ]]; then
            _mark WARN "NFS ${mountpoint}: mounted soft — EIO on server blip; remount with hard"
        elif [[ "$opts" != *"hard"* ]]; then
            _mark WARN "NFS ${mountpoint}: mount options missing 'hard' — may get EIO under load"
        fi

        if [[ "$opts" != *"timeo="* ]]; then
            _mark WARN "NFS ${mountpoint}: no timeo set — kernel default (7 = 0.7s) is too aggressive for bulk I/O"
        else
            local timeo
            timeo=$(echo "$opts" | grep -oP 'timeo=\K[0-9]+')
            if [ -n "$timeo" ] && [ "$timeo" -lt 100 ]; then
                _mark WARN "NFS ${mountpoint}: timeo=${timeo} (< 100) — raise to 600+ for model weight I/O"
            fi
        fi
    done

    if [ "${#checked[@]}" -eq 0 ]; then
        _mark PASS "No NFS mounts under model/cache paths"
    else
        local any_fail=0
        for mp in "${checked[@]}"; do
            local opts
            opts=$(awk -v mp="$mp" '$2 == mp {print $4}' /proc/mounts)
            if [[ "$opts" == *"hard"* ]] && [[ "$opts" == *"timeo="* ]]; then
                local timeo
                timeo=$(echo "$opts" | grep -oP 'timeo=\K[0-9]+')
                [ -n "$timeo" ] && [ "$timeo" -ge 100 ] && continue
            fi
            any_fail=1
        done
        [ "$any_fail" -eq 0 ] && _mark PASS "NFS mounts healthy (hard, sane timeo)"
    fi
}

# ── 6. Attention backend & JIT toolchain ────────────────────────────────

check_attention_backend() {
    local major="${GPU_COMPUTE_CAP%%.*}"
    local minor="${GPU_COMPUTE_CAP#*.}"
    local sm=$(( major * 10 + minor ))

    local backend=""
    if [ "$sm" -ge 90 ]; then
        backend="FLASHINFER"
    elif [ "$sm" -ge 80 ]; then
        backend="FLASH_ATTN"
    fi
    [ -n "$ATTENTION_BACKEND_OVERRIDE" ] && backend="$ATTENTION_BACKEND_OVERRIDE"

    if [ -z "$backend" ]; then
        _mark PASS "SM${sm}: no attention backend forced — vLLM default applies"
        return 0
    fi

    if [ "$backend" = "FLASHINFER" ] && [ "$sm" -lt 75 ]; then
        _bail "FlashInfer requires SM75+ (Turing); this GPU is SM${sm}"
    fi
    if [ "$backend" = "FLASH_ATTN" ] && [ "$sm" -lt 80 ]; then
        _bail "FLASH_ATTN requires SM80+ (Ampere); this GPU is SM${sm}"
    fi

    if [ "$backend" = "FLASHINFER" ]; then
        # AOT kernels present?
        if "$VLLM_PYTHON" -c \
            'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public' \
            &>/dev/null; then
            _mark PASS "Attention backend: FlashInfer (AOT kernels matched, no JIT needed)"
            return 0
        fi

        # JIT path — need compiler toolchain
        local venv_bin missing=()
        venv_bin="$(dirname "$VLLM_BIN")"

        if ! "$venv_bin/ninja" --version &>/dev/null; then
            missing+=("ninja (run: ${venv_bin}/pip install ninja)")
        fi
        if ! command -v nvcc &>/dev/null; then
            missing+=("nvcc (install cuda-toolkit)")
        else
            local nvcc_ver torch_cuda_ver
            nvcc_ver=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' | head -1) || true
            torch_cuda_ver=$("$VLLM_PYTHON" -c 'import torch; print(".".join(torch.version.cuda.split(".")[:2]))' 2>/dev/null) || true
            if [ -n "$nvcc_ver" ] && [ -n "$torch_cuda_ver" ] && [ "$nvcc_ver" != "$torch_cuda_ver" ]; then
                missing+=("nvcc ${torch_cuda_ver} (installed: ${nvcc_ver}; must match torch.version.cuda)")
            fi
        fi
        if ! command -v gcc &>/dev/null; then
            missing+=("gcc")
        fi

        if [ ${#missing[@]} -gt 0 ]; then
            printf '  [\e[31mFAIL\e[0m] FlashInfer JIT requires:\n' >&2
            printf '           - %s\n' "${missing[@]}" >&2
            (( _fail++ ))
            _rc=1
            return 1
        fi
        _mark PASS "Attention backend: FlashInfer (JIT toolchain present)"
    else
        _mark PASS "Attention backend: ${backend} (SM${sm})"
    fi
}

# ── 7. Persistence mode ────────────────────────────────────────────────

check_persistence_mode() {
    local pm
    pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader -i 0 | xargs)
    if [ "$pm" = "Enabled" ]; then
        _mark PASS "Persistence mode enabled"
    else
        _mark WARN "Persistence mode disabled — enable with: nvidia-smi -pm 1 (avoids driver reload latency)"
    fi
}

# ── Orchestrator ────────────────────────────────────────────────────────

run_preflight() {
    echo "vLLM preflight checks"
    echo "─────────────────────"
    check_driver
    check_fabric
    check_cuda_devices
    check_model
    check_nfs_mounts
    check_attention_backend
    check_persistence_mode
    _summary
    return "$_rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [ "${1:-}" = "--check-only" ]; then
        run_preflight
        exit $?
    elif [ $# -gt 0 ]; then
        echo "Usage: $0 [--check-only]" >&2
        exit 2
    fi
    run_preflight
fi
