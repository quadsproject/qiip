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
NFS_EXPORT="${AUTOVLLM_NFS_EXPORT:-}"

# Shared storage primitives (source/fstype/option verification, capacity).
# Resolved from the node bundle layout first, then the installed location used
# by systemd's ExecStartPre (setup.sh installs qiip-setup-base.sh).
# shellcheck disable=SC1091 source=../common/setup-base.sh
_COMMON_SH=""
if [ -f "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common/setup-base.sh" ]; then
    _COMMON_SH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common/setup-base.sh"
elif [ -f /usr/local/bin/qiip-setup-base.sh ]; then
    _COMMON_SH="/usr/local/bin/qiip-setup-base.sh"
fi
if [ -n "$_COMMON_SH" ]; then
    # shellcheck disable=SC1090
    source "$_COMMON_SH"
else
    echo "WARNING: setup-base.sh not found; storage verification disabled" >&2
fi

# Callers can pre-set these (start-vllm.sh does after detect_gpu_info /
# configure_vllm_params); otherwise preflight detects them itself.
EXPECTED_GPU_COUNT="${AUTOVLLM_EXPECTED_GPU_COUNT:-}"
EXPECTED_TENSOR_PARALLEL="${AUTOVLLM_TENSOR_PARALLEL:-}"
MODEL_PATH="${AUTOVLLM_MODEL:-}"
GPU_DEVICES_OVERRIDE="${AUTOVLLM_GPU_DEVICES:-}"

# The selected subset must be visible to torch for both the sourced path and
# the systemd ExecStartPre `vllm-preflight --check-only` path, so export here
# rather than only in start-vllm.sh main(). nvidia-smi ignores this variable,
# so physical inventory below stays the full list.
if [ -n "$GPU_DEVICES_OVERRIDE" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES_OVERRIDE"
fi

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

# Validates an explicit device list: format, physical range, duplicates.
# Shared with start-vllm.sh detect_gpu_info so an invalid first token cannot
# reach an nvidia-smi -i probe before this actionable message can fire.
validate_gpu_devices_list() {
    local count="$1"
    if [ -z "$GPU_DEVICES_OVERRIDE" ]; then
        return 0
    fi
    if ! [[ "$GPU_DEVICES_OVERRIDE" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        _bail "GPU device list '${GPU_DEVICES_OVERRIDE}' is not a comma-separated list of numeric indices (e.g. 0,2)"
    fi
    local -a devices=()
    local token
    IFS=',' read -r -a devices <<< "$GPU_DEVICES_OVERRIDE"
    for token in "${devices[@]}"; do
        if [ "$token" -ge "$count" ]; then
            _bail "GPU device ${token} not present; physical devices are 0..$((count - 1)) (nvidia-smi --list-gpus)"
        fi
    done
    local -a seen=()
    for token in "${devices[@]}"; do
        if printf '%s\n' "${seen[@]}" | grep -qxF "$token"; then
            _bail "GPU device ${token} listed more than once in AUTOVLLM_GPU_DEVICES"
        fi
        seen+=("$token")
    done
}

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
    validate_gpu_devices_list "$count"

    # Capability probes follow the same first-selected-device rule as
    # detect_gpu_info so backend selection describes the card the engine runs.
    local probe_index=0
    if [ -n "$GPU_DEVICES_OVERRIDE" ]; then
        probe_index="${GPU_DEVICES_OVERRIDE%%,*}"
    fi
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$probe_index" | xargs)
    # Profile sizing needs the card name; detect_gpu_info already set it for
    # the sourced path (same first-selected-device rule), so only query here
    # when running standalone.
    GPU_MODEL="${GPU_MODEL:-$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$probe_index" | xargs)}"
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

# ── 3. Tensor-parallel default & CUDA device count ──────────────────────

# Mirrors configure_vllm_params in start-vllm.sh: datacenter cards spread
# across the selected devices; consumer/unknown cards use a single GPU.
profile_tensor_parallel() {
    local device_count="${GPU_DEVICE_COUNT:-}"
    if [ -z "$device_count" ]; then
        if [ -n "$GPU_DEVICES_OVERRIDE" ]; then
            device_count=$(awk -F, '{print NF}' <<< "$GPU_DEVICES_OVERRIDE") || true
        else
            device_count="$GPU_COUNT"
        fi
    fi
    case "$GPU_MODEL" in
        *"H100"*|*"A100"*|*"A30"*|*"A40"*|*"V100"*) echo "$device_count" ;;
        *) echo 1 ;;
    esac
}

# The tensor-parallel size the launch will use: start-vllm.sh sets
# EXPECTED_TENSOR_PARALLEL to its resolved value; the standalone --check-only
# path must reproduce the default so a device subset is validated against the
# allocation, not against the physical inventory.
effective_tensor_parallel() {
    if [ -n "$EXPECTED_TENSOR_PARALLEL" ]; then
        echo "$EXPECTED_TENSOR_PARALLEL"
    else
        profile_tensor_parallel
    fi
}

# ── CUDA device count vs tensor-parallel ────────────────────────────────

check_cuda_devices() {
    local tp
    tp="$(effective_tensor_parallel)"
    local cuda_count
    cuda_count=$("$VLLM_PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null) || true

    if [ -z "$cuda_count" ]; then
        _bail "torch.cuda.device_count() failed — check PyTorch/CUDA install in ${VLLM_PYTHON}"
    fi
    if [ -z "$tp" ]; then
        tp="$GPU_COUNT"
    fi
    if [ "$tp" -lt 1 ]; then
        _bail "tensor-parallel-size must be >= 1 (got ${tp})"
    fi
    if [ "$tp" -gt "$cuda_count" ]; then
        _bail "tensor-parallel-size ${tp} exceeds ${cuda_count} CUDA-visible device(s) of ${GPU_COUNT} physical — check AUTOVLLM_GPU_DEVICES / CUDA_VISIBLE_DEVICES or add GPUs"
    fi
    if [ "$tp" -lt "$cuda_count" ]; then
        _mark WARN "using ${tp} of ${cuda_count} CUDA-visible device(s); the engine allocates the first ${tp}"
    fi
    if [ "$cuda_count" -lt "$GPU_COUNT" ] && [ -z "$GPU_DEVICES_OVERRIDE" ]; then
        _mark WARN "CUDA-visible ${cuda_count} < physical ${GPU_COUNT} without AUTOVLLM_GPU_DEVICES; the model was sized from the physical inventory"
    fi
    _mark PASS "physical=${GPU_COUNT} (nvidia-smi), CUDA-visible=${cuda_count}, allocated=${tp} for tensor-parallel"
}

# ── 4. Model path / space ───────────────────────────────────────────────

# Mirrors the gateway's ModelCatalogService._snapshot_state (catalog.py): the
# local-only snapshot_download call checks the cached tree manifest and rejects
# snapshots with missing files, so a partial cache dir is never declared ready.
# huggingface_hub 1.x only checks completeness when the tree manifest
# (trees/<commit>.json) is cached: with the manifest absent, an existing
# snapshot dir is returned as-is. Require the manifest here, otherwise a
# manifest-less snapshot is reported as complete and the download that could
# repair it is skipped.
# Exit 0 complete, 2 incomplete (missing files), 3 not present, 4 probe timed out.
verify_hf_snapshot() {
    local model="$1" cache_dir="$2"
    local out rc repo_dir commit manifest
    rc=0
    repo_dir="${cache_dir}/models--${model//\//'--'}"
    if timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -r "${repo_dir}/refs/main"; then
        commit=$(timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" cat "${repo_dir}/refs/main") || return 4
        commit=$(printf '%s' "$commit" | tr -d '[:space:]')
        [ -z "$commit" ] && return 3
        manifest="${repo_dir}/trees/${commit}.json"
        [ -r "$manifest" ] || return 3
    fi
    out=$(timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" "$VLLM_PYTHON" -c '
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

model, cache_dir = sys.argv[1], sys.argv[2]
try:
    snapshot_download(model, repo_type="model", cache_dir=cache_dir, local_files_only=True)
except IncompleteSnapshotError:
    sys.exit(2)
except (LocalEntryNotFoundError, RepositoryNotFoundError):
    sys.exit(3)
' "$model" "$cache_dir" 2>&1) || rc=$?
    case "$rc" in
        0) return 0 ;;
        2) return 2 ;;
        3) return 3 ;;
        124) return 4 ;;
        *)
            echo "FATAL: cannot verify cached model ${model}: $(printf '%s\n' "$out" | tail -1)" >&2
            return 1
            ;;
    esac
}

check_model() {
    if [ -z "$MODEL_PATH" ]; then
        _mark WARN "No model specified (AUTOVLLM_MODEL) — skipping model-path check"
        return 0
    fi

    # Local directory — already downloaded
    if timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -d "$MODEL_PATH"; then
        if [ ! -r "$MODEL_PATH" ]; then
            _bail "Model directory ${MODEL_PATH} exists but is not readable"
        fi
        _mark PASS "Model ${MODEL_PATH} present locally"
        return 0
    fi

    # HF hub model — verify completeness of the local snapshot first, then
    # capacity for a download when the snapshot is absent. Hub resolves the
    # repo under <cache_dir>/models--<slug>: the cache root is the hub dir,
    # not the NFS mount root (the gateway stages under <mount>/hub, and
    # HF_HOME=<mount> for downloads resolves to the same place).
    local cache_dir="${NFS_MOUNT_POINT}/hub"
    if [ -L "$HF_CACHE_LINK" ]; then
        local link_target
        link_target=$(timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" readlink -f "$HF_CACHE_LINK" 2>/dev/null) || true
        [ -n "$link_target" ] && cache_dir="${link_target}/hub"
    fi

    if verify_hf_snapshot "$MODEL_PATH" "$cache_dir"; then
        _mark PASS "Model ${MODEL_PATH} verified complete in cache at ${cache_dir}"
        return 0
    else
        local verify_rc=$?
        if [ "$verify_rc" -eq 2 ]; then
            _bail "Model ${MODEL_PATH} snapshot is incomplete in ${cache_dir}; remove the partial repo directory and re-run setup"
        fi
        if [ "$verify_rc" -eq 4 ]; then
            _mark WARN "Model ${MODEL_PATH} cache verification timed out (storage slow or unreachable); preparation will re-check"
        elif [ "$verify_rc" -ne 3 ]; then
            _bail "Model ${MODEL_PATH} could not be verified (rc=${verify_rc})"
        fi
    fi

    local slug="${MODEL_PATH//\//'--'}"
    if timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -d "${cache_dir}/models--${slug}"; then
        _mark WARN "Model ${MODEL_PATH} cache directory exists but could not be verified; a download will repair it"
    fi

    # Hub creates <cache_dir> on first write and downloads under <mount>/hub,
    # so a missing hub subdirectory is fine as long as the mount is writable.
    local write_root="$cache_dir"
    if ! timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -d "$write_root"; then
        write_root="${NFS_MOUNT_POINT}"
    fi
    if ! timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -d "$write_root" \
        || ! timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" test -w "$write_root"; then
        _bail "Cache ${cache_dir} missing or not writable — model ${MODEL_PATH} needs download"
    fi

    local avail_bytes model_bytes
    avail_bytes=$(timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" df --output=avail -B1 "$cache_dir" 2>/dev/null | tail -1 | xargs) || true
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

# Mirrors vLLM's shardability rule so an unsupported tensor-parallel size
# fails with an actionable message instead of a mid-start error. KV heads
# replicate when fewer than TP (vLLM allows tp % kv == 0 in that case).
# rc: 0 pass, 1 reject (message printed), 2 not verifiable (no config.json).
verify_model_topology() {
    local model="$1" tp="$2"
    local config_file=""

    if [ "$tp" -le 1 ]; then
        return 0
    fi
    if [ -d "$model" ]; then
        config_file="${model}/config.json"
    else
        local slug="${model//\//'--'}"
        local cache_dir="${NFS_MOUNT_POINT}"
        if [ -L "$HF_CACHE_LINK" ]; then
            cache_dir=$(timeout --kill-after=2 "${AUTOVLLM_PROBE_TIMEOUT:-10}" readlink -f "$HF_CACHE_LINK") || true
        fi
        local model_root="${cache_dir}/hub/models--${slug}"
        local rev=""
        if [ -f "${model_root}/refs/main" ]; then
            rev=$(xargs < "${model_root}/refs/main") || true
        fi
        # Only the requested revision is authoritative; falling back to an
        # arbitrary cached snapshot would judge this model by another branch
        # or an older layout, and can reject a valid pending revision.
        if [ -n "$rev" ] && [ -f "${model_root}/snapshots/${rev}/config.json" ]; then
            config_file="${model_root}/snapshots/${rev}/config.json"
        fi
    fi
    if [ -z "$config_file" ] || [ ! -r "$config_file" ]; then
        return 2
    fi

    # All parsing, validation, and arithmetic stay in Python: config.json is
    # untrusted model metadata, and shell arithmetic recursively evaluates
    # embedded command substitutions before the divisibility checks run.
    # rc: 0 pass, 1 reject (FATAL on stderr), 2 not verifiable.
    local rc=0
    "$VLLM_PYTHON" - "$config_file" "$tp" "$model" <<'PY' || rc=$?
import json
import sys

try:
    with open(sys.argv[1]) as fh:
        cfg = json.load(fh)
    tp = int(sys.argv[2])
    model = sys.argv[3]
except Exception:
    sys.exit(2)

heads = cfg.get("num_attention_heads")
kv = cfg.get("num_key_value_heads")


def pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


if not pos_int(heads):
    sys.exit(2)
if not pos_int(kv) or kv > heads:
    kv = heads
if heads % tp:
    print(
        f"FATAL: model {model} has {heads} attention head(s), "
        f"not divisible by tensor-parallel-size {tp}",
        file=sys.stderr,
    )
    sys.exit(1)
if kv >= tp:
    if kv % tp:
        print(
            f"FATAL: model {model} has {kv} KV head(s), "
            f"not divisible by tensor-parallel-size {tp}",
            file=sys.stderr,
        )
        sys.exit(1)
elif tp % kv:
    print(
        f"FATAL: model {model} has {kv} KV head(s); "
        f"tensor-parallel-size {tp} is not a multiple of it",
        file=sys.stderr,
    )
    sys.exit(1)
PY
    case "$rc" in
        0|1|2) return "$rc" ;;
        *) return 2 ;;
    esac
}

check_model_topology() {
    local tp
    tp="$(effective_tensor_parallel)"
    if [ -z "$MODEL_PATH" ]; then
        return 0
    fi
    local rc=0
    verify_model_topology "$MODEL_PATH" "$tp" || rc=$?
    case "$rc" in
        0) _mark PASS "Model topology verified for TP=${tp}" ;;
        1) _bail "Unsupported model/topology parallelism (see message above)" ;;
        2) _mark WARN "Model topology not verifiable pre-launch; engine will validate" ;;
    esac
}

# ── 5. NFS mount health ────────────────────────────────────────────────

check_nfs_mounts() {
    # Managed runs may carry the exact export (AUTOVLLM_NFS_EXPORT); when
    # present, verify source, filesystem type, and required options against it
    # and fail closed. With no export (operator/direct run) the legacy
    # warning-only path below applies.
    if [ -n "$NFS_EXPORT" ]; then
        if verify_nfs_storage; then
            _mark PASS "NFS mount verified: source=${NFS_EXPORT}, NFSv3, required options"
            return 0
        else
            _bail "NFS storage verification failed at ${NFS_MOUNT_POINT} (see message above)"
        fi
    fi

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
        if [ "$any_fail" -eq 0 ]; then
            _mark PASS "NFS mounts healthy (hard, sane timeo)"
        fi
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
            missing+=("ninja (run: dnf install ninja-build / apt install ninja-build)")
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

    # vLLM v1 uses FlashInfer internally regardless of the attention backend.
    # When AOT kernels aren't available, JIT compilation needs ninja.
    if [ "$backend" != "FLASHINFER" ] \
        && "$VLLM_PYTHON" -c 'import flashinfer' &>/dev/null \
        && ! "$VLLM_PYTHON" -c \
            'from importlib.metadata import version; from packaging.version import Version; import flashinfer_cubin; assert Version(version("flashinfer-cubin")).public == Version(version("flashinfer-python")).public' \
            &>/dev/null; then
        local venv_bin
        venv_bin="$(dirname "$VLLM_BIN")"
        if ! "$venv_bin/ninja" --version &>/dev/null && ! command -v ninja &>/dev/null; then
            _bail "FlashInfer JIT needs ninja — run: dnf install ninja-build / apt install ninja-build"
        fi
    fi
}

# ── 7. Persistence mode ────────────────────────────────────────────────

check_persistence_mode() {
    local probe_index=0
    if [ -n "$GPU_DEVICES_OVERRIDE" ]; then
        probe_index="${GPU_DEVICES_OVERRIDE%%,*}"
    fi
    local pm
    pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader -i "$probe_index" | xargs)
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
    check_model_topology
    check_nfs_mounts
    # Model-cache space is verified in check_model only when a download is
    # needed; JIT kernel cache space is the only write at start time. The
    # directory is created here so the check never probes zero filesystems.
    mkdir -p "$FLASHINFER_CACHE"
    check_storage_capacity "$FLASHINFER_CACHE" || {
        local cap_rc=$?
        if [ "$cap_rc" -eq 1 ]; then
            _bail "Insufficient free space for the FlashInfer JIT cache (see above)"
        fi
        _mark WARN "Could not verify storage capacity (see above)"
    }
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
