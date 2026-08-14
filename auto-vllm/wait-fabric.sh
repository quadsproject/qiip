#!/bin/bash
set -euo pipefail

# Standalone wait for NVSwitch fabric training. Used as systemd ExecStartPre
# so vLLM cannot launch before CUDA peer access is available.

TIMEOUT="${AUTOVLLM_FM_TIMEOUT:-120}"

nvswitch_count=$(lspci 2>/dev/null | grep -ci nvswitch || true)
if [ "$nvswitch_count" -eq 0 ]; then
    exit 0
fi

elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    state=$(nvidia-smi -q 2>/dev/null \
        | grep -A2 'Fabric' | grep 'State' | head -1 \
        | awk -F: '{print $2}' | xargs) || true
    if [ "$state" = "Completed" ]; then
        exit 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "FATAL: NVSwitch fabric training did not complete within ${TIMEOUT}s; check /var/log/fabricmanager.log" >&2
exit 1
