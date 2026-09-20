#!/bin/bash
set -euo pipefail

# Standalone wait for NVSwitch fabric training. Used as systemd ExecStartPre
# so vLLM cannot launch before CUDA peer access is available. The shared
# implementation lives in common/setup-base.sh (wait_nvswitch_fabric).

# shellcheck disable=SC1091
_qiip_lib="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common/setup-base.sh"
[ -f "$_qiip_lib" ] || _qiip_lib=/usr/local/bin/qiip-setup-base.sh
# shellcheck disable=SC1090
source "$_qiip_lib"
unset _qiip_lib

wait_nvswitch_fabric
