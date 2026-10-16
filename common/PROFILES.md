# Runtime profile compatibility matrix

Selection is driven by `common/profiles.sh`: the node measures compute
capability (`nvidia-smi --query-gpu=compute_cap`), per-device VRAM, driver
version, OS/ABI and NVSwitch presence, then matches one tested profile. GPU
marketing names are never used for selection; they appear only in logs and
the recorded reason. Measurement uses the first SELECTED device when an
`AUTOVLLM_GPU_DEVICES` subset is pinned (the gateway's subset semantics),
or from the caller's `CUDA_VISIBLE_DEVICES` selection, otherwise physical
device 0; queries are physical, so the subset is sized
against its own cards.

The CUDA execution proof preserves that selection and targets the selected
profile's architecture, with PTX for newer selected cards. It does not compile
for or execute on unselected cards. Driver mutations still require the entire
NVIDIA stack to be idle.

Status is honest: rows are `candidate` unless validated on a real fleet
node. The consumer-ada (SM 8.9) row is validated by the matched L4 pilot,
which provisioned a real node and returned `READY`; see
[docs/fleet-reliability.md](docs/fleet-reliability.md#matched-l4-pilot).
No other row has been validated in this repository. Before trusting a row
for production, run the fleet validation procedure below and record the result.

## Profiles

| Profile | Engine | SM | Min VRAM/device | Reusable driver policy; CUDA proof required | CUDA toolkit | Backend | OS/ABI | Fabric | Status |
|---|---|---|---|---|---|---|---|---|---|
| hopper | vllm, llamacpp | 9.0 | 80 GB | ≥580.65.06 | 13.0 | FlashInfer (vllm) | rhel9+, x86_64, glibc 2.34+ | when NVSwitch present | candidate |
| ampere-a100 | vllm, llamacpp | 8.0 | 40 GB | ≥580.65.06 | 13.0 | FlashAttn (vllm) | rhel9+, x86_64, glibc 2.34+ | when NVSwitch present | candidate |
| ampere-a30 | vllm, llamacpp | 8.0 | 24 GB | ≥580.65.06 | 13.0 | FlashAttn (vllm) | rhel9+, x86_64, glibc 2.34+ | when NVSwitch present | candidate |
| ga102-dc | vllm, llamacpp | 8.6 | 48 GB (reported ≥44 GiB) | ≥580.65.06 | 13.0 | FlashAttn (vllm) | rhel9+, x86_64, glibc 2.34+ | when NVSwitch present | candidate |
| consumer-ada | vllm, llamacpp | 8.9 | any (below DC cutoff) | ≥580.65.06 | 13.0 | FlashAttn (vllm) | rhel9+, x86_64, glibc 2.34+ | not required | candidate |
| consumer-ampere | vllm, llamacpp | 8.6 | any (below DC cutoff) | ≥580.65.06 | 13.0 | FlashAttn (vllm) | rhel9+, x86_64, glibc 2.34+ | not required | candidate |
| turing | vllm, llamacpp | 7.5 | 16 GB (reported ≥14 GiB) | ≥580.65.06 | 13.0 | default (vllm) | rhel9+, x86_64, glibc 2.34+ | not required | candidate |
| consumer-turing | vllm, llamacpp | 7.5 | any (below DC cutoff) | ≥580.65.06 | 13.0 | default (vllm) | rhel9+, x86_64, glibc 2.34+ | not required | candidate |
| volta | llamacpp | 7.0 | 16 GB | ≥575.57.08, branch ≤580 | 12.9 | n/a (source-built) | rhel9+, x86_64, glibc 2.34+ | not required | candidate |

Driver limits are compatibility evidence, not new fleet validation results.
The tested replacement artifact remains 580.126.09. NVIDIA documents
[CUDA 13.x compatibility with R580 and newer](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
and [CUDA 12.9 Update 2's full driver requirement](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/).
[R580 is the final branch supporting Volta](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/).
Every reuse decision also requires a CUDA kernel execution probe. Setup
compiles for the measured architectures, runs the kernel on each CUDA device,
and records the driver API/runtime versions. Fabric Manager must still match
the actual installed driver on NVSwitch hosts. No cuda-compat shim is assumed.
Compatible reuse does not require kernel headers or a replacement download.
Repair and reboot handling are described in the
[upgrade guide](../UPGRADING.md#13-prepare-for-strict-driver-and-cache-checks).

Unlisted combinations (for example Blackwell sm 10.0/12.0, GTX 10xx sm 6.1,
P100 sm 6.0, sm 8.0 with less than 24 GB, sm 9.0 with less than 80 GB) are
rejected as unsupported: `[REJECT:unsupported_hardware:...]` marker and exit
code 3, with the measured signature and the supported bucket list in the
message. They are not silently given conservative settings.

## Behavior changes vs marketing-name selection

The following cards switch from the old `case "$GPU_MODEL"` outcomes to the
profile arms above. Each is a deliberate `candidate` change, not a validated
regression:

- H200/H20 (sm 9.0): old unmatched `*` arm (TP=1, 0.75, 4096, eager, 7B)
  becomes the hopper arm (throughput tuning).
- L40/L40S (sm 8.9): old unmatched `*` arm becomes consumer-ada.
- A10/A2 (sm 8.6, below 48 GB): old unmatched `*` arm becomes
  consumer-ampere.
- RTX A6000 (sm 8.6, 48 GB): old RTX arm (TP=1, 0.80, 4096, eager) becomes
  the ga102-dc arm (TP=all, 0.90, 32768, no eager), matching A40.
- Any other card that previously hit the unmatched `*` arm now lands on a
  tuned arm: examples are A800 (sm 8.0, 80 GB), GH200 (sm 9.0, 141 GB), L20
  (sm 8.9, 48 GB), L4 (sm 8.9, 24 GB). The payloads are the profile arms
  above; the old `*` arm no longer exists.
- OS/ABI gate: non-RHEL9 or non-x86_64 nodes (for example RHEL 8 or Fedora)
  and vLLM hosts with glibc below 2.34 are now rejected at setup and start
  with exit code 3 where they previously ran.
- Volta (V100, sm 7.0) toolkit: CUDA 13.0 removed Maxwell/Pascal/Volta
  (offline compilation and library support), so the volta profile pins the
  CUDA 12.9 toolkit and is llama.cpp-only. The frozen vLLM stack (torch
  2.11.0 CUDA-13.0) cannot target SM70, so `volta` with the vllm engine is
  rejected up front with exit code 3 instead of failing the CUDA proof later.
- SM 7.5 cutoff: a 16 GB T4 reports 15,079 MiB usable, which rounds to 15 GB
  with the marketing figure; the turing cutoff therefore keys on reported
  MiB (≥ 14 GiB) so a real T4 stays on the turing arm instead of
  consumer-turing.
- SM 8.6 cutoff: an A40 with ECC reports 46,068 MiB (44.98 GiB), which rounds
  to 45 GB and would miss a 48 GB cutoff; the ga102-dc cutoff therefore keys
  on reported MiB (≥ 44 GiB) so a data-center GA102 card stays on the tuned
  arm instead of consumer-ampere, mirroring the turing cutoff.

These follow from measured SM and VRAM; two cards that differ only by name
are indistinguishable to the profile, so the data-center arm wins for 48 GB
GA102 hardware.

## Node upgrade note

Existing nodes keep the previous `/usr/local/bin` helper copies
(`wait-nvswitch-fabric`, `vllm-preflight`) and the systemd unit until
`setup.sh` is re-run. Re-run the node's `setup.sh` after deploying this
bundle so the installed copies of `setup-base.sh` and `profiles.sh`
(`/usr/local/bin/qiip-setup-base.sh`, `/usr/local/bin/qiip-profiles.sh`)
refresh and the profile gates apply.

The NVIDIA RHEL9 repository installs nvcc under
`/usr/local/cuda-<version>/bin` without a `/usr/local/cuda` symlink; setup
creates that symlink after installing the exact toolkit so both layouts
work.

## Fleet validation procedure

One-time, per hardware family on a representative node:

1. Set up with the profile: run `setup.sh`; confirm the
   `[PROFILE:select:<name>]` marker and `[STEP:cuda_proof:OK]`.
2. Confirm `nvcc --version` reports the profile CUDA version and that the
   tiny CUDA execution probe passes (setup already fails otherwise).
3. Start the engine, confirm the `[PROFILE:select:...]` marker and the
   profile line in the launch banner.
4. Run a representative inference request; record model, tensor parallel,
   context length, GPU memory utilization and observed throughput.
5. Record the node signature (SM, VRAM/device, driver, OS, NVSwitch count)
   and the result in this document, then flip the row status to `validated`.

A failure must distinguish reproduced code-level failures (reproducible in
the controlled regression suite) from causes confirmed on the real node
(hardware, topology, vendor driver behavior).
