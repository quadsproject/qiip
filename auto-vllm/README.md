# auto-vllm

Provision and run vLLM directly on bare-metal GPU nodes.

## Prerequisites

- RHEL 9-compatible Linux on x86_64 with CPython 3.12 available as
  `/usr/bin/python3.12`
- NVIDIA GPU; `setup.sh` installs the configured driver when none is present
- Reachable NFS export for the shared HuggingFace cache

## Reproducible Python environment

Node Python packages are managed only by the dedicated `uv` project in this
directory. `setup.sh` installs the pinned `uv` release after verifying its
published SHA-256, then performs an exact, frozen, wheel-only synchronization
into `/opt/vllm-venv`. Re-running setup removes packages that are not present in
`uv.lock`.

The `uv` version is recorded in `.uv-version`. Its checksum comes from Astral's
published release asset:

```text
https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-unknown-linux-gnu.tar.gz.sha256
```

Regenerate the node lock from the repository root with that exact `uv` binary:

```bash
uv lock --project auto-vllm --python 3.12 --upgrade
uv lock --check --project auto-vllm
```

The Linux x86_64 CPython constraint is declarative in `pyproject.toml`; `uv
lock` does not accept `--python-platform`. Provisioning selects the matching
installation artifacts with `--python-platform x86_64-manylinux_2_34` and
refuses source builds. If locking or the wheel-only dry-run fails, choose the
newest acceptable dependency release that publishes a compatible wheel. Do not
weaken the global no-source-build policy without separate review.

The FlashInfer index is part of the project and frozen lock. A deployment using
a package mirror must ship a deliberately regenerated `auto-vllm` bundle rather
than overriding the index at runtime.

The current frozen node runtime contains vLLM 0.26.0 and matching FlashInfer
Python/AOT packages at 0.6.14. Updating those versions and regenerating the lock
is one reviewed change; do not edit the lock or runtime versions independently.

The committed default artifact digests originate from the vendors' published
checksum files:

- NVIDIA 580.126.09: `NVIDIA-Linux-x86_64-580.126.09.run.sha256sum`
- LLMFit 1.1.6: `llmfit-v1.1.6-x86_64-unknown-linux-musl.tar.gz.sha256`

## Sources and mirrors

The node setup has no global mirror setting. Each source has a distinct trust
boundary:

- `NVIDIA_DRIVER_URL` and `LLMFIT_URL` are node-side environment variables read
  by `setup.sh`. Byte-identical mirrors work with the committed digests. The
  gateway passes the selected versions and digests, but it does not forward
  these URL variables.
- The LLMFit runner's separate on-demand installation path uses the gateway
  setting `INFERENCE_PROXY_LLMFIT__INSTALL_URL`. Configure that URL as well when
  recommendations must use a mirror.
- The uv bootstrap URL is pinned in `setup.sh`; its official checksum is stored
  beside the script. An air-gapped node can preinstall the exact version from
  `.uv-version` at `/usr/local/bin/uv` so bootstrap does not download it.
- Python package indexes are owned by `pyproject.toml` and `uv.lock`. A custom
  PyPI or FlashInfer source requires regenerating and reviewing the bundle.
  `FLASHINFER_INDEX_URL` is retired and makes setup fail rather than silently
  invalidating the lock's source assumptions.
- DNF packages use node repository policy and RPM signature verification.
  `setup.sh` also enables NVIDIA's RHEL 9 CUDA repository. Mirror that traffic
  through node repository management or a reviewed script customization; the
  gateway has no DNF mirror setting.

See the repository [upgrade guide](../UPGRADING.md#artifact-sources-and-mirror-policy)
for the full operator-facing policy.

## Setup

Run the setup script to install drivers, uv, the locked vLLM environment,
LLMFit, mount NFS, and open the firewall:

```bash
AUTOVLLM_NFS_EXPORT=storage.example.com:/exports/huggingface ./setup.sh
```

An already-installed NVIDIA driver must exactly match
`AUTOVLLM_NVIDIA_DRIVER_VERSION`; setup refuses to hot-swap a different live
kernel driver. Custom NVIDIA or LLMFit versions require matching SHA-256 values.

## Run

Start vLLM (auto-detects GPU and selects a model):

```bash
./start-vllm.sh
```

vLLM runs as a background process. PID is written to `/var/run/vllm.pid`, logs to `/var/log/vllm-serve.log`.

Script-specific launch overrides use the `AUTOVLLM_*` namespace, including
`AUTOVLLM_MODEL`, `AUTOVLLM_API_PORT`, `AUTOVLLM_TENSOR_PARALLEL`,
`AUTOVLLM_GPU_MEM_UTIL`, `AUTOVLLM_MAX_MODEL_LEN`,
`AUTOVLLM_MAX_BATCHED_TOKENS`, and `AUTOVLLM_EXTRA_ARGS`. Do not use
`VLLM_PORT` for the API port; vLLM reserves that name for internal
communication.

## Health check

```bash
curl -s http://{HOSTNAME}:8000/health
```

## Stop

```bash
./stop-vllm.sh
```

The stop helper verifies process identity, attempts TERM before KILL, scans for
orphaned vLLM processes, and preserves the PID file when termination cannot be
verified.
