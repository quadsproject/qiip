# auto-llamacpp

Provision and run llama.cpp (llama-server) directly on bare-metal GPU nodes.

## Prerequisites

- RHEL 9-compatible Linux on x86_64
- NVIDIA GPU (optional; CPU-only inference is supported)
- Reachable NFS export for the shared HuggingFace cache with GGUF models
  in the `gguf/` subdirectory

## Binary acquisition

`setup.sh` downloads a pre-built CUDA binary from the llama.cpp GitHub
releases and verifies its SHA-256 when a digest is configured. The binary
is installed to `/usr/local/bin/llama-server`. The `llama-quantize` tool is
also installed when present in the release archive.

The version is a build tag (e.g. `b10242`). llama.cpp ships multiple builds
per day with no LTS, so pinning is essential.

### Bumping the llama.cpp version

1. Pick a release tag from https://github.com/ggml-org/llama.cpp/releases
2. Download the CUDA binary for your platform and compute the SHA-256:
   ```bash
   curl -fSL -o llamacpp.tar.gz \
     "https://github.com/ggml-org/llama.cpp/releases/download/b12345/llama-b12345-bin-ubuntu-x64-cuda-cu12.2.tar.gz"
   sha256sum llamacpp.tar.gz
   ```
3. Update the provisioning settings:
   ```dotenv
   INFERENCE_PROXY_PROVISIONING__LLAMACPP_VERSION=b12345
   INFERENCE_PROXY_PROVISIONING__LLAMACPP_SHA256=<hash-from-step-2>
   ```
   Or update the defaults in `inference_proxy/config/settings.py`.
4. Test provisioning on a single node before fleet rollout.

The `install_llamacpp()` function is idempotent and skips download when the
installed version already matches.

## Shared setup infrastructure

Shared setup functions (NVIDIA driver, CUDA toolkit, NFS mount, firewall,
llmfit) live in `common/setup-base.sh`. Both `auto-vllm/setup.sh` and
`auto-llamacpp/setup.sh` source this file. Do not duplicate shared logic
in engine-specific scripts.

## Setup

Run the setup script to install drivers, llama-server, mount NFS, and open
the firewall:

```bash
AUTOVLLM_NFS_EXPORT=storage.example.com:/exports/huggingface ./setup.sh
```

Shared environment variables use the `AUTOVLLM_` prefix for backward
compatibility with existing provisioning infrastructure.

## Run

Start llama-server (auto-detects GPU and selects a GGUF model):

```bash
./start-llamacpp.sh
```

llama-server runs as a background process. PID is written to
`/var/run/llamacpp.pid`, logs to `/var/log/llamacpp-serve.log`.

Script-specific launch overrides use the `AUTOLLAMACPP_*` namespace:
`AUTOLLAMACPP_MODEL`, `AUTOLLAMACPP_PORT`, `AUTOLLAMACPP_GPU_LAYERS`,
`AUTOLLAMACPP_CTX_SIZE`, `AUTOLLAMACPP_PARALLEL`, `AUTOLLAMACPP_BATCH_SIZE`,
`AUTOLLAMACPP_QUANTIZATION`, and `AUTOLLAMACPP_EXTRA_ARGS`.

Context size is divided across parallel slots. If `AUTOLLAMACPP_CTX_SIZE=8192`
and `AUTOLLAMACPP_PARALLEL=4`, each slot gets 2048 tokens.

### GGUF model storage

GGUF files are stored under `<NFS_MOUNT_POINT>/gguf/` in directories named
after the HuggingFace repo (e.g. `Qwen--Qwen2.5-7B-Instruct-GGUF/`). The
start script selects a file matching the configured quantization level
(default `Q4_K_M`).

### Model name aliasing

The `--alias` flag sets the canonical model name reported by `/v1/models`.
The start script derives the alias by stripping the `-GGUF` suffix from the
directory name and converting `--` back to `/`, so a vLLM node and a
llama.cpp node serving the same model report the same name to the router.

## Health check

```bash
curl -s http://{HOSTNAME}:8000/health
```

## Stop

```bash
./stop-llamacpp.sh
```

Pass `--force` for immediate SIGKILL.
