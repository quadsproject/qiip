# auto-llamacpp

Provision and run CUDA-enabled llama.cpp (`llama-server`) directly on
bare-metal NVIDIA GPU nodes.

## Prerequisites

- RHEL 9-compatible Linux on x86_64
- An NVIDIA GPU and access to the configured NVIDIA package repositories;
  setup installs or validates the driver and CUDA toolkit
- Reachable NFS export containing the native Hugging Face Hub cache and its
  GGUF snapshots

QIIP-managed provisioning requires CUDA and fails before starting an engine if
GPU verification fails. The start script retains a standalone CPU branch for
direct development use, but CPU-only nodes are not a supported QIIP-managed
deployment target.

## Verified source build

Linux CUDA archives are not published for the pinned `b10242` release.
`setup.sh` therefore downloads the pinned GitHub tag source archive, verifies
its committed SHA-256 before extraction, and compiles `llama-server` and
`llama-quantize` with `GGML_CUDA=ON` and the attached GPUs' native CUDA
architecture.

Installations are immutable and build-identified under
`/opt/llama.cpp/<version>-<identity>`. The two public binaries in
`/usr/local/bin` are replaced with same-directory atomic symlink renames only
after the new build reports the configured version. Repeating setup with the
same source digest and GPU capabilities reuses that installation.

The full llama.cpp setup command has a separate two-hour default deadline
(`INFERENCE_PROXY_PROVISIONING__LLAMACPP_SETUP_TIMEOUT`) because it includes
package setup and a CUDA compilation. Its 15-minute SSH inactivity deadline
still applies, so build output remains a liveness signal.

### Bumping the llama.cpp version

1. Pick a `b<number>` tag from <https://github.com/ggml-org/llama.cpp/releases>.
2. Download the tag source archive and compute its SHA-256:

   ```bash
   curl -fSL -o llama.cpp-b12345.tar.gz \
     "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12345.tar.gz"
   sha256sum llama.cpp-b12345.tar.gz
   ```

3. Configure the matching version and digest:

   ```dotenv
   INFERENCE_PROXY_PROVISIONING__LLAMACPP_VERSION=b12345
   INFERENCE_PROXY_PROVISIONING__LLAMACPP_SHA256=<hash-from-step-2>
   ```

   A custom version without an explicitly configured digest fails validation.
   A mirror can be selected with
   `INFERENCE_PROXY_PROVISIONING__LLAMACPP_SOURCE_URL`, but the downloaded bytes
   must match the configured digest.

4. Validate the CUDA build and launch on representative fleet GPU models before
   rollout.

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

Start llama-server with an exact path relative to the mounted export root and
an explicit model alias. For an export whose Hub cache is in `hub/`:

```bash
AUTOLLAMACPP_GGUF_PATH=hub/models--org--model-GGUF/snapshots/<commit-sha>/model-Q4_K_M.gguf \
AUTOLLAMACPP_MODEL_ALIAS=org/model \
  ./start-llamacpp.sh
```

llama-server runs as a background process. PID is written to
`/var/run/llamacpp.pid`, logs to `/var/log/llamacpp-serve.log`.

Script-specific launch overrides use the `AUTOLLAMACPP_*` namespace:
`AUTOLLAMACPP_GGUF_PATH`, `AUTOLLAMACPP_MODEL_ALIAS`, `AUTOLLAMACPP_PORT`,
`AUTOLLAMACPP_GPU_LAYERS`, `AUTOLLAMACPP_CTX_SIZE`,
`AUTOLLAMACPP_PARALLEL`, `AUTOLLAMACPP_BATCH_SIZE`, and
`AUTOLLAMACPP_EXTRA_ARGS`.

The launcher clears inherited `LLAMA_ARG_*` variables before invoking
`llama-server`; only the reviewed command-line arguments and the
`AUTOLLAMACPP_*` inputs above control the child process. CUDA is required by
default. A standalone user deliberately exercising the retained CPU path must
opt out explicitly with `AUTOLLAMACPP_REQUIRE_CUDA=0`.

Context size is divided across parallel slots. If `AUTOLLAMACPP_CTX_SIZE=8192`
and `AUTOLLAMACPP_PARALLEL=4`, each slot gets 2048 tokens.

### GGUF model storage

QIIP does not create a parallel GGUF directory. It discovers standalone `.gguf`
files and complete llama.cpp split families directly in native Hugging Face
snapshot directories. The gateway maps the entrypoint from its configured
`HUGGINGFACE__SHARED_ROOT` to the node's `NFS_MOUNT_POINT`, preserving the
snapshot filename so llama.cpp can locate sibling shards.

The launcher receives one exact path; it never scans globally or chooses the
first quantization match. Snapshot generations are not removed automatically.
Older commits may still back running or restartable nodes, so operators must
coordinate Hugging Face cache retention deliberately.

### Model name aliasing

The `--alias` flag sets the canonical model name reported by `/v1/models`.
QIIP uses the exact Hugging Face repository ID as the managed artifact alias
and verifies that the launcher reports the same value. Directory names are
never decoded to recover that identity, so repository names containing
repeated `--` or `---` remain exact.

## Health check

```bash
curl -s http://{HOSTNAME}:8000/health
```

## Stop

```bash
./stop-llamacpp.sh
```

Pass `--force` for immediate SIGKILL.
