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
its committed SHA-256 before extraction, applies one digest-pinned CLI
allowlist transformation, and compiles `llama-server`, `llama-fit-params`, and
`llama-quantize` with `GGML_CUDA=ON` and the attached GPUs' native CUDA
architecture. The supporting CPU backend uses `GGML_NATIVE=OFF`: managed
inference is CUDA-only, and the portable CPU profile avoids coupling builds to
host-specific compiler and assembler feature support.
The build uses CMake's explicit Unix Makefiles generator with parallel jobs, so
it depends only on the `make` package available from the standard RHEL
repositories and does not require CodeReady Builder or `ninja-build`.

Installations are immutable and build-identified under
`/opt/llama.cpp/<version>-<identity>`. The three public binaries in
`/usr/local/bin` are replaced with same-directory atomic symlink renames only
after the new build reports the configured version. Repeating setup with the
same source digest, transformation digest, build profile, and GPU capabilities
reuses that installation.

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

The sizing overrides above are for standalone use. QIIP sets
`AUTOLLAMACPP_MANAGED=1`, rejects all five sizing and extra-argument overrides,
and invokes the installed `llama-fit-params` estimator before launch. The
planner first maximizes guaranteed per-request context up to the model's
trained length, then finds the largest slot count that keeps every layer on GPU
and preserves the requested free-memory margin. The only concurrency ceiling is
llama.cpp's 256-sequence limit. Configure the per-GPU free-memory target with
`INFERENCE_PROXY_PROVISIONING__LLAMACPP_FIT_TARGET_MIB` (default: 1024 MiB).
The pinned helper emits model training context only through its normal fitting
path at debug verbosity 5. QIIP accepts a nonzero fit status after that metadata
is present because the all-layer probe can legitimately exceed an undersized
GPU; missing or malformed training-context metadata still fails closed.

Managed launch passes the selected `--ctx-size`, `--parallel`, unified-KV, and
full-offload values explicitly and disables the server's second fitting pass.
The aggregate context is at least `context_per_slot * slots`; idle slots leave
their share available to active requests, while all slots can simultaneously
reach the guaranteed context. QIIP waits for `/health`, then refuses healthy
registration unless the runtime matches the plan, KV is unified, every model
layer was offloaded to GPU, and actual free VRAM still meets the target.

With unified KV, llama.cpp internally reports `n_ctx_seq` as the aggregate
pool. When that exceeds the model training context, b10242 emits its expected
`possible training context overflow` and slot-capping warnings, then caps each
request to the training context. QIIP validates those exact records as benign;
it still rejects `failed to fit params to free device memory`. The provisioning
record distinguishes `context_per_slot` (capacity guaranteed simultaneously to
every selected slot), `slot_context_limit` (llama.cpp's maximum for one
request), and `aggregate_context` (the unified pool).

The pinned b10242 estimator already implements unified-KV memory accounting but
does not expose that option in the `llama-fit-params` CLI allowlist. The
versioned `cuda-portable-cpu-v2-fit-concurrency` build profile exposes the
existing option so estimation and serving use the same KV mode. Its exact
transformation digest is part of `BUILD-INFO` and the installation identity,
and setup exercises every planner option against the built helper before
publication. A changed transformation or incompatible future source fails the
build closed.

Trace verbosity 4 is required for the pinned build's context and offload evidence
and remains enabled for the lifetime of the server. QIIP tails that log into the
bounded provisioning buffer only until `/health` succeeds, so startup trace can
evict earlier setup entries but later request traffic cannot. If measurement
shows the startup record exceeds the defaults, raise
`INFERENCE_PROXY_PROVISIONING__LOG_MAX_ENTRIES_PER_HOST` or
`INFERENCE_PROXY_PROVISIONING__LOG_MAX_BYTES_PER_HOST`. Runtime trace continues
to accumulate in `/var/log/llamacpp-serve.log`; include that file in the node's
normal log-rotation policy.

Standalone context size is divided across its configured parallel slots. If
`AUTOLLAMACPP_CTX_SIZE=8192` and `AUTOLLAMACPP_PARALLEL=4`, each slot gets 2048
tokens.

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
