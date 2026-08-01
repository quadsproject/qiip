# auto-vllm

Provision and run vLLM directly on bare-metal GPU nodes.

## Prerequisites

- NVIDIA GPU with driver installed (setup.sh handles this)
- NFS-mounted Hugging Face cache at `/srv/hf-cache`

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

The committed default artifact digests originate from the vendors' published
checksum files:

- NVIDIA 580.126.09: `NVIDIA-Linux-x86_64-580.126.09.run.sha256sum`
- LLMFit 1.1.6: `llmfit-v1.1.6-x86_64-unknown-linux-musl.tar.gz.sha256`

## Setup

Run the setup script to install drivers, Python, vLLM, mount NFS, and open the firewall:

```bash
./setup.sh
```

## Run

Start vLLM (auto-detects GPU and selects model):

```bash
./start-vllm.sh
```

vLLM runs as a background process. PID is written to `/var/run/vllm.pid`, logs to `/var/log/vllm-serve.log`.

## Health check

```bash
curl -s http://{HOSTNAME}:8000/health
```

## Stop

```bash
kill $(cat /var/run/vllm.pid)
```
