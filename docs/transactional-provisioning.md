# Transactional node provisioning

Engine installations and setup bundles are staged independently. Setup selects
them together only after verification. This applies to both vLLM and llama.cpp.

## Published files

The gateway snapshots the selected engine directory, shared scripts, recorder,
log store, and diagnostic collector before uploading. `BUNDLE.json` contains a
SHA-256 for every file. Its digest identifies the bundle. Uploads go to unique
`.qiip/bundles/.upload-*` directories in the SSH user's home. The node verifies
the complete file set, digests, shell syntax, and Python syntax before renaming
the directory to `.qiip/bundles/<digest>`. Existing completed bundles are
verified and reused. A damaged completed bundle is quarantined as `.corrupt-*`
and replaced by a verified upload under the publication lock; interrupted repair
can be retried. Recorder workers execute from their immutable bundle,
including imports; separate uploads cannot replace a running worker's code.

vLLM environments are created at their final paths under
`/opt/vllm-venv-generations/<identity>`. Entry-point shebangs retain those paths.
The identity includes the project and lock digests, Python and uv versions, and
selected runtime profile/toolkit. Frozen synchronization is allowed only for an
incomplete installation. Before completion, setup imports vLLM, Torch, and
FlashInfer, checks matching FlashInfer Python/cubin versions, and verifies
Python, ninja, and the managed `vllm serve` options.

llama.cpp installations retain the source digest, version, transformation,
build profile, compiler, OS/ABI, GPU capabilities, CUDA architecture/toolkit, and runtime profile
in `BUILD-INFO`. Setup checks the installed server and fit-planner CLI and seals
all three tools in `RUNTIME.json`. A sealed installation is verified and
reused. Failed verification does not overwrite a sealed installation.
New builds also seal a compiled CUDA execution probe, `BUILD-INFO`, and bundled
CUDA/compiler libraries. A pinned artifact catalog can supply these packages
without installing a compilation toolchain. Source and artifact installations
copy a complete verified package to an inactive staging directory on the
installation filesystem, flush it, and publish it under a lock before selection.
An interrupted copy leaves the active generation untouched. See the
[artifact producer and fallback guide](../auto-llamacpp/README.md#verified-build-artifacts).

`RUNTIME.json` records dependency/profile identity and executable digests.
External executable symlinks, including uv's RPM-managed Python interpreter,
record their link targets instead of hashing system files. Their targets must
remain executable; routine interpreter package updates do not invalidate the seal.
`GENERATION.json` joins the bundle, runtime manifest, and setup configuration.
Generation links and the vLLM service file are also verified. Generations live
under `/opt/qiip/<engine>/generations/<digest>`, where the engine is `vllm` or
`llama_cpp`. One atomic `current` symlink selects the entire generation;
`previous` retains the prior selection. Manifest writes and publication are
flushed before activation. Bundle publication and generation activation use
filesystem locks to serialize competing publishers.

Start and stop resolve `current` once, then use that generation's scripts and
absolute runtime paths. Stop verifies the selected scripts and configuration
without requiring the damaged or missing runtime to pass integrity checks.
Relaunch and teardown can upload a new bundle without
selecting it; they continue to use the installed generation. The vLLM systemd
unit dispatches through `current` even when systemd has the prior unit loaded.
Effective launch settings are saved atomically in that generation's `vllm.env`.
Only service dispatch restores these settings, consistently for all Exec lines,
including preflight. Gateway and manual starts use their supplied settings and
defaults. Setup clears the selected generation's saved settings, including when
reactivating the same generation. Other retained generations keep their settings.
Legacy nodes remain stoppable and relaunchable through the verified fallback
bundle until setup creates their first generation. Existing legacy runtimes are
left in place.

## Configuration and evidence

`provisioning.generation_root` defaults to `/opt/qiip` on the node and must be a
dedicated absolute directory. Standalone scripts accept `QIIP_GENERATION_ROOT`
for the engine directory. vLLM also accepts `AUTOVLLM_RUNTIME_ROOT` and
`AUTOVLLM_BOOTSTRAP_PYTHON`; llama.cpp retains `AUTOLLAMACPP_INSTALL_ROOT`.

Attempt manifests include `staged_bundle` and `selected_generation`, with the
generation ID, bundle digest, runtime identity/path, and setup configuration.
`bundle_version` retains the candidate setup bundle identity. A failed new
installation can record the prior selection without attributing its setup
failure to that older bundle.
The node recorder commits selection evidence and the gateway retrieves it
after a lost SSH acknowledgement. Diagnostics use the recorded runtime path,
so a later activation cannot substitute its binaries in an earlier attempt.
The immutable `recorder_path` is persisted with each attempt so collection works
after a gateway restart. Attempts without a selected generation explicitly report
that no runtime was selected rather than probing a legacy installation.

## Interrupted operations and rollback

A partial upload remains inactive. A failed package installation has no
completion manifest. Retries upload another complete snapshot and reconcile
incomplete runtime installations at their final paths. An interrupted activation
leaves `current` pointing to the whole prior or new verified generation. Retries
complete an interrupted switch and preserve the rollback pointer. Inactive
upload and generation staging directories may remain for inspection. Completed
bundles, installations, and previous generations are not automatically deleted.

To select the previous vLLM generation on the node:

```bash
python3 /opt/qiip/vllm/current/common/generations.py rollback /opt/qiip/vllm
systemctl daemon-reload
```

For llama.cpp, substitute `llama_cpp` for `vllm`. Rollback changes the selected
files; it does not restart an already-running server. Use the managed lifecycle
to stop and launch it with the desired model and runtime policy. The old
generation's persisted vLLM settings remain available.
An interrupted rollback retains its target in `ROLLBACK.json`; a retry or a
subsequent setup completes that switch under the activation lock.
Only the destination must pass verification; corruption in the generation being
abandoned does not prevent rollback or replacement activation.

## Validation scope

Controlled regressions run the shipped vLLM setup/start/stop scripts with
fixture GPU, package, and storage operations. Fault injection covers package
failure, import/CLI rejection, disk exhaustion, partial upload, and interruption
before and after the activation commit. Checks also cover retry, rollback,
sealed-file corruption, and node/gateway selection evidence after lost
acknowledgements. These establish code-level behavior; they do not claim GPU
fleet validation or reproduce filesystem exhaustion on a production node.
