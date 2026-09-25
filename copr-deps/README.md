# QIIP COPR dependency project

System packages only: the qiip RPM must never install Python dependencies via
pip or uv. Most runtime dependencies are packaged in Fedora 43/44; the
specs here cover the three that are not (or are below the versions QIIP
needs), built into the COPR project `quadsdev/qiip-deps` and enabled as an
additional repository in the `quadsdev/qiip` and `quadsdev/qiip-dev`
projects.

| Spec | Why |
|------|------|
| `structlog.spec` | structlog is retired from Fedora (no package in f43-f45); QIIP needs >= 26.1.0. |
| `huggingface-hub.spec` | Fedora 43/44 ship 0.30.2 / 1.24.0; QIIP needs >= 1.25 (uses `huggingface_hub.errors.IncompleteSnapshotError`). |
| `fastapi.spec` | Fedora 43 ships 0.127.1; QIIP needs >= 0.135 (`fastapi.sse`), and 0.135 is the first release carrying it. |
| `etcd3gw.spec` | Fedora 43/44 ship 2.4.1 / 2.5.0; QIIP's floor is 2.7.0 (guarded by a repo test: the exercised watch/revision/lease API needs 2.7). |
| `uvicorn.spec` | Fedora 43/44 ship 0.38.0 / 0.40.0; QIIP needs >= 0.45. |
| `click.spec` | huggingface-hub >= 1.25 requires click >= 8.4.2; Fedora 43/44 ship 8.1.7. |

Nothing else: Fedora 43/44 already satisfy the remaining floors
(pydantic 2.12.5, pydantic-settings 2.15.0, httpx 0.28.1, asyncssh,
authlib, itsdangerous, jinja2, PyYAML, httpx-sse). All transitive
dependencies of these six (starlette, typing-inspection, annotated-doc,
fsspec, tqdm, filelock, packaging, requests, pbr, futurist, h11) are also
packaged in Fedora 43+. One deliberate omission: `hf-xet` (a Rust extension
that huggingface-hub 1.25 requires on x86_64) has no Fedora provider; QIIP
only downloads models, and huggingface_hub falls back to plain HTTPS with a
warning when `hf_xet` is absent (the Xet upload path is unused by QIIP).

EL10/EPEL10 and AlmaLinux 10 chroots are intentionally not supported yet:
they additionally lack etcd3gw, httpx-sse and huggingface-hub, and ship
fastapi/pydantic/uvicorn/pydantic-settings well below the floors. Revisit
when upstream packaging catches up.

## Publish once, bump on demand

The six versions here are pinned and published to `quadsdev/qiip-deps`
once; users never build anything, they only enable the repo (`dnf copr
enable quadsdev/qiip-deps`). The qiip/qiip-dev COPR builds consume these
RPMs as-is; they are not rebuilt per qiip release.

To publish (initial or after bumping a pinned version), use the `COPR
dependencies` workflow manually (`.github/workflows/copr-deps.yml`,
workflow_dispatch), or the commands below on a Fedora host with
`rpm-build`, `pyproject-rpm-macros`, and `pip`:

```bash
cd copr-deps

# Fetch each Source0 sdist locally so the SRPM embeds it (COPR builds the
# SRPM, it does not download sources for it).
for pkg in structlog huggingface-hub fastapi etcd3gw uvicorn click; do
    version="$(awk '/^Version: /{print $2; exit}' "${pkg}.spec")"
    pip download --quiet --no-deps --no-binary=:all: "${pkg}==${version}" -d .
done

# one SRPM per spec
for spec in *.spec; do
    rpmbuild -bs --define "_sourcedir $PWD" --define "_srcrpmdir $PWD" "$spec"
done

# submit to the dependency project
for srpm in *.src.rpm; do
    copr build quadsdev/qiip-deps "$srpm"
done
```

The sdist sources are downloaded from PyPI and embedded in the SRPMs; no
sources are kept in git.

## Enable for the qiip projects

In the COPR project settings for `quadsdev/qiip` and `quadsdev/qiip-dev`:

1. Add the qiip-deps repository to the fedora-43/44 chroots:
   `https://download.copr.fedorainfracloud.org/results/quadsdev/qiip-deps/fedora-$releasever-$basearch/`
2. Remove the `epel-10-x86_64` and `almalinux-10-x86_64_v2` chroots (not
   supported, see above).

Users installing the RPMs need the same two repos enabled:

```bash
sudo dnf copr enable quadsdev/qiip-deps
sudo dnf copr enable quadsdev/qiip
# or: sudo dnf copr enable quadsdev/qiip-dev (development train)
sudo dnf install qiip
```

## Updating

When a dependency needs a newer version (e.g. structlog 26.x lands fixes),
bump `Version:` in the spec, rebuild, and submit again to
`quadsdev/qiip-deps`; the qiip builds then pick it up through the repo.
