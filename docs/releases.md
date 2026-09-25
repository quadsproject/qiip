# QIIP releases: stable and development trains

QIIP publishes two release trains that map to two COPR RPMs:

| Train | Branch | GitHub releases | COPR project | RPM |
|-------|--------|-----------------|--------------|-----|
| Stable | `main` | `vX.Y.Z` (release) | `quadsdev/qiip` | `qiip` |
| Development | `development` | `vX.Y.Z-dev.N` (prerelease) | `quadsdev/qiip-dev` | `qiip-dev` |

Both trains are driven by the `Release` workflow (`.github/workflows/release.yml`)
using [python-semantic-release](https://python-semantic-release.readthedocs.io/).
A push or merge to either branch runs the pipeline; pull requests never do.

## Versioning rules

- `feat:` commits bump the minor version; `fix:`/`perf:` bump the patch;
  breaking changes (`feat!:`) bump the minor while QIIP is at 0.x.
- `chore:`/`docs:`/`refactor:` and markdown-only changes produce no release:
  no tag, no GitHub release, no COPR build.
- The development train releases prereleases of the next version
  (`v0.2.0-dev.1`, `v0.2.0-dev.2`, ...), so its tags never collide with the
  stable train's `v0.2.0`.
- When `development` is merged into `main`, the next stable release
  finalizes the prerelease: `v0.2.0-dev.N` lands as `v0.2.0`.
- First stable release is `v0.1.0`.
- python-semantic-release maintains `CHANGELOG.md` on both trains. Merging
  `development` into `main` can conflict on the changelog and the version
  line in `pyproject.toml`; resolve by keeping both changelog entries (the
  next release run rewrites them from git history).

## COPR RPMs

The RPMs are built from the release tag on merge only. Both require the
`quadsdev/qiip-deps` COPR repository for six dependencies that Fedora does
not carry at the version QIIP needs (or at all): see
[`copr-deps/README.md`](../copr-deps/README.md). The runtime installs system
packages only; nothing uses pip or uv.

Enable and install:

```bash
sudo dnf copr enable quadsdev/qiip-deps      # one-time; dependency RPMs
sudo dnf copr enable quadsdev/qiip           # stable train
sudo dnf install qiip
# development train (cannot sit beside the stable RPM):
sudo dnf copr enable quadsdev/qiip-dev
sudo dnf install qiip-dev
```

`qiip-dev` and `qiip` own the same files (`/usr/share/qiip`, the systemd
unit, `/etc/qiip`), so `qiip-dev` conflicts with `qiip`; use one train per
host.

## Badges

The README badges are live: GitHub release badges come from the `Release`
workflow (stable release and latest dev prerelease), and the COPR badges
show the last build of each project. Both COPR projects build Fedora 43/44
only; EPEL 10 / AlmaLinux 10 chroots are intentionally not enabled (their
Python packages are below QIIP's dependency floors and several are missing
entirely; revisit when upstream packaging catches up).
