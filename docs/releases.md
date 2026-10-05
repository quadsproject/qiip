# QIIP releases: stable and development trains

QIIP publishes two release trains that map to two COPR RPMs:

| Train | Branch | GitHub releases | COPR project | RPM |
|-------|--------|-----------------|--------------|-----|
| Stable | `main` | `vX.Y.Z` (release) | `quadsdev/qiip` | `qiip` |
| Development | `development` | `vX.Y.Z-dev.N` (prerelease) | `quadsdev/qiip` | `qiip-dev` |

Both trains are driven by the `Release` workflow (`.github/workflows/release.yml`)
using [python-semantic-release](https://python-semantic-release.readthedocs.io/).
A push or merge to either branch runs the pipeline; pull requests never do.

## Stable release flow

`main` requires the Quality and Python 3.13 checks before any push, so the
version bump cannot be pushed directly. On a push to `main` the workflow runs
python-semantic-release locally (commit and tag only), refreshes `uv.lock`,
pushes a `release-<version>` branch, and opens a `chore: release <version>`
pull request against `main`. The PR carries the required checks. Merging it
re-triggers the workflow, which then creates the `v<version>` tag and the
GitHub release and submits the `qiip` COPR build from that tag (the tag is
created at the merge head, so the COPR build sees exactly what `main` has).
The post-merge run detects that the stable version is already on `main` with
no tag yet and runs python-semantic-release without commit or changelog
generation, so the reviewed merge head is tagged as-is. The organization
disallows Actions-created pull requests, so opening the release PR needs a
`RELEASE_TOKEN` secret (a release identity with `repo` scope) or the
organization setting allowing it.
Development train releases still push directly; `development` is not
protected.

## Development train sync

The repo allows only squash or rebase merges, and both create new SHAs. After
a `development` -> `main` merge, `development` therefore does not contain the
stable release commit or its `v<version>` tag. python-semantic-release would
then keep computing the same prerelease as already released and release
nothing (the stale `v0.1.0-dev.*` tags from before any history rewrite have
the same effect). Reset `development` to `main` right after the merge:

```bash
git checkout development
git reset --hard origin/main
git push --force-with-lease origin development
```

The next development release is then `v0.2.0-dev.1` (the next minor after
`main`'s version). Do the reset only immediately after the merge, before new
development commits land, and never rewrite `development` afterwards: the
dev tags must stay reachable or the prerelease revision stops advancing.

## Versioning rules

- `feat:` commits bump the minor version; `fix:`/`perf:` bump the patch;
  breaking changes (`feat!:`) bump the minor while QIIP is at 0.x.
- `chore:`/`docs:`/`refactor:` and markdown-only changes produce no release:
  no tag, no GitHub release, no COPR build.
- The development train releases prereleases of the next version
  (`v0.2.0-dev.1`, `v0.2.0-dev.2`, ...), so its tags never collide with the
  stable train's `v0.2.0`.
- When `development` is merged into `main`, the next stable release
  finalizes the prerelease: `v0.2.0-dev.N` lands as `v0.2.0` through the
  release pull request described above.
- First stable release is `v0.1.0`.
- python-semantic-release maintains `CHANGELOG.md` on both trains. Merging
  `development` into `main` can conflict on the changelog and the version
  line in `pyproject.toml`; resolve by keeping both changelog entries (the
  next release run rewrites them from git history).

## COPR RPMs

The RPMs are built from the release tag on merge only, in the
`quadsdev/qiip` COPR project. Six dependencies that Fedora does not carry
at the version QIIP needs (or at all) are prebuilt into the same project:
see [`copr-deps/README.md`](../copr-deps/README.md). The runtime installs
system packages only; nothing uses pip or uv.

Enable and install:

```bash
sudo dnf copr enable quadsdev/qiip
sudo dnf install qiip
# development train (same repository; cannot sit beside the stable RPM):
sudo dnf install qiip-dev
```

`qiip-dev` and `qiip` own the same files (`/usr/share/qiip`, the systemd
unit, `/etc/qiip`), so `qiip-dev` conflicts with `qiip`; use one train per
host.

## Badges

The README badges are live: GitHub release badges come from the `Release`
workflow (stable release and latest dev prerelease), and the COPR badges
show the last build of each package. The `quadsdev/qiip` project builds
Fedora 43/44 only; EPEL 10 / AlmaLinux 10 chroots are intentionally not
enabled: their Python packages are below QIIP's dependency floors and
several are missing entirely (revisit when upstream packaging catches up).
