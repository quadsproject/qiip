# Maintainer release cycle: development to main

This document is for QIIP repository maintainers. It covers the once-per-release
cycle that moves `development` into `main` and leaves both branches ready for
the next prerelease.

Scope and assumptions:

- The target hosts are already on a supported Fedora release where the RPM
  layout is viable. Host installation and provisioning are out of scope.
- `main` is protected. `development` is not.
- This repository rejects merge commits and permits rebase merges for this
  cycle. Do not squash. Do not create a merge commit.
- GitHub cannot rebase-merge a branch that contains merge commits. If
  `development` contains one, replay its non-merge commits onto a new linear
  sync branch based on `main`. Do not squash the work into one commit.
- The Release workflow runs on every push to `main` and `development`. It uses
  python-semantic-release and opens the stable release PR after the development
  sync lands on `main`.
- A maintainer has write access to `quadsproject/qiip` and an authenticated
  `gh` CLI.

## Why this cycle

- Rebase merges preserve every reviewed commit as its own commit on `main`.
  Squashing collapses the work into one opaque commit and erases the review and
  semantic-release history.
- The two release trains stay aligned. `development` publishes
  `vX.Y.Z-dev.N` prereleases and `main` publishes `vX.Y.Z` stable releases.
  Resetting `development` to `main` after the stable tag bases the next
  prerelease on the released code.
- A linear sync branch avoids rewriting `development` and its prerelease tags
  before the stable release. The merge is still a rebase merge, and every
  unique commit keeps its author and message. Commit SHAs change when the
  commits are replayed.
- A merge commit cannot be rebase-merged by GitHub and is rejected by this
  repository. Squash is not an acceptable substitute.

The cycle has four phases:

1. [Create the linear sync branch](#1-create-the-linear-sync-branch)
2. [Open and rebase-merge the sync PR](#2-open-and-rebase-merge-the-sync-pr)
3. [Merge the release PR](#3-merge-the-release-pr)
4. [Reset development](#4-reset-development-to-main)

Freeze development merges for the duration of the cycle. Do not land new
`development` commits after the sync branch is verified until the release reset
is complete.

## 1. Create the linear sync branch

Start from a clean checkout and fetch the current branches and tags.

```bash
git fetch origin --tags
git status --short
VERSION=0.2.0
SYNC="release-sync-$VERSION"
```

Create the third branch from `main`, then replay every non-merge commit from
`main..development` onto it:

```bash
git switch -c "$SYNC" origin/main
git rev-list --reverse --no-merges origin/main..origin/development \
  | git cherry-pick --stdin
```

A commit whose patch is already on `main` can replay as empty, as happened
with the duplicate browser-test fix that `main` received through another PR.
Git will stop and offer `git cherry-pick --skip`; skip that duplicate. Resolve
genuine conflicts normally. Do not squash and do not create a merge commit.

Restore the stable version and lockfile from `main`, then refresh the lockfile
against the replayed code:

```bash
git checkout origin/main -- pyproject.toml
uv lock
git add pyproject.toml uv.lock
git commit -m "chore: restore stable version for the sync branch"
```

If the version line itself changed during the replay, keep the version that
`main` had before the sync. The Release workflow computes the next stable
version after the sync lands.

Verify the branch before pushing it:

```bash
git log --merges --oneline origin/main.."$SYNC"      # must be empty
git log --oneline --decorate origin/main.."$SYNC" | head
git diff --stat origin/development -- . \
  ':(exclude)pyproject.toml' ':(exclude)uv.lock'    # must be empty
uv lock --check
```

The branch must contain every unique development commit, no merge commits, the
authors and messages intact, and a tree identical to `development` except for
`pyproject.toml` and `uv.lock`. Push the new branch:

```bash
git push -u origin "$SYNC"
```

Do not force-push `development` in this phase. Keeping `development` unchanged
preserves its prerelease tags until the reset after the stable release.

## 2. Open and rebase-merge the sync PR

Open the sync PR from the third branch:

```bash
gh pr create --repo quadsproject/qiip \
  --base main --head "$SYNC" \
  --title "chore: sync development into main for v$VERSION" \
  --body "Linear replay of development's non-merge commits onto main. No squash; no merge commits."
```

If an older sync PR exists that cannot be rebase-merged, close it as superseded
and link the new PR in the close comment. Its discussion and commit history
remain available for review.

Wait for the required checks, then merge with rebase:

```bash
gh pr checks <sync-pr-number> --repo quadsproject/qiip --watch
gh pr merge <sync-pr-number> --repo quadsproject/qiip --rebase
```

Do not squash. Do not create a merge commit. The repository history must stay
linear.

After the merge, `main` contains the replayed SHAs, so the local `development`
branch is stale. Do not push to it again until the reset phase.

## 3. Merge the release PR

Confirm that `RELEASE_TOKEN` or the equivalent organization credential is
available. The stable release PR cannot be opened with `GITHUB_TOKEN` when the
organization disallows Actions-created pull requests.

The push to `main` triggers the Release workflow. It computes the stable
version and opens a `release-<version>` pull request against `main`.

Find it and wait for checks:

```bash
gh pr list --repo quadsproject/qiip --base main --state open \
  --json number,title,headRefName
gh pr checks <release-pr-number> --repo quadsproject/qiip --watch
```

Merge it with rebase:

```bash
gh pr merge <release-pr-number> --repo quadsproject/qiip --rebase
```

The post-merge Release run creates the `v<version>` tag, the GitHub release,
and the COPR build. Wait for the tag to exist before continuing:

```bash
gh release view v<version> --repo quadsproject/qiip
git fetch origin --tags
git rev-parse "v<version>^{commit}"
```

If the release run fails, fix the cause through a normal pull request against
`main`. Do not rewrite `main` and do not reset `development` until the tag and
release exist.

## 4. Reset development to main

After the stable release is tagged, reset `development` to the new `main` so it
contains the release commit and tag:

```bash
git fetch origin --tags
git switch development
git reset --hard origin/main
git push --force-with-lease origin development
```

Verify the reset:

```bash
git merge-base --is-ancestor "v<version>" origin/development
git log --oneline origin/development..origin/main
git log --oneline origin/main..origin/development
```

The last two commands should print nothing immediately after the reset. The
first command should exit 0.

Resume normal development merges only after this check passes. Do not rewrite
`development` again until the next release cycle.

## Failure recovery

- A replay conflict cannot be resolved: abort with `git cherry-pick --abort`,
  fix the source or the resolution, and recreate the sync branch from
  `origin/main`. Do not force-push a partial branch.
- A replay stops on an empty commit: inspect it. If its patch is already on
  `main` under another SHA, `git cherry-pick --skip`; otherwise investigate
  before continuing.
- The sync branch contains a merge commit: recreate it from `origin/main`;
  GitHub will not rebase-merge it.
- `main` advances before the sync PR merges: rebase the sync branch onto the
  new `origin/main`, force-push only the sync branch, and re-run checks. Never
  force-push `development` during this phase.
- The release PR is merged with squash by mistake: stop the cycle. The history
  is no longer linear; restore the intended head through the repository's
  normal review process before resetting development.
- The reset happened before the release tag existed: fetch tags, reset
  `development` to `origin/main` again, and force-push before any new
  development commits land.
- A force-push is rejected: someone else updated the branch. Fetch, inspect,
  and retry. Never use a plain `--force`.

## Checklist

- [ ] Development merges frozen for the cycle.
- [ ] Linear sync branch created from `origin/main`.
- [ ] All non-merge development commits replayed; authors and messages intact.
- [ ] Patch-identical commits already on `main` skipped and documented.
- [ ] No merge commits on the sync branch.
- [ ] Stable `pyproject.toml` version restored and `uv.lock` refreshed.
- [ ] Sync branch tree differs from development only in the version files.
- [ ] Checks pass on the sync branch.
- [ ] Sync PR merged with rebase.
- [ ] Release PR merged with rebase.
- [ ] `v<version>` tag and GitHub release exist.
- [ ] COPR build published.
- [ ] Development reset to `origin/main`.
- [ ] `origin/main..origin/development` and `origin/development..origin/main`
      are empty.
- [ ] Development merges resumed.
