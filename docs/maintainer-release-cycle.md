# Maintainer release cycle: development to main

This document is for QIIP repository maintainers. It covers the once-per-release
cycle that moves `development` into `main` and leaves both branches ready for
the next prerelease.

Scope and assumptions:

- The target hosts are already on a supported Fedora release where the RPM
  layout is viable. Host installation and provisioning are out of scope.
- `main` is protected. `development` is not.
- The repository uses rebase merges only. Do not squash and do not create merge
  commits.
- The Release workflow runs on every push to `main` and `development`. It uses
  python-semantic-release and opens the stable release PR after the development
  sync lands on `main`.
- A maintainer has write access to `quadsproject/qiip` and an authenticated
  `gh` CLI.

## Why this cycle

- Rebase merges preserve every reviewed commit on `main`. Squashing would
  collapse the development work into one opaque commit and erase the review
  and semantic-release history.
- The two release trains stay aligned. `development` publishes `vX.Y.Z-dev.N`
  prereleases and `main` publishes `vX.Y.Z` stable releases. Resetting
  `development` to `main` after the stable tag bases the next prerelease on
  the released code.
- A merge commit or a squash creates SHAs that `development` does not contain.
  Without the reset, the next prerelease repeats commits or misses the stable
  tag.

The cycle has four phases:

1. [Rewrite](#1-rewrite-development-onto-main)
2. [Sync pull request](#2-open-and-rebase-merge-the-sync-pr)
3. [Release pull request](#3-merge-the-release-pr)
4. [Reset development](#4-reset-development-to-main)

Freeze development merges for the duration of the cycle. Do not land new
`development` commits after the rewrite until the reset is complete.

## 1. Rewrite development onto main

Start from a clean checkout and fetch the current branches and tags.

```bash
git fetch origin --tags
git status --short
git checkout development
git rebase origin/main
```

Resolve conflicts as follows:

- `CHANGELOG.md`: keep both sides. The release workflow rewrites the changelog
  from git history after the merge.
- `pyproject.toml`: keep the stable version from `main`. Do not carry a
  `-dev` version into `main`; the release workflow computes the next version.
- `uv.lock`: after resolving `pyproject.toml`, run `uv lock` and include the
  result.
- Other conflicts: resolve normally, then run the repository checks.

Verify the rewrite before publishing it:

```bash
git log --oneline --decorate origin/main..development | head
uv lock --check
```

Push the rewritten branch with a lease:

```bash
git push --force-with-lease origin development
```

This is the one authorized rewrite of `development` in the cycle. If the push
is rejected because someone else pushed, fetch and rebase again. Do not force
over commits you have not reviewed.

## 2. Open and rebase-merge the sync PR

```bash
gh pr create --repo quadsproject/qiip \
  --base main --head development \
  --title "chore: sync development into main" \
  --body "Release sync: rebased development onto main. No new development merges until the release reset completes."
```

Wait for the required checks, then merge with rebase:

```bash
gh pr checks <pr-number> --repo quadsproject/qiip --watch
gh pr merge <pr-number> --repo quadsproject/qiip --rebase
```

Do not squash. Do not merge locally with a merge commit. The repository history
must stay linear.

After the merge, `main` contains rewritten SHAs, so the local `development`
branch is stale. Do not push to it again until the reset phase.

## 3. Merge the release PR

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
git checkout development
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

- Rebase conflicts are not resolvable: abort with `git rebase --abort`, fix the
  branch state, and restart the phase. Do not force-push a partial rewrite.
- The sync PR cannot merge because `main` advanced: fetch, rebase
  `development` onto the new `origin/main`, force-push, and re-run checks.
- The release PR is merged with squash by mistake: stop the cycle. The history
  is no longer linear; restore the intended head through the repository's
  normal review process before resetting development.
- The reset happened before the release tag existed: fetch tags, reset
  `development` to `origin/main` again, and force-push before any new
  development commits land.
- A force-push is rejected: someone else updated the branch. Fetch, inspect,
  and rebase. Never use a plain `--force`.

## Checklist

- [ ] Development merges frozen for the cycle.
- [ ] `development` rebased onto `origin/main`.
- [ ] `CHANGELOG.md`, `pyproject.toml`, and `uv.lock` conflicts resolved.
- [ ] Checks pass on the rewritten branch.
- [ ] Sync PR merged with rebase.
- [ ] Release PR merged with rebase.
- [ ] `v<version>` tag and GitHub release exist.
- [ ] Development reset to `origin/main`.
- [ ] `origin/main..origin/development` and `origin/development..origin/main`
      are empty.
- [ ] Development merges resumed.
