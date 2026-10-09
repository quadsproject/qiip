# CHANGELOG

<!-- version list -->

## v0.2.0-dev.4 (2026-10-09)

### Chores

- Refresh uv.lock after version bump
  ([`4f2d2ef`](https://github.com/quadsproject/qiip/commit/4f2d2efb0abfee4d0d36bcdbd04efb0f9a941047))

### Features

- Auto-rediscover self-setup model drift in health cycle
  ([`6c66d25`](https://github.com/quadsproject/qiip/commit/6c66d25813fec78dc7da675fa943f8489f9392b7))


## v0.2.0-dev.3 (2026-10-08)

### Chores

- Refresh uv.lock after version bump
  ([`d501b3f`](https://github.com/quadsproject/qiip/commit/d501b3f77bf02b9a5ce5cad4eb093884ad847187))

### Features

- Add user leaderboard and token dashboard
  ([`b41d846`](https://github.com/quadsproject/qiip/commit/b41d846818f061810a2480e943611d16618d2a7b))


## Unreleased

### Features

- Leaderboard and token dashboard for normal users: usage ranking of
  non-admin users, own-token create/delete, and per-harness config download
  ([#156](https://github.com/quadsproject/qiip/issues/156))

### Documentation

- Add a user guide (`docs/user-guide.md`) covering sign-in, onboarding,
  tokens, and the leaderboard

## v0.2.0-dev.2 (2026-10-08)

### Bug Fixes

- Address llama.cpp artifact review findings
  ([`03736a1`](https://github.com/quadsproject/qiip/commit/03736a13d9f58ca2e49b1224ca004485d478d865))

- Address native CPU and artifact runtime review
  ([`556680c`](https://github.com/quadsproject/qiip/commit/556680c2327a83809fe897a304768f2f432f9faa))

### Chores

- Refresh uv.lock after version bump
  ([`c55c4cd`](https://github.com/quadsproject/qiip/commit/c55c4cd47ba6761af1e820e81d67a472dfe758ed))

### Features

- Reuse verified llama.cpp builds with bounded source fallback
  ([`2ebf719`](https://github.com/quadsproject/qiip/commit/2ebf7199b949abf54539af6692f893a1e22f1008))

## v0.2.0-dev.1 (2026-10-07)

### Bug Fixes

- Address GPU setup review findings ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`71aada2`](https://github.com/quadsproject/qiip/commit/71aada25870595039899bcdced281a19a6ddec71))

- Copy driver reboot markers directly ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`71aada2`](https://github.com/quadsproject/qiip/commit/71aada25870595039899bcdced281a19a6ddec71))

### Chores

- Absorb transient Chrome hangs in browser tests
  ([#215](https://github.com/quadsproject/qiip/pull/215),
  [`482fe78`](https://github.com/quadsproject/qiip/commit/482fe7833407a235fea9babd0298b9e0414d2e79))

- Drop duplicate dev release badge from README
  ([`f800bba`](https://github.com/quadsproject/qiip/commit/f800bba67f4c2e2eb85a520ce1dc41575aacb74b))

### Features

- Reuse compatible GPU drivers during setup ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`71aada2`](https://github.com/quadsproject/qiip/commit/71aada25870595039899bcdced281a19a6ddec71))


## v0.1.0 (2026-10-05)

- Initial Release

## v0.1.0-dev.8 (2026-09-29)

### Bug Fixes

- Disable uv cache in semantic release job
  ([`9190735`](https://github.com/quadsproject/qiip/commit/9190735fb3fc2fc5bb58acdcde918e321a4899ad))

### Chores

- Enable copr-deps dispatch from main
  ([`e37765a`](https://github.com/quadsproject/qiip/commit/e37765a18a77d5eba1074b8bbf642cd37582eeb9))

- Refresh uv.lock after version bump
  ([`02d3037`](https://github.com/quadsproject/qiip/commit/02d3037224f6449a84a645dbc4e386842033ebc1))


## v0.1.0-dev.7 (2026-09-29)

### Chores

- Refresh uv.lock after version bump
  ([`ea52ab6`](https://github.com/quadsproject/qiip/commit/ea52ab6841d01d8277c95a5dce70d379bb0ab503))

### Features

- QUADS-style nginx and uvicorn tuning in RPM service
  ([`cc7394d`](https://github.com/quadsproject/qiip/commit/cc7394d213123455b29ed85b56e773acb6220759))


## v0.1.0-dev.6 (2026-09-29)

### Bug Fixes

- Commit uv.lock refresh with git identity
  ([`ad04fd0`](https://github.com/quadsproject/qiip/commit/ad04fd0f5cb9af49f4797422c7cefbbb700d00fe))

### Chores

- Refresh uv.lock after version bump
  ([`21c307c`](https://github.com/quadsproject/qiip/commit/21c307c288184b399a560edb33bc8060824de073))


## v0.1.0-dev.5 (2026-09-29)

### Bug Fixes

- Strip hf-xet from huggingface-hub build deps
  ([`7fe8250`](https://github.com/quadsproject/qiip/commit/7fe82508e74ae6c026b5c028e2d2f4a362f8413b))

### Chores

- Refresh uv.lock after version bump
  ([`57a1c1e`](https://github.com/quadsproject/qiip/commit/57a1c1ea226d7762e5b2754f8821405aff5afbf3))


## v0.1.0-dev.4 (2026-09-29)

### Bug Fixes

- Use consistent readable model labels
  ([`e7678c5`](https://github.com/quadsproject/qiip/commit/e7678c5fe5de612f84f69e0e8717b6ac68c40098))


## v0.1.0-dev.3 (2026-09-29)

### Bug Fixes

- Preflight COPR dependency RPMs in release
  ([`1866c74`](https://github.com/quadsproject/qiip/commit/1866c7410c95b49efb3ef96fcbeeb236db5eef34))

- Refresh uv.lock after each semantic release
  ([`03a70a0`](https://github.com/quadsproject/qiip/commit/03a70a029791cd64dc6095805b9427fd747ca9ba))

### Chores

- Refresh uv.lock after version bump
  ([`436190e`](https://github.com/quadsproject/qiip/commit/436190ececbd1adfd813eb45630261a22d9cbba6))


## v0.1.0-dev.2 (2026-09-29)

### Bug Fixes

- Publish all RPMs to the quadsdev/qiip COPR project
  ([`d17428a`](https://github.com/quadsproject/qiip/commit/d17428a8921fea02645ddb3162c2febe949d5ef6))

- Single-line dnf install in release workflow
  ([`49ea288`](https://github.com/quadsproject/qiip/commit/49ea288501fd7cada59c8f684e08e939461f718f))

### Chores

- Refresh uv.lock after version bump
  ([`a6148e4`](https://github.com/quadsproject/qiip/commit/a6148e412a6c8f244b713138a0b9d231e86a36ac))


## v0.1.0-dev.1 (2026-09-29)

- Initial Release
