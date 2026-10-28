# CHANGELOG

<!-- version list -->

## v0.2.0 (2026-10-28)

### Bug Fixes

- Address 0.2.0 roll-up review findings
  ([`a468960`](https://github.com/quadsproject/qiip/commit/a468960a755712fae690a1b930ee86339cf0c9ab))

- Address GPU setup review findings ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`9666bfd`](https://github.com/quadsproject/qiip/commit/9666bfd668854f682842b08e4b2f5071c1914c96))

- Address llama.cpp artifact review findings
  ([`b3745c8`](https://github.com/quadsproject/qiip/commit/b3745c8ac9e38728dd5ba51d06aea0ed2999ae93))

- Address local OIDC provider review findings
  ([`40a9228`](https://github.com/quadsproject/qiip/commit/40a9228dc1ca00326db36f2231ef3fb26fbc797b))

- Address native CPU and artifact runtime review
  ([`f46a47c`](https://github.com/quadsproject/qiip/commit/f46a47c6ef860439cde47df8b81d1fe410c5eb35))

- Address review findings on redfish power gate and bmc resolver
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Copy driver reboot markers directly ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`9666bfd`](https://github.com/quadsproject/qiip/commit/9666bfd668854f682842b08e4b2f5071c1914c96))

- Document new setup settings and satisfy strict typing
  ([`13a454e`](https://github.com/quadsproject/qiip/commit/13a454ef8aae2b00021e063176eb308ebd50944f))

- Drop the non-admin fleet client path
  ([`7d50717`](https://github.com/quadsproject/qiip/commit/7d50717cbbffc35c79e81817ac6dcb5c9c5845c6))

- Enforce the stale bound only where placement runs
  ([`e2f5f00`](https://github.com/quadsproject/qiip/commit/e2f5f00ac88d20e6bfa65021b0cde4bb6cae06ce))

- Gate the Google sign-in option on its provider
  ([`ff436ce`](https://github.com/quadsproject/qiip/commit/ff436cee6bcd662040c7545b16d2ae20fadcb231))

- Give admins the leaderboard ranking
  ([`01bd37f`](https://github.com/quadsproject/qiip/commit/01bd37f07a6fa7bb7078f2de159e97a51445cbe5))

- Guard Copr publishes against duplicate NVRs
  ([`118ea2b`](https://github.com/quadsproject/qiip/commit/118ea2b1ee303ae8ef64671e666a3c485e9bd15f))

- Guard the same-issuer refresh and check the ID-token audience
  ([`cc3b2c4`](https://github.com/quadsproject/qiip/commit/cc3b2c40905a6ef797c09c4bc8cc65d615d2b967))

- Harden the issuer migration crash recovery
  ([`1f778e2`](https://github.com/quadsproject/qiip/commit/1f778e2e9a3ae1d3ff33918e70495392299aed75))

- Price token estimates at Sonnet 5 rates
  ([`9f273dd`](https://github.com/quadsproject/qiip/commit/9f273dd2b31bd4200161b6ae18d024b5214654e7))

- Require quads.stale_after_seconds to be at least twice poll_interval
  ([`b1cd32d`](https://github.com/quadsproject/qiip/commit/b1cd32d4bca09a1f060efaa75753d12e03339d4a))

- Run the chat playground on the user's token
  ([`78c9d22`](https://github.com/quadsproject/qiip/commit/78c9d22a7d8c9a344daf430bc21ff2c05a88ce55))

- Tell a Copr outage apart from missing dependency RPMs
  ([`10c3af6`](https://github.com/quadsproject/qiip/commit/10c3af61d78d86b6ca56804679496979f81ba19f))

- Unify the account cluster across the top bars
  ([`ef8f040`](https://github.com/quadsproject/qiip/commit/ef8f04081dadf8bbfb89ff7beca9c0ea15c20077))

- **provisioning**: Address second review round
  ([`99708ca`](https://github.com/quadsproject/qiip/commit/99708ca609b7edf22484c926fa251338447b283a))

- **provisioning**: Address setup resume review feedback
  ([`d9edd07`](https://github.com/quadsproject/qiip/commit/d9edd0754c0ddfcc8406d78d939839d7fe441742))

- **provisioning**: Retry bare 5xx download failures
  ([`424837f`](https://github.com/quadsproject/qiip/commit/424837f963d72904387747f02c7818db5cf10d2d))

- **vllm**: Address review on memory-aware sizing
  ([`df589f3`](https://github.com/quadsproject/qiip/commit/df589f35a48bda553cb834bd5005998daadffb21))

- **vllm**: Address the second review round on memory-aware sizing
  ([`12f6fea`](https://github.com/quadsproject/qiip/commit/12f6feab757471acbe05b5274a6bfea16103b404))

- **vllm**: Keep the retried plan reason consistent with the effective context
  ([`4b3de5f`](https://github.com/quadsproject/qiip/commit/4b3de5fbc2d19553a5e8e04f07b1faa487de7732))

### Chores

- Absorb transient Chrome hangs in browser tests (#215)
  ([#216](https://github.com/quadsproject/qiip/pull/216),
  [`25c3f7d`](https://github.com/quadsproject/qiip/commit/25c3f7d911100343880240d8cdd641d6153f5467))

- Drop duplicate dev release badge from README
  ([`f800bba`](https://github.com/quadsproject/qiip/commit/f800bba67f4c2e2eb85a520ce1dc41575aacb74b))

- Refresh uv.lock after version bump
  ([`e78bb2d`](https://github.com/quadsproject/qiip/commit/e78bb2db1ff7dcca9459acdaf97d487be1f9313f))

- Refresh uv.lock after version bump
  ([`952cc33`](https://github.com/quadsproject/qiip/commit/952cc33a87c2230500ab5d421852b41f0328a7a4))

- Refresh uv.lock after version bump
  ([`67ec929`](https://github.com/quadsproject/qiip/commit/67ec9291bff747ee57edc7f959891ce0948beb8b))

- Refresh uv.lock after version bump
  ([`670e94e`](https://github.com/quadsproject/qiip/commit/670e94efca795783252f4bbcb96613a8f6a43408))

- Refresh uv.lock after version bump
  ([`6791fa6`](https://github.com/quadsproject/qiip/commit/6791fa67138fb6d994903444222aa48f3502e669))

- Refresh uv.lock after version bump
  ([`3585431`](https://github.com/quadsproject/qiip/commit/35854316fb57bdb9c90af588e4e04cc70a2e9e4a))

- Refresh uv.lock after version bump
  ([`99d8201`](https://github.com/quadsproject/qiip/commit/99d8201af31ac1bc57102a4bb3da678e4b2ed873))

- Refresh uv.lock after version bump
  ([`59f704f`](https://github.com/quadsproject/qiip/commit/59f704ffd7aa3b90a983e74320cf553361846d85))

- Refresh uv.lock after version bump
  ([`180352d`](https://github.com/quadsproject/qiip/commit/180352d2d0159dd1e14142d06a2cd01eec1be1ba))

- Refresh uv.lock after version bump
  ([`0496752`](https://github.com/quadsproject/qiip/commit/04967527f24d3d496c93957eaacf176fc47ac2bc))

- Refresh uv.lock after version bump
  ([`441c732`](https://github.com/quadsproject/qiip/commit/441c732fc37b05ac8a03eba8a438d1e3aba53a6e))

- Refresh uv.lock after version bump
  ([`7c0b199`](https://github.com/quadsproject/qiip/commit/7c0b19902146dc396468a87bfc085c3986e6780a))

- Refresh uv.lock after version bump
  ([`912a104`](https://github.com/quadsproject/qiip/commit/912a104dee3a2f8b5b239333985eff23a16e0aa6))

- Refresh uv.lock after version bump
  ([`d7e5102`](https://github.com/quadsproject/qiip/commit/d7e5102ea9623ab4ddef2d9eaba5c781aa230a7a))

- Refresh uv.lock after version bump
  ([`7ae4097`](https://github.com/quadsproject/qiip/commit/7ae4097be2dcea2763734fc613775622c887082a))

- Refresh uv.lock after version bump
  ([`d5b291d`](https://github.com/quadsproject/qiip/commit/d5b291de10001e7be650aec6958a3a06529a8fb2))

- Refresh uv.lock after version bump
  ([`a94a1e9`](https://github.com/quadsproject/qiip/commit/a94a1e913926832d0ab60fbd6a49a98750243801))

- Refresh uv.lock after version bump
  ([`042781d`](https://github.com/quadsproject/qiip/commit/042781d19d3ac76b6235b5f56a3a647e4088494c))

- Refresh uv.lock after version bump
  ([`c56a684`](https://github.com/quadsproject/qiip/commit/c56a684a4c539fc2800661dcc9e70ec3fa1a660f))

- Refresh uv.lock after version bump
  ([`d8840bc`](https://github.com/quadsproject/qiip/commit/d8840bce717c0a8236961e8d50dc22f77740687a))

- Refresh uv.lock after version bump
  ([`d226d84`](https://github.com/quadsproject/qiip/commit/d226d84c89463abf5d6ed6addfda7b066f976169))

- Refresh uv.lock after version bump
  ([`aded796`](https://github.com/quadsproject/qiip/commit/aded796dea0aa602557c5617412304962536c859))

- Refresh uv.lock after version bump
  ([`dceb92d`](https://github.com/quadsproject/qiip/commit/dceb92d4cf51fcbd08ee02f6cc7a2ee0d24866af))

- Restore stable version for the part 1 sync branch
  ([`365ed0a`](https://github.com/quadsproject/qiip/commit/365ed0a50afe64df6045f9fee28699624423256f))

- Stop diagnostics tests flaking under load
  ([`f5160ac`](https://github.com/quadsproject/qiip/commit/f5160ac570aaad43588d095abeaef87acd98bc60))

### Code Style

- **tests**: Ruff format for local OIDC provider change
  ([`531b381`](https://github.com/quadsproject/qiip/commit/531b381f60d2fd052ffee698a9678182c88592aa))

### Documentation

- Describe bmc credential key normalization accurately
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Warn that a global SSL_CERT_FILE replaces the trust store
  ([`32c451e`](https://github.com/quadsproject/qiip/commit/32c451e1341fa8255ba40adcea74b7fbb2772038))

### Features

- Add user leaderboard and token dashboard
  ([`d617c5f`](https://github.com/quadsproject/qiip/commit/d617c5fe1ae23d9e23111e984ef87f25a3be8a78))

- Auto-rediscover self-setup model drift in health cycle
  ([`65ef1b7`](https://github.com/quadsproject/qiip/commit/65ef1b7b12c805aa61fa73e390fee1303a5827ef))

- Copy the user's own token from the pages
  ([`af4fad0`](https://github.com/quadsproject/qiip/commit/af4fad0eecc1ef91d2515eddda1ac49812f8cbdc))

- Link the release RPM packages in the release notes
  ([`170ff07`](https://github.com/quadsproject/qiip/commit/170ff07e1cd58793f6313a03a5006300bed43d12))

- Local Login merges the admin and local oauth sign-in paths
  ([`95632c4`](https://github.com/quadsproject/qiip/commit/95632c4e4056f816f7b293b742ca5e6a05d9656f))

- Opt-in local OIDC auth provider (RFE #230)
  ([`2bf1c9f`](https://github.com/quadsproject/qiip/commit/2bf1c9fd083368a56365086bf3c8c44f3501db3d))

- Power on new nodes via redfish and resolve bmc creds per lab
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Reuse compatible GPU drivers during setup ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`9666bfd`](https://github.com/quadsproject/qiip/commit/9666bfd668854f682842b08e4b2f5071c1914c96))

- Reuse verified llama.cpp builds with bounded source fallback
  ([`e9e12cd`](https://github.com/quadsproject/qiip/commit/e9e12cd38195943ea5765f20087d23f0e7451464))

- Show token cost on the leaderboard
  ([`8556123`](https://github.com/quadsproject/qiip/commit/85561232b3b0fead8f46a380df09371ceca08317))

- Size automatic vLLM launches from available memory
  ([`fb12176`](https://github.com/quadsproject/qiip/commit/fb1217696ca469245ca2c7d484025fa91e7552b2))

- Sort the token and usage tables by column
  ([`27980b6`](https://github.com/quadsproject/qiip/commit/27980b63ff5462fc95a5d2a209d2e1bbbe51e2e4))

- Wait for QUADS to finish releasing a host before placing on it
  ([`f1a412a`](https://github.com/quadsproject/qiip/commit/f1a412aec9bb6526f6a6d23c131089dccf6d2e1f))

- **provisioning**: Resume verified stages and retry transient failures
  ([`d0b2f56`](https://github.com/quadsproject/qiip/commit/d0b2f56217033d286370ffe88ec98b5a816bf2f0))

### Testing

- Guard nested BmcCredential password secrecy in repr
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Satisfy strict mypy for the new auth tests
  ([`1f20dbf`](https://github.com/quadsproject/qiip/commit/1f20dbf698fab6a8dfc281e55dcc7943ea1a17bd))

- Widen timing-sensitive liveness timeouts
  ([`06512f0`](https://github.com/quadsproject/qiip/commit/06512f055a8668f6ac413c0034aa3d407a692abf))


## v0.2.0-dev.24 (2026-10-20)

### Chores

- Refresh uv.lock after version bump
  ([`154971a`](https://github.com/quadsproject/qiip/commit/154971ad05619a25b7b01abca243e11be9eef238))

### Features

- Show token cost on the leaderboard
  ([`ff311dd`](https://github.com/quadsproject/qiip/commit/ff311dda68e0f077da4bd7edc09adf5b05c9e33d))


## v0.2.0-dev.23 (2026-10-16)

### Bug Fixes

- Drop the non-admin fleet client path
  ([`49f53af`](https://github.com/quadsproject/qiip/commit/49f53afe68e264629e67109091260a30e6816f58))

### Chores

- Refresh uv.lock after version bump
  ([`9654394`](https://github.com/quadsproject/qiip/commit/9654394375bc2dd734b65fb68adedc80c075c9f2))


## v0.2.0-dev.22 (2026-10-16)

### Bug Fixes

- Give admins the leaderboard ranking
  ([`13eee30`](https://github.com/quadsproject/qiip/commit/13eee3020c2d801013a441c8ca33e95a49f72086))

### Chores

- Refresh uv.lock after version bump
  ([`19a6a17`](https://github.com/quadsproject/qiip/commit/19a6a179aa2a4962f021954b9d772dcb312d01a3))


## v0.2.0-dev.21 (2026-10-16)

### Bug Fixes

- Gate the Google sign-in option on its provider
  ([`a6c1445`](https://github.com/quadsproject/qiip/commit/a6c14451f4997d8832b8e714cac9cca553a04a7c))

### Chores

- Refresh uv.lock after version bump
  ([`2ce74f2`](https://github.com/quadsproject/qiip/commit/2ce74f296c083b87993461c422b47dcf40a41209))


## v0.2.0-dev.20 (2026-10-16)

### Bug Fixes

- Guard the same-issuer refresh and check the ID-token audience
  ([`52e17b7`](https://github.com/quadsproject/qiip/commit/52e17b73d23b402545ebc601f05c79e1dcf1e154))

### Chores

- Refresh uv.lock after version bump
  ([`f79e442`](https://github.com/quadsproject/qiip/commit/f79e442f9343df7abab15e1dae4669a76fab887d))


## v0.2.0-dev.19 (2026-10-16)

### Bug Fixes

- Address 0.2.0 roll-up review findings
  ([`8cea3e1`](https://github.com/quadsproject/qiip/commit/8cea3e1e9d2723fc8a6218f1e2f06eee6d8bceab))

### Chores

- Refresh uv.lock after version bump
  ([`14c1f87`](https://github.com/quadsproject/qiip/commit/14c1f875e389c9c801e7395301a56c6d305b1db3))


## v0.2.0-dev.18 (2026-10-15)

### Bug Fixes

- Tell a Copr outage apart from missing dependency RPMs
  ([`0df4381`](https://github.com/quadsproject/qiip/commit/0df43816825912ab4e666b80dee922df1d9f3a2b))

### Chores

- Absorb transient Chrome hangs in browser tests (#215)
  ([#216](https://github.com/quadsproject/qiip/pull/216),
  [`25c3f7d`](https://github.com/quadsproject/qiip/commit/25c3f7d911100343880240d8cdd641d6153f5467))

- Refresh uv.lock after version bump
  ([`5c5a405`](https://github.com/quadsproject/qiip/commit/5c5a405c71b23b3d5f44f0e47907e6f9b85511f8))


## v0.2.0-dev.17 (2026-10-15)

### Chores

- Refresh uv.lock after version bump
  ([`42d7896`](https://github.com/quadsproject/qiip/commit/42d7896147918d11732130f96eeff09bb3fc0dce))

### Features

- Link the release RPM packages in the release notes
  ([`814ac1d`](https://github.com/quadsproject/qiip/commit/814ac1d418c7b5639f7ab6ef348e2e1caed62a1d))


## v0.2.0-dev.16 (2026-10-15)

### Bug Fixes

- Guard Copr publishes against duplicate NVRs
  ([`a15b4c7`](https://github.com/quadsproject/qiip/commit/a15b4c702d1b2e39f95a176659fef90700ac78ba))

### Chores

- Refresh uv.lock after version bump
  ([`2c9e354`](https://github.com/quadsproject/qiip/commit/2c9e35442cc7802ce3be47818548d2e8292a885a))


## v0.2.0-dev.15 (2026-10-15)

### Bug Fixes

- Run the chat playground on the user's token
  ([`56b8c40`](https://github.com/quadsproject/qiip/commit/56b8c406f588299b54f635715419ac7807e7d53b))

### Chores

- Refresh uv.lock after version bump
  ([`554942c`](https://github.com/quadsproject/qiip/commit/554942cb1e46f2a87d75f4a5376c70cd52412607))


## v0.2.0-dev.14 (2026-10-15)

### Bug Fixes

- Unify the account cluster across the top bars
  ([`120ec7c`](https://github.com/quadsproject/qiip/commit/120ec7c5a2f9bbafa0e37be27c769a0a31f434bc))

### Chores

- Refresh uv.lock after version bump
  ([`a1328e8`](https://github.com/quadsproject/qiip/commit/a1328e8f29beb8a0a44ba8e988af831c8bc93ea2))


## v0.2.0-dev.13 (2026-10-15)

### Chores

- Refresh uv.lock after version bump
  ([`564ff60`](https://github.com/quadsproject/qiip/commit/564ff60254c0fdbc001909bce59e5c30c0ec91c0))

### Features

- Copy the user's own token from the pages
  ([`d1bcb53`](https://github.com/quadsproject/qiip/commit/d1bcb535d833228262fad8230f8df7d6b6cb7c15))


## Unreleased

### Features

- Copy the user's own token from the leaderboard token row and the `/start`
  token home: a clipboard icon next to the abbreviated prefix reveals the raw
  value through `POST /onboarding/token/reveal` and confirms the copy
  ([#247](https://github.com/quadsproject/qiip/issues/247))

## v0.2.0-dev.12 (2026-10-15)

### Chores

- Refresh uv.lock after version bump
  ([`abcf1f4`](https://github.com/quadsproject/qiip/commit/abcf1f4923abd887236e5bc6cf9ce9c29441fd0e))

- Stop diagnostics tests flaking under load
  ([`3b7fcc4`](https://github.com/quadsproject/qiip/commit/3b7fcc450ba40a1316b40ac8682b33023a14ad34))

### Features

- Sort the token and usage tables by column
  ([`1473fdf`](https://github.com/quadsproject/qiip/commit/1473fdfbc752dadd68979ba9b575fce62e859f5d))

## v0.2.0-dev.11 (2026-10-14)

### Bug Fixes

- Document new setup settings and satisfy strict typing
  ([`87d41b3`](https://github.com/quadsproject/qiip/commit/87d41b3f4463eeb60553fe384fa41eaeee7ccbb9))

- **provisioning**: Address second review round
  ([`dc93cd4`](https://github.com/quadsproject/qiip/commit/dc93cd4214fab5452d403b736410b2b1aab3ed46))

- **provisioning**: Address setup resume review feedback
  ([`337837e`](https://github.com/quadsproject/qiip/commit/337837e6801e72960500f0744338d7203802eadd))

- **provisioning**: Retry bare 5xx download failures
  ([`196add1`](https://github.com/quadsproject/qiip/commit/196add1e00feec34ac1626860dea81b642ee1ae3))

### Chores

- Refresh uv.lock after version bump
  ([`bf62aa7`](https://github.com/quadsproject/qiip/commit/bf62aa75e4d8c959eb76c50c45d2e0c30204a891))

### Features

- **provisioning**: Resume verified stages and retry transient failures
  ([`e14cc11`](https://github.com/quadsproject/qiip/commit/e14cc11d576aeef6fe2e06b8ff49900db05563bc))


## v0.2.0-dev.10 (2026-10-14)

### Bug Fixes

- Price token estimates at Sonnet 5 rates
  ([`9c16bde`](https://github.com/quadsproject/qiip/commit/9c16bdeba338acca564aad9b698c1bad7d3252c0))

### Chores

- Refresh uv.lock after version bump
  ([`33ef768`](https://github.com/quadsproject/qiip/commit/33ef76848b68d09a77f54dd19d1e3737ad4f1080))


## v0.2.0-dev.9 (2026-10-14)

### Bug Fixes

- **vllm**: Address review on memory-aware sizing
  ([`2f3e0e2`](https://github.com/quadsproject/qiip/commit/2f3e0e24d136f31e25704e5c0032d2bd40f3102d))

- **vllm**: Address the second review round on memory-aware sizing
  ([`66b65fe`](https://github.com/quadsproject/qiip/commit/66b65fecdd9811e59ac03658ab8f576cdec15f2d))

- **vllm**: Keep the retried plan reason consistent with the effective context
  ([`45faef2`](https://github.com/quadsproject/qiip/commit/45faef2eda304af6467880bce52c828d91b30cea))

### Chores

- Refresh uv.lock after version bump
  ([`c5f43d2`](https://github.com/quadsproject/qiip/commit/c5f43d205c087b01fa53822ba9856f8805cb7e1c))

### Features

- Size automatic vLLM launches from available memory
  ([`a56b1f1`](https://github.com/quadsproject/qiip/commit/a56b1f135d8652280b99b8969ecfb812c7c896cf))

### Testing

- Widen timing-sensitive liveness timeouts
  ([`e0c53b0`](https://github.com/quadsproject/qiip/commit/e0c53b0fbb6219878274cdd27ae04917fb1bdcfd))


## v0.2.0-dev.8 (2026-10-13)

### Bug Fixes

- Enforce the stale bound only where placement runs
  ([`982deb9`](https://github.com/quadsproject/qiip/commit/982deb996bb903e370d94962e51eb1ec6f7b755d))


## v0.2.0-dev.7 (2026-10-13)

### Bug Fixes

- Require quads.stale_after_seconds to be at least twice poll_interval
  ([`34fe664`](https://github.com/quadsproject/qiip/commit/34fe6643dfa3fccf89a6f0137ed270d99ed5faba))

### Chores

- Refresh uv.lock after version bump
  ([`6b04141`](https://github.com/quadsproject/qiip/commit/6b04141abecb6c4a3bca18fdd8c51020f2f2bbab))

### Features

- Wait for QUADS to finish releasing a host before placing on it
  ([`08842a6`](https://github.com/quadsproject/qiip/commit/08842a65b0d6ebf19359b204c07f686ff6d606d0))


## v0.2.0-dev.6 (2026-10-13)

### Bug Fixes

- Address local OIDC provider review findings
  ([`cbded89`](https://github.com/quadsproject/qiip/commit/cbded892ac8e76a3d0fbcc546872f68cc9b7d992))

- Harden the issuer migration crash recovery
  ([`02a133d`](https://github.com/quadsproject/qiip/commit/02a133d6ba4c59d4d6f5a0efc76769b447bc09a0))

### Chores

- Refresh uv.lock after version bump
  ([`6d1eff9`](https://github.com/quadsproject/qiip/commit/6d1eff9d299b8c1e05930037ddf556d2dd938219))

### Code Style

- **tests**: Ruff format for local OIDC provider change
  ([`f41abf6`](https://github.com/quadsproject/qiip/commit/f41abf6b619bf9578ff65a8357485e77ae7614dc))

### Documentation

- Warn that a global SSL_CERT_FILE replaces the trust store
  ([`6894d5b`](https://github.com/quadsproject/qiip/commit/6894d5b24e60f6118f1e3e359a140d7136934439))

### Features

- Local Login merges the admin and local oauth sign-in paths
  ([`13b3906`](https://github.com/quadsproject/qiip/commit/13b39068715a0fc9048a0a0ea04c099cb0d53cae))

- Opt-in local OIDC auth provider (RFE #230)
  ([`435077a`](https://github.com/quadsproject/qiip/commit/435077a1e204a782b0e9ec9454b135056ad8d1a5))

### Testing

- Satisfy strict mypy for the new auth tests
  ([`cee2e76`](https://github.com/quadsproject/qiip/commit/cee2e765a583acba0252fa3091c8a261bbbee9ea))


## v0.2.0-dev.5 (2026-10-09)

### Bug Fixes

- Address review findings on redfish power gate and bmc resolver
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`9e5339c`](https://github.com/quadsproject/qiip/commit/9e5339c9fdf8981d91a4bcbbdbf83c689e2603c1))

### Chores

- Refresh uv.lock after version bump
  ([`69de6a4`](https://github.com/quadsproject/qiip/commit/69de6a4fd07976744e7a744955ddba3755e50d99))

### Documentation

- Describe bmc credential key normalization accurately
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`9e5339c`](https://github.com/quadsproject/qiip/commit/9e5339c9fdf8981d91a4bcbbdbf83c689e2603c1))

### Features

- Power on new nodes via redfish and resolve bmc creds per lab
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`9e5339c`](https://github.com/quadsproject/qiip/commit/9e5339c9fdf8981d91a4bcbbdbf83c689e2603c1))

### Testing

- Guard nested BmcCredential password secrecy in repr
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`9e5339c`](https://github.com/quadsproject/qiip/commit/9e5339c9fdf8981d91a4bcbbdbf83c689e2603c1))


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
