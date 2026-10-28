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

- Address profile review findings
  ([`5b46ec8`](https://github.com/quadsproject/qiip/commit/5b46ec898895a7dbec1628bfa7fa07c37b952a80))

- Address review findings on redfish power gate and bmc resolver
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Allow teardown of unregistered stuck tasks
  ([`1541157`](https://github.com/quadsproject/qiip/commit/154115740c573f7b1b261fd65b938f998c23d7ee))

- Avoid dead home link for full-access users
  ([`1430082`](https://github.com/quadsproject/qiip/commit/1430082040d3626b0e72c03e06ac6dd1ed380dbb))

- Copy driver reboot markers directly ([#218](https://github.com/quadsproject/qiip/pull/218),
  [`9666bfd`](https://github.com/quadsproject/qiip/commit/9666bfd668854f682842b08e4b2f5071c1914c96))

- Correct onboarding and leaderboard navigation
  ([`a306031`](https://github.com/quadsproject/qiip/commit/a306031967eb3215c2584e6d3815abb88791269f))

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

- Harden leaderboard toggle, cover review criteria, update docs (refs #271)
  ([`021356d`](https://github.com/quadsproject/qiip/commit/021356dec57c32f05a6c92f504b5327bb07081ec))

- Harden the issuer migration crash recovery
  ([`1f778e2`](https://github.com/quadsproject/qiip/commit/1f778e2e9a3ae1d3ff33918e70495392299aed75))

- Ignore benign journal and split-log issues
  ([`76b87aa`](https://github.com/quadsproject/qiip/commit/76b87aa0294a05ba87d416fdf229889d932608a7))

- Ignore stale leaderboard load failures
  ([`29ce190`](https://github.com/quadsproject/qiip/commit/29ce1909105c2cff275f21dc9dd26e3c8f0ad99b))

- Keep multibyte log splits out of truncation
  ([`ea8904b`](https://github.com/quadsproject/qiip/commit/ea8904b4319898e7b5866a5b07b6be0f965f0184))

- Keep provisioning state and teardown labels consistent
  ([`3e3659b`](https://github.com/quadsproject/qiip/commit/3e3659ba454030d914106cfc3002b4061c2abc60))

- Keep teardown engine and recovery guidance honest
  ([`6d88b46`](https://github.com/quadsproject/qiip/commit/6d88b46ab257424c67fa52e5990119cd73dde870))

- Name the prior provisioning task explicitly
  ([`4476bf9`](https://github.com/quadsproject/qiip/commit/4476bf9af84dc28dcb1dcd7c710ae88778fd7fb8))

- Price token estimates at Sonnet 5 rates
  ([`9f273dd`](https://github.com/quadsproject/qiip/commit/9f273dd2b31bd4200161b6ae18d024b5214654e7))

- Reconnect logs for a new task after a terminal stream
  ([`2e3e254`](https://github.com/quadsproject/qiip/commit/2e3e254de65d81b30cd713d26b89af2f02cbc54b))

- Render last successful response and guard stale failure revert (refs #271)
  ([`c2dea29`](https://github.com/quadsproject/qiip/commit/c2dea293f8d477f2e0f1525062f57cf8b938658f))

- Require quads.stale_after_seconds to be at least twice poll_interval
  ([`b1cd32d`](https://github.com/quadsproject/qiip/commit/b1cd32d4bca09a1f060efaa75753d12e03339d4a))

- Require teardown after a failed teardown
  ([`a110c12`](https://github.com/quadsproject/qiip/commit/a110c12e3d3ec72ba003e22c057532a0bf448e79))

- Restore rendered scope on toggle failure; cover double-failure desync (refs #271)
  ([`2231c72`](https://github.com/quadsproject/qiip/commit/2231c72339401b76f43f4eaaa1b0725f81846276))

- Revert toggle control on failed fetch; pin request-id guard in harness (refs #271)
  ([`d4b49fc`](https://github.com/quadsproject/qiip/commit/d4b49fc1ca8a3f3a77addfef65a07635006f5a55))

- Run the chat playground on the user's token
  ([`78c9d22`](https://github.com/quadsproject/qiip/commit/78c9d22a7d8c9a344daf430bc21ff2c05a88ce55))

- Show non-admin users their profile
  ([`5109380`](https://github.com/quadsproject/qiip/commit/5109380798cd0b129e1ba10c5dcd58b04f5a460d))

- Surface provisioning task failures and available-host actions
  ([`8a09c35`](https://github.com/quadsproject/qiip/commit/8a09c35039396daf60f0b66961730c2233e773ff))

- Switch log stream when a new task replaces an open stream
  ([`cdf100f`](https://github.com/quadsproject/qiip/commit/cdf100f204313370a6ad53d6f75b06231726282c))

- Sync leaderboard scope and cover local-admin widen
  ([`6237bac`](https://github.com/quadsproject/qiip/commit/6237bac1a00792d6eabe1f3fc65e6248aa29a61a))

- Tell a Copr outage apart from missing dependency RPMs
  ([`10c3af6`](https://github.com/quadsproject/qiip/commit/10c3af61d78d86b6ca56804679496979f81ba19f))

- Treat journal parse failure as an issue
  ([`f737294`](https://github.com/quadsproject/qiip/commit/f737294e9f8f8db9340e37bc77a05968317151bb))

- Unify the account cluster across the top bars
  ([`ef8f040`](https://github.com/quadsproject/qiip/commit/ef8f04081dadf8bbfb89ff7beca9c0ea15c20077))

- Wait for the new task attempt before opening logs
  ([`f265907`](https://github.com/quadsproject/qiip/commit/f2659077dd138e0a0607722583330a5c98281a39))

- **provisioning**: Address second review round
  ([`99708ca`](https://github.com/quadsproject/qiip/commit/99708ca609b7edf22484c926fa251338447b283a))

- **provisioning**: Address setup resume review feedback
  ([`d9edd07`](https://github.com/quadsproject/qiip/commit/d9edd0754c0ddfcc8406d78d939839d7fe441742))

- **provisioning**: Close vLLM review nits
  ([`c3f4c61`](https://github.com/quadsproject/qiip/commit/c3f4c6156464c64d61a6c61d1420c72f8e4c1e77))

- **provisioning**: Retry bare 5xx download failures
  ([`424837f`](https://github.com/quadsproject/qiip/commit/424837f963d72904387747f02c7818db5cf10d2d))

- **provisioning**: Tighten vLLM serving gate evidence
  ([`a723b3a`](https://github.com/quadsproject/qiip/commit/a723b3a20f1984ab3081e1b83aecd84ef2af9812))

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

- Document linear sync release flow
  ([`2ac462a`](https://github.com/quadsproject/qiip/commit/2ac462aa09809dcbf2726006f889e9cfb940cfd2))

- Drop duplicate dev release badge from README
  ([`f800bba`](https://github.com/quadsproject/qiip/commit/f800bba67f4c2e2eb85a520ce1dc41575aacb74b))

- Preserve superseded sync PR records
  ([`74ed239`](https://github.com/quadsproject/qiip/commit/74ed239cb4e24f70e2575353e9f4cd54423a4ed5))

- Refresh uv.lock after version bump
  ([`5a39828`](https://github.com/quadsproject/qiip/commit/5a39828610603342f72886acc6e62774cc582675))

- Refresh uv.lock after version bump
  ([`3b48afe`](https://github.com/quadsproject/qiip/commit/3b48afe250a8ace28d92e8f63ccfb87983ec684b))

- Refresh uv.lock after version bump
  ([`67c44c5`](https://github.com/quadsproject/qiip/commit/67c44c5f4a1654c2b7011892335e0f8e3251e037))

- Refresh uv.lock after version bump
  ([`6b743fb`](https://github.com/quadsproject/qiip/commit/6b743fb7ac53dd5072f5ccae64fec1a21989e8b4))

- Refresh uv.lock after version bump
  ([`c526dc7`](https://github.com/quadsproject/qiip/commit/c526dc7c2bc456fa1de6708f5486b2d7d5373057))

- Refresh uv.lock after version bump
  ([`7bb32c7`](https://github.com/quadsproject/qiip/commit/7bb32c76e62963b543bbbe374017efad61652eac))

- Refresh uv.lock after version bump
  ([`5b7a0e4`](https://github.com/quadsproject/qiip/commit/5b7a0e4e91376fb7bfc5fec53ecdebb6c68412f9))

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

- Restore stable version for the part 2 sync branch
  ([`f9f298f`](https://github.com/quadsproject/qiip/commit/f9f298fba229100a8aa383f1cf04ce3c7ab9df54))

- Stop diagnostics tests flaking under load
  ([`f5160ac`](https://github.com/quadsproject/qiip/commit/f5160ac570aaad43588d095abeaef87acd98bc60))

### Code Style

- **tests**: Ruff format for local OIDC provider change
  ([`531b381`](https://github.com/quadsproject/qiip/commit/531b381f60d2fd052ffee698a9678182c88592aa))

### Documentation

- Describe bmc credential key normalization accurately
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Document the development-to-main release cycle
  ([`12a3431`](https://github.com/quadsproject/qiip/commit/12a3431ddda501d3cdcea8acfcc74287f30e98b5))

- Record #121 acceptance status and split fleet validation (#277)
  ([#278](https://github.com/quadsproject/qiip/pull/278),
  [`7ce1b79`](https://github.com/quadsproject/qiip/commit/7ce1b79db43ed13532982188bf31bf5ae88189f8))

- Warn that a global SSL_CERT_FILE replaces the trust store
  ([`32c451e`](https://github.com/quadsproject/qiip/commit/32c451e1341fa8255ba40adcea74b7fbb2772038))

- **provisioning**: Clarify vLLM probe timeout scope
  ([`14b46f2`](https://github.com/quadsproject/qiip/commit/14b46f23615eba3ae61c1a148402a2ec27b24939))

### Features

- Add admin toggle to include admins in usage leaderboard (fixes #271)
  ([`e912521`](https://github.com/quadsproject/qiip/commit/e912521b9ad21041dd96e84f24a7f96e8bda8e05))

- Add user leaderboard and token dashboard
  ([`d617c5f`](https://github.com/quadsproject/qiip/commit/d617c5fe1ae23d9e23111e984ef87f25a3be8a78))

- Auto-rediscover self-setup model drift in health cycle
  ([`65ef1b7`](https://github.com/quadsproject/qiip/commit/65ef1b7b12c805aa61fa73e390fee1303a5827ef))

- Copy the user's own token from the pages
  ([`af4fad0`](https://github.com/quadsproject/qiip/commit/af4fad0eecc1ef91d2515eddda1ac49812f8cbdc))

- Delete cached models from the admin UI
  ([`75f5a71`](https://github.com/quadsproject/qiip/commit/75f5a7188ed3e335f96e4fa3d8d633b6c65c62af))

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

- **provisioning**: Verify vLLM identity and inference before routing
  ([`41e2e95`](https://github.com/quadsproject/qiip/commit/41e2e95e1f32d14491bcd882e2472b658dc55855))

### Testing

- Allow slower CI for relaunch transitions
  ([`adf6857`](https://github.com/quadsproject/qiip/commit/adf6857bc81aa6f7b4fdd0ed8e7dbcffe13fc173))

- Cover profile token capability
  ([`cfc3dd9`](https://github.com/quadsproject/qiip/commit/cfc3dd9cc94e8d9d450b0719484d359629cacac3))

- Exercise the stale-failure guard with overlapping in-flight toggles (refs #271)
  ([`c800a0b`](https://github.com/quadsproject/qiip/commit/c800a0bdd845bcd5e0018305a69b2243e58aa1ca))

- Guard nested BmcCredential password secrecy in repr
  ([#228](https://github.com/quadsproject/qiip/pull/228),
  [`e1f1dd9`](https://github.com/quadsproject/qiip/commit/e1f1dd94e490f112f50b3a9e2248a92e3dd15e27))

- Satisfy strict mypy for the new auth tests
  ([`1f20dbf`](https://github.com/quadsproject/qiip/commit/1f20dbf698fab6a8dfc281e55dcc7943ea1a17bd))

- Widen timing-sensitive liveness timeouts
  ([`06512f0`](https://github.com/quadsproject/qiip/commit/06512f055a8668f6ac413c0034aa3d407a692abf))

- **provisioning**: Keep failed-store guard meaningful
  ([`d1d2372`](https://github.com/quadsproject/qiip/commit/d1d23729adc178de04a14c454bc2a63291d25c11))


## v0.2.0-dev.30 (2026-10-27)

### Bug Fixes

- Address profile review findings
  ([`7f7d493`](https://github.com/quadsproject/qiip/commit/7f7d493c4de7e6f3447bb3df099b6fa4111b42fe))

- Avoid dead home link for full-access users
  ([`6a16da4`](https://github.com/quadsproject/qiip/commit/6a16da4485bf732c41a8e2b5b3e550163d429936))

- Correct onboarding and leaderboard navigation
  ([`b586d53`](https://github.com/quadsproject/qiip/commit/b586d53cc162f435c57b7b77b979f60e6fc02c13))

- Show non-admin users their profile
  ([`557d17b`](https://github.com/quadsproject/qiip/commit/557d17b2f96c72c7390b903cc9e233a2cedf8d6b))

### Chores

- Refresh uv.lock after version bump
  ([`4fd6b06`](https://github.com/quadsproject/qiip/commit/4fd6b064f80d2c6b6a3d026cc4c1156c8673d105))

### Testing

- Cover profile token capability
  ([`4bf40c9`](https://github.com/quadsproject/qiip/commit/4bf40c9da112754218e31e1f68b36baccea3dcf0))


## v0.2.0-dev.29 (2026-10-27)

### Bug Fixes

- Ignore benign journal and split-log issues
  ([`a29f8a9`](https://github.com/quadsproject/qiip/commit/a29f8a92c4823e6bef6e3a016c56d20ea17d0759))

- Keep multibyte log splits out of truncation
  ([`89354fd`](https://github.com/quadsproject/qiip/commit/89354fde9057bd089f61917b8737f37c6ca9e469))

- Reconnect logs for a new task after a terminal stream
  ([`21eca71`](https://github.com/quadsproject/qiip/commit/21eca717023be08e7a0433cc255d87b211b8db56))

- Switch log stream when a new task replaces an open stream
  ([`4bc55a5`](https://github.com/quadsproject/qiip/commit/4bc55a529dc3ba8a785cc5ecaf4ba8a901e00268))

- Treat journal parse failure as an issue
  ([`9e91f29`](https://github.com/quadsproject/qiip/commit/9e91f2918bdcd77f578a5db40cb078f1f25e95bc))

- Wait for the new task attempt before opening logs
  ([`28f1c11`](https://github.com/quadsproject/qiip/commit/28f1c11e96c867f9668883663dc8e53b409d3319))

### Chores

- Refresh uv.lock after version bump
  ([`0914035`](https://github.com/quadsproject/qiip/commit/0914035f63c9b52966fbe2fa964143b216b357cb))

### Features

- Delete cached models from the admin UI
  ([`8e62ab7`](https://github.com/quadsproject/qiip/commit/8e62ab7689466009a641cc677950154976605e94))


## v0.2.0-dev.28 (2026-10-27)

### Bug Fixes

- Allow teardown of unregistered stuck tasks
  ([`9f1879e`](https://github.com/quadsproject/qiip/commit/9f1879efcf71cb30a1839089423ee201a1e5ead2))

- Keep provisioning state and teardown labels consistent
  ([`4637c54`](https://github.com/quadsproject/qiip/commit/4637c54ad8496653450b80f6a5447f47c32497ca))

- Keep teardown engine and recovery guidance honest
  ([`499d43d`](https://github.com/quadsproject/qiip/commit/499d43d666459ea6aeb83c31861da9bb89adbf87))

- Name the prior provisioning task explicitly
  ([`ee5addc`](https://github.com/quadsproject/qiip/commit/ee5addc56e7cccbb3708f57128ce9dbe5a46fbb0))

- Require teardown after a failed teardown
  ([`1f2f344`](https://github.com/quadsproject/qiip/commit/1f2f34437bbc7ac504e665d74901422a5ea6cb0b))

- Surface provisioning task failures and available-host actions
  ([`080f5c5`](https://github.com/quadsproject/qiip/commit/080f5c577f0abe0fb10c18603e131fc531c1292d))

### Chores

- Refresh uv.lock after version bump
  ([`98c1476`](https://github.com/quadsproject/qiip/commit/98c147642c13889a4d438039b7665905ea993b64))

### Documentation

- Document the development-to-main release cycle
  ([`e249d62`](https://github.com/quadsproject/qiip/commit/e249d62e5947d9cfd2efd5b56e00266883dbb777))

- Record #121 acceptance status and split fleet validation (#277)
  ([#278](https://github.com/quadsproject/qiip/pull/278),
  [`ffebc31`](https://github.com/quadsproject/qiip/commit/ffebc31f3ccc7b4b843053ce3dc68d456cd8ea12))

### Testing

- Allow slower CI for relaunch transitions
  ([`ba7ebc1`](https://github.com/quadsproject/qiip/commit/ba7ebc1c191315460c31cd9cb3e2521ed0267f9c))


## v0.2.0-dev.27 (2026-10-22)

### Bug Fixes

- **provisioning**: Close vLLM review nits
  ([`87f9f22`](https://github.com/quadsproject/qiip/commit/87f9f22d43255275219d7c31e475be05a807793e))

- **provisioning**: Tighten vLLM serving gate evidence
  ([`4aca3c0`](https://github.com/quadsproject/qiip/commit/4aca3c0fc950b166b72675d6b27d3b7bf7d4cf23))

### Chores

- Refresh uv.lock after version bump
  ([`f2fddbd`](https://github.com/quadsproject/qiip/commit/f2fddbd43a64aba22626a864a6e0f05b16a211a3))

### Documentation

- **provisioning**: Clarify vLLM probe timeout scope
  ([`adb94d1`](https://github.com/quadsproject/qiip/commit/adb94d160ad740c257bcf001ed97e2bfd659d9d8))

### Features

- **provisioning**: Verify vLLM identity and inference before routing
  ([`23a7d49`](https://github.com/quadsproject/qiip/commit/23a7d497be97382c0a7b8a36c4b35da8f02c09e6))

### Testing

- **provisioning**: Keep failed-store guard meaningful
  ([`a758701`](https://github.com/quadsproject/qiip/commit/a75870105ad6ab007fafcbee6dd5eca4fee42cdd))


## v0.2.0-dev.26 (2026-10-20)

### Bug Fixes

- Ignore stale leaderboard load failures
  ([`c2592fd`](https://github.com/quadsproject/qiip/commit/c2592fd8a4f1daa25b4cad5209dcf57dbd69d388))

### Chores

- Refresh uv.lock after version bump
  ([`19e9a13`](https://github.com/quadsproject/qiip/commit/19e9a137974d0e06e7fddf7c75200df8d3572805))


## v0.2.0-dev.25 (2026-10-20)

### Bug Fixes

- Harden leaderboard toggle, cover review criteria, update docs (refs #271)
  ([`f2f4d4c`](https://github.com/quadsproject/qiip/commit/f2f4d4cce4c97d6059c13424520145dfdf3fe04b))

- Render last successful response and guard stale failure revert (refs #271)
  ([`abb6159`](https://github.com/quadsproject/qiip/commit/abb615947d4884ee950b2e81e8c3930e1d880d00))

- Restore rendered scope on toggle failure; cover double-failure desync (refs #271)
  ([`60ce3c9`](https://github.com/quadsproject/qiip/commit/60ce3c9b76d099dd1112e0e62381a4823677e402))

- Revert toggle control on failed fetch; pin request-id guard in harness (refs #271)
  ([`b594513`](https://github.com/quadsproject/qiip/commit/b5945134b99215df5001eff5cfb70ef1f5fcd925))

- Sync leaderboard scope and cover local-admin widen
  ([`f28b736`](https://github.com/quadsproject/qiip/commit/f28b73681c50ae1cb855aaca091ba64d93584a9d))

### Chores

- Refresh uv.lock after version bump
  ([`bfd1ee4`](https://github.com/quadsproject/qiip/commit/bfd1ee4741a25459f09464014816c3123ee81242))

### Features

- Add admin toggle to include admins in usage leaderboard (fixes #271)
  ([`cca9722`](https://github.com/quadsproject/qiip/commit/cca9722faad4c24a94eb9f30e57b9cf685c7d142))

### Testing

- Exercise the stale-failure guard with overlapping in-flight toggles (refs #271)
  ([`991bcbb`](https://github.com/quadsproject/qiip/commit/991bcbb174c1bc11ec978d4a31eb4cbd52748d0e))


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
