# Vendored frontend dependencies

These browser assets are served locally so the administrative origin does not
execute code from a CDN at runtime.

| Package | Version | Source package integrity | Vendored browser asset SHA-256 |
|---|---:|---|---|
| Marked | 18.0.7 | `sha512-iDVQ5ldaiKXn6b2JroX5kgRfmwgqolW7NpaEzTl1k/2Zh1njIEN9yniyLV/mOvWwtsE8OGgkjsCYvijuPk1dtA==` | `7a1f8c5e7226b75ff16644bdb2c0130d2ae7371e7ea3106c2d6dac77ab0ff7b6` |
| DOMPurify | 3.4.12 | `sha512-zQvGet8Z2sWbQhCmfFz/T5QWH2oBmjnqK3qvOjaqaNLrLEF912WamU+ohnTp0TCep/MFVHpdJuCZEdFOdTnEFg==` | `c45ba939765574f96cbf35ee9b6d89f73756a17921814425e74b82f7c54603ce` |

The files come from the packages published at
`https://registry.npmjs.org/marked/-/marked-18.0.7.tgz` and
`https://registry.npmjs.org/dompurify/-/dompurify-3.4.12.tgz`. Each package's
license is stored beside its browser asset.

## Updating vendored assets

Treat a frontend dependency update as a reviewed supply-chain change:

1. Select a released package version and review its security and compatibility
   notes.
2. Download the registry package, verify the registry integrity value, and
   extract only the required browser asset and license.
3. Recompute the vendored asset SHA-256 and update this table, template script
   reference, asset, and license in the same commit.
4. Run `uv run pytest tests/frontend` and the full quality gates.

`test_chat_uses_pinned_local_frontend_dependencies` intentionally fails when
an asset, version reference, or recorded digest changes independently. Do not
silence it or reintroduce a CDN fallback; the local-only dependency is part of
the authenticated admin origin's XSS boundary and air-gapped behavior.
