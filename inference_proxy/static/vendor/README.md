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
