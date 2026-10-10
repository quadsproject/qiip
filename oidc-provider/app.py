"""Toy OIDC provider for local QIIP testing.

Speaks just enough OpenID Connect for the qiip internal_oidc auth plugin:
discovery, authorization-code flow with a password login form, RS256-signed
ID tokens, JWKS, and a userinfo endpoint. Test-only: plaintext password file,
in-memory codes/tokens, and (by default) a self-signed TLS terminator in front
via nginx at https://your-qiip-host.localdomain/oidc.
"""

from __future__ import annotations

import base64
import hmac
import html
import os
import secrets
import time
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

BASE = os.environ.get(
    "OIDC_BASE_URL", "https://your-qiip-host.localdomain/oidc"
).rstrip("/")
PREFIX = os.environ.get("OIDC_BASE_PATH", "/oidc")
CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "my-qiip-client")
CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "change-me")
APP_REDIRECT_URI = os.environ.get(
    "OIDC_REDIRECT_URI", "https://your-qiip-host.localdomain/auth/callback"
)
USERS_FILE = Path(
    os.environ.get("OIDC_USERS_FILE", str(Path(__file__).with_name("users.txt")))
)
KEY_FILE = Path(
    os.environ.get("OIDC_KEY_FILE", str(Path(__file__).with_name("key.pem")))
)
DOMAIN = os.environ.get("OIDC_DOMAIN", "localdomain")
TOKEN_TTL = 600

# ---------------------------------------------------------------------------
# RSA signing key (persisted; regenerated if the file is missing)
# ---------------------------------------------------------------------------


def _load_or_create_key() -> tuple[object, str]:
    if KEY_FILE.exists():
        key = serialization.load_pem_private_key(KEY_FILE.read_bytes(), password=None)
        return key, "local"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(KEY_FILE, 0o600)
    return key, "local"


_PRIVATE_KEY, KID = _load_or_create_key()


def _jwk() -> dict:
    pub = _PRIVATE_KEY.public_key()
    numbers = pub.public_numbers()

    def b64url(n: int) -> str:
        return (
            base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big"))
            .rstrip(b"=")
            .decode()
        )

    return {
        "kty": "RSA",
        "kid": KID,
        "use": "sig",
        "alg": "RS256",
        "n": b64url(numbers.n),
        "e": b64url(numbers.e),
    }


def _sign_id_token(claims: dict) -> str:
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})


# ---------------------------------------------------------------------------
# User store: USERS_FILE, one "username:password" per line (# comments allowed)
# ---------------------------------------------------------------------------

# A local user whose username equals qiip's configured admin username is never
# a provider account: qiip's admin password wins. Setting OIDC_ADMIN_USERNAME
# makes this provider ignore that entry too (mirrors qiip's local login).
ADMIN_USERNAME = os.environ.get("OIDC_ADMIN_USERNAME", "").strip().lower()


def load_users() -> dict[str, str]:
    users: dict[str, str] = {}
    if not USERS_FILE.exists():
        return users
    for line in USERS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, password = line.partition(":")
        name = name.strip().lower()
        if not name or name == ADMIN_USERNAME:
            continue
        users[name] = password
    return users


def check_login(username: str, password: str) -> str | None:
    """Return the canonical username on success, else None."""
    users = load_users()
    name = username.strip().lower()
    if name not in users:
        return None
    # compare_digest only accepts ASCII str or bytes: non-ASCII passwords
    # (UTF-8) must be compared as bytes or they raise TypeError.
    if not hmac.compare_digest(users[name].encode("utf-8"), password.encode("utf-8")):
        return None
    return name


def user_email(name: str) -> str:
    return name if "@" in name else f"{name}@{DOMAIN}"


# In-memory state (single worker).
_codes: dict[
    str, dict
] = {}  # code -> {username, client_id, redirect_uri, nonce, expires}
_tokens: dict[str, dict] = {}  # access_token -> {username, expires}

app = FastAPI(title="qiip toy oidc", openapi_url=None, docs_url=None, redoc_url=None)


@app.get(f"{PREFIX}/.well-known/openid-configuration")
def discovery() -> JSONResponse:
    return JSONResponse(
        {
            "issuer": BASE,
            "authorization_endpoint": f"{BASE}/authorize",
            "token_endpoint": f"{BASE}/token",
            "userinfo_endpoint": f"{BASE}/userinfo",
            "jwks_uri": f"{BASE}/jwks",
            "response_types_supported": ["code"],
            "response_modes_supported": ["query"],
            "grant_types_supported": ["authorization_code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
            "scopes_supported": ["openid", "email", "profile"],
            # Non-standard claim consumed by qiip's internal_oidc plugin so
            # the Local Login form and this provider mint identical emails
            # for the same bare username (single source; OIDC_DOMAIN).
            "user_email_domain": DOMAIN,
            "claims_supported": ["sub", "email", "email_verified", "name"],
            "code_challenge_methods_supported": [],
        }
    )


@app.get(f"{PREFIX}/jwks")
def jwks() -> JSONResponse:
    return JSONResponse({"keys": [_jwk()]})


@app.get(f"{PREFIX}/qiip.svg")
def qiip_mark() -> Response:
    # Derived copy of assets/qiip.svg (same artwork, cropped to 56x46 and
    # flattened to the brand teal for the login card). Kept in-tree so the
    # vendored provider is self-contained: the RPM installs it standalone
    # under /usr/share/qiip/oidc-provider without the gateway package.
    return FileResponse(
        Path(__file__).with_name("qiip.svg"), media_type="image/svg+xml"
    )


# Styling mirrors qiip's dashboard.css: same :root/[data-theme="dark"] token
# swatches, same sign-in card classes, same theme-toggle icons and the same
# persistence rule (localStorage key "theme", default dark, applied to
# <html data-theme> before first paint).
_CSS = """
:root {
  --primary: #106C75;
  --primary-hover: #0F646C;
  --primary-soft: rgba(16, 108, 117, 0.08);
  --primary-soft-strong: rgba(16, 108, 117, 0.16);
  --warning: #D97706;
  --warning-bg: rgba(217, 119, 6, 0.08);
  --surface: #FFFFFF;
  --bg: #F3F4F6;
  --text: #111827;
  --text-light: #6B7280;
  --border: #E5E7EB;
  --border-strong: #D1D5DB;
  --radius-sm: 0.25rem;
  --radius: 0.5rem;
  --radius-lg: 0.75rem;
  --font-display: 'Outfit', 'Manrope', system-ui, -apple-system, sans-serif;
  --font-sans: 'Manrope', system-ui, -apple-system, sans-serif;
  /* Text-on-accent-fill colors, matching qiip's WCAG AA pairing. */
  --on-primary: #FFFFFF;
}

[data-theme="dark"] {
  color-scheme: dark;
  --primary: #29D6D0;
  --primary-hover: #3AD9D4;
  --primary-soft: rgba(41, 214, 208, 0.1);
  --primary-soft-strong: rgba(41, 214, 208, 0.2);
  --warning: #FBBF24;
  --warning-bg: rgba(251, 191, 36, 0.12);
  --surface: #1B2632;
  --bg: #10161D;
  --text: #F3F6F6;
  --text-light: #8B98A5;
  --border: #2A3744;
  --border-strong: #3D4B59;
  --on-primary: #0A1114;
}

*, *::before, *::after { box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  margin: 0;
  font-family: var(--font-sans);
  font-size: 0.875rem;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

button:focus-visible,
input:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: 2px;
}

/* --- Sign-in card (same classes/layout as qiip's signin page) --- */
.signin-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 2rem;
}

.signin-card {
  max-width: 26rem;
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.5rem 2rem;
  text-align: center;
}

.signin-mark { margin-bottom: 0.75rem; }

.signin-card h1 {
  font-family: var(--font-display);
  font-size: 1.375rem;
  margin: 0 0 0.5rem;
}

.signin-notice {
  color: var(--warning);
  background: var(--warning-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.75rem;
  font-size: 0.8125rem;
  margin: 0 0 1rem;
}

.muted-status { color: var(--text-light); font-size: 0.875rem; margin: 0 0 1rem; }

.signin-form {
  display: grid;
  gap: 0.625rem;
  text-align: left;
  margin-top: 1.25rem;
}

.signin-label { font-size: 0.8125rem; font-weight: 600; color: var(--text-light); }

.signin-input {
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 0.875rem;
}

.signin-input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--primary-soft);
}

/* --- Buttons --- */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.4375rem 0.8125rem;
  border: 1px solid transparent;
  border-radius: var(--radius);
  font-family: var(--font-sans);
  font-size: 0.8125rem;
  font-weight: 600;
  line-height: 1.3;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.1s;
}

.btn:active { transform: scale(0.96); }

.btn-primary { background: var(--primary-soft); color: var(--primary); }
.btn-primary:hover { background: var(--primary); color: var(--on-primary); }

.signin-submit { justify-content: center; padding: 0.75rem 1rem; font-size: 0.875rem; }

/* --- Theme toggle (same icons/behavior as qiip) --- */
.theme-toggle {
  position: fixed;
  top: 1rem;
  right: 1rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  width: 2rem;
  height: 2rem;
  padding: 0;
  cursor: pointer;
  color: var(--text-light);
  transition: border-color 0.15s, color 0.15s, transform 0.1s;
}

.theme-toggle:hover { border-color: var(--text); color: var(--text); }
.theme-toggle:active { transform: scale(0.94); }
.theme-toggle svg { width: 16px; height: 16px; }
.theme-toggle .icon-moon { display: none; }
[data-theme="dark"] .theme-toggle .icon-sun { display: none; }
[data-theme="dark"] .theme-toggle .icon-moon { display: block; }
"""

_TOGGLE_JS = """
(function () {
  var t = document.getElementById("qiip-theme");
  if (!t) return;
  t.addEventListener("click", function () {
    var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (err) { /* private mode */ }
  });
})();
"""

_LOGIN_FORM = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QIIP - Sign In</title>
<link rel="icon" type="image/svg+xml" href="{prefix}/qiip.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet">
<style>{css}</style>
<script>var __t=localStorage.getItem('theme')||'dark';document.documentElement.dataset.theme=__t;</script>
</head>
<body>
<main class="signin-page">
<div class="signin-card">
<img class="signin-mark" src="{prefix}/qiip.svg" alt="QIIP" width="56" height="46">
<h1>Local QIIP OIDC sign-in</h1>
{error}
<p class="muted-status">Enter a test user (the @{domain} suffix is added automatically).</p>
<form method="post" action="{prefix}/login" class="signin-form">
<input type="hidden" name="client_id" value="{client_id}">
<input type="hidden" name="redirect_uri" value="{redirect_uri}">
<input type="hidden" name="state" value="{state}">
<input type="hidden" name="scope" value="{scope}">
<input type="hidden" name="nonce" value="{nonce}">
<label class="signin-label" for="oidc-username">Username</label>
<input class="signin-input" id="oidc-username" name="username" autofocus required autocomplete="username">
<label class="signin-label" for="oidc-password">Password</label>
<input class="signin-input" id="oidc-password" type="password" name="password" required autocomplete="current-password">
<button class="btn btn-primary signin-submit" type="submit">Sign in</button>
</form>
</div>
</main>
<button class="theme-toggle" type="button" id="qiip-theme" aria-label="Toggle color theme">
<svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
<svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
</button>
<script>{toggle_js}</script>
</body></html>"""


_ERROR_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QIIP - OIDC error</title>
<style>{css}</style>
</head>
<body>
<main class="signin-page">
<div class="signin-card">
<img class="signin-mark" src="{prefix}/qiip.svg" alt="QIIP" width="56" height="46">
<h1>Sign-in request rejected</h1>
<p class="signin-notice">{error}</p>
</div>
</main>
</body></html>"""


def _bad_request(error: str) -> HTMLResponse:
    """Render a 400 page for invalid authorize/login requests.

    Validation failures must never redirect to a caller-supplied URI (open
    redirect, review #231): only a validated ``redirect_uri`` may receive
    an OAuth error redirect.
    """
    return HTMLResponse(
        _ERROR_PAGE.format(
            prefix=PREFIX,
            css=_CSS,
            error=html.escape(error, quote=True),
        ),
        status_code=400,
    )


@app.get(f"{PREFIX}/authorize")
def authorize(
    response_type: str = "",
    client_id: str = "",
    redirect_uri: str = "",
    state: str = "",
    scope: str = "",
    nonce: str = "",
) -> Response:
    if response_type != "code":
        return _bad_request(
            "unsupported_response_type: only the authorization code flow is supported."
        )
    if client_id != CLIENT_ID:
        return _bad_request("unauthorized_client: unknown client_id.")
    if redirect_uri != APP_REDIRECT_URI:
        return _bad_request(
            "invalid_request: redirect_uri does not match the registered callback."
        )
    # Plaintext redirect (no OAuth parameter leakage beyond code/state).
    page = _LOGIN_FORM.format(
        domain=DOMAIN,
        prefix=PREFIX,
        css=_CSS,
        toggle_js=_TOGGLE_JS,
        error="",
        client_id=html.escape(client_id, quote=True),
        redirect_uri=html.escape(redirect_uri, quote=True),
        state=html.escape(state, quote=True),
        scope=html.escape(scope, quote=True),
        nonce=html.escape(nonce, quote=True),
    )
    return HTMLResponse(page)


@app.post(f"{PREFIX}/login")
async def login(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
    client_id: str = Form(""),
    redirect_uri: str = Form(""),
    state: str = Form(""),
    scope: str = Form(""),
    nonce: str = Form(""),
) -> Response:
    # Re-validate the (caller-supplied) client and callback before doing
    # anything: the success redirect must never go to an unregistered URI
    # (open redirect, review #231). Only validated values are echoed back.
    if client_id != CLIENT_ID or redirect_uri != APP_REDIRECT_URI:
        return _bad_request(
            "invalid_request: client_id or redirect_uri is not registered."
        )
    name = check_login(username, password)
    if name is None:
        page = _LOGIN_FORM.format(
            domain=DOMAIN,
            prefix=PREFIX,
            css=_CSS,
            toggle_js=_TOGGLE_JS,
            error='<p class="signin-notice">Invalid username or password.</p>',
            client_id=html.escape(client_id, quote=True),
            redirect_uri=html.escape(redirect_uri, quote=True),
            state=html.escape(state, quote=True),
            scope=html.escape(scope, quote=True),
            nonce=html.escape(nonce, quote=True),
        )
        return HTMLResponse(page, status_code=401)
    code = secrets.token_urlsafe(24)
    _codes[code] = {
        "username": name,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "nonce": nonce,
        "expires": time.time() + 300,
    }
    from urllib.parse import urlencode

    sep = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(
        f"{redirect_uri}{sep}{urlencode({'code': code, 'state': state})}",
        status_code=302,
    )


@app.post(f"{PREFIX}/token")
async def token(request: Request) -> JSONResponse:
    form = dict(await request.form())
    auth = request.headers.get("authorization", "")
    client_id = form.get("client_id") or ""
    client_secret = form.get("client_secret") or ""
    if auth.lower().startswith("basic "):
        import base64 as b64

        try:
            decoded = b64.b64decode(auth.split(" ", 1)[1]).decode()
            client_id, _, client_secret = decoded.partition(":")
        except Exception:
            pass
    # Non-ASCII secrets must be compared as UTF-8 bytes (compare_digest
    # rejects non-ASCII str, review #231).
    if client_id != CLIENT_ID or not hmac.compare_digest(
        client_secret.encode("utf-8"), CLIENT_SECRET.encode("utf-8")
    ):
        return _token_error("invalid_client")
    grant_type = form.get("grant_type", "")
    code = form.get("code", "")
    if grant_type != "authorization_code" or code not in _codes:
        return _token_error("invalid_grant")
    entry = _codes.pop(code)
    if entry["expires"] < time.time():
        return _token_error("invalid_grant")
    if entry["client_id"] != client_id:
        return _token_error("invalid_grant")
    # The redirect_uri is required and must match the one the code was
    # issued for (review #231), not merely checked when present.
    if form.get("redirect_uri") != entry["redirect_uri"]:
        return _token_error("invalid_grant")

    username = entry["username"]
    email = user_email(username)
    now = int(time.time())
    id_token = _sign_id_token(
        {
            "iss": BASE,
            "sub": email,
            "aud": client_id,
            "iat": now,
            "exp": now + TOKEN_TTL,
            "email": email,
            "email_verified": True,
            "name": username,
            "nonce": entry["nonce"] or None,
        }
    )
    access_token = secrets.token_urlsafe(32)
    _tokens[access_token] = {
        "username": username,
        "email": email,
        "expires": now + TOKEN_TTL,
    }
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": TOKEN_TTL,
            "id_token": id_token,
        }
    )


@app.get(f"{PREFIX}/userinfo")
def userinfo(request: Request) -> JSONResponse:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    entry = _tokens.get(token)
    if entry is None or entry["expires"] < time.time():
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    email = entry["email"]
    return JSONResponse(
        {
            "sub": email,
            "email": email,
            "email_verified": True,
            "name": entry["username"],
        }
    )


def _token_error(code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8081)
