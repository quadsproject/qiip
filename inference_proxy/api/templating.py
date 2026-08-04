"""Shared Jinja environment with content-versioned static asset URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

_BASE_DIR = Path(__file__).resolve().parent.parent
_STATIC_DIR = (_BASE_DIR / "static").resolve()


def static_asset_url(request: Request, path: str) -> str:
    """Return a static URL whose query changes with the file contents.

    The dashboard JavaScript files are interdependent. Content versions keep a
    newly rendered HTML shell from running against an older browser-cached
    script generation after a deployment.
    """
    asset = (_STATIC_DIR / path).resolve()
    try:
        asset.relative_to(_STATIC_DIR)
    except ValueError as exc:
        raise ValueError("static asset path escapes the static directory") from exc
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
    return f"{request.url_for('static', path=path)}?v={digest}"


templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
templates.env.globals["static_asset_url"] = static_asset_url
