"""Run the gateway worker farm from resolved settings.

The RPM unit executes this module instead of calling uvicorn directly so the
YAML ``server:`` block (with ``INFERENCE_PROXY_SERVER__*`` overrides, like
every other QIIP setting) drives the ASGI server tuning.
"""

from __future__ import annotations

from typing import Any

import uvicorn

from inference_proxy.config.dependencies import get_settings


def uvicorn_kwargs() -> dict[str, Any]:
    """Return uvicorn.run() arguments from the resolved application settings."""
    server = get_settings().server
    return {
        "app": "inference_proxy.main:create_app",
        "factory": True,
        "host": server.host,
        "port": server.port,
        "workers": server.workers,
        "limit_concurrency": server.limit_concurrency,
        "limit_max_requests": server.max_requests,
        "limit_max_requests_jitter": server.max_requests_jitter,
        "log_level": server.log_level,
    }


def main() -> None:
    uvicorn.run(**uvicorn_kwargs())


if __name__ == "__main__":
    main()
