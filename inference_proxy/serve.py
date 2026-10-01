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
    kwargs: dict[str, Any] = {
        "app": "inference_proxy.main:create_app",
        "factory": True,
        "host": server.host,
        "port": server.port,
        "workers": server.workers,
        "limit_concurrency": server.limit_concurrency,
        "log_level": server.log_level,
        # Give in-flight requests (long SSE streams) time to finish on
        # shutdown instead of dropping them when the process restarts.
        "timeout_graceful_shutdown": 30,
        # The app logs every request itself, with setup-link ids redacted
        # (RequestLoggingMiddleware). Uvicorn's access log would record them.
        "access_log": False,
    }
    # A single worker runs without a supervisor, so enforcing the request
    # limit would terminate the process (outage) instead of recycling.
    # Only recycle when more than one worker is run.
    if server.workers > 1:
        kwargs["limit_max_requests"] = server.max_requests
        kwargs["limit_max_requests_jitter"] = server.max_requests_jitter
    return kwargs


def main() -> None:
    uvicorn.run(**uvicorn_kwargs())


if __name__ == "__main__":
    main()
