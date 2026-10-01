"""Shared model labels for UI and harness configuration."""

import re


def model_display_name(model: str) -> str:
    """Return a user-facing label without changing the API model identity."""
    return re.sub(r"(?:-MTP)?-GGUF$", "", model.rsplit("/", 1)[-1], flags=re.I) or model
