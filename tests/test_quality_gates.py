"""Structural guards for repository-wide quality gates."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_ci_enforces_branch_coverage_floor() -> None:
    """Pin CI coverage collection and its floor, not behavioral correctness."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    coverage = project["tool"]["coverage"]

    assert coverage["run"]["branch"] is True
    assert coverage["run"]["source"] == ["inference_proxy"]
    assert coverage["report"]["fail_under"] == 91.5
    assert coverage["report"]["precision"] == 2
    assert coverage["report"]["show_missing"] is True

    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert workflow.count("uv run --frozen coverage run -m pytest") == 1
    assert workflow.count("uv run --frozen coverage report") == 1
