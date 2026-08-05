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
    assert coverage["report"]["fail_under"] == 92
    assert coverage["report"]["precision"] == 2
    assert coverage["report"]["show_missing"] is True

    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert workflow.count("uv run --frozen coverage run -m pytest") == 1
    assert workflow.count("uv run --frozen coverage report") == 1


def test_ci_exercises_badge_extraction_without_blocking_on_publication() -> None:
    """PRs exercise extraction; optional Gist writes cannot fail Quality."""
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    quality = workflow.split("  quality:\n", 1)[1].split("  publish-badges:\n", 1)[0]
    extraction = quality.split("      - name: Extract badge values\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    publication = workflow.split("  publish-badges:\n", 1)[1].split(
        "  compatibility:\n", 1
    )[0]

    assert "        id: badges" in extraction
    assert "        if:" not in extraction
    assert "dynamic-badges-action" not in quality

    assert "if: github.event_name == 'push'" in publication
    assert "github.ref == 'refs/heads/main'" in publication
    assert "needs: quality" in publication
    assert publication.count("continue-on-error: true") == 3
    assert publication.count("schneegans/dynamic-badges-action@") == 3
