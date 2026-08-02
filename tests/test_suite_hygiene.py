"""Structural policies for tests that target the current application API."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

TESTS_ROOT = Path(__file__).parent
_FORBIDDEN_COMPAT_EXCEPTIONS = {
    "ImportError",
    "ModuleNotFoundError",
    "TypeError",
}


def _exception_names(expression: ast.expr | None) -> Iterator[str]:
    if isinstance(expression, ast.Name):
        yield expression.id
    elif isinstance(expression, ast.Attribute):
        yield expression.attr
    elif isinstance(expression, ast.Tuple):
        for item in expression.elts:
            yield from _exception_names(item)


def _inspect_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_names: set[str] = set()
    signature_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "inspect":
                    module_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "inspect":
            for alias in node.names:
                if alias.name == "signature":
                    signature_names.add(alias.asname or alias.name)
    return module_names, signature_names


def _is_inspect_signature_call(
    node: ast.Call,
    module_names: set[str],
    signature_names: set[str],
) -> bool:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id in signature_names
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "signature"
        and isinstance(function.value, ast.Name)
        and function.value.id in module_names
    )


def test_suite_has_no_banned_historical_compatibility_patterns() -> None:
    """Reject import-error catches, type-error catches, and signature probes.

    This intentionally does not ban every use of ``hasattr`` or ``getattr``;
    those have legitimate assertion and data-access uses that require review.
    Historical API comparisons belong in worktrees rather than the live suite.
    """
    violations: list[str] = []

    for path in sorted(TESTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module_names, signature_names = _inspect_aliases(tree)
        relative_path = path.relative_to(TESTS_ROOT.parent)

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                caught = _FORBIDDEN_COMPAT_EXCEPTIONS.intersection(
                    _exception_names(node.type)
                )
                if caught:
                    names = ", ".join(sorted(caught))
                    violations.append(f"{relative_path}:{node.lineno}: catches {names}")
            elif isinstance(node, ast.Call) and _is_inspect_signature_call(
                node,
                module_names,
                signature_names,
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: calls inspect.signature"
                )

    assert violations == [], "Historical compatibility shims found:\n" + "\n".join(
        violations
    )
