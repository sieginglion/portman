#!/usr/bin/env python3
"""Report recorded direct callees by cumulative source-line count.

The call graph is read from ``portman-valuation-call-edges.json``, which is
created by ``backend.call_recorder``.  A callee's cumulative count includes
the source lines of that callee and every recorded callable reachable from it,
with each callable counted once.

Examples:
    uv run python bin/call_tree_lines.py
    uv run python bin/call_tree_lines.py fetch_xps --edges /tmp/edges.json
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDGES = PROJECT_ROOT / "portman-valuation-call-edges.json"
DEFAULT_SOURCE = PROJECT_ROOT / "backend/valuation.py"
DEFAULT_CALLABLE = "fetch_resolved_income_statement_quarters"


@dataclass(frozen=True)
class Scope:
    """A lexical scope that contributes to a code object's qualified name."""

    name: str
    is_function: bool


class CallableLineCollector(ast.NodeVisitor):
    """Map source-level code-object qualified names to their source lines."""

    def __init__(self) -> None:
        self.line_counts: dict[str, int] = defaultdict(int)
        self._scopes: list[Scope] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node, node.name, is_function=False)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_callable(node, "<lambda>")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_callable(node, "<listcomp>")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_callable(node, "<setcomp>")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_callable(node, "<dictcomp>")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_callable(node, "<genexpr>")

    def _visit_callable(self, node: ast.AST, name: str) -> None:
        self.line_counts[self._qualname(name)] += self._line_count(node)
        self._visit_scoped(node, name, is_function=True)

    def _visit_scoped(self, node: ast.AST, name: str, *, is_function: bool) -> None:
        self._scopes.append(Scope(name, is_function))
        self.generic_visit(node)
        self._scopes.pop()

    def _qualname(self, name: str) -> str:
        parts: list[str] = []
        for scope in self._scopes:
            parts.append(scope.name)
            if scope.is_function:
                parts.append("<locals>")
        parts.append(name)
        return ".".join(parts)

    @staticmethod
    def _line_count(node: ast.AST) -> int:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if not isinstance(lineno, int) or not isinstance(end_lineno, int):
            raise ValueError("callable AST node has no source location")
        return end_lineno - lineno + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "callable",
        nargs="?",
        default=DEFAULT_CALLABLE,
        help=f"recorded root callable (default: {DEFAULT_CALLABLE})",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=DEFAULT_EDGES,
        help=f"recorded call-edge JSON file (default: {DEFAULT_EDGES})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Python source used to count lines (default: {DEFAULT_SOURCE})",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    """Resolve ``path`` and ensure it is a regular file."""
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def load_graph(path: Path) -> dict[str, set[str]]:
    """Load a caller-to-callees graph from the recorder's edge JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid call-edge JSON in {path}: {error}") from None

    if not isinstance(data, list):
        raise SystemExit(f"invalid call-edge JSON in {path}: expected a list of pairs")

    graph: dict[str, set[str]] = defaultdict(set)
    for index, edge in enumerate(data, start=1):
        if not isinstance(edge, list) or len(edge) != 2:
            raise SystemExit(
                f"invalid call-edge JSON in {path}: edge {index} is not a string pair"
            )
        caller, callee = edge
        if not isinstance(caller, str) or not isinstance(callee, str):
            raise SystemExit(
                f"invalid call-edge JSON in {path}: edge {index} is not a string pair"
            )
        graph[caller].add(callee)
        graph.setdefault(callee, set())
    return graph


def collect_line_counts(source: Path) -> dict[str, int]:
    """Return source-line counts keyed by Python code-object qualified name."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    collector = CallableLineCollector()
    collector.visit(tree)
    return collector.line_counts


def reachable_callables(root: str, graph: dict[str, set[str]]) -> set[str]:
    """Return all recorded callables reachable from ``root``, including it."""
    reachable: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(graph.get(current, ()))
    return reachable


def cumulative_line_count(
    root: str, graph: dict[str, set[str]], line_counts: dict[str, int], source: Path
) -> int:
    """Return ``root`` plus all its unique recorded descendants' source lines."""
    callables = reachable_callables(root, graph)
    missing = sorted(callables - line_counts.keys())
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"recorded callable(s) not found in {source}: {names}; "
            "record the graph again after updating the source"
        )
    return sum(line_counts[callable_name] for callable_name in callables)


def main() -> None:
    args = parse_args()
    edges_path = require_file(args.edges, "call-edge file")
    source_path = require_file(args.source, "source file")
    graph = load_graph(edges_path)
    line_counts = collect_line_counts(source_path)

    if args.callable not in line_counts:
        raise SystemExit(f"callable {args.callable!r} was not found in {source_path}")

    direct_callees = graph.get(args.callable, set())
    ranked_callees = sorted(
        (
            (
                callee,
                cumulative_line_count(callee, graph, line_counts, source_path),
            )
            for callee in direct_callees
        ),
        key=lambda item: (-item[1], item[0]),
    )

    print(f"Direct recorded callees of {args.callable} by cumulative source lines:")
    if not ranked_callees:
        print("  none")
        return
    for callee, lines in ranked_callees:
        print(f"  {callee} (cumulative: {lines} lines)")


if __name__ == "__main__":
    main()
