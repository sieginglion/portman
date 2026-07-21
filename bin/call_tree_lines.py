#!/usr/bin/env python3
"""List a Python function's reachable local helpers by cumulative line count.

By default this analyses ``resolve_us_income_statement_quarters`` in
``backend/valuation.py``. It follows calls that can be resolved statically to
functions defined in that same file. Calls to other modules, builtins, and
dynamic callables are intentionally excluded.

Examples:
    python bin/call_tree_lines.py
    python bin/call_tree_lines.py backend/valuation.py fetch_xps
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('backend/valuation.py')
DEFAULT_FUNCTION = 'resolve_us_income_statement_quarters'


@dataclass(frozen=True)
class FunctionInfo:
    """Metadata for a function definition."""

    qualname: str
    class_qualname: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    own_lines: int


class DefinitionCollector(ast.NodeVisitor):
    """Collect every function definition without treating classes as calls."""

    def __init__(self) -> None:
        self.functions: list[FunctionInfo] = []
        self._scope: list[str] = []
        self._class_qualname: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous_class_qualname = self._class_qualname
        self._scope.append(node.name)
        self._class_qualname = '.'.join(self._scope)
        self.generic_visit(node)
        self._class_qualname = previous_class_qualname
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = '.'.join((*self._scope, node.name))
        self.functions.append(
            FunctionInfo(
                qualname=qualname,
                class_qualname=self._class_qualname,
                node=node,
                own_lines=node.end_lineno - node.lineno + 1,
            )
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


class CallCollector(ast.NodeVisitor):
    """Collect resolved local calls in one function body."""

    def __init__(
        self,
        function: FunctionInfo,
        functions: dict[str, FunctionInfo],
    ) -> None:
        self.function = function
        self.functions = functions
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = resolve_call(node.func, self.function, self.functions)
        if target and target not in self.calls:
            self.calls.append(target)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def dotted_name(node: ast.expr) -> str | None:
    """Return an identifier/attribute chain, such as ``self.to_date``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f'{prefix}.{node.attr}' if prefix else None
    return None


def scan_file(source: Path) -> dict[str, FunctionInfo]:
    """Collect function definitions from ``source`` only."""
    functions: dict[str, FunctionInfo] = {}
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    collector = DefinitionCollector()
    collector.visit(tree)

    for function in collector.functions:
        if function.qualname in functions:
            raise ValueError(f'duplicate function definition: {function.qualname}')
        functions[function.qualname] = function
    return functions


def local_name_candidates(function: FunctionInfo, name: str) -> Iterable[str]:
    """Yield names visible from innermost lexical scope through module scope."""
    scope = function.qualname.split('.')
    for end in range(len(scope), -1, -1):
        prefix = '.'.join(scope[:end])
        yield f'{prefix}.{name}' if prefix else name


def resolve_bare_call(
    name: str,
    function: FunctionInfo,
    functions: dict[str, FunctionInfo],
) -> str | None:
    """Resolve an unqualified call through enclosing local scopes."""
    return next(
        (
            candidate
            for candidate in local_name_candidates(function, name)
            if candidate in functions
        ),
        None,
    )


def resolve_bound_method_call(
    name: str,
    function: FunctionInfo,
    functions: dict[str, FunctionInfo],
) -> str | None:
    """Resolve a ``self`` or ``cls`` method call in the enclosing class."""
    root_name, suffix = name.split('.', 1)
    if root_name not in {'self', 'cls'} or not function.class_qualname:
        return None

    candidate = f'{function.class_qualname}.{suffix}'
    return candidate if candidate in functions else None


def resolve_call(
    expression: ast.expr,
    function: FunctionInfo,
    functions: dict[str, FunctionInfo],
) -> str | None:
    """Resolve a supported local call expression to a function key."""
    name = dotted_name(expression)
    if name is None:
        return None
    if '.' not in name:
        return resolve_bare_call(name, function, functions)
    return resolve_bound_method_call(name, function, functions)


def collect_calls(functions: dict[str, FunctionInfo]) -> dict[str, tuple[str, ...]]:
    """Return the resolved local calls made by each function."""
    calls_by_function: dict[str, tuple[str, ...]] = {}
    for key, function in functions.items():
        collector = CallCollector(function, functions)
        for statement in function.node.body:
            collector.visit(statement)
        calls_by_function[key] = tuple(collector.calls)
    return calls_by_function


def reachable_functions(
    root: str, calls_by_function: dict[str, tuple[str, ...]]
) -> set[str]:
    """Return the set of unique local functions reachable from ``root``."""
    reachable: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(calls_by_function[current])
    return reachable


def cumulative_line_count(
    root: str,
    functions: dict[str, FunctionInfo],
    calls_by_function: dict[str, tuple[str, ...]],
) -> int:
    """Return the combined line count of root and its reachable functions."""
    return sum(
        functions[target].own_lines
        for target in reachable_functions(root, calls_by_function)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'source',
        nargs='?',
        default=DEFAULT_SOURCE,
        type=Path,
        help=f'source file containing the root function (default: {DEFAULT_SOURCE})',
    )
    parser.add_argument(
        'function',
        nargs='?',
        default=DEFAULT_FUNCTION,
        help=f'root function name (default: {DEFAULT_FUNCTION})',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=PROJECT_ROOT,
        help='repository root used to identify local modules',
    )
    return parser.parse_args()


def resolve_source_path(root: Path, source: Path) -> tuple[Path, Path]:
    """Resolve and validate a source file beneath the repository root."""
    root = root.resolve()
    source = source if source.is_absolute() else root / source
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f'source file does not exist: {source}')
    try:
        source.relative_to(root)
    except ValueError:
        raise SystemExit(f'source file must be below --root: {source}') from None
    return root, source


def find_root_key(
    functions: dict[str, FunctionInfo],
    function_name: str,
    source: Path,
) -> str:
    """Return the requested root key or explain which functions are available."""
    if function_name in functions:
        return function_name

    available = sorted(function.qualname for function in functions.values())
    choices = ', '.join(available)
    raise SystemExit(
        f'function {function_name!r} was not found in {source}; '
        f'available functions: {choices}'
    )


def main() -> None:
    args = parse_args()
    _, source = resolve_source_path(args.root, args.source)
    functions = scan_file(source)
    calls_by_function = collect_calls(functions)
    root_key = find_root_key(functions, args.function, source)

    callees = reachable_functions(root_key, calls_by_function) - {root_key}
    cumulative = {
        key: cumulative_line_count(key, functions, calls_by_function) for key in callees
    }
    ranked_callees = sorted(callees, key=lambda key: (-cumulative[key], key))
    print('Cumulative totals count each reachable function in this file once.')
    print('Reachable local functions by cumulative lines, excluding the root:')
    if ranked_callees:
        for key in ranked_callees:
            print(f'  {key} (cumulative: {cumulative[key]} lines)')
    else:
        print('  none')


if __name__ == '__main__':
    main()
