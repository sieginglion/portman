#!/usr/bin/env python3
"""Print a Python file's function call tree with cumulative source line counts.

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

    key: str
    module: str
    qualname: str
    class_qualname: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    own_lines: int


class DefinitionCollector(ast.NodeVisitor):
    """Collect every function definition without treating classes as calls."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.functions: list[FunctionInfo] = []
        self._scope: list[str] = []
        self._class_scopes: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self._class_scopes.append('.'.join(self._scope))
        self.generic_visit(node)
        self._class_scopes.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = '.'.join((*self._scope, node.name))
        self.functions.append(
            FunctionInfo(
                key=f'{self.module}.{qualname}',
                module=self.module,
                qualname=qualname,
                class_qualname=self._class_scopes[-1] if self._class_scopes else None,
                node=node,
                own_lines=node.end_lineno - node.lineno + 1,
            )
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


class CallCollector(ast.NodeVisitor):
    """Collect calls in one function body, excluding nested callable bodies."""

    def __init__(self) -> None:
        self.called_expressions: list[ast.expr] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.called_expressions.append(node.func)
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


def module_name_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix('')
    parts = list(relative.parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def scan_file(root: Path, source: Path) -> dict[str, FunctionInfo]:
    """Collect function definitions from ``source`` only."""
    functions: dict[str, FunctionInfo] = {}
    module = module_name_for(source, root)
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    collector = DefinitionCollector(module)
    collector.visit(tree)

    for function in collector.functions:
        if function.key in functions:
            raise ValueError(f'duplicate function definition: {function.key}')
        functions[function.key] = function
    return functions


def local_name_candidates(function: FunctionInfo, name: str) -> Iterable[str]:
    """Yield names visible from innermost lexical scope through module scope."""
    scope = function.qualname.split('.')
    for end in range(len(scope), -1, -1):
        prefix = '.'.join(scope[:end])
        yield (
            f'{function.module}.{prefix}.{name}'
            if prefix
            else f'{function.module}.{name}'
        )


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

    candidate = f'{function.module}.{function.class_qualname}.{suffix}'
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
        collector = CallCollector()
        for statement in function.node.body:
            collector.visit(statement)
        calls: list[str] = []
        for expression in collector.called_expressions:
            target = resolve_call(expression, function, functions)
            if target and target not in calls:
                calls.append(target)
        calls_by_function[key] = tuple(calls)
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


def cumulative_line_counts(
    functions: dict[str, FunctionInfo], calls_by_function: dict[str, tuple[str, ...]]
) -> dict[str, int]:
    return {
        key: sum(
            functions[target].own_lines
            for target in reachable_functions(key, calls_by_function)
        )
        for key in functions
    }


def reachable_callees_by_cumulative_lines(
    root: str,
    calls_by_function: dict[str, tuple[str, ...]],
    cumulative: dict[str, int],
) -> list[str]:
    """Return root's reachable local callees, ordered by cumulative lines."""
    callees = reachable_functions(root, calls_by_function) - {root}
    return sorted(callees, key=lambda key: (-cumulative[key], key))


def function_label(function: FunctionInfo, cumulative_lines: int) -> str:
    return f'{function.key} (cumulative: {cumulative_lines} lines)'


def render_tree(
    root: str,
    functions: dict[str, FunctionInfo],
    calls_by_function: dict[str, tuple[str, ...]],
    cumulative: dict[str, int],
) -> list[str]:
    """Render a call DAG as a tree, marking repeated functions and cycles."""
    lines = [function_label(functions[root], cumulative[root])]
    expanded = {root}

    def visit(key: str, prefix: str, ancestors: set[str]) -> None:
        children = calls_by_function[key]
        for index, child in enumerate(children):
            last_child = index == len(children) - 1
            branch = '└── ' if last_child else '├── '
            label = function_label(functions[child], cumulative[child])
            if child in ancestors:
                lines.append(f'{prefix}{branch}{label} [cycle]')
                continue
            if child in expanded:
                lines.append(f'{prefix}{branch}{label} [shared helper; shown above]')
                continue
            lines.append(f'{prefix}{branch}{label}')
            expanded.add(child)
            visit(
                child,
                prefix + ('    ' if last_child else '│   '),
                ancestors | {child},
            )

    visit(root, '', {root})
    return lines


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
    source_module: str,
    function_name: str,
    source: Path,
) -> str:
    """Return the requested root key or explain which functions are available."""
    root_key = f'{source_module}.{function_name}'
    if root_key in functions:
        return root_key

    available = sorted(
        function.qualname
        for function in functions.values()
        if function.module == source_module
    )
    choices = ', '.join(available)
    raise SystemExit(
        f'function {function_name!r} was not found in {source}; '
        f'available functions: {choices}'
    )


def main() -> None:
    args = parse_args()
    root, source = resolve_source_path(args.root, args.source)
    functions = scan_file(root, source)
    calls_by_function = collect_calls(functions)
    source_module = module_name_for(source, root)
    root_key = find_root_key(functions, source_module, args.function, source)

    cumulative = cumulative_line_counts(functions, calls_by_function)
    callees = reachable_callees_by_cumulative_lines(
        root_key, calls_by_function, cumulative
    )
    print('Cumulative totals count each reachable function in this file once.')
    print('Reachable local functions by cumulative lines, excluding the root:')
    if callees:
        for key in callees:
            print(f'  {function_label(functions[key], cumulative[key])}')
    else:
        print('  none')
    print()
    print('\n'.join(render_tree(root_key, functions, calls_by_function, cumulative)))


if __name__ == '__main__':
    main()
