#!/usr/bin/env python3
"""Print a local Python function call tree with cumulative source line counts.

By default this analyses ``resolve_us_income_statement_quarters`` in
``backend/valuation.py``.  It follows calls that can be resolved statically to
functions in the same local package.  Calls to third-party modules, builtins,
and dynamic callables are intentionally excluded.

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
    """A function definition and the local calls made by its body."""

    key: str
    module: str
    qualname: str
    class_qualname: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    own_lines: int
    calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleImports:
    """Names imported into a module and their fully qualified targets."""

    modules: dict[str, str]
    symbols: dict[str, str]


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
    """Return an identifier/attribute chain, such as ``shared.to_date``."""
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


def absolute_import_module(
    current_module: str, level: int, imported_module: str | None
) -> str:
    """Resolve an ``ImportFrom`` module name without importing project code."""
    package_parts = current_module.split('.')[:-1]
    if level:
        package_parts = package_parts[: len(package_parts) - (level - 1)]
    imported_parts = imported_module.split('.') if imported_module else []
    return '.'.join((*package_parts, *imported_parts))


def collect_imports(tree: ast.Module, module: str) -> ModuleImports:
    modules: dict[str, str] = {}
    symbols: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for imported in statement.names:
                local_name = imported.asname or imported.name.split('.')[0]
                modules[local_name] = imported.name if imported.asname else local_name
        elif isinstance(statement, ast.ImportFrom):
            imported_module = absolute_import_module(
                module, statement.level, statement.module
            )
            for imported in statement.names:
                if imported.name == '*':
                    continue
                local_name = imported.asname or imported.name
                target = f'{imported_module}.{imported.name}'
                # ``from . import shared`` imports the sibling module itself.
                if statement.module is None:
                    modules[local_name] = target
                else:
                    symbols[local_name] = target
    return ModuleImports(modules=modules, symbols=symbols)


def scan_package(
    root: Path, package_directory: Path
) -> tuple[dict[str, FunctionInfo], dict[str, ModuleImports]]:
    functions: dict[str, FunctionInfo] = {}
    module_imports: dict[str, ModuleImports] = {}
    collected: list[FunctionInfo] = []

    for path in sorted(package_directory.rglob('*.py')):
        module = module_name_for(path, root)
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        module_imports[module] = collect_imports(tree, module)
        collector = DefinitionCollector(module)
        collector.visit(tree)
        collected.extend(collector.functions)

    for function in collected:
        if function.key in functions:
            raise ValueError(f'duplicate function definition: {function.key}')
        functions[function.key] = function
    return functions, module_imports


def local_name_candidates(function: FunctionInfo, name: str) -> Iterable[str]:
    """Yield names visible from innermost lexical scope through module scope."""
    scope = function.qualname.split('.')[:-1]
    for end in range(len(scope), -1, -1):
        prefix = '.'.join(scope[:end])
        yield f'{function.module}.{prefix}.{name}' if prefix else f'{function.module}.{name}'


def resolve_call(
    expression: ast.expr,
    function: FunctionInfo,
    functions: dict[str, FunctionInfo],
    imports: ModuleImports,
) -> str | None:
    name = dotted_name(expression)
    if name is None:
        return None

    if '.' not in name:
        for candidate in local_name_candidates(function, name):
            if candidate in functions:
                return candidate
        imported_symbol = imports.symbols.get(name)
        return imported_symbol if imported_symbol in functions else None

    root_name, suffix = name.split('.', 1)
    if root_name in {'self', 'cls'} and function.class_qualname:
        candidate = f'{function.module}.{function.class_qualname}.{suffix}'
    elif root_name in imports.modules:
        candidate = f'{imports.modules[root_name]}.{suffix}'
    elif root_name in imports.symbols:
        candidate = f'{imports.symbols[root_name]}.{suffix}'
    else:
        candidate = name
    return candidate if candidate in functions else None


def add_calls(
    functions: dict[str, FunctionInfo], module_imports: dict[str, ModuleImports]
) -> dict[str, FunctionInfo]:
    resolved: dict[str, FunctionInfo] = {}
    for key, function in functions.items():
        collector = CallCollector()
        for statement in function.node.body:
            collector.visit(statement)
        calls: list[str] = []
        for expression in collector.called_expressions:
            target = resolve_call(
                expression, function, functions, module_imports[function.module]
            )
            if target and target not in calls:
                calls.append(target)
        resolved[key] = FunctionInfo(
            key=function.key,
            module=function.module,
            qualname=function.qualname,
            class_qualname=function.class_qualname,
            node=function.node,
            own_lines=function.own_lines,
            calls=tuple(calls),
        )
    return resolved


def reachable_functions(root: str, functions: dict[str, FunctionInfo]) -> set[str]:
    """Return the set of unique local functions reachable from ``root``."""
    reachable: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(functions[current].calls)
    return reachable


def cumulative_line_counts(functions: dict[str, FunctionInfo]) -> dict[str, int]:
    return {
        key: sum(functions[target].own_lines for target in reachable_functions(key, functions))
        for key in functions
    }


def function_label(function: FunctionInfo, cumulative_lines: int) -> str:
    return (
        f'{function.key} '
        f'(own: {function.own_lines} lines; cumulative: {cumulative_lines} lines)'
    )


def render_tree(
    root: str, functions: dict[str, FunctionInfo], cumulative: dict[str, int]
) -> list[str]:
    """Render a call DAG as a tree, marking repeated functions and cycles."""
    lines = [function_label(functions[root], cumulative[root])]
    expanded = {root}

    def visit(key: str, prefix: str, ancestors: set[str]) -> None:
        children = functions[key].calls
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


def package_root_for(source: Path) -> Path:
    """Return the outermost package directory containing ``source``."""
    package_root = source.parent
    while (package_root.parent / '__init__.py').is_file():
        package_root = package_root.parent
    return package_root


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f'source file does not exist: {source}')
    try:
        source.relative_to(root)
    except ValueError:
        raise SystemExit(f'source file must be below --root: {source}') from None

    package_directory = package_root_for(source)
    functions, module_imports = scan_package(root, package_directory)
    functions = add_calls(functions, module_imports)
    source_module = module_name_for(source, root)
    root_key = f'{source_module}.{args.function}'
    if root_key not in functions:
        available = sorted(
            function.qualname
            for function in functions.values()
            if function.module == source_module
        )
        choices = ', '.join(available)
        raise SystemExit(
            f'function {args.function!r} was not found in {source}; '
            f'available functions: {choices}'
        )

    cumulative = cumulative_line_counts(functions)
    print('Line metric: physical source lines from `def` through its final body line.')
    print('Cumulative totals count each reachable local function once.')
    print()
    print('\n'.join(render_tree(root_key, functions, cumulative)))


if __name__ == '__main__':
    main()
