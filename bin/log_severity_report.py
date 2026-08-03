#!/usr/bin/env python3
"""Summarize warnings and errors emitted by ``backend/valuation.py``.

Logging call sites are discovered directly from ``backend/valuation.py`` using
its syntax tree.  The log location embedded by Loguru is then matched against
the discovered ``logger.warning``, ``logger.error``, and ``logger.exception``
calls.  This keeps the report synchronized when logging calls are added,
removed, or moved, without depending on dynamic message values such as a CIK,
symbol, or date.

Explicit ``raise`` statements are discovered as error sites too.  Uncaught
errors are identified from traceback frames in ``valuation.py`` and grouped by
their deepest validated valuation frame, source message template, and terminal
exception type.  Warnings and errors from all other modules are ignored.

Examples:
    uv run python bin/log_severity_report.py
    uv run python bin/log_severity_report.py backend.log
    uv run python bin/log_severity_report.py portman.log

Unknown or stale valuation warnings and errors always return a non-zero status.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "portman.log"
DEFAULT_VALUATION_SOURCE = PROJECT_ROOT / "backend/valuation.py"
DEFAULT_MODULE = "backend.valuation"
LOG_RECORD = re.compile(
    r"\|\s+(?P<level>WARNING|ERROR)\s+\|\s+"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_.]*):"
    r"(?P<function>[^:]+):(?P<line>\d+)\s+-"
)
ERROR_BLOCK_MARKER = re.compile(r"^ERROR:")
TRACEBACK_FRAME = re.compile(
    r'^\s*File "(?P<file>.+)", line (?P<line>\d+), in (?P<function>\S+)'
)
EXCEPTION_LINE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*"
    r"(?:Error|Exception|KeyboardInterrupt|SystemExit|GeneratorExit))"
    r"(?::\s*(?P<message>.*))?$"
)
LOGGER_LEVELS = {
    "warning": "WARNING",
    "error": "ERROR",
    "exception": "ERROR",
}


@dataclass(frozen=True)
class FunctionLocation:
    """The source range occupied by a function."""

    name: str
    first_line: int
    last_line: int


@dataclass(frozen=True)
class LogSite:
    """A logging call discovered in the valuation source."""

    level: str
    function: str
    line: int
    template: str

    @property
    def category(self) -> str:
        return f"{self.function}:{self.line} - {self.template}"


@dataclass(frozen=True)
class RaiseSite:
    """An explicit raise statement discovered in the valuation source."""

    exception: str
    function: str
    line: int
    template: str | None

    def category(self, terminal_exception: str) -> str:
        message = f": {self.template}" if self.template is not None else ""
        return f"{self.function}:{self.line} - {terminal_exception}{message}"


@dataclass(frozen=True)
class SourceMetadata:
    """Logging and function locations discovered from the valuation source."""

    log_sites: dict[tuple[str, int], LogSite]
    raise_sites: dict[int, RaiseSite]
    functions: tuple[FunctionLocation, ...]
    line_count: int


@dataclass
class ErrorBlock:
    """Traceback details collected after an ``ERROR:`` log marker."""

    terminal_exception: str | None = None
    valuation_frames: list[tuple[int, str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help=f"log file to analyze (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--valuation-source",
        type=Path,
        default=DEFAULT_VALUATION_SOURCE,
        help=f"source used to discover log sites (default: {DEFAULT_VALUATION_SOURCE})",
    )
    parser.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"module name expected in Loguru records (default: {DEFAULT_MODULE})",
    )
    return parser.parse_args()


def enclosing_function(line: int, functions: tuple[FunctionLocation, ...]) -> str:
    """Return the innermost function containing ``line``."""
    matches = [
        function
        for function in functions
        if function.first_line <= line <= function.last_line
    ]
    if not matches:
        return "<module>"
    return min(
        matches, key=lambda function: function.last_line - function.first_line
    ).name


def string_template(message: ast.expr) -> str:
    """Return a compact display form of a string or f-string expression."""
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        return message.value.replace("\n", r"\n")
    return ast.unparse(message).replace("\n", r"\n")


def log_template(call: ast.Call) -> str:
    """Return a logging call's source message template."""
    if not call.args:
        return "<no message>"
    return string_template(call.args[0])


def raise_site(node: ast.Raise, function: str) -> RaiseSite:
    """Describe an explicit raise statement from its syntax tree node."""
    if isinstance(node.exc, ast.Call):
        expression, arguments = node.exc.func, node.exc.args
    else:
        expression, arguments = node.exc, ()
    exception = ast.unparse(expression) if expression else "<reraised exception>"
    return RaiseSite(
        exception=exception,
        function=function,
        line=node.lineno,
        template=string_template(arguments[0]) if arguments else None,
    )


class SourceVisitor(ast.NodeVisitor):
    """Collect source metadata while tracking the enclosing function."""

    def __init__(self) -> None:
        self.functions: list[FunctionLocation] = []
        self.log_sites: dict[tuple[str, int], LogSite] = {}
        self.raise_sites: dict[int, RaiseSite] = {}
        self.function_stack: list[str] = []

    @property
    def function(self) -> str:
        return self.function_stack[-1] if self.function_stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(
            FunctionLocation(node.name, node.lineno, node.end_lineno or node.lineno)
        )
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raise_sites[node.lineno] = raise_site(node, self.function)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and (level := LOGGER_LEVELS.get(node.func.attr)) is not None
        ):
            line = node.func.lineno
            self.log_sites[level, line] = LogSite(
                level=level,
                function=self.function,
                line=line,
                template=log_template(node),
            )
        self.generic_visit(node)


def load_source_metadata(source: Path) -> SourceMetadata:
    """Discover functions and warning/error logging call sites in ``source``."""
    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))
    visitor = SourceVisitor()
    visitor.visit(tree)
    return SourceMetadata(
        log_sites=visitor.log_sites,
        raise_sites=visitor.raise_sites,
        functions=tuple(visitor.functions),
        line_count=len(source_text.splitlines()),
    )


def classify_log_record(
    line: str,
    module: str,
    metadata: SourceMetadata,
) -> tuple[str, str] | None:
    """Classify a Loguru valuation record by its discovered source call site."""
    match = LOG_RECORD.search(line)
    if match is None or match.group("module") != module:
        return None

    level = match.group("level")
    source_line = int(match.group("line"))
    site = metadata.log_sites.get((level, source_line))
    if site is None or site.function != match.group("function"):
        return level, f"Stale {level.lower()} source location"
    return level, site.category


def traceback_source_matches(frame_file: str, valuation_source: Path) -> bool:
    """Return whether a traceback path refers to the valuation source."""
    frame_path = Path(frame_file)
    try:
        if frame_path.resolve() == valuation_source:
            return True
    except OSError:
        pass
    # Logs remain recognizable after the project checkout has been moved.
    source_parts = valuation_source.parts[-2:]
    return len(frame_path.parts) >= 2 and frame_path.parts[-2:] == source_parts


def valuation_traceback_frame(
    line: str, valuation_source: Path
) -> tuple[int, str] | None:
    """Return a valuation traceback line and function, if present."""
    match = TRACEBACK_FRAME.match(line)
    if match is None or not traceback_source_matches(
        match.group("file"), valuation_source
    ):
        return None
    return int(match.group("line")), match.group("function")


def classify_error_block(
    error_block: ErrorBlock, metadata: SourceMetadata
) -> tuple[str, bool] | None:
    """Return an error category and whether it is unknown or stale."""
    if not error_block.valuation_frames:
        return None

    source_line, traceback_function = error_block.valuation_frames[-1]
    current_function = enclosing_function(source_line, metadata.functions)
    if source_line > metadata.line_count or current_function != traceback_function:
        return "Stale traceback source location", True
    if error_block.terminal_exception is None:
        return "ERROR without terminal exception", True

    site = metadata.raise_sites.get(source_line)
    terminal_exception = error_block.terminal_exception
    terminal_name = terminal_exception.rsplit(".", 1)[-1]
    if (
        site is not None
        and site.function == traceback_function
        and site.exception.rsplit(".", 1)[-1] == terminal_name
    ):
        return site.category(terminal_exception), False
    return (
        "Unknown error source: "
        f"{traceback_function}:{source_line} - {terminal_exception}",
        True,
    )


def summarize(
    log: Path,
    valuation_source: Path,
    module: str,
    metadata: SourceMetadata,
) -> tuple[Counter[str], Counter[str], int, int]:
    """Classify valuation Loguru records and uncaught error traceback blocks."""
    warnings: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    unknown_warnings = 0
    unknown_errors = 0
    error_block: ErrorBlock | None = None

    def finish_error_block() -> None:
        nonlocal error_block, unknown_errors
        if error_block is None:
            return
        if result := classify_error_block(error_block, metadata):
            category, is_unknown = result
            errors[category] += 1
            unknown_errors += is_unknown
        error_block = None

    with log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            record = classify_log_record(line, module, metadata)
            if record is not None:
                level, category = record
                if level == "WARNING":
                    warnings[category] += 1
                    if category.startswith("Stale "):
                        unknown_warnings += 1
                else:
                    errors[category] += 1
                    if category.startswith("Stale "):
                        unknown_errors += 1

            if ERROR_BLOCK_MARKER.match(line):
                finish_error_block()
                error_block = ErrorBlock()
                continue

            if error_block is not None:
                if frame := valuation_traceback_frame(line, valuation_source):
                    error_block.valuation_frames.append(frame)
                if match := EXCEPTION_LINE.match(line.rstrip()):
                    error_block.terminal_exception = match.group("name")

    finish_error_block()
    return warnings, errors, unknown_warnings, unknown_errors


def print_counts(title: str, counts: Counter[str]) -> None:
    """Print categories in descending frequency."""
    print(f"{title}: {sum(counts.values())}")
    for name, count in counts.most_common():
        print(f"  {count:>4}  {name}")


def main() -> int:
    args = parse_args()
    log = args.log.resolve()
    if not log.is_file():
        raise SystemExit(f"log file does not exist: {log}")
    valuation_source = args.valuation_source.resolve()
    if not valuation_source.is_file():
        raise SystemExit(f"valuation source does not exist: {valuation_source}")

    metadata = load_source_metadata(valuation_source)
    warnings, errors, unknown_warnings, unknown_errors = summarize(
        log, valuation_source, args.module, metadata
    )
    print(f"Log: {log}")
    print(f"Source: {valuation_source}")
    print(
        "Discovered source sites: "
        f"{sum(site.level == 'WARNING' for site in metadata.log_sites.values())} warnings, "
        f"{sum(site.level == 'ERROR' for site in metadata.log_sites.values())} logged errors, "
        f"{len(metadata.raise_sites)} raises"
    )
    print_counts(f"Warnings from {args.module}", warnings)
    print_counts(f"Errors from {args.module}", errors)

    unknown = unknown_warnings + unknown_errors
    if unknown:
        print(
            f"\nFailed: {unknown} unknown or stale valuation entries "
            f"({unknown_warnings} warnings, {unknown_errors} errors)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
