#!/usr/bin/env python3
"""Summarize warning and error types emitted by ``backend/valuation.py``.

Warning types are identified from the ``backend.valuation:function:line``
location embedded by Loguru, then checked against ``backend/valuation.py``.
This avoids depending on dynamic values inside a warning message, such as a
CIK, symbol, or date.

Examples:
    uv run python bin/log_severity_report.py
    uv run python bin/log_severity_report.py backend.log
    uv run python bin/log_severity_report.py portman.log --strict

Warnings and errors from all other modules are ignored.  ``--strict`` returns
a non-zero status if a valuation entry does not match a known type or no
longer points at a valid source location.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "portman.log"
DEFAULT_VALUATION_SOURCE = PROJECT_ROOT / "backend/valuation.py"
WARNING_MARKER = re.compile(r"\| WARNING\s+\|")
WARNING_SOURCE = re.compile(
    r"\| WARNING\s+\|\s+"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_.]*):"
    r"(?P<function>[A-Za-z_][A-Za-z0-9_]*):(?P<line>\d+)\s+-"
)
ERROR_MARKER = re.compile(r"^ERROR:")
TRACEBACK_FRAME = re.compile(r'^\s*File "(?P<file>.+)", line (?P<line>\d+), in ')
EXCEPTION_LINE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(?P<message>.*)$"
)
CONSENSUS_FIELD = re.compile(r"^no consensus .*\bfield=(?P<field>\w+)$")
ALIGNED_QUARTERS = re.compile(r"^need (?P<quarters>\d+) aligned quarters")


@dataclass(frozen=True)
class WarningType:
    """A warning's semantic type and its emitting valuation function."""

    name: str
    function: str
    description: str


WARNING_TYPES = (
    WarningType(
        "SEC concept fetch failed",
        "fetch_first_available_sec_field",
        "An SEC XBRL concept request failed (usually a 404); the concept is skipped and other sources remain usable.",
    ),
    WarningType(
        "SEC concept returned empty units",
        "fetch_sec_concept_rows",
        "The SEC response contained the concept but no values in the expected unit; that concept is skipped.",
    ),
    WarningType(
        "Latest quarter dropped (insufficient sources)",
        "drop_incomplete_latest_us_quarter",
        "The newest reported quarter had fewer than two usable data sources, so it is excluded from valuation inputs.",
    ),
    WarningType(
        "Quarter consensus unavailable",
        "resolve_us_quarter_consensus",
        "No usable consensus could be formed for a required field in a quarter; downstream quarter resolution may fail.",
    ),
    WarningType(
        "SEC lookup skipped (unsupported forms)",
        "dedupe_sec_rows",
        "The available SEC facts used unsupported filing forms, so no SEC reference values were added.",
    ),
    WarningType(
        "SEC reference unavailable (missing CIK)",
        "add_sec_reference_values",
        "No CIK was available from FMP or Massive, so SEC reference data could not be requested.",
    ),
)
WARNING_TYPE_BY_FUNCTION = {warning_type.function: warning_type for warning_type in WARNING_TYPES}
WARNING_DESCRIPTIONS = {
    warning_type.name: warning_type.description for warning_type in WARNING_TYPES
}
ERROR_DESCRIPTIONS = {
    "ValueError: no consensus": (
        "A required field had no usable cross-source consensus for its quarter; the endpoint request fails."
    ),
    "ValueError: insufficient aligned quarters": (
        "Fewer resolved quarters were available than the requested growth calculation requires; the endpoint request fails."
    ),
    "ValueError: other": "A valuation ValueError did not match a known failure mode.",
    "ERROR without terminal exception": "The valuation traceback ended without a recognizable exception line.",
    "Stale valuation traceback location": (
        "The traceback references a line beyond the current valuation source, suggesting the log and source revision differ."
    ),
}


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
        "--strict",
        action="store_true",
        help="exit 1 when a warning or error has no known type or source",
    )
    parser.add_argument(
        "--valuation-source",
        type=Path,
        default=DEFAULT_VALUATION_SOURCE,
        help=f"source used to verify warning locations (default: {DEFAULT_VALUATION_SOURCE})",
    )
    return parser.parse_args()


def load_source_metadata(source: Path) -> tuple[set[int], int]:
    """Return ``logger.warning`` lines and the total line count in ``source``."""
    source_lines = source.read_text(encoding="utf-8").splitlines()
    warning_lines = {
        line_number
        for line_number, line in enumerate(source_lines, 1)
        if "logger.warning" in line
    }
    return warning_lines, len(source_lines)


def classify_warning(line: str, source_warning_lines: set[int]) -> str | None:
    """Classify a valuation warning by its validated call site, or ignore it."""
    match = WARNING_SOURCE.search(line)
    if match is None or match.group("module") != "backend.valuation":
        return None

    warning_type = WARNING_TYPE_BY_FUNCTION.get(match.group("function"))
    if warning_type is None:
        return "Unknown warning source"

    if int(match.group("line")) not in source_warning_lines:
        return "Stale warning source location"

    return warning_type.name


def classify_error(exception: tuple[str, str] | None) -> tuple[str, str | None]:
    """Return an error type and optional subtype from a traceback's final exception."""
    if exception is None:
        return "ERROR without terminal exception", None

    name, message = exception
    if name != "ValueError":
        return name, None

    consensus = CONSENSUS_FIELD.match(message)
    if consensus:
        return "ValueError: no consensus", consensus.group("field")

    quarters = ALIGNED_QUARTERS.match(message)
    if quarters:
        return "ValueError: insufficient aligned quarters", quarters.group("quarters")

    return "ValueError: other", None


def valuation_traceback_line(line: str, valuation_source: Path) -> int | None:
    """Return a valuation traceback line number, if ``line`` is such a frame."""
    match = TRACEBACK_FRAME.match(line)
    if match is None:
        return None
    try:
        frame_source = Path(match.group("file")).resolve()
    except OSError:
        return None
    if frame_source != valuation_source:
        return None
    return int(match.group("line"))


def summarize(
    log: Path,
    valuation_source: Path,
    source_warning_lines: set[int],
    source_line_count: int,
) -> tuple[Counter[str], Counter[str], dict[str, Counter[str]]]:
    """Classify valuation warning lines and ERROR traceback blocks in ``log``."""
    warnings: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    subtypes: dict[str, Counter[str]] = defaultdict(Counter)
    terminal_exception: tuple[str, str] | None = None
    in_error_block = False
    has_valuation_frame = False
    has_stale_valuation_frame = False

    def finish_error_block() -> None:
        nonlocal terminal_exception, in_error_block, has_valuation_frame, has_stale_valuation_frame
        if not in_error_block:
            return
        if has_valuation_frame:
            if has_stale_valuation_frame:
                errors["Stale valuation traceback location"] += 1
            else:
                category, subtype = classify_error(terminal_exception)
                errors[category] += 1
                if subtype is not None:
                    subtypes[category][subtype] += 1
        terminal_exception = None
        in_error_block = False
        has_valuation_frame = False
        has_stale_valuation_frame = False

    with log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if WARNING_MARKER.search(line):
                category = classify_warning(line, source_warning_lines)
                if category is not None:
                    warnings[category] += 1

            if ERROR_MARKER.match(line):
                finish_error_block()
                in_error_block = True
                continue

            if in_error_block:
                frame_line = valuation_traceback_line(line, valuation_source)
                if frame_line is not None:
                    has_valuation_frame = True
                    if frame_line > source_line_count:
                        has_stale_valuation_frame = True

                if match := EXCEPTION_LINE.match(line.rstrip()):
                    terminal_exception = match.group("name"), match.group("message")

    finish_error_block()
    return warnings, errors, subtypes


def print_counts(
    title: str,
    counts: Counter[str],
    subtypes: dict[str, Counter[str]],
    descriptions: dict[str, str],
) -> None:
    """Print categories in descending frequency, with any available subtypes."""
    total = sum(counts.values())
    print(f"{title}: {total}")
    for name, count in counts.most_common():
        print(f"  {count:>4}  {name}")
        if description := descriptions.get(name):
            print(f"        {description}")
        for subtype, subtype_count in subtypes.get(name, Counter()).most_common():
            print(f"        {subtype_count:>4}  {subtype}")


def main() -> int:
    args = parse_args()
    log = args.log.resolve()
    if not log.is_file():
        raise SystemExit(f"log file does not exist: {log}")
    valuation_source = args.valuation_source.resolve()
    if not valuation_source.is_file():
        raise SystemExit(f"valuation source does not exist: {valuation_source}")

    source_warning_lines, source_line_count = load_source_metadata(valuation_source)
    warnings, errors, subtypes = summarize(
        log, valuation_source, source_warning_lines, source_line_count
    )
    print(f"Log: {log}")
    print(f"Source: {valuation_source}")
    print_counts(
        "Warnings from backend/valuation.py", warnings, {}, WARNING_DESCRIPTIONS
    )
    print_counts(
        "Errors from backend/valuation.py", errors, subtypes, ERROR_DESCRIPTIONS
    )

    unknown = (
        warnings["Unknown warning source"]
        + warnings["Stale warning source location"]
        + errors["Stale valuation traceback location"]
        + errors["ERROR without terminal exception"]
        + errors["ValueError: other"]
    )
    if args.strict and unknown:
        print(f"\nStrict mode failed: {unknown} unclassified or ambiguous entries.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
