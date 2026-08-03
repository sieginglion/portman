import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "bin/log_severity_report.py"


class LogSeverityReportTests(unittest.TestCase):
    def test_discovers_and_classifies_source_logging_and_traceback_sites(self):
        source_text = """
from loguru import logger

def warn():
    logger.warning("Dynamic warning {}", "value")

def logged_error():
    logger.exception("Dynamic failure {}", "value")

def fail():
    raise LookupError("boom")
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            warning_line = self._line_containing(source_text, "logger.warning")
            logged_error_line = self._line_containing(source_text, "logger.exception")
            raise_line = self._line_containing(source_text, "raise LookupError")
            log = Path(directory) / "app.log"
            log.write_text(
                "\n".join(
                    (
                        "2026-01-01 | WARNING | "
                        f"backend.valuation:warn:{warning_line} - Dynamic warning value",
                        "2026-01-01 | ERROR | "
                        "backend.valuation:logged_error:"
                        f"{logged_error_line} - Dynamic failure value",
                        "ERROR: Exception in application",
                        "Traceback (most recent call last):",
                        f'  File "{source}", line {raise_line}, in fail',
                        '    raise LookupError("boom")',
                        "LookupError: boom",
                        'INFO: 127.0.0.1 - "GET / HTTP/1.1" 500',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Discovered source sites: 1 warnings, 1 logged errors, 1 raises",
            result.stdout,
        )
        self.assertIn("Warnings from backend.valuation: 1", result.stdout)
        self.assertIn(
            f"warn:{warning_line} - Dynamic warning {{}}",
            result.stdout,
        )
        self.assertIn("Errors from backend.valuation: 2", result.stdout)
        self.assertIn(
            f"logged_error:{logged_error_line} - Dynamic failure {{}}",
            result.stdout,
        )
        self.assertIn(f"fail:{raise_line} - LookupError: boom", result.stdout)

    def test_ignores_records_from_other_modules(self):
        source_text = """
from loguru import logger

def warn():
    logger.warning("Dynamic warning")
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            warning_line = self._line_containing(source_text, "logger.warning")
            log = Path(directory) / "app.log"
            log.write_text(
                "2026-01-01 | WARNING | "
                f"backend.other:warn:{warning_line} - Dynamic warning\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Warnings from backend.valuation: 0", result.stdout)
        self.assertIn("Errors from backend.valuation: 0", result.stdout)

    def test_preserves_f_string_expression_and_multiline_templates(self):
        source_text = '''
from loguru import logger

def f_string(value):
    logger.warning(f"Value: {value!r:>8}")

def expression(message):
    logger.warning(message)

def multiline():
    logger.warning("first\\nsecond")
'''.lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            f_string_line = self._line_containing(source_text, 'f"Value')
            expression_line = self._line_containing(source_text, "logger.warning(message)")
            multiline_line = self._line_containing(source_text, '"first\\nsecond"')
            log = Path(directory) / "app.log"
            log.write_text(
                "\n".join(
                    (
                        "2026-01-01 | WARNING | "
                        f"backend.valuation:f_string:{f_string_line} - Value: 123",
                        "2026-01-01 | WARNING | "
                        f"backend.valuation:expression:{expression_line} - Dynamic value",
                        "2026-01-01 | WARNING | "
                        f"backend.valuation:multiline:{multiline_line} - first second",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"f_string:{f_string_line} - f'Value: {{value!r:>8}}'", result.stdout)
        self.assertIn(f"expression:{expression_line} - message", result.stdout)
        self.assertIn(f"multiline:{multiline_line} - first\\nsecond", result.stdout)

    def test_classifies_all_explicit_raise_forms(self):
        source_text = """
class errors:
    class DomainError(Exception):
        pass

def raise_class():
    raise ValueError

def raise_without_message():
    raise RuntimeError()

def raise_qualified():
    raise errors.DomainError("bad")

def reraise():
    try:
        raise ValueError("original")
    except ValueError:
        raise
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            class_line = self._line_containing(source_text, "raise ValueError")
            empty_line = self._line_containing(source_text, "raise RuntimeError()")
            qualified_line = self._line_containing(source_text, "raise errors.DomainError")
            reraise_line = next(
                line_number
                for line_number, line in enumerate(source_text.splitlines(), 1)
                if line.strip() == "raise"
            )
            log = Path(directory) / "app.log"
            log.write_text(
                "\n".join(
                    (
                        "ERROR: Exception in application",
                        f'  File "{source}", line {class_line}, in raise_class',
                        "ValueError",
                        "ERROR: Exception in application",
                        f'  File "{source}", line {empty_line}, in raise_without_message',
                        "RuntimeError",
                        "ERROR: Exception in application",
                        f'  File "{source}", line {qualified_line}, in raise_qualified',
                        "errors.DomainError: bad",
                        "ERROR: Exception in application",
                        f'  File "{source}", line {reraise_line}, in reraise',
                        "ValueError: original",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"raise_class:{class_line} - ValueError", result.stdout)
        self.assertIn(f"raise_without_message:{empty_line} - RuntimeError", result.stdout)
        self.assertIn(
            f"raise_qualified:{qualified_line} - errors.DomainError: bad", result.stdout
        )
        self.assertIn(
            f"Unknown error source: reraise:{reraise_line} - ValueError", result.stdout
        )

    def test_stale_logging_location_fails(self):
        source_text = """
from loguru import logger

def warn():
    logger.warning("Dynamic warning")
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            warning_line = self._line_containing(source_text, "logger.warning")
            log = Path(directory) / "app.log"
            log.write_text(
                "2026-01-01 | WARNING | "
                f"backend.valuation:warn:{warning_line + 1} - Dynamic warning\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Stale warning source location", result.stdout)
        self.assertIn(
            "Failed: 1 unknown or stale valuation entries (1 warnings, 0 errors).",
            result.stdout,
        )

    def test_unknown_propagated_error_fails(self):
        source_text = """
def accidental_error():
    return {}["missing"]
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            error_line = self._line_containing(source_text, 'return {}["missing"]')
            log = Path(directory) / "app.log"
            log.write_text(
                "\n".join(
                    (
                        "ERROR: Exception in application",
                        "Traceback (most recent call last):",
                        f'  File "{source}", line {error_line}, in accidental_error',
                        '    return {}["missing"]',
                        "KeyError: 'missing'",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            f"Unknown error source: accidental_error:{error_line} - KeyError",
            result.stdout,
        )
        self.assertIn(
            "Failed: 1 unknown or stale valuation entries (0 warnings, 1 errors).",
            result.stdout,
        )

    def test_module_level_raise_is_a_known_error_site(self):
        source_text = """
from loguru import logger
logger.warning("warning")
raise ValueError("boom")
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "backend/valuation.py"
            source.parent.mkdir()
            source.write_text(source_text, encoding="utf-8")
            log = Path(directory) / "app.log"
            log.write_text(
                "\n".join(
                    (
                        "2026-01-01 | WARNING | "
                        "backend.valuation:<module>:2 - warning",
                        "ERROR: Exception in application",
                        "Traceback (most recent call last):",
                        f'  File "{source}", line 3, in <module>',
                        '    raise ValueError("boom")',
                        "ValueError: boom",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run(log, source)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<module>:2 - warning", result.stdout)
        self.assertIn("<module>:3 - ValueError: boom", result.stdout)

    def _run(self, log: Path, source: Path, *extra_args: str):
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                str(log),
                "--valuation-source",
                str(source),
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _line_containing(source: str, text: str) -> int:
        return next(
            line_number
            for line_number, line in enumerate(source.splitlines(), 1)
            if text in line
        )


if __name__ == "__main__":
    unittest.main()
