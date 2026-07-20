import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / 'bin/call_tree_lines.py'
SOURCE = REPOSITORY_ROOT / 'backend/valuation.py'
TARGET = 'resolve_us_income_statement_quarters'


class CallTreeLineScriptTests(unittest.TestCase):
    def test_default_target_includes_direct_helpers_and_cumulative_count(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )

        root_line = next(
            line for line in result.stdout.splitlines() if line.startswith('backend.')
        )
        expected_own_lines = self._function_lines(SOURCE, TARGET)
        match = re.search(r'own: (\d+) lines; cumulative: (\d+) lines', root_line)

        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), expected_own_lines)
        self.assertGreater(int(match.group(2)), expected_own_lines)
        self.assertIn('backend.valuation.required_xps_fields', result.stdout)
        self.assertIn('backend.valuation.add_sec_reference_values', result.stdout)
        self.assertIn('backend.valuation.resolve_us_quarter_consensus', result.stdout)

    @staticmethod
    def _function_lines(path: Path, function_name: str) -> int:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        return function.end_lineno - function.lineno + 1
