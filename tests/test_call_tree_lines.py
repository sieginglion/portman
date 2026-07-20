import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / 'bin/call_tree_lines.py'
SOURCE = REPOSITORY_ROOT / 'backend/valuation.py'
TARGET = 'resolve_us_income_statement_quarters'


class CallTreeLineScriptTests(unittest.TestCase):
    def test_default_target_includes_file_local_helpers_and_cumulative_count(self):
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
        self.assertNotIn('backend.shared.cached_get', result.stdout)

    def test_omits_functions_imported_from_another_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'sample'
            package.mkdir()
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'shared.py').write_text(
                'def external() -> int:\n    return 2\n', encoding='utf-8'
            )
            source = package / 'main.py'
            source.write_text(
                'from .shared import external\n\n'
                'def helper() -> int:\n'
                '    return 1\n\n'
                'def entry() -> int:\n'
                '    return helper() + external()\n',
                encoding='utf-8',
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    'entry',
                    '--root',
                    str(root),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )

        self.assertIn('sample.main.entry', result.stdout)
        self.assertIn('sample.main.helper', result.stdout)
        self.assertNotIn('sample.shared.external', result.stdout)
        self.assertIn('cumulative: 4 lines', result.stdout)

    def test_shared_helper_is_counted_once_and_rendered_once(self):
        result = self._run_source(
            'def shared() -> int:\n'
            '    return 1\n\n'
            'def left() -> int:\n'
            '    return shared()\n\n'
            'def right() -> int:\n'
            '    return shared()\n\n'
            'def entry() -> int:\n'
            '    return left() + right()\n',
            'entry',
        )

        self.assertIn('sample.entry (own: 2 lines; cumulative: 8 lines)', result)
        self.assertIn('sample.shared (own: 2 lines; cumulative: 2 lines)', result)
        self.assertIn('[shared helper; shown above]', result)

    def test_cycle_is_marked_without_repeating_its_subtree(self):
        result = self._run_source(
            'def first() -> int:\n'
            '    return second()\n\n'
            'def second() -> int:\n'
            '    return first()\n',
            'first',
        )

        self.assertIn('sample.first (own: 2 lines; cumulative: 4 lines)', result)
        self.assertIn('sample.second (own: 2 lines; cumulative: 4 lines)', result)
        self.assertIn('[cycle]', result)

    def test_resolves_self_and_cls_method_calls(self):
        result = self._run_source(
            'class Helpers:\n'
            '    def entry(self) -> int:\n'
            '        return self.instance_helper() + self.class_entry()\n\n'
            '    def instance_helper(self) -> int:\n'
            '        return 1\n\n'
            '    @classmethod\n'
            '    def class_entry(cls) -> int:\n'
            '        return cls.class_helper()\n\n'
            '    @classmethod\n'
            '    def class_helper(cls) -> int:\n'
            '        return 2\n',
            'Helpers.entry',
        )

        self.assertIn('sample.Helpers.entry (own: 2 lines; cumulative: 8 lines)', result)
        self.assertIn('sample.Helpers.instance_helper', result)
        self.assertIn('sample.Helpers.class_entry', result)
        self.assertIn('sample.Helpers.class_helper', result)

    def test_resolves_nested_helper_before_module_helper(self):
        result = self._run_source(
            'def helper() -> int:\n'
            '    return 1\n\n'
            'def entry() -> int:\n'
            '    def helper() -> int:\n'
            '        return 2\n'
            '    return helper()\n',
            'entry',
        )

        self.assertIn('sample.entry (own: 4 lines; cumulative: 6 lines)', result)
        self.assertIn('sample.entry.helper (own: 2 lines; cumulative: 2 lines)', result)
        self.assertNotIn('sample.helper (own:', result)

    @staticmethod
    def _run_source(source_text: str, function: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sample.py'
            source.write_text(source_text, encoding='utf-8')
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    function,
                    '--root',
                    str(root),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )
        return result.stdout

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
