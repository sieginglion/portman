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
        match = re.search(r'cumulative: (\d+) lines', root_line)

        self.assertIsNotNone(match)
        self.assertGreater(int(match.group(1)), expected_own_lines)
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

        self.assertIn('sample.entry (cumulative: 8 lines)', result)
        self.assertIn('sample.shared (cumulative: 2 lines)', result)
        self.assertIn('[shared helper; shown above]', result)

    def test_lists_all_reachable_functions_by_cumulative_lines(self):
        result = self._run_source(
            'def leaf() -> int:\n'
            '    first = 1\n'
            '    second = 2\n'
            '    return first + second\n\n'
            'def wrapper() -> int:\n'
            '    return leaf()\n\n'
            'def own_longest() -> int:\n'
            '    first = 1\n'
            '    second = 2\n'
            '    return first + second\n\n'
            'def entry() -> int:\n'
            '    return wrapper() + own_longest()\n',
            'entry',
        )

        self.assertIn(
            'Reachable local functions by cumulative lines, excluding the root:\n'
            '  sample.wrapper (cumulative: 6 lines)\n'
            '  sample.leaf (cumulative: 4 lines)\n'
            '  sample.own_longest (cumulative: 4 lines)',
            result,
        )

    def test_cycle_is_marked_without_repeating_its_subtree(self):
        result = self._run_source(
            'def first() -> int:\n'
            '    return second()\n\n'
            'def second() -> int:\n'
            '    return first()\n',
            'first',
        )

        self.assertIn('sample.first (cumulative: 4 lines)', result)
        self.assertIn('sample.second (cumulative: 4 lines)', result)
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

        self.assertIn('sample.Helpers.entry (cumulative: 8 lines)', result)
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

        self.assertIn('sample.entry (cumulative: 6 lines)', result)
        self.assertIn('sample.entry.helper (cumulative: 2 lines)', result)
        self.assertNotIn('sample.helper (cumulative:', result)

    def test_rejects_missing_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'missing.py'
            result = self._run_script(str(source), 'entry', '--root', str(root))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertEqual(
            result.stderr, f'source file does not exist: {source}\n'
        )

    def test_rejects_source_outside_root(self):
        with (
            tempfile.TemporaryDirectory() as root_directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            root = Path(root_directory).resolve()
            source = Path(external_directory).resolve() / 'sample.py'
            source.write_text('def entry() -> None:\n    pass\n', encoding='utf-8')
            result = self._run_script(str(source), 'entry', '--root', str(root))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertEqual(
            result.stderr, f'source file must be below --root: {source}\n'
        )

    def test_reports_available_functions_for_unknown_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'sample.py'
            source.write_text('def entry() -> None:\n    pass\n', encoding='utf-8')
            result = self._run_script(str(source), 'missing', '--root', str(root))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')
        self.assertEqual(
            result.stderr,
            f"function 'missing' was not found in {source}; available functions: entry\n",
        )

    @staticmethod
    def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )

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
