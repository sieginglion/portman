import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / 'bin/call_tree_lines.py'
class CallTreeLineScriptTests(unittest.TestCase):
    def test_default_target_lists_direct_file_local_helpers_by_cumulative_count(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertIn(
            'Direct local functions called by the root, by cumulative lines:',
            result.stdout,
        )
        self.assertIn('required_xps_fields', result.stdout)
        self.assertIn('add_sec_reference_values', result.stdout)
        self.assertIn('resolve_us_quarter_consensus', result.stdout)
        self.assertNotIn('cached_get', result.stdout)

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

        self.assertIn('helper', result.stdout)
        self.assertNotIn('external', result.stdout)
        self.assertIn('helper (cumulative: 2 lines)', result.stdout)

    def test_shared_helper_is_counted_once_per_direct_callee_total(self):
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

        self.assertIn('left (cumulative: 4 lines)', result)
        self.assertIn('right (cumulative: 4 lines)', result)
        self.assertNotIn('\n  shared (cumulative:', result)

    def test_lists_only_direct_functions_by_cumulative_lines(self):
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
            'Direct local functions called by the root, by cumulative lines:\n'
            '  wrapper (cumulative: 6 lines)\n'
            '  own_longest (cumulative: 4 lines)',
            result,
        )
        self.assertNotIn('\n  leaf (cumulative:', result)

    def test_only_calculates_counts_for_root_direct_callees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sample.py'
            source.write_text(
                'def leaf() -> int:\n'
                '    return 1\n\n'
                'def reachable() -> int:\n'
                '    return leaf()\n\n'
                'def unrelated_leaf() -> int:\n'
                '    return 2\n\n'
                'def unrelated() -> int:\n'
                '    return unrelated_leaf()\n\n'
                'def entry() -> int:\n'
                '    return reachable()\n',
                encoding='utf-8',
            )
            module_name = 'call_tree_lines_test_module'
            spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                with (
                    patch.object(
                        module,
                        'cumulative_line_count',
                        wraps=module.cumulative_line_count,
                    ) as count,
                    patch.object(
                        sys,
                        'argv',
                        [str(SCRIPT), str(source), 'entry', '--root', str(root)],
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    module.main()
            finally:
                sys.modules.pop(module_name, None)

        counted_functions = [call.args[0] for call in count.call_args_list]
        self.assertCountEqual(counted_functions, ['reachable'])

    def test_cycle_has_a_finite_cumulative_count(self):
        result = self._run_source(
            'def first() -> int:\n'
            '    return second()\n\n'
            'def second() -> int:\n'
            '    return first()\n',
            'first',
        )

        self.assertIn('second (cumulative: 4 lines)', result)
        self.assertNotIn('[cycle]', result)

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

        self.assertIn('Helpers.instance_helper', result)
        self.assertIn('Helpers.class_entry', result)
        self.assertNotIn('Helpers.class_helper', result)

    def test_ignores_calls_inside_nested_callable_bodies(self):
        result = self._run_source(
            'def hidden() -> int:\n'
            '    return 1\n\n'
            'def entry() -> int:\n'
            '    def nested() -> int:\n'
            '        return hidden()\n'
            '    callback = lambda: hidden()\n'
            '    class Deferred:\n'
            '        def run(self) -> int:\n'
            '            return hidden()\n'
            '    return 0\n',
            'entry',
        )

        self.assertNotIn('hidden (cumulative:', result)

    def test_restores_outer_class_context_after_nested_class(self):
        result = self._run_source(
            'class Outer:\n'
            '    class Inner:\n'
            '        def value(self) -> int:\n'
            '            return 1\n\n'
            '    def entry(self) -> int:\n'
            '        return self.outer_helper()\n\n'
            '    def outer_helper(self) -> int:\n'
            '        return 2\n',
            'Outer.entry',
        )

        self.assertIn('Outer.outer_helper', result)

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

        self.assertIn('entry.helper (cumulative: 2 lines)', result)
        self.assertNotIn('\n  helper (cumulative:', result)

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
