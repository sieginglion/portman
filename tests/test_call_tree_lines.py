import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "bin/call_tree_lines.py"


class CallTreeLinesTests(unittest.TestCase):
    def test_reports_a_direct_callees_cumulative_lines(self):
        result = self._run(
            """
def c():
    return 1

def b():
    return c()

def a():
    return b()
""".lstrip(),
            [["a", "b"], ["b", "c"]],
            "a",
        )

        self.assertEqual(result.stdout, "b 4\n")

    def test_all_lists_every_reachable_descendant_without_line_counts(self):
        result = self._run(
            """
def c():
    return 1

def b():
    return c()

def a():
    return b()
""".lstrip(),
            [["a", "b"], ["b", "c"]],
            "a",
            extra_args=["--all", "--source", "/does-not-need-to-exist.py"],
        )

        self.assertEqual(result.stdout, "b\nc\n")

    def test_counts_shared_descendants_once_per_cumulative_total(self):
        result = self._run(
            """
def shared():
    return 1

def left():
    return shared()

def right():
    return shared()

def b():
    return left() + right()

def a():
    return b()
""".lstrip(),
            [
                ["a", "b"],
                ["b", "left"],
                ["b", "right"],
                ["left", "shared"],
                ["right", "shared"],
            ],
            "a",
        )

        self.assertEqual(result.stdout, "b 8\n")

    def test_counts_nested_callables_with_recorder_qualified_names(self):
        result = self._run(
            """
def c():
    return 1

def a():
    def b():
        return c()
    return b()
""".lstrip(),
            [["a", "a.<locals>.b"], ["a.<locals>.b", "c"]],
            "a",
        )

        self.assertEqual(result.stdout, "a.<locals>.b 4\n")

    def test_counts_methods_in_a_nested_class(self):
        result = self._run(
            """
def c():
    return 1

def a():
    class Worker:
        def b(self):
            return c()
    return Worker().b()
""".lstrip(),
            [["a", "a.<locals>.Worker.b"], ["a.<locals>.Worker.b", "c"]],
            "a",
        )

        self.assertEqual(result.stdout, "a.<locals>.Worker.b 4\n")

    def test_counts_lambdas_and_comprehensions_with_qualified_names(self):
        result = self._run(
            """
def helper():
    return 1

def a():
    return (lambda: helper())() + sum([helper() for _ in range(1)])
""".lstrip(),
            [
                ["a", "a.<locals>.<lambda>"],
                ["a", "a.<locals>.<listcomp>"],
                ["a.<locals>.<lambda>", "helper"],
                ["a.<locals>.<listcomp>", "helper"],
            ],
            "a",
        )

        self.assertEqual(
            result.stdout,
            "a.<locals>.<lambda> 3\na.<locals>.<listcomp> 3\n",
        )

    def test_ranks_direct_callees_by_lines_then_name(self):
        result = self._run(
            """
def ant():
    return 1

def zebra():
    value = 1
    value += 1
    return value

def a():
    return ant() + zebra()
""".lstrip(),
            [["a", "ant"], ["a", "zebra"]],
            "a",
        )

        self.assertEqual(result.stdout, "zebra 4\nant 2\n")

    def test_rejects_invalid_edge_pairs(self):
        source = """
def a():
    return 1
""".lstrip()
        for edges in (["a"], [["a"]], [["a", 1]]):
            with self.subTest(edges=edges):
                result = self._run(source, edges, "a", check=False)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("edge 1 is not a string pair", result.stderr)

    @staticmethod
    def _run(
        source_text: str,
        edges: object,
        callable_name: str,
        *,
        check: bool = True,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source = directory_path / "sample.py"
            edges_path = directory_path / "edges.json"
            source.write_text(source_text, encoding="utf-8")
            edges_path.write_text(json.dumps(edges), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    callable_name,
                    "--source",
                    str(source),
                    "--edges",
                    str(edges_path),
                    *(extra_args or []),
                ],
                cwd=REPOSITORY_ROOT,
                check=check,
                capture_output=True,
                text=True,
            )
