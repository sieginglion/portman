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

        self.assertIn(
            "Direct recorded callees of a by cumulative source lines:\n"
            "  b (cumulative: 4 lines)\n",
            result.stdout,
        )
        self.assertNotIn("\n  c (cumulative:", result.stdout)

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

        self.assertIn("b (cumulative: 8 lines)", result.stdout)

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

        self.assertIn("a.<locals>.b (cumulative: 4 lines)", result.stdout)

    @staticmethod
    def _run(
        source_text: str, edges: list[list[str]], callable_name: str
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
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
