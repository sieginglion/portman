import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from backend import valuation
from backend.call_recorder import ValuationCallRecorder


class ValuationCallRecorderTests(unittest.TestCase):
    def test_records_each_internal_call_edge_once(self):
        recorder = ValuationCallRecorder(valuation)
        recorder.enable()
        try:
            self.assertTrue(valuation.field_has_consensus("revenue", 1, 1))
            self.assertTrue(valuation.field_has_consensus("revenue", 2, 2))
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {("field_has_consensus", "source_log_diff")},
        )

    def test_records_enabled_instance_static_class_and_property_methods(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def helper():
    return 1

class Worker:
    def method(self):
        return helper()

    @staticmethod
    def static_method():
        return helper()

    @classmethod
    def class_method(cls):
        return helper()

    @property
    def value(self):
        return helper()

def caller():
    worker = Worker()
    return (
        worker.method(),
        Worker.static_method(),
        Worker.class_method(),
        worker.value,
    )
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        try:
            self.assertEqual(module.caller(), (1, 1, 1, 1))
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {
                ("caller", "Worker.method"),
                ("caller", "Worker.static_method"),
                ("caller", "Worker.class_method"),
                ("Worker.method", "helper"),
                ("Worker.static_method", "helper"),
                ("Worker.class_method", "helper"),
                ("Worker.value", "helper"),
            },
        )

    def test_excludes_a_same_source_callee_that_is_not_enabled(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def caller():
    return late_callee()
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        try:
            exec(
                compile(
                    """
def late_callee():
    return 1
""",
                    source,
                    "exec",
                ),
                vars(module),
            )
            self.assertEqual(module.caller(), 1)
        finally:
            recorder.disable()

        self.assertEqual(recorder.edges, set())

    def test_writes_only_the_unique_edges(self):
        recorder = ValuationCallRecorder(valuation)
        recorder.edges.add(("caller", "callee"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "edges.json"
            recorder.write(output)

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                [["caller", "callee"]],
            )
