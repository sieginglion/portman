import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from backend import call_recorder, valuation
from backend.call_recorder import ValuationCallRecorder


class ValuationCallRecorderTests(unittest.TestCase):
    def test_requires_a_module_source_file(self):
        with self.assertRaisesRegex(ValueError, "source file"):
            ValuationCallRecorder(ModuleType("dynamic_module"))

    def test_disabling_an_inactive_recorder_is_a_no_op(self):
        with patch.object(call_recorder.sys, "monitoring") as monitoring:
            recorder = ValuationCallRecorder(valuation)
            recorder.disable()

        monitoring.set_local_events.assert_not_called()
        monitoring.register_callback.assert_not_called()
        monitoring.free_tool_id.assert_not_called()

    def test_records_each_internal_call_edge_once(self):
        recorder = ValuationCallRecorder(valuation)
        recorder.enable()
        recorder.disable()
        recorder.enable()
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

    def test_records_nested_function_calls(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def caller():
    def helper():
        return 1

    return helper()
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        try:
            self.assertEqual(module.caller(), 1)
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {("caller", "caller.<locals>.helper")},
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

    def test_records_partial_and_callable_instance_calls(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
from functools import partial

def helper():
    return 1

class Worker:
    def __call__(self):
        return helper()

def caller():
    return partial(helper)() + Worker()()
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        try:
            self.assertEqual(module.caller(), 2)
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {
                ("caller", "helper"),
                ("caller", "Worker.__call__"),
                ("Worker.__call__", "helper"),
            },
        )

    def test_skips_members_of_imported_classes(self):
        class ImportedMeta(type):
            def __getattribute__(cls, name: str) -> object:
                if name == "__dict__":
                    raise AssertionError("imported class members should not be inspected")
                return super().__getattribute__(name)

        class Imported(metaclass=ImportedMeta):
            pass

        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        vars(module)["Imported"] = Imported

        recorder = ValuationCallRecorder(module)

        self.assertEqual(recorder._module_codes(), set())

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
