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

        monitoring.set_events.assert_not_called()
        monitoring.register_callback.assert_not_called()
        monitoring.free_tool_id.assert_not_called()

    def test_context_manager_enables_global_call_events(self):
        with patch.object(call_recorder.sys, "monitoring") as monitoring:
            monitoring.PROFILER_ID = 2
            monitoring.get_tool.return_value = None
            recorder = ValuationCallRecorder(valuation)
            with recorder as active_recorder:
                self.assertIs(active_recorder, recorder)

        monitoring.set_events.assert_any_call(2, monitoring.events.CALL)
        monitoring.set_events.assert_any_call(2, 0)
        monitoring.set_local_events.assert_not_called()

    def test_context_manager_disables_after_an_exception(self):
        with patch.object(call_recorder.sys, "monitoring") as monitoring:
            monitoring.PROFILER_ID = 2
            monitoring.get_tool.return_value = None
            recorder = ValuationCallRecorder(valuation)

            with self.assertRaisesRegex(ValueError, "expected"):
                with recorder:
                    raise ValueError("expected")

        monitoring.set_events.assert_any_call(2, 0)
        monitoring.register_callback.assert_any_call(
            2,
            monitoring.events.CALL,
            None,
        )

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

    def test_records_comprehension_callers(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def helper(value):
    return value

def caller():
    return (
        [helper(value) for value in (1,)],
        {helper(value) for value in (2,)},
        {value: helper(value) for value in (3,)},
        tuple(helper(value) for value in (4,)),
        [[helper(value) for value in values] for values in ((5,),)],
    )
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        recorder.disable()
        recorder.enable()
        try:
            self.assertEqual(module.caller(), ([1], {2}, {3: 3}, (4,), [[5]]))
        finally:
            recorder.disable()

        self.assertEqual(
            recorder.edges,
            {
                ("caller", "caller.<locals>.<genexpr>"),
                ("caller", "helper"),
                ("caller.<locals>.<genexpr>", "helper"),
            },
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
                ("caller", "Worker"),
                ("caller", "Worker.method"),
                ("caller", "Worker.static_method"),
                ("caller", "Worker.class_method"),
                ("Worker.method", "helper"),
                ("Worker.static_method", "helper"),
                ("Worker.class_method", "helper"),
                ("Worker.value", "helper"),
            },
        )

    def test_records_a_same_source_callee_defined_after_enable(self):
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

        self.assertEqual(recorder.edges, {("caller", "late_callee")})

    def test_records_callable_instance_calls(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def helper():
    return 1

class Worker:
    def __call__(self):
        return helper()

def caller():
    return Worker()()
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
            {
                ("caller", "Worker"),
                ("caller", "Worker.__call__"),
                ("Worker.__call__", "helper"),
            },
        )

    def test_ignores_partial_calls(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
from functools import partial

def helper():
    return 1

def caller():
    return partial(helper)()
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

        self.assertEqual(recorder.edges, set())

    def test_excludes_a_callable_from_another_module(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())

        class Imported:
            pass

        vars(module)["Imported"] = Imported
        exec(
            compile(
                """
def caller():
    return Imported()
""",
                source,
                "exec",
            ),
            vars(module),
        )

        recorder = ValuationCallRecorder(module)
        recorder.enable()
        try:
            module.caller()
        finally:
            recorder.disable()

        self.assertEqual(recorder.edges, set())

    def test_records_a_caller_defined_after_enable(self):
        module = ModuleType("test_valuation")
        module.__file__ = "/tmp/test_valuation.py"
        source = str(Path(module.__file__).resolve())
        exec(
            compile(
                """
def helper():
    return 1
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
def caller():
    return helper()
""",
                    source,
                    "exec",
                ),
                vars(module),
            )
            self.assertEqual(module.caller(), 1)
        finally:
            recorder.disable()

        self.assertEqual(recorder.edges, {("caller", "helper")})

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
