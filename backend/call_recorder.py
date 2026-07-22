"""Record unique source-level calls made within ``backend.valuation``."""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import CodeType, ModuleType


class ValuationCallRecorder:
    """Collect unique caller-to-callee pairs without building a tree."""

    _COMPREHENSION_NAMES = frozenset(
        {"<dictcomp>", "<genexpr>", "<listcomp>", "<setcomp>"}
    )

    def __init__(self, module: ModuleType) -> None:
        self._module = module
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError("the recorded module must have a source file")
        self._module_source = str(Path(module_file).resolve())
        self._tool_id = sys.monitoring.PROFILER_ID
        self._monitored_codes: set[CodeType] = set()
        self._comprehension_parents: dict[CodeType, CodeType] = {}
        self._active = False
        self.edges: set[tuple[str, str]] = set()

    def _is_module_code(self, code: CodeType | None) -> bool:
        return code is not None and code.co_filename == self._module_source

    def _callee_code(self, target: object) -> CodeType | None:
        while isinstance(target, partial):
            target = target.func

        for candidate in (
            target,
            getattr(target, "__func__", None),
            getattr(type(target), "__call__", None),
        ):
            code = getattr(candidate, "__code__", None)
            if isinstance(code, CodeType):
                return code

        return None

    def _record_call(
        self,
        caller: CodeType,
        _instruction_offset: int,
        target: object,
        _arg0: object,
    ) -> None:
        callee = self._callee_code(target)
        if callee in self._comprehension_parents:
            return

        caller = self._comprehension_parents.get(caller, caller)
        if (
            callee is not None
            and caller in self._monitored_codes
            and callee in self._monitored_codes
        ):
            self.edges.add((caller.co_qualname, callee.co_qualname))

    def _nested_codes(
        self,
        code: CodeType,
        parent: CodeType | None = None,
    ) -> Iterator[CodeType]:
        if code.co_name in self._COMPREHENSION_NAMES:
            assert parent is not None
            self._comprehension_parents[code] = parent
        else:
            parent = code

        yield code
        for value in code.co_consts:
            if isinstance(value, CodeType):
                yield from self._nested_codes(value, parent)

    def _module_codes(self) -> set[CodeType]:
        self._comprehension_parents.clear()
        return {
            code
            for value in vars(self._module).values()
            for code in self._value_codes(value)
        }

    def _value_codes(self, value: object) -> Iterator[CodeType]:
        if inspect.isfunction(value):
            if self._is_module_code(value.__code__):
                yield from self._nested_codes(value.__code__)
            return

        if isinstance(value, (staticmethod, classmethod)):
            yield from self._value_codes(value.__func__)
            return

        if isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if accessor is not None:
                    yield from self._value_codes(accessor)
            return

        if inspect.isclass(value):
            if value.__module__ != self._module.__name__:
                return

            for member in vars(value).values():
                yield from self._value_codes(member)

    def enable(self) -> None:
        if self._active:
            return

        if sys.monitoring.get_tool(self._tool_id) is not None:
            raise RuntimeError("the Python profiler monitoring tool is already in use")

        sys.monitoring.use_tool_id(self._tool_id, "portman-valuation-call-recorder")
        self._active = True
        try:
            sys.monitoring.register_callback(
                self._tool_id,
                sys.monitoring.events.CALL,
                self._record_call,
            )
            self._monitored_codes = self._module_codes()
            for code in self._monitored_codes:
                sys.monitoring.set_local_events(
                    self._tool_id,
                    code,
                    sys.monitoring.events.CALL,
                )
        except BaseException:
            self.disable()
            raise

    def disable(self) -> None:
        if not self._active:
            return

        for code in self._monitored_codes:
            sys.monitoring.set_local_events(self._tool_id, code, 0)
        self._monitored_codes.clear()
        self._comprehension_parents.clear()
        sys.monitoring.register_callback(
            self._tool_id,
            sys.monitoring.events.CALL,
            None,
        )
        if sys.monitoring.get_tool(self._tool_id) is not None:
            sys.monitoring.free_tool_id(self._tool_id)
        self._active = False

    def write(self, destination: Path) -> None:
        destination.write_text(
            json.dumps(sorted(self.edges), indent=2) + "\n",
            encoding="utf-8",
        )
