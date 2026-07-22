"""Record unique source-level calls made within ``backend.valuation``."""

from __future__ import annotations

import inspect
import json
import sys
from functools import partial
from pathlib import Path
from types import CodeType, ModuleType


class ValuationCallRecorder:
    """Collect unique caller-to-callee pairs without building a tree."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module
        self._source = str(Path(module.__file__).resolve())
        self._tool_id = sys.monitoring.PROFILER_ID
        self._enabled_codes: set[CodeType] = set()
        self.edges: set[tuple[str, str]] = set()

    def _is_valuation_code(self, code: CodeType | None) -> bool:
        return code is not None and code.co_filename == self._source

    def _callee_code(self, target: object) -> CodeType | None:
        if isinstance(target, partial):
            return self._callee_code(target.func)

        code = getattr(target, "__code__", None)
        if isinstance(code, CodeType):
            return code

        function = getattr(target, "__func__", None)
        code = getattr(function, "__code__", None)
        if isinstance(code, CodeType):
            return code

        call_method = getattr(type(target), "__call__", None)
        code = getattr(call_method, "__code__", None)
        return code if isinstance(code, CodeType) else None

    def _record_call(
        self,
        caller: CodeType,
        instruction_offset: int,
        target: object,
        arg0: object,
    ) -> None:
        callee = self._callee_code(target)
        if self._is_valuation_code(caller) and self._is_valuation_code(callee):
            self.edges.add((caller.co_qualname, callee.co_qualname))

    def _nested_codes(self, code: CodeType) -> set[CodeType]:
        codes = {code}
        for value in code.co_consts:
            if isinstance(value, CodeType):
                codes.update(self._nested_codes(value))
        return codes

    def _module_codes(self) -> set[CodeType]:
        codes: set[CodeType] = set()
        for value in vars(self._module).values():
            if inspect.isfunction(value) and self._is_valuation_code(value.__code__):
                codes.update(self._nested_codes(value.__code__))
        return codes

    def enable(self) -> None:
        if sys.monitoring.get_tool(self._tool_id) is not None:
            raise RuntimeError("the Python profiler monitoring tool is already in use")

        sys.monitoring.use_tool_id(self._tool_id, "portman-valuation-call-recorder")
        try:
            sys.monitoring.register_callback(
                self._tool_id,
                sys.monitoring.events.CALL,
                self._record_call,
            )
            self._enabled_codes = self._module_codes()
            for code in self._enabled_codes:
                sys.monitoring.set_local_events(
                    self._tool_id,
                    code,
                    sys.monitoring.events.CALL,
                )
        except BaseException:
            self.disable()
            raise

    def disable(self) -> None:
        for code in self._enabled_codes:
            sys.monitoring.set_local_events(self._tool_id, code, 0)
        self._enabled_codes.clear()
        sys.monitoring.register_callback(
            self._tool_id,
            sys.monitoring.events.CALL,
            None,
        )
        if sys.monitoring.get_tool(self._tool_id) is not None:
            sys.monitoring.free_tool_id(self._tool_id)

    def write(self, destination: Path) -> None:
        destination.write_text(
            json.dumps(sorted(self.edges), indent=2) + "\n",
            encoding="utf-8",
        )
