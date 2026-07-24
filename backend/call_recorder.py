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
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError("the recorded module must have a source file")
        self._module_source = str(Path(module_file).resolve())
        self._tool_id = sys.monitoring.PROFILER_ID
        self._active = False
        self.edges: set[tuple[str, str]] = set()

    def _is_module_code(self, code: CodeType | None) -> bool:
        return code is not None and code.co_filename == self._module_source

    def _unwrap_partial(self, target: object) -> object:
        while isinstance(target, partial):
            target = target.func
        return target

    def _callee_code(self, target: object) -> CodeType | None:
        target = self._unwrap_partial(target)

        for candidate in (
            target,
            getattr(target, "__func__", None),
            getattr(type(target), "__call__", None),
        ):
            code = getattr(candidate, "__code__", None)
            if isinstance(code, CodeType):
                return code

        return None

    def _callee_source_file(self, target: object) -> str | None:
        """Return the best-known source file for a callable target."""
        target = self._unwrap_partial(target)

        if inspect.isclass(target):
            try:
                source_file = inspect.getfile(target)
            except (OSError, TypeError):
                # Synthetic modules used by callers can have a valid __file__
                # without being registered in sys.modules.  Classes do not
                # retain their class-body code object, so the module name is
                # the only source-location fallback available in that case.
                return (
                    self._module_source
                    if target.__module__ == self._module.__name__
                    else None
                )
        else:
            code = self._callee_code(target)
            if code is None:
                return None
            source_file = code.co_filename

        return str(Path(source_file).resolve())

    def _callee_name(self, target: object) -> str | None:
        """Return the qualified name for a callable from the module."""
        if not callable(target):
            return None

        target = self._unwrap_partial(target)

        if self._callee_source_file(target) != self._module_source:
            return None

        if inspect.isclass(target):
            return target.__qualname__

        code = self._callee_code(target)
        if code is not None:
            return code.co_qualname
        return None

    def _record_call(
        self,
        caller: CodeType,
        _instruction_offset: int,
        target: object,
        _arg0: object,
    ) -> None:
        if not self._is_module_code(caller):
            return

        callee = self._callee_name(target)
        if callee is not None:
            self.edges.add((caller.co_qualname, callee))

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
            sys.monitoring.set_events(
                self._tool_id,
                sys.monitoring.events.CALL,
            )
        except BaseException:
            self.disable()
            raise

    def disable(self) -> None:
        if not self._active:
            return

        sys.monitoring.set_events(self._tool_id, 0)
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
