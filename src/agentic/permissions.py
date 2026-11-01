"""Permission gate. Every tool call passes through here before it runs."""
from __future__ import annotations

from enum import Enum
from typing import Callable

from .tools import Risk, Tool


class Mode(str, Enum):
    ASK_ALL = "ask-all"              # confirm every call, reads included
    ASK_ON_WRITE = "ask-on-write"    # reads auto-approved, anything else asks
    READONLY = "readonly"            # reads auto-approved, everything else denied


class Gate:
    def __init__(self, mode: Mode, ask: Callable[[str], bool]):
        self.mode = mode
        self.ask = ask
        self.tainted = False  # set once untrusted content has entered the model's context

    def allow(self, tool: Tool, args: dict) -> tuple[bool, str]:
        risk = tool.risk_for(args)
        note = " (context contains untrusted content)" if self.tainted else ""
        summary = f"[{risk.value}] {tool.name} {args}{note}"
        auto_ok = risk is Risk.READ and not (self.tainted and tool.name == "run_shell")
        if auto_ok and self.mode is not Mode.ASK_ALL:
            return True, "auto"
        if self.mode is Mode.READONLY:
            return False, "denied: readonly mode"
        return (True, "approved") if self.ask(summary) else (False, "denied by user")
