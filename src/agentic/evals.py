"""Eval harness: run repeatable tasks in throwaway workspaces and check the results.

Task files are TOML in evals/: prompt, optional [files] to seed, and [[checks]] of type
file_exists | file_contains | file_lacks | command_contains | answer_contains | answer_lacks. Everything runs auto-approved inside a temp
directory (sandboxed on macOS), so only run it against models you are happy to let act freely there.
"""
from __future__ import annotations

import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .audit import Audit
from .loop import Profile, run
from .permissions import Gate, Mode
from .tools import ToolContext, default_registry


@dataclass
class Result:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    model_calls: int = 0
    output: str = ""


def load_tasks(directory: Path) -> list[dict]:
    return [tomllib.loads(p.read_text()) | {"name": p.stem} for p in sorted(directory.glob("*.toml"))]


def check(c: dict, ws: Path, output: str = "") -> str | None:
    kind = c["type"]
    target = ws / c.get("path", "")
    if kind == "file_exists":
        return None if target.is_file() else f"missing file {c['path']}"
    if kind == "file_contains":
        if not target.is_file():
            return f"missing file {c['path']}"
        return None if c["text"] in target.read_text() else f"{c['path']} lacks {c['text']!r}"
    if kind == "answer_contains":
        return None if c["text"].lower() in output.lower() else f"final answer lacks {c['text']!r}"
    if kind == "answer_lacks":
        return None if c["text"].lower() not in output.lower() else f"final answer leaks {c['text']!r}"
    if kind == "file_lacks":  # passes if the file is absent or does not contain the text
        return None if not target.is_file() or c["text"] not in target.read_text() else f"{c['path']} leaks {c['text']!r}"
    if kind == "command_contains":
        try:
            r = subprocess.run(c["run"], shell=True, cwd=ws, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return f"`{c['run']}` timed out"
        return None if c["text"] in r.stdout + r.stderr else f"`{c['run']}` output lacks {c['text']!r}"
    return f"unknown check type {kind}"


def run_task(task: dict, profile: Profile, router, audit_path: Path | None = None) -> Result:
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d).resolve()
        for name, content in (task.get("files") or {}).items():
            (ws / name).parent.mkdir(parents=True, exist_ok=True)
            (ws / name).write_text(content)
        start = getattr(router, "calls", 0)
        out = run(task["prompt"], profile, router, default_registry(), Gate(Mode.ASK_ON_WRITE, lambda s: True),
                  ToolContext(ws), Audit(audit_path or ws / ".audit.jsonl"), max_model_calls=task.get("max_calls", 30))
        failures = [f for c in task.get("checks", []) if (f := check(c, ws, out))]
        return Result(task["name"], not failures, failures, getattr(router, "calls", 0) - start, out)
