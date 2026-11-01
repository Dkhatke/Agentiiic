"""Built-in tools: filesystem and shell, all confined to the workspace."""
from __future__ import annotations

import os
import re
import subprocess

from ..safety.sandbox import wrap_command
from .registry import Registry, Risk, Tool, ToolContext, ToolError

MAX_OUTPUT = 8000
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
PROTECTED = re.compile(r"(^|/)(\.env(\..*)?|id_rsa.*|id_ed25519.*|.*\.pem|.*\.p12|credentials(\.json)?|\.netrc)$")
SAFE_SHELL = re.compile(r"^(ls|pwd|git status|git diff|git log)( -[\w-]+)*$")


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n...[truncated {len(text) - MAX_OUTPUT} chars]"


def read_file(ctx: ToolContext, path: str) -> str:
    p = ctx.resolve(path)
    if PROTECTED.search(str(p)):
        raise ToolError(f"protected file (secrets): {path}")
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    return _clip(p.read_text(errors="replace"))


def list_dir(ctx: ToolContext, path: str = ".") -> str:
    p = ctx.resolve(path)
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries if e.name != ".git") or "(empty)"


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    p = ctx.resolve(path)
    if PROTECTED.search(str(p)):
        raise ToolError(f"protected file (secrets): {path}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def edit_file(ctx: ToolContext, path: str, old: str, new: str) -> str:
    p = ctx.resolve(path)
    if PROTECTED.search(str(p)):
        raise ToolError(f"protected file (secrets): {path}")
    text = p.read_text()
    if text.count(old) != 1:
        raise ToolError(f"'old' must match exactly once (found {text.count(old)})")
    p.write_text(text.replace(old, new))
    return f"edited {path}"


def run_shell(ctx: ToolContext, command: str) -> str:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    argv = wrap_command(command, ctx.root, ctx.sandbox_network) if ctx.sandbox else None
    try:
        r = subprocess.run(argv or command, shell=argv is None, cwd=ctx.root, env=env, capture_output=True,
                           text=True, timeout=ctx.shell_timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {ctx.shell_timeout}s")
    return _clip(f"exit={r.returncode}\n{r.stdout}{r.stderr}")


def _shell_risk(args: dict) -> Risk:
    return Risk.READ if SAFE_SHELL.match(args.get("command", "").strip()) else Risk.DESTRUCTIVE


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}


def default_registry() -> Registry:
    reg = Registry()
    reg.register(Tool("read_file", "Read a text file in the workspace.", _obj({"path": _S}, ["path"]), Risk.READ, read_file, untrusted=True))
    reg.register(Tool("list_dir", "List a directory in the workspace.", _obj({"path": _S}, []), Risk.READ, list_dir))
    reg.register(Tool("write_file", "Create or overwrite a file.", _obj({"path": _S, "content": _S}, ["path", "content"]), Risk.WRITE, write_file))
    reg.register(Tool("edit_file", "Replace one exact occurrence of text in a file.", _obj({"path": _S, "old": _S, "new": _S}, ["path", "old", "new"]), Risk.WRITE, edit_file))
    reg.register(Tool("run_shell", "Run a shell command in the workspace.", _obj({"command": _S}, ["command"]), Risk.DESTRUCTIVE, run_shell, _shell_risk))
    return reg
