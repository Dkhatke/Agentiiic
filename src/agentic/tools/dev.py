"""Developer tools: repo map and guarded git commit."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .builtin import PROTECTED, _S, _clip, _obj
from .registry import Registry, Risk, Tool, ToolContext, ToolError

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".pytest_cache"}
SYMBOLS = {
    ".py": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w+"),
    ".js": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const\s+\w+\s*=\s*\()\s*\w*"),
    ".ts": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+\w+"),
}
PLAYBOOK_DIR = Path(__file__).parent.parent / "playbooks"
PROTECTED_BRANCHES = {"main", "master"}
MAX_FILES = 200


def repo_map(ctx: ToolContext, path: str = ".") -> str:
    base = ctx.resolve(path)
    lines, count = [], 0
    for f in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in f.relative_to(ctx.root).parts) or not f.is_file():
            continue
        count += 1
        if count > MAX_FILES:
            lines.append(f"...[more than {MAX_FILES} files, narrow `path`]")
            break
        lines.append(str(f.relative_to(ctx.root)))
        pat = SYMBOLS.get(f.suffix)
        if pat and f.stat().st_size < 200_000:
            for line in f.read_text(errors="replace").splitlines():
                if pat.match(line):
                    lines.append("    " + line.strip()[:100])
    return _clip("\n".join(lines) or "(empty)")


def grep(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regex: {e}") from e
    base = ctx.resolve(path)
    hits = []
    for f in sorted(base.rglob("*") if base.is_dir() else [base]):
        if not f.is_file() or any(p in SKIP_DIRS for p in f.relative_to(ctx.root).parts):
            continue
        if PROTECTED.search(str(f)) or f.stat().st_size > 500_000:
            continue
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(lines, 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(ctx.root)}:{n}: {line.strip()[:160]}")
                if len(hits) >= 100:
                    return "\n".join(hits) + "\n...[stopped at 100 matches, narrow the pattern or path]"
    return "\n".join(hits) or "no matches"


def _git(ctx: ToolContext, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ctx.root, capture_output=True, text=True, timeout=ctx.shell_timeout)
    if r.returncode != 0:
        raise ToolError(r.stderr.strip() or f"git {args[0]} failed")
    return r.stdout.strip()


def git_commit(ctx: ToolContext, message: str) -> str:
    branch = _git(ctx, "symbolic-ref", "--short", "HEAD")
    if branch in PROTECTED_BRANCHES and not ctx.allow_main_commit:
        raise ToolError(f"refusing to commit on '{branch}'; create a feature branch first (git_branch tool)")
    _git(ctx, "add", "-A")
    return _git(ctx, "commit", "-m", message)


def git_branch(ctx: ToolContext, name: str) -> str:
    if not re.fullmatch(r"[\w./-]{1,60}", name) or name.startswith("-"):
        raise ToolError("invalid branch name")
    _git(ctx, "checkout", "-b", name)
    return f"on new branch {name}"


def load_playbook(ctx: ToolContext, name: str = "") -> str:
    available = sorted(p.stem for p in PLAYBOOK_DIR.glob("*.md"))
    if not name:
        return "available playbooks: " + ", ".join(available)
    if not re.fullmatch(r"[a-z0-9-]{1,40}", name) or name not in available:
        raise ToolError(f"unknown playbook '{name}' (available: {', '.join(available)})")
    return (PLAYBOOK_DIR / f"{name}.md").read_text()


def semantic_search(ctx: ToolContext, query: str) -> str:
    from .. import embeddings
    if not (ctx.embedder and ctx.store):
        raise ToolError("semantic search is not configured (needs an embeddings key; see [embeddings] in config)")
    return _clip(embeddings.search(ctx.root, query, ctx.embedder, ctx.store))


def register(reg: Registry) -> None:
    reg.register(Tool("grep", "Search files for a regex; returns file:line: text.",
                      _obj({"pattern": _S, "path": _S}, ["pattern"]), Risk.READ, grep, untrusted=True))
    reg.register(Tool("semantic_search", "Find the code or text most related to a query by meaning (sends file text to the embeddings provider).",
                      _obj({"query": _S}, ["query"]), Risk.NETWORK, semantic_search, untrusted=True))
    reg.register(Tool("load_playbook", "Load a step-by-step recipe (call with no name to list them).",
                      _obj({"name": _S}, []), Risk.READ, load_playbook))
    reg.register(Tool("repo_map", "List project files with top-level function/class names.",
                      _obj({"path": _S}, []), Risk.READ, repo_map, untrusted=True))
    reg.register(Tool("git_branch", "Create and switch to a new git branch.",
                      _obj({"name": _S}, ["name"]), Risk.WRITE, git_branch))
    reg.register(Tool("git_commit", "Stage all changes and commit (refused on main/master).",
                      _obj({"message": _S}, ["message"]), Risk.WRITE, git_commit))
