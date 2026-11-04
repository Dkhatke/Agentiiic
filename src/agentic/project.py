"""Project mode: how medium projects get done with small free models.

A planner model splits the spec into a few small milestones, each with a shell command that proves it works. Each milestone
is built in a fresh session (small context, fewer tokens, less confusion), checked by running its command, and committed as a
git checkpoint. After ATTEMPTS failed tries the person is asked for a one-line hint instead of letting the model spin."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .tools import ToolContext, ToolError
from .tools.builtin import run_shell
from .tools.dev import _git, git_commit

PLAN_FILE = ".agentic/project.json"
MAX_MILESTONES = 8
ATTEMPTS = 2            # automatic tries per milestone before asking the person
FAILURE_CHARS = 1500    # tail of the failing check shown to the next attempt
NOTES_CHARS = 700       # each finished milestone's own summary, passed to later ones
MILESTONE_STEPS = 20    # per attempt: a small milestone needs fewer; stuck ones reach the person sooner
PLAN_PROMPT = """You are planning a software project that a small AI model will build one milestone at a time.
Split the spec below into 3 to {max} milestones, in build order. Each milestone must be small: 1 to 3 files, finishable in
about 15 tool calls, and leave the project working. Each needs a `check`: ONE shell command, run from the project folder with
no network, that exits 0 only when the milestone works (usually a test command such as `python3 -m pytest -q tests/test_x.py`;
the milestone itself should write those tests). Do not include setup that needs the internet (pip install, npm install).
Each milestone is built in a fresh session that cannot see your plan's reasoning, so also write `conventions`: the decisions
every milestone must follow so their pieces fit: file and module names, main function names and signatures, data format
or table schema, and how tests should run the code. For a command line program, write one literal example invocation per
command, with every argument and option in its place (e.g. `python3 app.py search TEXT --db PATH`), and the exact output format
of each command; in live runs most failures were milestones guessing these differently. Tests must never share or
touch the real data file: the code takes its data path from an option or environment variable, and every test uses a fresh
temporary one. Tests should compare output with whitespace stripped.
Reply with ONLY a JSON object, no prose:
{{"conventions": "...", "milestones": [{{"title": "short name", "goal": "what to build and how it should behave", "check": "shell command"}}]}}

Spec:
{spec}"""


@dataclass
class Milestone:
    title: str
    goal: str
    check: str
    status: str = "pending"  # pending | done | skipped
    attempts: int = 0
    notes: str = ""          # what the worker said it built, for later milestones


def _json_in(text: str):
    """The plan in a reply: an object with milestones (and conventions), or a bare milestone array."""
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) or (isinstance(data, dict) and "milestones" in data):
            return data
    return None


def parse_conventions(text: str) -> str:
    data = _json_in(text)
    c = data.get("conventions") if isinstance(data, dict) else ""
    return c.strip() if isinstance(c, str) else json.dumps(c) if c else ""


def parse_plan(text: str) -> list[Milestone] | None:
    """Pull the milestones out of a model reply; None if there is nothing usable."""
    data = _json_in(text)
    if isinstance(data, dict):
        data = data.get("milestones")
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in ("title", "goal", "check")):
            out.append(Milestone(item["title"].strip()[:80], item["goal"].strip(), item["check"].strip()))
    return out[:MAX_MILESTONES] or None


def make_plan(spec: str, planner: Callable[[str], str]) -> tuple[str, list[Milestone]]:
    """Returns (conventions, milestones)."""
    prompt = PLAN_PROMPT.format(max=MAX_MILESTONES, spec=spec)
    for _ in range(2):  # one retry: small models sometimes wrap the JSON in prose or cut it short
        reply = planner(prompt)
        plan = parse_plan(reply)
        if plan:
            return parse_conventions(reply), plan
        prompt += "\n\nYour last reply was not valid JSON in the format above. Reply with the JSON object only."
    raise ToolError("the planner did not return a usable milestone list; try again or simplify the spec")


def save(root: Path, spec: str, plan: list[Milestone], conventions: str = "") -> None:
    f = root / PLAN_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    ignore = f.parent / ".gitignore"
    if not ignore.exists():  # agent bookkeeping stays out of the checkpoints
        ignore.write_text("project.json\ntmp/\n")
    f.write_text(json.dumps({"spec": spec, "conventions": conventions, "milestones": [asdict(m) for m in plan]}, indent=2))


def load(root: Path) -> tuple[str, str, list[Milestone]] | None:
    """Returns (spec, conventions, milestones)."""
    try:
        data = json.loads((root / PLAN_FILE).read_text())
        return data["spec"], data.get("conventions", ""), [Milestone(**m) for m in data["milestones"]]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def describe(plan: list[Milestone], conventions: str = "") -> str:
    mark = {"done": "x", "skipped": "-", "pending": " "}
    lines = [f"[{mark.get(m.status, ' ')}] {i}. {m.title}\n      check: {m.check}" for i, m in enumerate(plan, 1)]
    return "\n".join(([f"Conventions: {conventions}"] if conventions else []) + lines)


def milestone_task(spec: str, plan: list[Milestone], i: int, failure: str = "", hint: str = "", conventions: str = "") -> str:
    m = plan[i]
    done = [p for p in plan[:i] if p.status == "done"]
    parts = [f"You are building one milestone of a larger project. Project spec (for context only):\n{spec}\n"]
    if conventions:
        parts.append(f"Project conventions (follow them exactly so the pieces fit):\n{conventions}\n")
    if done:
        parts.append("Already finished, and their code is in the workspace (read it before you add to it; keep its "
                     "interfaces and command syntax unless your milestone says otherwise):")
        parts += [f"- {p.title}: {p.notes[:NOTES_CHARS]}" if p.notes else f"- {p.title}" for p in done]
    else:
        parts.append("This is the first milestone.")
    parts += [
              f"\nYOUR MILESTONE ({i + 1}/{len(plan)}): {m.title}\n{m.goal}",
              f"\nIt is done when this command exits 0: `{m.check}`. Run it yourself before you finish. "
              "Do only this milestone; later milestones will handle the rest. End with a short summary of the files, "
              "functions and command syntax you added, for the milestones after you."]
    if failure:
        parts.append(f"\nA previous attempt failed. The check printed (end of output):\n{failure[-FAILURE_CHARS:]}\n"
                     "Decide whether the code or the test is wrong before changing anything.")
    if hint:
        parts.append(f"\nHint from the person: {hint}")
    return "\n".join(parts)


def run_check(ctx: ToolContext, command: str) -> tuple[bool, str]:
    try:
        out = run_shell(ctx, command)
    except ToolError as e:
        return False, f"error: {e}"
    return out.startswith("exit=0"), out


def is_repo(root: Path) -> bool:
    return (root / ".git").exists()


def prepare_git(ctx: ToolContext, branch: str) -> str:
    """Make sure checkpoints have a non-main branch to land on. Returns a line for the person."""
    if not is_repo(ctx.root):
        _git(ctx, "init", "-q")
        _git(ctx, "checkout", "-q", "-b", branch)
        return f"started a git repository on branch {branch} for checkpoints"
    try:
        current = _git(ctx, "symbolic-ref", "--short", "HEAD")
    except ToolError:
        current = ""
    if current in {"main", "master"}:
        _git(ctx, "checkout", "-q", "-b", branch)
        return f"checkpoints go on new branch {branch} (your {current} is untouched)"
    return f"checkpoints go on branch {current}"


def checkpoint(ctx: ToolContext, message: str) -> str:
    try:
        return git_commit(ctx, message).splitlines()[0]
    except ToolError as e:
        return f"checkpoint skipped: {e}"


def undo(ctx: ToolContext) -> str:
    """Set aside everything since the last checkpoint. `git stash` keeps it recoverable (git stash list / pop)."""
    try:
        _git(ctx, "stash", "push", "-u", "-q", "-m", "agentic: undone milestone attempt")
        return "changes since the last checkpoint were stashed (recover with `git stash pop`)"
    except ToolError as e:
        return f"undo failed: {e}"


def run_project(spec: str, plan: list[Milestone], ctx: ToolContext, worker: Callable[[str], str],
                ask_stuck: Callable[[Milestone, str], str], on_event: Callable[[str], None] = print,
                conventions: str = "") -> str:
    """Build every pending milestone. ask_stuck returns a hint, or 's' (skip), 'u' (undo and retry), 'q' (stop)."""
    for i, m in enumerate(plan):
        if m.status != "pending":
            continue
        failure, hint, tries = "", "", 0
        while True:
            on_event(f"milestone {i + 1}/{len(plan)}: {m.title}" + (f" (attempt {tries + 1})" if tries else ""))
            summary = worker(milestone_task(spec, plan, i, failure, hint, conventions)) or ""
            m.attempts += 1
            tries += 1
            ok, out = run_check(ctx, m.check)
            if ok:
                m.status, m.notes = "done", summary.strip()[:NOTES_CHARS]
                on_event(f"  check passed: {m.check}")
                if is_repo(ctx.root):
                    on_event("  " + checkpoint(ctx, f"milestone {i + 1}: {m.title}"))
                break
            failure = out
            on_event(f"  check failed: {m.check}")
            save(ctx.root, spec, plan, conventions)
            if tries < ATTEMPTS:
                continue
            answer = ask_stuck(m, out[-FAILURE_CHARS:]).strip()
            if answer.lower() in {"q", "quit"}:
                save(ctx.root, spec, plan, conventions)
                return f"Stopped at milestone {i + 1}. Continue later with: agentic project --continue\n\n{describe(plan, conventions)}"
            if answer.lower() in {"s", "skip"}:
                m.status = "skipped"
                break
            if answer.lower() in {"u", "undo"}:
                on_event("  " + undo(ctx))
                failure, hint, tries = "", "", 0
                continue
            hint, tries = answer, 0  # a hint earns a fresh set of attempts
        save(ctx.root, spec, plan, conventions)
    done = sum(m.status == "done" for m in plan)
    return f"Project finished: {done}/{len(plan)} milestones passed their checks.\n\n{describe(plan, conventions)}"
