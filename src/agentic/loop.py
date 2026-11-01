"""The agent loop: model -> tool call -> permission gate -> execute -> observe -> repeat."""
from __future__ import annotations

import json
import platform
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .audit import Audit
from .permissions import Gate
from .router import Router
from .safety import redact
from .tools import Registry, ToolContext, ToolError


MAX_REPEATS = 2           # identical tool calls allowed before we push back
CONTEXT_CHARS = 48_000    # rough budget (~12k tokens) so small free-model contexts don't overflow
KEEP_RECENT = 6           # newest messages are never compacted


def env_note(ctx: ToolContext) -> str:
    """Facts about this machine so the model doesn't waste steps guessing (e.g. `python` vs `python3`)."""
    py = "python3" if shutil.which("python3") else ("python" if shutil.which("python") else "no python")
    git = "a git repository" if (ctx.root / ".git").exists() else "not a git repository"
    return (f"\n\nEnvironment: {platform.system()} {platform.machine()}; run Python with `{py}` "
            f"(`{py} -m pip` for packages); workspace is {git}. Shell commands run in the workspace with no network "
            f"unless the user enabled it.")


def compact(messages: list[dict]) -> None:
    """Shrink old tool outputs in place once the conversation outgrows the budget."""
    if sum(len(m.get("content") or "") for m in messages) <= CONTEXT_CHARS:
        return
    for m in messages[:-KEEP_RECENT]:
        c = m.get("content") or ""
        if m["role"] == "tool" and len(c) > 400:
            m["content"] = c[:300] + f"\n...[older output compacted, {len(c) - 300} chars dropped]"


@dataclass
class Profile:
    name: str
    system_prompt: str
    tools: list[str]
    max_steps: int = 25
    mode: str | None = None          # default permission mode for this profile
    keywords: tuple[str, ...] = ()   # used by the orchestrator to pick this profile


def run(task: str, profile: Profile, router: Router, registry: Registry, gate: Gate,
        ctx: ToolContext, audit: Audit, max_model_calls: int = 60,
        on_event: Callable[[str], None] = lambda _: None, store=None, task_id: int | None = None,
        messages: list[dict] | None = None) -> str:
    messages = _repair(messages) if messages else [
        {"role": "system", "content": profile.system_prompt + env_note(ctx)}, {"role": "user", "content": task}]
    save = (lambda status: store.save_task(task_id, messages, status)) if store and task_id else (lambda _: None)
    schemas = registry.schemas(profile.tools)
    calls_at_start = getattr(router, "calls", 0)  # budget is per run, not per router lifetime
    audit.record("task_start", task=task, profile=profile.name)

    seen: Counter = Counter()
    for step in range(profile.max_steps):
        if getattr(router, "calls", 0) - calls_at_start >= max_model_calls:
            audit.record("budget_stop", reason="max_model_calls")
            save("stopped")
            return "Stopped: model call budget reached."
        compact(messages)
        reply = router.chat(messages, schemas)
        msg = {"role": "assistant", "content": reply.message.get("content")}  # drop provider-specific extras
        if reply.message.get("tool_calls"):
            msg["tool_calls"] = reply.message["tool_calls"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            audit.record("task_done", steps=step + 1, provider=reply.provider)
            save("done")
            return msg.get("content") or ""
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": _execute(call, profile, registry, gate, ctx, audit, on_event, seen)})
        save("running")

    audit.record("budget_stop", reason="max_steps")
    save("stopped")
    return "Stopped: step limit reached."


def _repair(messages: list[dict]) -> list[dict]:
    """Drop a trailing assistant tool-call turn whose results never got recorded (crash mid-step)."""
    while messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
        messages.pop()
    return messages


def _execute(call, profile, registry, gate, ctx, audit, on_event, seen: Counter) -> str:
    name = call["function"]["name"]
    key = (name, call["function"].get("arguments") or "")
    seen[key] += 1
    if seen[key] > MAX_REPEATS:
        audit.record("stuck", tool=name)
        return (f"error: you have already made this exact {name} call {MAX_REPEATS} times. "
                "Change your approach, or finish and explain what is blocking you.")
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
        if name not in profile.tools:
            raise ToolError(f"tool not allowed for this profile: {name}")
        tool = registry.get(name)
    except (json.JSONDecodeError, ToolError) as e:
        audit.record("tool_error", tool=name, error=str(e))
        return f"error: {e}"
    ok, why = gate.allow(tool, args)
    audit.record("tool_call", tool=name, args=json.loads(redact(json.dumps(args))), decision=why)
    on_event(f"{name} {args} -> {why}")
    if not ok:
        return why
    try:
        out = redact(tool.func(ctx, **args))
    except (ToolError, TypeError, OSError) as e:
        audit.record("tool_error", tool=name, error=str(e))
        return f"error: {e}"
    if tool.untrusted:
        gate.tainted = True
        out = f"<untrusted_data>\n{out}\n</untrusted_data>"
    return out
