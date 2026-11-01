"""The agent loop: model -> tool call -> permission gate -> execute -> observe -> repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .audit import Audit
from .permissions import Gate
from .router import Router
from .safety import redact
from .tools import Registry, ToolContext, ToolError


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
        on_event: Callable[[str], None] = lambda _: None) -> str:
    messages = [{"role": "system", "content": profile.system_prompt}, {"role": "user", "content": task}]
    schemas = registry.schemas(profile.tools)
    audit.record("task_start", task=task, profile=profile.name)

    for step in range(profile.max_steps):
        if router.calls >= max_model_calls:
            audit.record("budget_stop", reason="max_model_calls")
            return "Stopped: model call budget reached."
        reply = router.chat(messages, schemas)
        msg = reply.message
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            audit.record("task_done", steps=step + 1, provider=reply.provider)
            return msg.get("content") or ""
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": _execute(call, profile, registry, gate, ctx, audit, on_event)})

    audit.record("budget_stop", reason="max_steps")
    return "Stopped: step limit reached."


def _execute(call, profile, registry, gate, ctx, audit, on_event) -> str:
    name = call["function"]["name"]
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
