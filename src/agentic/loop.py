"""The agent loop: model -> tool call -> permission gate -> execute -> observe -> repeat."""
from __future__ import annotations

import json
import platform
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from . import notes
from .audit import Audit
from .permissions import Gate
from .router import Router
from .safety import redact
from .tools import Registry, ToolContext, ToolError


MAX_REPEATS = 2           # identical tool calls allowed before we push back
CONTEXT_CHARS = 48_000    # rough budget (~12k tokens) so small free-model contexts don't overflow
KEEP_RECENT = 6           # newest messages are never compacted
PLAN_MIN_WORDS = 20       # only plan for tasks long enough to need it (saves model calls)
WRITE_TOOLS = {"write_file", "edit_file"}
CITATION_NUDGE = ("Your answer cites URLs you never fetched, so you cannot know what they say:\n{urls}\n"
                  "Either fetch them now, or remove them and keep only sources you actually read.")
URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")
VERIFY_NUDGE = ("You changed files but have not run anything since. Verify now: run the project's tests or the code. "
                "If you truly cannot verify, say exactly why. Then give your final summary.")
FINAL_STEP = ("This is your last step. Do not call any more tools. Answer now using what you have found, "
              "and state clearly what is still unverified or unfinished.")
MIN_STEPS_FOR_FORCED_ANSWER = 4
PLAN_REQUEST = ("Before acting, write a short numbered plan (at most 6 steps) that ends with how you will verify the result. "
                "Do not call tools yet.")


def env_note(ctx: ToolContext) -> str:
    """Facts about this machine so the model doesn't waste steps guessing (e.g. `python` vs `python3`)."""
    py = "python3" if shutil.which("python3") else ("python" if shutil.which("python") else "no python")
    git = "a git repository" if (ctx.root / ".git").exists() else "not a git repository"
    return (f"\n\nEnvironment: {platform.system()} {platform.machine()}; run Python with `{py}` "
            f"(`{py} -m pip` for packages); workspace is {git}. Shell commands run in the workspace with no network "
            f"unless the user enabled it.")


def _norm_url(u: str) -> str:
    return u.split("#")[0].rstrip(".,;:/")


def unfetched_urls(text: str, ctx: ToolContext) -> list[str]:
    seen = {_norm_url(u) for u in ctx.fetched}
    out = []
    for u in URL_RE.findall(text):
        if _norm_url(u) not in seen and u not in out:
            out.append(u)
    return out


def compact(messages: list[dict], budget: int = CONTEXT_CHARS) -> None:
    """Shrink tool outputs in place once the conversation outgrows the budget: old ones first, then the largest."""
    size = lambda: sum(len(m.get("content") or "") for m in messages)  # noqa: E731
    if size() <= budget:
        return
    for m in messages[:-KEEP_RECENT]:
        c = m.get("content") or ""
        if m["role"] == "tool" and len(c) > 400:
            m["content"] = c[:300] + f"\n...[older output compacted, {len(c) - 300} chars dropped]"
        for call in m.get("tool_calls") or []:  # big file bodies in old write calls
            raw = call["function"].get("arguments") or ""
            if len(raw) > 600:
                try:
                    args = json.loads(raw)
                    call["function"]["arguments"] = json.dumps(
                        {k: (v[:200] + "...[compacted]" if isinstance(v, str) and len(v) > 200 else v) for k, v in args.items()})
                except (json.JSONDecodeError, AttributeError):
                    pass
    while size() > budget:  # still too big: halve the largest tool output (never the system prompt or user turns)
        tools = [m for m in messages if m["role"] == "tool" and len(m.get("content") or "") > 1500]
        if not tools:
            return
        biggest = max(tools, key=lambda m: len(m["content"]))
        biggest["content"] = biggest["content"][:len(biggest["content"]) // 2] + "\n...[output shortened to fit the model's input limit]"


@dataclass
class Profile:
    name: str
    system_prompt: str
    tools: list[str]
    max_steps: int = 25
    mode: str | None = None          # default permission mode for this profile
    keywords: tuple[str, ...] = ()   # used by the orchestrator to pick this profile
    plan: bool = False               # for long tasks, write a short plan before acting
    checks: tuple[str, ...] = ()     # extra pre-finish checks: "citations"


def run(task: str, profile: Profile, router: Router, registry: Registry, gate: Gate,
        ctx: ToolContext, audit: Audit, max_model_calls: int = 60,
        on_event: Callable[[str], None] = lambda _: None, store=None, task_id: int | None = None,
        messages: list[dict] | None = None) -> str:
    resuming = bool(messages)
    messages = _repair(messages) if messages else [
        {"role": "system", "content": profile.system_prompt + env_note(ctx) + notes.load(ctx.root)}, {"role": "user", "content": task}]
    save = (lambda status: store.save_task(task_id, messages, status)) if store and task_id else (lambda _: None)
    schemas = registry.schemas(profile.tools)
    calls_at_start = getattr(router, "calls", 0)  # budget is per run, not per router lifetime
    audit.record("task_start", task=task, profile=profile.name)

    seen: Counter = Counter()
    if profile.plan and not resuming and len(task.split()) >= PLAN_MIN_WORDS:
        plan = router.chat(messages + [{"role": "user", "content": PLAN_REQUEST}], None).message.get("content")
        if plan:
            messages.append({"role": "assistant", "content": plan})
            on_event("plan: " + plan.splitlines()[0][:100])
    dirty, nudged = False, set()  # dirty: files changed since the last command ran; nudged: one-shot checks used
    for step in range(profile.max_steps):
        if getattr(router, "calls", 0) - calls_at_start >= max_model_calls:
            audit.record("budget_stop", reason="max_model_calls")
            save("stopped")
            return "Stopped: model call budget reached."
        compact(messages, getattr(router, "context_budget", None) or CONTEXT_CHARS)
        last = profile.max_steps >= MIN_STEPS_FOR_FORCED_ANSWER and step == profile.max_steps - 1
        if last:  # guarantee an answer instead of running out of steps mid-research
            messages.append({"role": "user", "content": FINAL_STEP})
        reply = router.chat(messages, None if last else schemas)
        msg = {"role": "assistant", "content": reply.message.get("content")}  # drop provider-specific extras
        if reply.message.get("tool_calls"):
            msg["tool_calls"] = reply.message["tool_calls"]
        messages.append(msg)
        calls = [] if last else (msg.get("tool_calls") or [])
        if not calls and not last:
            nudge = None
            if dirty and "verify" not in nudged and "run_shell" in profile.tools:
                nudge = ("verify", VERIFY_NUDGE)
            elif "citations" in profile.checks and "citations" not in nudged:
                bad = unfetched_urls(msg.get("content") or "", ctx)
                if bad:
                    nudge = ("citations", CITATION_NUDGE.format(urls="\n".join(bad[:5])))
            if nudge:
                nudged.add(nudge[0])
                messages.append({"role": "user", "content": nudge[1]})
                audit.record("final_nudge", kind=nudge[0])
                continue
        if not calls:
            audit.record("task_done", steps=step + 1, provider=reply.provider)
            save("done")
            return msg.get("content") or ("Stopped: step limit reached." if last else "")
        for call in calls:
            result = _execute(call, profile, registry, gate, ctx, audit, on_event, seen)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if not result.startswith(("error", "denied")):
                name = call["function"]["name"]
                dirty = True if name in WRITE_TOOLS else (False if name == "run_shell" else dirty)
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
        if not isinstance(args, dict):
            raise ToolError(f"arguments for {name} must be a JSON object like {{\"path\": \"a.txt\"}}, got {type(args).__name__}")
        if name not in profile.tools:
            raise ToolError(f"tool not allowed for this profile: {name}. Available: {', '.join(profile.tools)}")
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
    except TypeError as e:
        props = tool.parameters.get("properties", {})
        audit.record("tool_error", tool=name, error=str(e))
        return (f"error: invalid arguments for {name}. Expected {json.dumps(list(props))} "
                f"(required: {tool.parameters.get('required', [])}), got {json.dumps(list(args))}.")
    except (ToolError, OSError) as e:
        audit.record("tool_error", tool=name, error=str(e))
        return f"error: {e}"
    except Exception as e:  # a tool bug or odd input must never take the whole run down
        audit.record("tool_error", tool=name, error=f"{type(e).__name__}: {e}")
        return f"error: {name} failed ({type(e).__name__}: {e}). Check the arguments and try a different approach."
    if tool.untrusted:
        gate.tainted = True
        out = out.replace("untrusted_data>", "untrusted-data>")  # content cannot close or fake the wrapper
        out = f"<untrusted_data>\n{out}\n</untrusted_data>"
    return out
