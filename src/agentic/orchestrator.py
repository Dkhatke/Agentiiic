"""Pick a profile for a request. Keyword scoring first; when that is ambiguous (no match, or a tie) and a router is
available, one short model call decides. Any failure there falls back to the keyword result."""
from __future__ import annotations

import re

from .loop import Profile
from .profiles import all_profiles, load_profile
from .router import RouterError

DEFAULT = "general"

ROUTE_PROMPT = """Pick the best agent for this request. Reply with only the agent name.

Agents:
{agents}

Request: {task}"""


def _ask_model(task: str, candidates: list[Profile], router) -> Profile | None:
    agents = "\n".join(f"- {p.name}: {', '.join(p.keywords[:8])}" for p in candidates)
    try:
        reply = router.chat([{"role": "user", "content": ROUTE_PROMPT.format(agents=agents, task=task[:1000])}])
    except RouterError:
        return None
    text = (reply.message.get("content") or "").lower()
    for p in candidates:
        if re.search(rf"\b{re.escape(p.name)}\b", text):
            return p
    return None


def select_profile(task: str, profiles: list[Profile] | None = None, router=None) -> Profile:
    text = task.lower()
    profiles = profiles or all_profiles()
    scored = [(sum(1 for k in p.keywords if re.search(rf"\b{re.escape(k)}", text)), p) for p in profiles]
    top = max(s for s, _ in scored)
    tied = [p for s, p in scored if s == top]
    if router is not None and len(tied) > 1:  # includes top == 0: nothing matched, so every profile is a candidate
        picked = _ask_model(task, tied, router)
        if picked:
            return picked
    return tied[0] if top > 0 else load_profile(DEFAULT)
