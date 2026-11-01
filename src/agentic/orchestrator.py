"""Pick a profile for a request. Keyword scoring first; no model call needed."""
from __future__ import annotations

import re

from .loop import Profile
from .profiles import all_profiles, load_profile

DEFAULT = "general"


def select_profile(task: str, profiles: list[Profile] | None = None) -> Profile:
    text = task.lower()
    scored = []
    for p in profiles or all_profiles():
        score = sum(1 for k in p.keywords if re.search(rf"\b{re.escape(k)}", text))
        scored.append((score, p))
    best = max(scored, key=lambda s: s[0])
    return best[1] if best[0] > 0 else load_profile(DEFAULT)
