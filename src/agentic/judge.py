"""Builds the callable behind `judge_ideas`: a router whose first choice differs from the main chain's first choice."""
from __future__ import annotations

from .router import Router, RouterError


def make_judge(cfg, store=None, client=None):
    providers = list(cfg.providers)
    if cfg.judge.provider:  # the user picked a specific judge
        providers.sort(key=lambda p: p.name != cfg.judge.provider)
    elif len(providers) > 1:  # otherwise rotate so the generator's favourite model is not its own judge
        providers = providers[1:] + providers[:1]
    router = Router(providers, client=client, store=store)

    def judge(prompt: str) -> str:
        try:
            return router.chat([{"role": "user", "content": prompt}]).message.get("content") or ""
        except RouterError as e:
            return f"judge unavailable: {e}"
    return judge
