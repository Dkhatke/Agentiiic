"""Model router: one OpenAI-compatible client, ordered fallback chain, cooldowns."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import Provider
from .textmode import parse_reply, to_text_messages

RETRYABLE = {408, 429, 500, 502, 503, 504}
AUTH_FAILED = {401, 403}


class RouterError(Exception):
    pass


@dataclass
class Reply:
    message: dict
    provider: str
    usage: dict


class Router:
    def __init__(self, providers: list[Provider], client: httpx.Client | None = None,
                 cooldown: float = 60.0, clock=time.monotonic, store=None):
        self.store = store
        self.providers = [p for p in providers if p.usable]
        self.client = client or httpx.Client(timeout=120)
        self.cooldown = cooldown
        self.clock = clock
        self._blocked_until: dict[str, float] = {}
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        if not self.providers:
            raise RouterError("no usable providers (set an API key env var or run a local model)")
        errors = []
        for p in self.providers:
            if self._blocked_until.get(p.name, 0) > self.clock():
                errors.append(f"{p.name}: cooling down")
                continue
            if self.store and p.daily_calls and self.store.calls_today(p.name) >= p.daily_calls:
                errors.append(f"{p.name}: daily budget reached")
                continue
            try:
                return self._call(p, messages, tools)
            except _Skip as e:
                self._blocked_until[p.name] = self.clock() + self.cooldown
                errors.append(f"{p.name}: {e}")
        raise RouterError("all providers failed: " + "; ".join(errors))

    def _call(self, p: Provider, messages, tools) -> Reply:
        text_mode = bool(tools) and p.tool_mode == "text"
        body: dict = {"model": p.model, "messages": to_text_messages(messages, tools) if text_mode else messages}
        if tools and not text_mode:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}
        self.calls += 1
        if self.store:
            self.store.add_usage(p.name, calls=1)
        try:
            r = self.client.post(f"{p.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise _Skip(f"unreachable ({type(e).__name__})") from e
        if r.status_code in RETRYABLE or r.status_code in AUTH_FAILED:
            raise _Skip(f"HTTP {r.status_code}")
        if r.status_code >= 400:
            raise _Skip(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise _Skip("malformed response") from e
        usage = data.get("usage") or {}
        if self.store and usage.get("total_tokens"):
            self.store.add_usage(p.name, tokens=int(usage["total_tokens"]))
        return Reply(parse_reply(msg) if text_mode else msg, p.name, usage)


class _Skip(Exception):
    """Provider failed in a way that justifies trying the next one."""
