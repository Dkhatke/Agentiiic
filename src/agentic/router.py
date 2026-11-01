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
    def __init__(self, msg: str, retryable: bool = False):
        super().__init__(msg)
        self.retryable = retryable  # every provider failed only for temporary reasons


@dataclass
class Reply:
    message: dict
    provider: str
    usage: dict


class Router:
    def __init__(self, providers: list[Provider], client: httpx.Client | None = None,
                 cooldown: float = 30.0, clock=time.monotonic, store=None, retries: int = 2,
                 sleep=time.sleep, max_wait: float = 45.0):
        self.store = store
        self.retries, self.sleep, self.max_wait = retries, sleep, max_wait
        self.providers = [p for p in providers if p.usable]
        self.client = client or httpx.Client(timeout=120)
        self.cooldown = cooldown
        self.clock = clock
        self._blocked_until: dict[str, float] = {}
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        for attempt in range(self.retries + 1):
            try:
                return self._try_all(messages, tools)
            except RouterError as e:
                if not e.retryable or attempt == self.retries:
                    raise
                now = self.clock()
                waits = [t - now for t in self._blocked_until.values() if t > now]
                self.sleep(min(max(min(waits, default=1.0), 1.0) + 0.5, self.max_wait))

    def _try_all(self, messages, tools) -> Reply:
        if not self.providers:
            raise RouterError("no usable providers (set an API key env var or run a local model)")
        errors, retryable = [], True
        for p in self.providers:
            if self._blocked_until.get(p.name, 0) > self.clock():
                errors.append(f"{p.name}: cooling down")
                continue
            if self.store and p.daily_calls and self.store.calls_today(p.quota or p.name) >= p.daily_calls:
                errors.append(f"{p.name}: daily budget reached")
                retryable = False
                continue
            try:
                return self._call(p, messages, tools)
            except _Skip as e:
                self._blocked_until[p.name] = self.clock() + self.cooldown
                errors.append(f"{p.name}: {e}")
                retryable = retryable and e.transient
        raise RouterError("all providers failed: " + "; ".join(errors), retryable)

    def _call(self, p: Provider, messages, tools) -> Reply:
        text_mode = bool(tools) and p.tool_mode == "text"
        body: dict = {"model": p.model, "messages": to_text_messages(messages, tools) if text_mode else messages}
        if tools and not text_mode:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}
        self.calls += 1
        if self.store:
            self.store.add_usage(p.quota or p.name, calls=1)
        try:
            r = self.client.post(f"{p.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise _Skip(f"unreachable ({type(e).__name__})", transient=True) from e
        if r.status_code in RETRYABLE:
            raise _Skip(f"HTTP {r.status_code}", transient=True)
        if r.status_code in AUTH_FAILED:
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
            self.store.add_usage(p.quota or p.name, tokens=int(usage["total_tokens"]))
        return Reply(parse_reply(msg) if text_mode else msg, p.name, usage)


class _Skip(Exception):
    """Provider failed in a way that justifies trying the next one."""

    def __init__(self, msg: str, transient: bool = False):
        super().__init__(msg)
        self.transient = transient
