"""Adapter between our OpenAI-style messages/tools and Anthropic's native Messages API."""
from __future__ import annotations

import json

VERSION = "2023-06-01"
MAX_TOKENS = 4096


def to_request(model: str, messages: list[dict], tools: list[dict] | None) -> dict:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":  # consecutive tool results share one user turn
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"][-1]["type"] == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for c in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                               "input": json.loads(c["function"]["arguments"] or "{}")})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
        else:
            out.append({"role": "user", "content": m.get("content") or ""})
    body: dict = {"model": model, "max_tokens": MAX_TOKENS, "messages": out}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                          "input_schema": t["function"]["parameters"]} for t in tools]
    return body


def from_response(data: dict) -> tuple[dict, dict]:
    """Return (OpenAI-style assistant message, usage with total_tokens)."""
    text, calls = [], []
    for b in data.get("content") or []:
        if b.get("type") == "text":
            text.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            calls.append({"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}})
    msg: dict = {"role": "assistant", "content": "".join(text)}
    if calls:
        msg["tool_calls"] = calls
    u = data.get("usage") or {}
    return msg, {"total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0)}
