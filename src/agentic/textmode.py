"""Text tool protocol for models without reliable native tool-calling.

The model emits <tool_call>{"name": ..., "arguments": {...}}</tool_call>; we parse it into the
same tool_calls structure native mode returns, so the loop never knows the difference.
"""
from __future__ import annotations

import json
import re

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

INSTRUCTIONS = """

You can use tools. To call one, reply with exactly one block and nothing after it:
<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>
You will get the result in the next message. When the task is finished, reply in plain text with no tool_call.
Available tools (JSON schemas):
"""


def safe_args(raw) -> dict:
    """Tool-call arguments as a dict; invalid or non-object JSON (a weaker model's output) becomes {}."""
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def to_text_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "system":
            specs = "\n".join(json.dumps(t["function"]) for t in tools)
            out.append({"role": "system", "content": m["content"] + INSTRUCTIONS + specs})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_result>\n{m['content']}\n</tool_result>"})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            blocks = "\n".join(
                '<tool_call>' + json.dumps({"name": c["function"]["name"],
                                            "arguments": safe_args(c["function"]["arguments"])})
                + '</tool_call>' for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + blocks})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def parse_reply(message: dict) -> dict:
    """Turn <tool_call> blocks in the content into native-style tool_calls."""
    content = message.get("content") or ""
    calls = []
    for i, raw in enumerate(CALL_RE.findall(content)):
        try:
            data = json.loads(raw)
            calls.append({"id": f"text-{i}", "type": "function", "function": {
                "name": data["name"], "arguments": json.dumps(data.get("arguments") or {})}})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    if not calls:
        return message
    return {"role": "assistant", "content": CALL_RE.sub("", content).strip() or None, "tool_calls": calls}
