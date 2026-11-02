"""Minimal MCP server over stdio (JSON-RPC, one message per line).

Exposes only READ-risk tools (read_file, list_dir, repo_map), confined to one workspace, with secrets
redacted and untrusted output wrapped. Writes stay with the agent's own approval loop, so an editor
can use this for context without being able to change anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO

from .safety import redact
from .tools import Risk, ToolContext, ToolError, default_registry

PROTOCOL = "2024-11-05"


def exposed_tools(registry):
    return {n: t for n, t in registry._tools.items() if t.risk is Risk.READ and t.risk_fn is None}


def handle(msg: dict, registry, ctx: ToolContext) -> dict | None:
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params")
    if not isinstance(params, dict):
        params = {}
    if not isinstance(mid, (str, int, type(None))):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid id"}}
    tools = exposed_tools(registry)

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, text):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}

    if mid is None:  # notification, no reply
        return None
    if method == "initialize":
        return ok({"protocolVersion": params.get("protocolVersion", PROTOCOL), "capabilities": {"tools": {}},
                   "serverInfo": {"name": "agentic", "version": "0.1.0"}})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": t.description, "inputSchema": t.parameters}
                             for n, t in tools.items()]})
    if method == "tools/call":
        name = params.get("name")
        tool = tools.get(name) if isinstance(name, str) else None
        args = params.get("arguments")
        if not tool:
            return err(-32602, f"unknown or non-read-only tool: {name}")
        try:
            text = redact(tool.func(ctx, **(args if isinstance(args, dict) else {})))
            if tool.untrusted:
                text = text.replace("untrusted_data>", "untrusted-data>")
                text = f"<untrusted_data>\n{text}\n</untrusted_data>"
            return ok({"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:  # report any tool failure to the editor instead of dying
            return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    return err(-32601, f"method not found: {method}")


def serve(workspace: Path, stdin: IO = sys.stdin, stdout: IO = sys.stdout) -> None:
    ctx = ToolContext(workspace.resolve())
    registry = default_registry()
    for line in stdin:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line), registry, ctx)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
