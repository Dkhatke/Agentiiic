import json

import httpx
import pytest

from agentic.audit import Audit
from agentic.config import Provider
from agentic.loop import Profile, run
from agentic.permissions import Gate, Mode
from agentic.router import Router, RouterError
from agentic.tools import ToolContext, ToolError, default_registry


def ctx(tmp_path):
    return ToolContext(tmp_path.resolve())


def test_path_escape_blocked(tmp_path):
    with pytest.raises(ToolError):
        ctx(tmp_path).resolve("../outside.txt")
    with pytest.raises(ToolError):
        ctx(tmp_path).resolve("/etc/passwd")


def test_gate_modes(tmp_path):
    reg = default_registry()
    asked = []
    gate = Gate(Mode.ASK_ON_WRITE, lambda s: asked.append(s) or False)
    assert gate.allow(reg.get("read_file"), {"path": "a"})[0] is True
    assert gate.allow(reg.get("write_file"), {"path": "a", "content": ""})[0] is False
    assert gate.allow(reg.get("run_shell"), {"command": "git status"})[0] is True      # safe command
    assert gate.allow(reg.get("run_shell"), {"command": "git status; rm -rf /"})[0] is False
    assert len(asked) == 2
    ro = Gate(Mode.READONLY, lambda s: True)
    assert ro.allow(reg.get("write_file"), {})[0] is False


def test_router_falls_back_and_cools_down():
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        if "bad" in str(req.url):
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    providers = [Provider("a", "http://bad/v1", "m"), Provider("b", "http://good/v1", "m")]
    r = Router(providers, httpx.Client(transport=httpx.MockTransport(handler)))
    assert r.chat([]).provider == "b"
    assert r.chat([]).provider == "b"
    assert sum("bad" in u for u in seen) == 1  # second call skipped the cooling-down provider


def test_router_all_fail():
    r = Router([Provider("a", "http://x/v1", "m")],
               httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500))))
    with pytest.raises(RouterError):
        r.chat([])


class ScriptedRouter:
    calls = 0

    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, tools=None):
        from agentic.router import Reply
        self.calls += 1
        return Reply(next(self.replies), "fake", {})


def tool_msg(name, args):
    return {"content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def profile():
    return Profile("t", "sys", ["read_file", "write_file"], max_steps=5)


def test_loop_write_requires_approval(tmp_path):
    for approve in (False, True):
        router = ScriptedRouter([tool_msg("write_file", {"path": "x.txt", "content": "hi"}), {"content": "done"}])
        out = run("t", profile(), router, default_registry(), Gate(Mode.ASK_ON_WRITE, lambda s: approve),
                  ctx(tmp_path), Audit(tmp_path / "audit.jsonl"))
        assert out == "done"
        assert (tmp_path / "x.txt").exists() is approve
    assert "tool_call" in (tmp_path / "audit.jsonl").read_text()


def test_loop_blocks_tool_outside_profile(tmp_path):
    router = ScriptedRouter([tool_msg("run_shell", {"command": "echo hi"}), {"content": "done"}])
    run("t", profile(), router, default_registry(), Gate(Mode.ASK_ALL, lambda s: True),
        ctx(tmp_path), Audit(tmp_path / "audit.jsonl"))
    assert "tool not allowed" in (tmp_path / "audit.jsonl").read_text()


def test_loop_step_cap(tmp_path):
    router = ScriptedRouter([tool_msg("read_file", {"path": "nope"})] * 10)
    out = run("t", profile(), router, default_registry(), Gate(Mode.ASK_ON_WRITE, lambda s: True),
              ctx(tmp_path), Audit(tmp_path / "audit.jsonl"))
    assert "step limit" in out
