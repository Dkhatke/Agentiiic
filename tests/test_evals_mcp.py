import io
import json
from pathlib import Path

from agentic.evals import load_tasks, run_task
from agentic.loop import Profile
from agentic.mcp_server import handle, serve
from agentic.router import Reply
from agentic.tools import ToolContext, default_registry

EVALS = Path(__file__).parent.parent / "evals"


class Scripted:
    calls = 0

    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, tools=None):
        self.calls += 1
        return Reply(next(self.replies), "fake", {})


def tc(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


PROFILE = Profile("t", "sys", ["read_file", "write_file", "edit_file", "run_shell"], max_steps=10)


def task(name):
    return next(t for t in load_tasks(EVALS) if t["name"] == name)


def test_eval_task_pass_and_fail():
    good = Scripted([tc("write_file", {"path": "hello.py", "content": "print('hello agent')"}), {"content": "done"}])
    assert run_task(task("hello-script"), PROFILE, good).passed
    bad = Scripted([{"content": "I did nothing"}])
    r = run_task(task("hello-script"), PROFILE, bad)
    assert not r.passed and r.failures


def test_eval_secret_task_fails_if_agent_leaks():
    leaky = Scripted([tc("write_file", {"path": "summary.txt", "content": "TOKEN=x"}), {"content": "done"}])
    assert not run_task(task("respects-secrets"), PROFILE, leaky).passed
    careful = Scripted([tc("read_file", {"path": ".env"}), {"content": "I cannot read that file"}])
    assert run_task(task("respects-secrets"), PROFILE, careful).passed


def test_all_shipped_tasks_parse():
    tasks = load_tasks(EVALS)
    assert len(tasks) >= 4 and all(t.get("checks") and t.get("prompt") for t in tasks)


def rpc(method, params=None, mid=1):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}


def test_mcp_lists_only_read_tools_and_calls(tmp_path):
    (tmp_path / "a.txt").write_text("key sk-abcdefghijklmnop1234 here")
    ctx, reg = ToolContext(tmp_path.resolve()), default_registry()
    names = {t["name"] for t in handle(rpc("tools/list"), reg, ctx)["result"]["tools"]}
    assert {"read_file", "list_dir", "repo_map"} <= names
    assert not names & {"write_file", "edit_file", "run_shell", "git_commit", "fetch_url"}
    out = handle(rpc("tools/call", {"name": "read_file", "arguments": {"path": "a.txt"}}), reg, ctx)["result"]
    assert "sk-abc" not in out["content"][0]["text"] and not out["isError"]
    assert "error" in handle(rpc("tools/call", {"name": "write_file", "arguments": {}}), reg, ctx)
    escape = handle(rpc("tools/call", {"name": "read_file", "arguments": {"path": "../x"}}), reg, ctx)["result"]
    assert escape["isError"]


def test_mcp_stdio_loop(tmp_path):
    lines = [rpc("initialize", {"protocolVersion": "2024-11-05"}), {"jsonrpc": "2.0", "method": "notifications/initialized"},
             rpc("ping", mid=2)]
    out = io.StringIO()
    serve(tmp_path, io.StringIO("\n".join(json.dumps(m) for m in lines) + "\nnot json\n"), out)
    replies = [json.loads(l) for l in out.getvalue().splitlines()]
    assert replies[0]["result"]["serverInfo"]["name"] == "agentic"
    assert replies[1]["result"] == {} and replies[2]["error"]["code"] == -32700 and len(replies) == 3
