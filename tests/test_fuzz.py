"""Seeded random-input tests: parsers and converters must never raise on junk and must keep their invariants."""
import copy
import io
import json
import random

import pytest

from agentic import anthropic_api
from agentic.loop import CONTEXT_CHARS, compact
from agentic.mcp_server import handle, serve
from agentic.safety.redact import redact
from agentic.textmode import parse_reply, to_text_messages
from agentic.tools import ToolContext, default_registry
from agentic.tools import web
from agentic.tools.builtin import _names_protected_file
from agentic.tools.ideas import parse_scores

FRAGMENTS = ["<tool_call>", "</tool_call>", '{"name": "read_file", "arguments": {"path": "a"}}', "{", "}", "[", "]", '"', "'", "\\",
             "\x00", "\n", " ", "é", "日本", "😀", "sk-", "Bearer ", "://", "@", "<untrusted_data>", "</untrusted_data>", "null", "5", "-1",
             "http://", "../", "~", "$(", ")", "`", ";", "|", "&", "*", "?", "[a]", ".env", "a" * 300, "<script>", "</div>", "<p", "&amp;"]


def junk(rng: random.Random, n: int | None = None) -> str:
    return "".join(rng.choice(FRAGMENTS) for _ in range(n if n is not None else rng.randint(0, 25)))


SEEDS = range(200)


@pytest.mark.parametrize("seed", SEEDS)
def test_parse_reply_never_raises_and_returns_a_valid_message(seed):
    rng = random.Random(seed)
    msg = rng.choice([{"content": junk(rng)}, {"content": None}, {}, {"role": "assistant", "content": junk(rng) + "<tool_call>" + junk(rng) + "</tool_call>"}])
    out = parse_reply(msg)
    assert isinstance(out, dict)
    for c in out.get("tool_calls") or []:
        assert isinstance(json.loads(c["function"]["arguments"]), (dict, list, str, int, float, type(None)))
        assert c["function"]["name"]


def random_history(rng: random.Random) -> list[dict]:
    msgs = [{"role": "system", "content": junk(rng)}, {"role": "user", "content": junk(rng)}]
    for i in range(rng.randint(0, 6)):
        kind = rng.choice(["assistant", "assistant_call", "tool", "assistant_none"])
        if kind == "assistant":
            msgs.append({"role": "assistant", "content": junk(rng)})
        elif kind == "assistant_none":
            msgs.append({"role": "assistant", "content": None})
        elif kind == "assistant_call":
            args = rng.choice(['{"path": "a"}', "", "{bad", "[1]", "null", junk(rng), "5", '{"a": ' + junk(rng, 3) + "}"])
            msgs.append({"role": "assistant", "content": rng.choice([None, junk(rng)]),
                         "tool_calls": [{"id": f"id{i}", "type": "function", "function": {"name": rng.choice(["read_file", "x", ""]), "arguments": args}}]})
        else:
            msgs.append({"role": "tool", "tool_call_id": f"id{i}", "content": rng.choice([junk(rng), None, ""])})
    return msgs


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("seed", SEEDS)
def test_message_converters_survive_any_history(seed):
    msgs = random_history(random.Random(seed))
    before = copy.deepcopy(msgs)
    assert isinstance(to_text_messages(msgs, TOOLS), list)
    body = anthropic_api.to_request("m", msgs, TOOLS)
    assert body["messages"] and all(m["role"] in {"user", "assistant"} for m in body["messages"])
    assert msgs == before  # converters never mutate the live conversation


@pytest.mark.parametrize("seed", SEEDS)
def test_anthropic_response_parser_survives_junk(seed):
    rng = random.Random(seed)
    data = rng.choice([{}, {"content": None}, {"content": [{"type": "text"}]}, {"content": [{"type": "tool_use", "id": "1", "name": "x", "input": None}]},
                       {"content": [{"type": "text", "text": junk(rng)}], "usage": {"input_tokens": None}}])
    try:
        msg, usage = anthropic_api.from_response(data)
        assert msg["role"] == "assistant" and "total_tokens" in usage
    except (KeyError, TypeError):
        pass  # the router turns these into a skipped provider ("malformed response"); anything else would be a crash


@pytest.mark.parametrize("seed", SEEDS)
def test_redact_is_total_and_idempotent(seed):
    rng = random.Random(seed)
    text = junk(rng, rng.randint(0, 60))
    once = redact(text)
    assert redact(once) == once and isinstance(once, str)


@pytest.mark.parametrize("seed", SEEDS)
def test_html_to_text_and_select_text_are_total_and_bounded(seed):
    rng = random.Random(seed)
    html = junk(rng, rng.randint(0, 80))
    text = web.html_to_text(html)
    focus = junk(rng, rng.randint(0, 6))
    out = web.select_text(text * rng.randint(1, 40), focus, rng.choice([0, 0, 3, 10**6]))
    assert isinstance(out, str) and len(out) <= web.PAGE_CHARS + 400


@pytest.mark.parametrize("seed", SEEDS)
def test_command_check_is_total(seed, tmp_path):
    rng = random.Random(seed)
    assert isinstance(_names_protected_file(junk(rng, 20), tmp_path), bool)


@pytest.mark.parametrize("seed", SEEDS)
def test_check_url_only_raises_tool_errors(seed):
    rng = random.Random(seed)
    url = rng.choice(["http://", "https://", "", "ht!tp://x", "http://[", "http://a b/", "http://%00/", "http://" + junk(rng, 4), junk(rng, 5)])
    try:
        web.check_url(url)
    except web.ToolError:
        pass


@pytest.mark.parametrize("seed", SEEDS)
def test_parse_scores_is_total(seed):
    rng = random.Random(seed)
    text = rng.choice([junk(rng), "[" + junk(rng) + "]", json.dumps([{"n": rng.choice([1, "x", None, 99]), "feasibility": rng.choice([3, "4", None, 9.7])}]),
                       '[{"n": 1, "feasibility": 3, "usefulness": 3, "learning": 3}]'])
    out = parse_scores(text, 3)
    assert out is None or all(1 <= r["feasibility"] <= 5 for r in out)


@pytest.mark.parametrize("seed", range(60))
def test_compact_keeps_structure_and_shrinks(seed):
    rng = random.Random(seed)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for i in range(rng.randint(3, 25)):
        big = "x" * rng.choice([100, 5000, 40000])
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": str(i), "type": "function", "function": {
            "name": "write_file", "arguments": rng.choice([json.dumps({"path": "a", "content": big}), "{bad", "[1]", ""])}}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": big})
    before = [(m["role"], m.get("tool_call_id")) for m in msgs]
    total = sum(len(m.get("content") or "") for m in msgs)
    compact(msgs, CONTEXT_CHARS)
    assert [(m["role"], m.get("tool_call_id")) for m in msgs] == before  # same turns, same order
    assert msgs[0]["content"] == "SYS" and msgs[1]["content"] == "TASK"
    assert sum(len(m.get("content") or "") for m in msgs) <= max(total, 1)


# --- the MCP stdio server must survive hostile JSON-RPC ---

BAD_MESSAGES = [[], 5, "x", None, {"id": 1}, {"id": 1, "method": 5}, {"id": 1, "method": "tools/call"}, {"id": 1, "method": "tools/call", "params": []},
                {"id": 1, "method": "tools/call", "params": {"name": "read_file", "arguments": []}},
                {"id": 1, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "a\x00b"}}},
                {"id": 1, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": None}}},
                {"id": 1, "method": "tools/call", "params": {"name": "grep", "arguments": {"pattern": "(", "path": 5}}},
                {"id": 1, "method": "tools/call", "params": {"name": "count_words", "arguments": {"path": "blob.bin"}}},
                {"id": 1, "method": "tools/call", "params": {"name": ["x"]}}, {"id": [1], "method": "ping"}, {"id": 1, "method": "initialize", "params": 5}]


@pytest.mark.parametrize("msg", BAD_MESSAGES, ids=range(len(BAD_MESSAGES)))
def test_mcp_survives_hostile_messages(msg, tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00" * 100)
    out = io.StringIO()
    serve(tmp_path, io.StringIO(json.dumps(msg) + "\n" + json.dumps({"id": 99, "method": "ping"}) + "\n"), out)
    replies = [json.loads(l) for l in out.getvalue().splitlines()]
    assert replies[-1]["id"] == 99  # the server is still alive and answering after the bad message


def test_mcp_never_exposes_write_or_shell_tools():
    from agentic.mcp_server import exposed_tools
    names = set(exposed_tools(default_registry()))
    assert not names & {"write_file", "edit_file", "run_shell", "git_commit", "git_branch", "fetch_url", "web_search", "remember"}
