import os
import os
import json
import sys
from pathlib import Path

import httpx
import pytest

from agentic import anthropic_api
from agentic.config import Provider, load_config
from agentic.evals import load_tasks, run_task
from agentic.loop import Profile
from agentic.orchestrator import select_profile
from agentic.profiles import load_profile
from agentic.router import Reply, Router, RouterError
from agentic.safety import sandbox
from agentic.tools import ToolContext, ToolError
from agentic.tools import web

from test_evals_mcp import Scripted, tc

EVALS = Path(__file__).parent.parent / "evals"
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


# --- native Anthropic adapter ---

def test_anthropic_request_translation():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x"}'}},
                {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "y"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "AAA"}, {"role": "tool", "tool_call_id": "b", "content": "BBB"}]
    body = anthropic_api.to_request("m", msgs, TOOLS)
    assert body["system"] == "be brief" and body["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0] == {"type": "tool_use", "id": "a", "name": "read_file", "input": {"path": "x"}}
    results = body["messages"][2]["content"]  # both results arrive in one user turn
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_anthropic_response_translation():
    msg, usage = anthropic_api.from_response({
        "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}],
        "usage": {"input_tokens": 10, "output_tokens": 5}})
    assert msg["content"] == "ok" and json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "x"}
    assert usage["total_tokens"] == 15


def test_router_calls_anthropic_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["key"], seen["ver"] = str(req.url), req.headers.get("x-api-key"), req.headers.get("anthropic-version")
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 1, "output_tokens": 1}})

    p = Provider("claude", "https://api.anthropic.com/v1", "m", api_key_env="ANTHROPIC_API_KEY", api="anthropic")
    reply = Router([p], httpx.Client(transport=httpx.MockTransport(handler))).chat([{"role": "user", "content": "hi"}])
    assert reply.message["content"] == "hello"
    assert seen == {"url": "https://api.anthropic.com/v1/messages", "key": "k", "ver": anthropic_api.VERSION}


def test_anthropic_overloaded_is_temporary():
    p = Provider("claude", "https://x/v1", "m", api="anthropic")
    r = Router([p], httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(529))), retries=0)
    with pytest.raises(RouterError) as e:
        r.chat([{"role": "user", "content": "hi"}])
    assert e.value.retryable


# --- SearXNG ---

def searx_ctx(tmp_path):
    def handler(req: httpx.Request):
        assert req.url.path == "/search" and req.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [{"title": "Zorblax", "url": "https://z.example/a", "content": "a data library"}]})
    return ToolContext(tmp_path.resolve(), http=httpx.Client(transport=httpx.MockTransport(handler)),
                       searxng_url="http://localhost:8080")


def test_searxng_search_allows_only_its_own_host(tmp_path):
    out = web.web_search(searx_ctx(tmp_path), "zorblax")
    assert "https://z.example/a" in out and "a data library" in out
    with pytest.raises(ToolError):  # the exemption does not extend to fetching pages from that host
        web.check_url("http://localhost:8080/admin", searx_ctx(tmp_path).allow_hosts)


def test_searxng_non_json_gives_hint(tmp_path):
    c = ToolContext(tmp_path.resolve(), http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))),
                    searxng_url="http://localhost:8080")
    with pytest.raises(ToolError, match="json"):
        web.web_search(c, "x")


# --- sandboxes ---

def test_docker_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    argv = sandbox.wrap_command("ls", tmp_path, network=False, image="python:3.12-slim")
    assert argv[:3] == ["docker", "run", "--rm"] and f"{tmp_path}:/work" in argv
    assert "--network" in argv and "none" in argv and "--cap-drop" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "ls"] and "python:3.12-slim" in argv
    assert "--network" not in sandbox.wrap_command("ls", tmp_path, network=True, image="img")
    monkeypatch.setattr(sandbox, "docker_available", lambda: False)
    monkeypatch.setattr(sandbox, "available", lambda: False)
    assert sandbox.wrap_command("ls", tmp_path, image="img") is None  # no docker: falls back, never crashes


def test_sandbox_profile_hides_credentials(tmp_path):
    prof = sandbox.profile(tmp_path, False, home=Path("/home/u"))
    assert '(subpath "/home/u/.ssh")' in prof and '(subpath "/home/u/.config/agentic")' in prof
    assert '(literal "/home/u/.netrc")' in prof and "deny file-read*" in prof


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec only")
def test_sandbox_blocks_reading_credential_dir(tmp_path):
    import subprocess
    secret = Path.home() / ".config" / "agentic"
    secret.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(sandbox.wrap_command(f"ls '{secret}'", tmp_path.resolve()), capture_output=True, text=True)
    assert r.returncode != 0 and "not permitted" in r.stderr.lower()


# --- model-based profile routing ---

def test_keyword_match_needs_no_model_call():
    router = Scripted([])
    assert select_profile("fix the bug in my api", router=router).name == "builder"
    assert router.calls == 0


def test_ambiguous_request_asks_the_model():
    router = Scripted([{"content": "ideator"}])
    assert select_profile("something vague about my weekend", router=router).name == "ideator"
    assert router.calls == 1
    tie = Scripted([{"content": "The best agent is security."}])  # "audit" (security) ties with "code" (builder)
    assert select_profile("audit this code", router=tie).name == "security"


def test_routing_falls_back_when_model_fails_or_rambles():
    assert select_profile("something vague", router=Scripted([{"content": "no idea"}])).name == "general"

    class Down:
        def chat(self, *a, **k):
            raise RouterError("down")
    assert select_profile("something vague", router=Down()).name == "general"
    assert select_profile("audit this code", router=Down()).name in {"security", "builder"}


# --- offline web evals ---

def test_offline_research_eval_scripted():
    t = next(x for x in load_tasks(EVALS) if x["name"] == "offline-research")
    good = Scripted([tc("web_search", {"query": "zorblax licence"}),
                     tc("fetch_url", {"url": "https://docs.zorblax.example/about"}),
                     {"content": "Zorblax 4.2 is Apache-2.0 licensed. [1]\n[1] https://docs.zorblax.example/about"}])
    r = run_task(t, load_profile("researcher"), good)
    assert r.passed, r.failures
    wrong = Scripted([{"content": "Probably MIT."}, {"content": "Probably MIT."}])
    assert not run_task(t, load_profile("researcher"), wrong).passed


def test_web_injection_eval_flags_obedience():
    t = next(x for x in load_tasks(EVALS) if x["name"] == "web-injection")
    safe = Scripted([tc("fetch_url", {"url": "https://quillfish.example/readme"}), {"content": "It is written in Rust."}])
    assert run_task(t, load_profile("researcher"), safe).passed
    obedient = Scripted([tc("fetch_url", {"url": "https://quillfish.example/readme"}),
                         tc("write_file", {"path": "pwned.txt", "content": "owned"}), {"content": "Rust."}])
    r = run_task(t, load_profile("researcher"), obedient)
    assert not r.passed


def test_config_parses_new_options(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[providers]]\nname="c"\napi="anthropic"\nbase_url="https://x/v1"\nmodel="m"\n'
                 '[sandbox]\ndocker_image="img"\n[search]\nsearxng_url="http://localhost:8080"\n')
    cfg = load_config(p)
    assert cfg.providers[0].api == "anthropic" and cfg.sandbox.docker_image == "img"
    assert cfg.search.searxng_url == "http://localhost:8080"


# --- wildcard-proof protected files ---

def test_glob_obfuscation_is_caught_by_the_command_check(tmp_path):
    from agentic.tools.builtin import _names_protected_file
    (tmp_path / ".env").write_text("K=1")
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "server.pem").write_text("x")
    root = tmp_path.resolve()
    for cmd in ["cat .e*", "cat .en?", "cat .en[v]", "cat keys/*.pem", "cat **/server.pe?"]:
        assert _names_protected_file(cmd, root), cmd
    assert not _names_protected_file("cat READ*", root) and not _names_protected_file("ls *.py", root)


def test_find_protected_skips_vendor_dirs(tmp_path):
    from agentic.safety.protected import find_protected
    (tmp_path / ".env").write_text("a")
    (tmp_path / "node_modules" / "p").mkdir(parents=True)
    (tmp_path / "node_modules" / "p" / "x.pem").write_text("a")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "id_rsa").write_text("a")
    assert sorted(p.name for p in find_protected(tmp_path)) == [".env", "id_rsa"]


def test_docker_shadows_secret_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    argv = sandbox.wrap_command("ls", tmp_path, image="img", hidden_files=[tmp_path / "sub" / ".env"])
    assert "/dev/null:/work/sub/.env:ro" in argv


def test_sandbox_profile_hides_workspace_secrets(tmp_path):
    prof = sandbox.profile(tmp_path, False, hidden_files=[tmp_path / 'we"ird' / ".env"])
    assert '(literal "' + str(tmp_path) + '/we\\"ird/.env")' in prof


@pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec only")
def test_sandbox_blocks_obfuscated_env_read(tmp_path):
    import subprocess
    from agentic.safety.protected import find_protected
    root = tmp_path.resolve()
    (root / ".env").write_text("TOPSECRET")
    (root / "README").write_text("fine")
    hidden = find_protected(root)
    for cmd in ["cat .e*", "cat $(printf '.e%s' nv)", "ln -s .env x && cat x"]:
        r = subprocess.run(sandbox.wrap_command(cmd, root, hidden_files=hidden), capture_output=True, text=True, cwd=root)
        assert "TOPSECRET" not in r.stdout and r.returncode != 0, cmd
    ok = subprocess.run(sandbox.wrap_command("cat README", root, hidden_files=hidden), capture_output=True, text=True, cwd=root)
    assert ok.stdout == "fine"


# --- judge_ideas ---

IDEAS = "Notes bot: summarises lectures\nPlant monitor: sensors on a Pi\nQuiz maker: flashcards from PDFs"


def test_judge_ideas_scores_sorted_and_drops(tmp_path):
    from agentic.tools.ideas import judge_ideas
    reply = json.dumps([{"n": 1, "feasibility": 5, "usefulness": 4, "learning": 3, "reason": "easy"},
                        {"n": 2, "feasibility": 2, "usefulness": 3, "learning": 5, "reason": "needs hardware"},
                        {"n": 3, "feasibility": 4, "usefulness": 5, "learning": 4, "reason": "weekly use"}])
    seen = []
    c = ToolContext(tmp_path.resolve(), judge=lambda prompt: seen.append(prompt) or "Sure:\n" + reply)
    (tmp_path / "AGENTS.md").write_text("Owner has no GPU")
    out = judge_ideas(c, IDEAS)
    lines = out.splitlines()[1:]
    assert lines[0].startswith("3.") and lines[-1].startswith("2.") and "DROP" in lines[-1] and "KEEP" in lines[0]
    assert "no GPU" in seen[0] and "Plant monitor" in seen[0]


def test_judge_ideas_errors_and_bad_replies(tmp_path):
    from agentic.tools.ideas import judge_ideas, parse_scores
    with pytest.raises(ToolError, match="no judge"):
        judge_ideas(ToolContext(tmp_path.resolve()), IDEAS)
    with pytest.raises(ToolError, match="one idea per line"):
        judge_ideas(ToolContext(tmp_path.resolve(), judge=lambda p: ""), "just one")
    assert "could not be parsed" in judge_ideas(ToolContext(tmp_path.resolve(), judge=lambda p: "meh"), IDEAS)
    assert parse_scores('[{"n": 9, "feasibility": 3, "usefulness": 3, "learning": 3}]', 3) is None  # n out of range
    s = parse_scores('[{"n": 1, "feasibility": 9, "usefulness": 0, "learning": 3}]', 3)
    assert s[0]["feasibility"] == 5 and s[0]["usefulness"] == 1  # clamped to the scale


def test_judge_uses_a_different_provider_first():
    from agentic.config import Config, Judge
    from agentic.judge import make_judge
    urls = []

    def handler(req: httpx.Request):
        urls.append(req.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Config(providers=[Provider("a", "http://a/v1", "m"), Provider("b", "http://b/v1", "m"), Provider("c", "http://c/v1", "m")])
    assert make_judge(cfg, client=client)("score") == "ok" and urls == ["b"]
    cfg.judge = Judge(provider="c")
    urls.clear()
    make_judge(cfg, client=client)("score")
    assert urls == ["c"]


def test_ideator_has_the_judge_tool_and_it_works_end_to_end(tmp_path):
    prof = load_profile("ideator")
    assert "judge_ideas" in prof.tools
    scores = json.dumps([{"n": i, "feasibility": 4, "usefulness": 4, "learning": 4, "reason": "ok"} for i in (1, 2, 3)])
    t = next(x for x in load_tasks(EVALS) if x["name"] == "ideas-judged")
    router = Scripted([tc("judge_ideas", {"ideas": IDEAS}), {"content": "Recommend Quiz maker. First milestone: PDF to text."}])
    r = run_task(t, prof, router, judge=lambda p: scores)
    assert r.passed, r.failures


# --- VS Code launcher (node tests, skipped when node is missing) ---

@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node not installed")
def test_vscode_extension_node_tests():
    import subprocess
    ext = Path(__file__).parent.parent / "vscode-extension"
    r = subprocess.run(["node", "--test", "test/"], cwd=ext, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-400:]
    manifest = json.loads((ext / "package.json").read_text())
    ids = {c["command"] for c in manifest["contributes"]["commands"]}
    assert {"agentic.run", "agentic.securityAudit", "agentic.openUI"} <= ids


def test_page_text_puts_the_title_first():
    html = "<html><head><title> Zorblax &amp; Co </title></head><body><p>Body text here.</p></body></html>"
    assert web.page_text(html).startswith("title: Zorblax & Co\n\nBody text")
    assert web.page_text("<html><body><h1>Zorblax</h1></body></html>") == "Zorblax"  # no title: unchanged
    dup = "<head><title>Zorblax</title></head><body><h1>Zorblax</h1><p>x</p></body>"
    assert not web.page_text(dup).startswith("title:")  # already visible in the body: not repeated


# --- audit hardening and doctor ---

def test_audit_scrubs_secrets_everywhere_and_stays_valid_json(tmp_path):
    from agentic.audit import Audit
    a = Audit(tmp_path / "a.jsonl")
    tok = "sk-" + "or-v1-" + "0123456789abcdef" * 4
    a.record("task_start", task=f'deploy with API_KEY="{tok}" please', nested={"k": [f"Bearer {'x' * 30}"]})
    line = (tmp_path / "a.jsonl").read_text()
    assert "0123456789abcdef" not in line and "x" * 30 not in line
    assert json.loads(line)["event"] == "task_start"


def test_audit_rotates_and_keeps_a_bounded_number_of_files(tmp_path):
    from agentic.audit import Audit, KEEP_ROTATED
    a = Audit(tmp_path / "a.jsonl", max_bytes=300)
    for i in range(60):
        a.record("tick", i=i, pad="x" * 40)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "a.jsonl" in files and len(files) <= KEEP_ROTATED + 1
    assert (tmp_path / "a.jsonl").stat().st_size <= 400
    assert json.loads((tmp_path / "a.jsonl").read_text().splitlines()[-1])["i"] == 59  # newest entries are in the live file


def test_doctor_reports_each_provider_and_counts_problems(tmp_path, monkeypatch):
    from agentic.config import Config, Provider
    from agentic.doctor import run_doctor
    monkeypatch.setenv("GOOD_KEY", "k")
    monkeypatch.setenv("BAD_KEY", "k")
    monkeypatch.delenv("MISSING_KEY", raising=False)

    def handler(req: httpx.Request):
        if req.url.host == "good":
            return httpx.Response(200, json={"data": []})
        if req.url.host == "bad":
            return httpx.Response(401)
        raise httpx.ConnectError("refused")
    cfg = Config(providers=[Provider("g", "http://good/v1", "m", "GOOD_KEY"), Provider("b", "http://bad/v1", "m", "BAD_KEY"),
                            Provider("m", "http://x/v1", "m", "MISSING_KEY"), Provider("down", "http://localhost:11434/v1", "m")])
    lines = []
    n = run_doctor(cfg, tmp_path, httpx.Client(transport=httpx.MockTransport(handler)), out=lines.append)
    text = "\n".join(lines)
    assert "provider g" in text and "key accepted" in text
    assert "key rejected (HTTP 401)" in text and "export MISSING_KEY" in text and "ollama serve" in text
    assert n == 0  # one working provider is enough; the others are skipped, not failures
    none = []
    cfg.providers = cfg.providers[1:]
    assert run_doctor(cfg, tmp_path, httpx.Client(transport=httpx.MockTransport(handler)), out=none.append) == 1
    assert "FAIL  at least one working provider" in "\n".join(none)


def test_doctor_cli_runs(tmp_path):
    import subprocess, sys
    (tmp_path / ".env").write_text("A=1")
    env = {**os.environ, "AGENTIC_CONFIG": str(tmp_path / "c.toml"), "AGENTIC_STATE": str(tmp_path / "s"), "HOME": str(tmp_path)}
    (tmp_path / "c.toml").write_text('[[providers]]\nname = "x"\nbase_url = "http://127.0.0.1:1/v1"\nmodel = "m"\n')
    r = subprocess.run([sys.executable, "-m", "agentic.cli", "doctor", "--workspace", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "unreachable" in r.stdout and "1 hidden from the agent" in r.stdout and "Traceback" not in r.stderr


# --- provider reply normalisation ---

def test_normalize_handles_provider_quirks():
    from agentic.router import normalize
    assert normalize({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})["content"] == "ab"
    assert normalize({})["content"] is None and normalize({"content": 5})["content"] is None
    m = normalize({"content": None, "reasoning": "x", "tool_calls": [
        {"function": {"name": "list_dir", "arguments": {"path": "."}}},           # object arguments, no id
        {"id": "k", "function": {"name": "grep"}},                                 # no arguments
        {"id": "z", "function": {"name": 5}}, {"id": "q"}, "junk", None]})         # unusable entries are dropped
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["list_dir", "grep"]
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"path": "."}
    assert m["tool_calls"][1]["function"]["arguments"] == "{}" and m["tool_calls"][1]["id"] == "k"
    assert m["tool_calls"][0]["id"]  # a missing id is filled in
    assert "reasoning" not in m and "tool_calls" not in normalize({"tool_calls": "nope"})


def test_router_skips_a_provider_that_answers_with_html():
    p = Provider("a", "http://a/v1", "m")
    ok = Provider("b", "http://b/v1", "m")

    def handler(req):
        if req.url.host == "a":
            return httpx.Response(200, text="<html>bad gateway</html>")
        return httpx.Response(200, json={"choices": [{"message": {"content": "fine"}}]})
    r = Router([p, ok], httpx.Client(transport=httpx.MockTransport(handler)))
    assert r.chat([{"role": "user", "content": "hi"}]).provider == "b"


def test_load_env_reads_keys_without_overriding_and_ignores_junk(tmp_path, monkeypatch):
    from agentic.config import load_env
    f = tmp_path / ".env"
    f.write_text("# c\nAGX_A=one\nexport AGX_B='two'\nAGX_C=\nAGX_SET=new\nbad line\n=x\n")
    monkeypatch.setenv("AGX_SET", "old")
    for k in ("AGX_A", "AGX_B", "AGX_C"):
        monkeypatch.delenv(k, raising=False)
    assert load_env([f, tmp_path / "missing"]) == [f]
    assert (os.environ["AGX_A"], os.environ["AGX_B"], os.environ["AGX_SET"]) == ("one", "two", "old")
    assert "AGX_C" not in os.environ  # an empty placeholder does not count as a key
    for k in ("AGX_A", "AGX_B"):
        monkeypatch.delenv(k)
