"""End-to-end: the real CLI and the real daemon, as subprocesses, talking over HTTP to a local fake model server.

Nothing inside agentic is mocked here, so this exercises argument parsing, config loading, the router's real HTTP client,
the permission prompt on stdin, SQLite state, the audit log and the web API the way a user hits them."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest


class FakeModel:
    """OpenAI-compatible /v1/chat/completions that plays back scripted replies (dict = message, int = HTTP status)."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                item = outer.replies.pop(0) if outer.replies else {"role": "assistant", "content": "(script ended)"}
                if isinstance(item, int):
                    self.send_response(item)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = json.dumps({"choices": [{"message": item}], "usage": {"total_tokens": 7}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def tc(name, args, cid="c1"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def say(text):
    return {"role": "assistant", "content": text}


@pytest.fixture
def box(tmp_path):
    """An isolated user: own config, state and workspace, pointing at a fake model."""
    made = []

    class Box:
        ws = tmp_path / "ws"
        state = tmp_path / "state"
        config = tmp_path / "config.toml"

        def model(self, replies):
            m = FakeModel(replies)
            made.append(m)
            self.config.write_text(f'[[providers]]\nname = "fake"\nbase_url = "http://127.0.0.1:{m.port}/v1"\nmodel = "m"\n'
                                   '[embeddings]\nenabled = false\n[limits]\nmax_model_calls = 20\n')
            return m

        def env(self):
            return {**os.environ, "AGENTIC_CONFIG": str(self.config), "AGENTIC_STATE": str(self.state), "HOME": str(tmp_path)}

        def cli(self, *args, stdin="", timeout=60):
            return subprocess.run([sys.executable, "-m", "agentic.cli", *args], input=stdin, capture_output=True, text=True,
                                  env=self.env(), cwd=self.ws, timeout=timeout)

    b = Box()
    b.ws.mkdir()
    yield b
    for m in made:
        m.close()


def test_run_creates_file_after_approval_and_records_everything(box):
    m = box.model([tc("write_file", {"path": "hello.py", "content": "print('hi')\n"}), say("Created hello.py."), say("Verified it by reading it back.")])
    r = box.cli("run", "create hello.py", "--profile", "builder", "--workspace", str(box.ws), stdin="y\n")
    assert r.returncode == 0, r.stderr
    assert (box.ws / "hello.py").read_text() == "print('hi')\n"
    assert "Verified it" in r.stdout and "Allow?" in r.stdout and "task #1" in r.stdout
    audit = [json.loads(l) for l in (box.state / "audit.jsonl").read_text().splitlines()]
    assert [a["event"] for a in audit if a["event"] == "tool_call"] and any(a.get("decision") == "approved" for a in audit)
    assert len(m.requests) == 3 and m.requests[0]["tools"]  # 3rd call = the verify nudge  # real tool schemas went over the wire
    assert "system" == m.requests[0]["messages"][0]["role"]
    tasks = box.cli("tasks").stdout
    assert "#1" in tasks and "builder" in tasks
    usage = box.cli("usage").stdout
    assert "fake" in usage and "3 calls" in usage


def test_run_denied_approval_writes_nothing(box):
    box.model([tc("write_file", {"path": "evil.txt", "content": "x"}), say("OK, I could not write it.")])
    r = box.cli("run", "write evil.txt", "--profile", "builder", "--workspace", str(box.ws), stdin="n\n")
    assert r.returncode == 0 and not (box.ws / "evil.txt").exists()
    assert "denied by user" in (box.state / "audit.jsonl").read_text()


def test_readonly_mode_never_prompts_and_blocks_writes(box):
    box.model([tc("write_file", {"path": "a.txt", "content": "x"}), say("done")])
    r = box.cli("run", "write a.txt", "--profile", "builder", "--mode", "readonly", "--workspace", str(box.ws))
    assert r.returncode == 0 and "Allow?" not in r.stdout and not (box.ws / "a.txt").exists()


def test_resume_continues_after_a_provider_failure(box):
    box.model([401] * 5)  # the provider rejects us: the run errors out, but the task is saved
    first = box.cli("run", "say hello", "--profile", "general", "--workspace", str(box.ws))
    assert first.returncode == 1 and "error:" in first.stdout and "Traceback" not in first.stderr
    box.model([say("Hello again!")])
    second = box.cli("resume", "1")
    assert second.returncode == 0 and "Hello again!" in second.stdout, second.stdout + second.stderr


def test_no_usable_provider_gives_a_clear_error(box):
    box.config.write_text('[[providers]]\nname = "x"\nbase_url = "http://127.0.0.1:1/v1"\nmodel = "m"\napi_key_env = "NOPE_KEY"\n')
    r = box.cli("run", "hi", "--profile", "general", "--workspace", str(box.ws))
    assert r.returncode == 1 and "no usable providers" in r.stdout and "Traceback" not in r.stderr
    assert "NOPE_KEY" in box.cli("providers").stdout


def test_init_providers_and_profiles(box):
    r = box.cli("init")
    assert r.returncode == 0 and box.config.exists()
    assert box.cli("init").returncode == 1  # never overwrites
    out = box.cli("providers").stdout
    assert "groq-gptoss" in out and "ollama" in out
    assert box.cli("run", "x", "--profile", "nonsense", "--workspace", str(box.ws)).returncode != 0


def test_eval_command_reports_pass_and_fail(box, tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "make-file.toml").write_text('prompt = "create a.txt"\n[[checks]]\ntype = "file_exists"\npath = "a.txt"\n')
    box.model([tc("write_file", {"path": "a.txt", "content": "x"}), say("done"), say("done")])
    ok = box.cli("eval", "--dir", str(evals))
    assert ok.returncode == 0 and "PASS" in ok.stdout and "1/1 passed" in ok.stdout, ok.stdout + ok.stderr
    box.model([say("I did nothing")])
    bad = box.cli("eval", "--dir", str(evals))
    assert bad.returncode == 1 and "FAIL" in bad.stdout


def test_auto_profile_uses_the_model_only_when_ambiguous(box):
    m = box.model([say("ideator"), say("Some ideas.")])
    r = box.cli("run", "something vague about my weekend", "--workspace", str(box.ws))
    assert "profile: ideator" in r.stdout and len(m.requests) == 2
    m2 = box.model([say("Fixed.")])
    r2 = box.cli("run", "fix the bug in my api", "--workspace", str(box.ws))
    assert "profile: builder" in r2.stdout and len(m2.requests) == 1


def test_model_that_sends_garbage_tool_arguments_does_not_crash_the_cli(box):
    bad = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "run_shell", "arguments": "[1, 2"}},
        {"id": "2", "type": "function", "function": {"name": "nonexistent", "arguments": "{}"}},
        {"id": "3", "type": "function", "function": {"name": "read_file", "arguments": "5"}}]}
    box.model([bad, say("sorry, recovered")])
    r = box.cli("run", "x", "--profile", "builder", "--workspace", str(box.ws))
    assert r.returncode == 0 and "recovered" in r.stdout and "Traceback" not in r.stderr


# --- the daemon ---

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_daemon_end_to_end(box):
    box.model([tc("write_file", {"path": "web.txt", "content": "from the daemon"}), say("Wrote web.txt."), say("Confirmed.")])
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "agentic.cli", "serve", "--port", str(port)], env=box.env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=box.ws)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("daemon did not start: " + proc.stderr.read().decode())
        token = (box.state / "token").read_text()
        assert oct((box.state / "token").stat().st_mode & 0o777) == "0o600"
        h = {"Authorization": f"Bearer {token}"}
        assert httpx.get(base + "/api/tasks").status_code == 401
        assert httpx.get(base + "/api/tasks", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert httpx.get(base + "/api/tasks", headers={**h, "Host": "evil.example.com"}).status_code == 403  # DNS rebinding
        assert httpx.post(base + "/api/tasks", headers=h, content=b"{not json").status_code == 400
        assert httpx.post(base + "/api/tasks", headers=h, json={"task": "x", "workspace": "/nonexistent"}).status_code == 400
        assert httpx.get(base + "/api/tasks/abc/events", headers=h).status_code == 400
        r = httpx.post(base + "/api/tasks", headers=h, json={"task": "write web.txt", "workspace": str(box.ws), "profile": "builder"})
        tid = r.json()["id"]
        after, approval, done = 0, None, None
        deadline = time.time() + 30
        while time.time() < deadline and not done:
            for ev in httpx.get(f"{base}/api/tasks/{tid}/events?after={after}", headers=h, timeout=30).json():
                after = ev["i"] + 1
                if ev["type"] == "approval":
                    approval = ev
                    assert not (box.ws / "web.txt").exists()  # nothing happens before the human says yes
                    assert httpx.post(f"{base}/api/approvals/{ev['id']}", headers=h, json={"approve": True}).json()["ok"]
                if ev["type"] in {"done", "error"}:
                    done = ev
        assert approval and done and done["type"] == "done", done
        assert (box.ws / "web.txt").read_text() == "from the daemon"
        assert any(t["id"] == tid for t in httpx.get(base + "/api/tasks", headers=h).json())
        assert httpx.get(base + "/api/usage", headers=h).json()[0]["calls"] == 3
        assert b"<title>" in httpx.get(base + "/").content  # the UI page itself is served
    finally:
        proc.terminate()
        proc.wait(10)


# --- a misbehaving model, over real HTTP ---

def test_endless_tool_calls_stop_at_the_step_limit(box):
    m = box.model([tc("list_dir", {"path": f"d{i}"}, cid=f"c{i}") for i in range(60)])
    r = box.cli("run", "loop forever", "--profile", "general", "--workspace", str(box.ws))
    assert r.returncode == 0 and "Traceback" not in r.stderr
    assert len(m.requests) <= 31, len(m.requests)  # general has a hard step cap; never an unbounded loop


def test_identical_repeated_calls_are_cut_off(box):
    m = box.model([tc("list_dir", {"path": "."}, cid=f"c{i}") for i in range(8)] + [say("giving up")])
    r = box.cli("run", "x", "--profile", "general", "--workspace", str(box.ws))
    assert r.returncode == 0
    tool_msgs = [x["content"] for req in m.requests[-1:] for x in req["messages"] if x["role"] == "tool"]
    assert any("exact" in t and "times" in t for t in tool_msgs)  # the model was told it is repeating itself


def test_model_call_budget_is_enforced(box):
    m = box.model([tc("list_dir", {"path": f"d{i}"}, cid=f"c{i}") for i in range(30)])
    box.config.write_text(box.config.read_text().replace("max_model_calls = 20", "max_model_calls = 3"))
    r = box.cli("run", "x", "--profile", "general", "--workspace", str(box.ws))
    assert r.returncode == 0 and "Traceback" not in r.stderr and "budget" in r.stdout
    assert len(m.requests) == 3


@pytest.mark.parametrize("reply", [
    {"role": "assistant", "content": ""},
    {"content": "no role field"},
    {"role": "assistant", "content": [{"type": "text", "text": "list of parts"}]},
    {"role": "assistant", "content": None, "tool_calls": []},
    {"role": "assistant", "content": "x", "tool_calls": None},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "list_dir", "arguments": {"path": "."}}}]},
    {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "list_dir"}}]},
    {"role": "assistant", "content": "ok", "reasoning": "hidden", "refusal": None, "annotations": []},
], ids=["empty", "no-role", "content-list", "empty-calls", "null-calls", "args-as-object", "no-call-id", "no-arguments", "extra-fields"])
def test_odd_but_legal_provider_replies_are_handled(box, reply):
    box.model([reply, say("finished")])
    r = box.cli("run", "x", "--profile", "general", "--workspace", str(box.ws))
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr[-800:]


def test_http_failures_are_reported_cleanly(box):
    for status in (400, 401, 403, 404, 413, 422):
        box.model([status])
        r = box.cli("run", "x", "--profile", "general", "--workspace", str(box.ws))
        assert r.returncode == 1 and "error:" in r.stdout and "Traceback" not in r.stderr, status


def test_provider_returning_html_or_garbage_json(box):
    class Garbage(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b"<html>502 Bad Gateway from some proxy</html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Garbage)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        box.config.write_text(f'[[providers]]\nname = "g"\nbase_url = "http://127.0.0.1:{httpd.server_address[1]}/v1"\nmodel = "m"\n[embeddings]\nenabled = false\n')
        r = box.cli("run", "x", "--profile", "general", "--workspace", str(box.ws))
        assert r.returncode == 1 and "Traceback" not in r.stderr, r.stderr[-800:]
    finally:
        httpd.shutdown()


def test_daemon_handles_concurrent_tasks_without_mixing_them_up(box):
    box.model([say(f"answer {i}") for i in range(8)])
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "agentic.cli", "serve", "--port", str(port)], env=box.env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=box.ws)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        h = {"Authorization": "Bearer " + (box.state / "token").read_text()}
        ids = []

        def submit(i):
            ids.append(httpx.post(base + "/api/tasks", headers=h, json={"task": f"say {i}", "workspace": str(box.ws), "profile": "general"}).json()["id"])
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(set(ids)) == 8  # distinct task ids, no lost inserts
        results = {}
        deadline = time.time() + 30
        while len(results) < 8 and time.time() < deadline:
            for tid in ids:
                if tid in results:
                    continue
                for ev in httpx.get(f"{base}/api/tasks/{tid}/events?after=0", headers=h, timeout=30).json():
                    if ev["type"] in {"done", "error"}:
                        results[tid] = ev
        assert len(results) == 8 and all(e["type"] == "done" for e in results.values()), results
        assert len(httpx.get(base + "/api/tasks", headers=h).json()) == 8
        assert httpx.get(base + "/api/usage", headers=h).json()[0]["calls"] == 8
    finally:
        proc.terminate()
        proc.wait(10)
