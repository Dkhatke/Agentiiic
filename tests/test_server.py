import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agentic import server
from agentic.router import Reply


class Scripted:
    calls = 0

    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, tools=None):
        self.calls += 1
        return Reply(next(self.replies), "fake", {})


def call(msg_name, args):
    return {"content": None, "tool_calls": [{"id": "1", "type": "function",
            "function": {"name": msg_name, "arguments": json.dumps(args)}}]}


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_STATE", str(tmp_path / "state"))
    monkeypatch.setattr("agentic.audit.STATE_DIR", tmp_path / "state")
    app = server.App(token="tok", store_path=tmp_path / "db.sqlite",
                     router_factory=lambda cfg, store: Scripted(
                         [call("write_file", {"path": "x.txt", "content": "hi"}), {"content": "all done"}]))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", tmp_path
    httpd.shutdown()


def req(url, method="GET", body=None, token="tok", host=None):
    r = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": f"Bearer {token}", **({"Host": host} if host else {})})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def test_auth_and_host_checks(live):
    base, _ = live
    assert req(base + "/api/usage", token="wrong")[0] == 401
    assert req(base + "/api/usage", host="evil.com")[0] == 403
    assert req(base + "/api/usage")[0] == 200
    with urllib.request.urlopen(base + "/") as r:        # the UI page itself needs no token
        assert b"<title>agentic</title>" in r.read()


@pytest.mark.parametrize("approve", [True, False])
def test_task_flow_with_approval(live, approve):
    base, tmp = live
    ws = tmp / f"ws{approve}"
    ws.mkdir()
    code, body = req(base + "/api/tasks", "POST", {"task": "write a file", "workspace": str(ws), "profile": "builder"})
    assert code == 200
    tid, after, approval, final = body["id"], 0, None, None
    while final is None:
        code, evs = req(base + f"/api/tasks/{tid}/events?after={after}")
        for e in evs:
            after = e["i"] + 1
            if e["type"] == "approval":
                approval = e
                assert req(base + f"/api/approvals/{e['id']}", "POST", {"approve": approve})[0] == 200
            if e["type"] in {"done", "error"}:
                final = e
    assert final["type"] == "done" and final["output"] == "all done"
    assert approval and "write_file" in approval["summary"]
    assert (ws / "x.txt").exists() is approve
    assert req(base + "/api/approvals/nope", "POST", {"approve": True})[0] == 404
    assert any(t["id"] == tid for t in req(base + "/api/tasks")[1])


def test_bad_workspace_rejected(live):
    base, _ = live
    assert req(base + "/api/tasks", "POST", {"task": "x", "workspace": "/definitely/not/here"})[0] == 400
