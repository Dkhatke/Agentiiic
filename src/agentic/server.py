"""Local daemon: HTTP API plus a static web UI, bound to 127.0.0.1, token-protected.

Endpoints (all /api/* need the token as `Authorization: Bearer <t>` or `?token=<t>`):
  POST /api/tasks                 {task, workspace, profile?, mode?} -> {id}
  GET  /api/tasks                 recent tasks
  GET  /api/tasks/<id>/events     ?after=N  long-poll for new events (tool calls, approvals, result)
  POST /api/approvals/<aid>       {approve: bool}
  GET  /api/usage                 today's provider usage
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .audit import Audit
from .config import STATE_DIR, load_config
from .embeddings import Embedder
from .loop import run
from .orchestrator import select_profile
from .permissions import Gate, Mode
from .profiles import load_profile
from .router import Router, RouterError
from .store import Store
from .tools import ToolContext, default_registry

UI_DIR = Path(__file__).parent / "ui"
APPROVAL_TIMEOUT = 600
MAX_BODY = 1_000_000


class TaskSession:
    def __init__(self):
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.pending: dict[str, list] = {}

    def emit(self, type_: str, **data) -> None:
        with self.cond:
            self.events.append({"i": len(self.events), "type": type_, **data})
            self.cond.notify_all()

    def wait_events(self, after: int, timeout: float) -> list[dict]:
        with self.cond:
            self.cond.wait_for(lambda: len(self.events) > after, timeout)
            return self.events[after:]

    def ask(self, summary: str) -> bool:
        aid = uuid.uuid4().hex[:12]
        slot = [threading.Event(), False]
        self.pending[aid] = slot
        self.emit("approval", id=aid, summary=summary)
        answered = slot[0].wait(APPROVAL_TIMEOUT)
        self.pending.pop(aid, None)
        decision = slot[1] if answered else False
        self.emit("approval_resolved", id=aid, approved=decision)
        return decision

    def decide(self, aid: str, approve: bool) -> bool:
        slot = self.pending.get(aid)
        if not slot:
            return False
        slot[1] = approve
        slot[0].set()
        return True


class App:
    def __init__(self, token: str | None = None, router_factory=None, store_path: Path | None = None):
        self.token = token or secrets.token_urlsafe(24)
        self.router_factory = router_factory or (lambda cfg, store: Router(cfg.providers, store=store))
        self.store_path = store_path
        self.sessions: dict[int, TaskSession] = {}
        self.approvals: dict[str, TaskSession] = {}

    def store(self) -> Store:  # sqlite connections are per-thread
        return Store(self.store_path)

    def start_task(self, task: str, workspace: str, profile_name: str | None, mode: str | None) -> int:
        ws = Path(workspace).expanduser().resolve()
        if not ws.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        profile = load_profile(profile_name) if profile_name and profile_name != "auto" else select_profile(task)
        store = self.store()
        tid = store.create_task(task, profile.name, str(ws))
        session = self.sessions[tid] = TaskSession()
        session.emit("started", id=tid, profile=profile.name)
        threading.Thread(target=self._run, args=(tid, task, profile, ws, mode, session), daemon=True).start()
        return tid

    def _run(self, tid, task, profile, ws, mode, session) -> None:
        try:
            cfg = load_config()
            store = self.store()
            ask = lambda summary: self._ask(session, summary)  # noqa: E731
            gate = Gate(Mode(mode or profile.mode or "ask-on-write"), ask)
            ctx = ToolContext(ws, cfg.limits.shell_timeout, cfg.sandbox.enabled, cfg.sandbox.network,
                              embedder=Embedder(cfg.embeddings) if cfg.embeddings.usable else None, store=store)
            profile.max_steps = min(profile.max_steps, cfg.limits.max_steps)
            out = run(task, profile, self.router_factory(cfg, store), default_registry(), gate, ctx,
                      Audit(), cfg.limits.max_model_calls, on_event=lambda e: session.emit("tool", text=e),
                      store=store, task_id=tid)
            session.emit("done", output=out)
        except (RouterError, ValueError) as e:
            session.emit("error", message=str(e))
        except Exception as e:  # keep the daemon alive whatever the task does
            session.emit("error", message=f"{type(e).__name__}: {e}")

    def _ask(self, session: TaskSession, summary: str) -> bool:
        return session.ask(summary)

    def find_approval(self, aid: str) -> TaskSession | None:
        return next((s for s in self.sessions.values() if aid in s.pending), None)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "agentic"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body, ctype: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self) -> bool:  # blocks DNS-rebinding
            return (self.headers.get("Host") or "").split(":")[0] in {"127.0.0.1", "localhost"}

        def _authed(self, query: dict) -> bool:
            auth = self.headers.get("Authorization", "")
            given = auth[7:] if auth.startswith("Bearer ") else (query.get("token") or [""])[0]
            return hmac.compare_digest(given.encode(), app.token.encode())

        def _json_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _route(self, method: str) -> None:
            if not self._host_ok():
                return self._send(403, {"error": "bad host"})
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if method == "GET" and u.path in {"/", "/index.html"}:
                return self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if not u.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if not self._authed(q):
                return self._send(401, {"error": "unauthorized"})
            parts = u.path.strip("/").split("/")[1:]
            try:
                if method == "POST" and parts == ["tasks"]:
                    b = self._json_body()
                    if not b.get("task") or not b.get("workspace"):
                        return self._send(400, {"error": "task and workspace required"})
                    return self._send(200, {"id": app.start_task(b["task"], b["workspace"], b.get("profile"), b.get("mode"))})
                if method == "GET" and parts == ["tasks"]:
                    return self._send(200, [dict(zip(("id", "status", "profile", "task"), r)) for r in app.store().list_tasks()])
                if method == "GET" and len(parts) == 3 and parts[0] == "tasks" and parts[2] == "events":
                    s = app.sessions.get(int(parts[1]))
                    if not s:
                        return self._send(404, {"error": "unknown task"})
                    return self._send(200, s.wait_events(int((q.get("after") or ["0"])[0]), 20))
                if method == "POST" and len(parts) == 2 and parts[0] == "approvals":
                    s = app.find_approval(parts[1])
                    ok = bool(s) and s.decide(parts[1], bool(self._json_body().get("approve")))
                    return self._send(200 if ok else 404, {"ok": ok})
                if method == "GET" and parts == ["usage"]:
                    return self._send(200, [dict(zip(("provider", "calls", "tokens"), r)) for r in app.store().usage_today()])
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return Handler


def serve(port: int = 8765, app: App | None = None) -> None:
    app = app or App()
    token_file = STATE_DIR / "token"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(app.token)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    print(f"agentic UI: http://127.0.0.1:{port}/#token={app.token}\n(token also in {token_file}; Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
