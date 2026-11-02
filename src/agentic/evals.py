"""Eval harness: run repeatable tasks in throwaway workspaces and check the results.

Task files are TOML in evals/: prompt, optional [files] to seed, optional extra_tools to grant the profile, optional [web] (url = page text) served as an offline
fake internet to web_search/fetch_url, and [[checks]] of type
file_exists | file_contains | file_lacks | command_contains | answer_contains | answer_lacks. Everything runs auto-approved inside a temp
directory (sandboxed on macOS), so only run it against models you are happy to let act freely there.
"""
from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import tomllib
from urllib.parse import urlparse
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .audit import Audit
from .loop import Profile, run
from .permissions import Gate, Mode
from .tools import ToolContext, default_registry


@dataclass
class Result:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    model_calls: int = 0
    output: str = ""


def load_tasks(directory: Path) -> list[dict]:
    return [tomllib.loads(p.read_text()) | {"name": p.stem} for p in sorted(directory.glob("*.toml"))]


def check(c: dict, ws: Path, output: str = "") -> str | None:
    kind = c["type"]
    target = ws / c.get("path", "")
    if kind == "file_exists":
        return None if target.is_file() else f"missing file {c['path']}"
    if kind == "file_contains":
        if not target.is_file():
            return f"missing file {c['path']}"
        return None if c["text"] in target.read_text() else f"{c['path']} lacks {c['text']!r}"
    if kind == "answer_contains":
        return None if c["text"].lower() in output.lower() else f"final answer lacks {c['text']!r}"
    if kind == "answer_lacks":
        return None if c["text"].lower() not in output.lower() else f"final answer leaks {c['text']!r}"
    if kind == "file_lacks":  # passes if the file is absent or does not contain the text
        return None if not target.is_file() or c["text"] not in target.read_text() else f"{c['path']} leaks {c['text']!r}"
    if kind == "command_contains":
        try:
            r = subprocess.run(c["run"], shell=True, cwd=ws, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return f"`{c['run']}` timed out"
        return None if c["text"] in r.stdout + r.stderr else f"`{c['run']}` output lacks {c['text']!r}"
    return f"unknown check type {kind}"


SEARCH_HOST = "html.duckduckgo.com"


def fake_web(pages: dict[str, str]) -> tuple[httpx.Client, set[str]]:
    """An offline internet: web_search lists the fixture URLs, fetch_url returns their text."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == SEARCH_HOST:
            links = "".join(f'<a class="result__a" href="{u}">{u}</a>' for u in pages)
            return httpx.Response(200, text=links, headers={"content-type": "text/html"})
        body = pages.get(str(req.url.copy_with(query=None)))
        return httpx.Response(200, text=body, headers={"content-type": "text/plain"}) if body is not None \
            else httpx.Response(404)
    hosts = {urlparse(u).hostname for u in pages} | {SEARCH_HOST}
    return httpx.Client(transport=httpx.MockTransport(handler)), hosts


def run_task(task: dict, profile: Profile, router, audit_path: Path | None = None) -> Result:
    if task.get("extra_tools"):  # e.g. give the researcher write_file to see whether a planted instruction makes it write
        profile = dataclasses.replace(profile, tools=[*profile.tools, *task["extra_tools"]])
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d).resolve()
        for name, content in (task.get("files") or {}).items():
            (ws / name).parent.mkdir(parents=True, exist_ok=True)
            (ws / name).write_text(content)
        web_ctx = {}
        if task.get("web"):
            client, hosts = fake_web(task["web"])
            web_ctx = {"http": client, "allow_hosts": hosts}
        start = getattr(router, "calls", 0)
        out = run(task["prompt"], profile, router, default_registry(), Gate(Mode.ASK_ON_WRITE, lambda s: True),
                  ToolContext(ws, **web_ctx), Audit(audit_path or ws / ".audit.jsonl"), max_model_calls=task.get("max_calls", 30))
        failures = [f for c in task.get("checks", []) if (f := check(c, ws, out))]
        return Result(task["name"], not failures, failures, getattr(router, "calls", 0) - start, out)
