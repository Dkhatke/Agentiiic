"""Live checks: the real internet, the real OS sandbox and (if a key is set) a real model.
Skipped unless AGENTIC_LIVE=1, because they depend on the network and on third parties staying up:
    AGENTIC_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -v"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentic.config import load_config, load_env
from agentic.safety import sandbox
from agentic.tools import ToolContext, ToolError
from agentic.tools import web
from agentic.tools.builtin import run_shell

if os.environ.get("AGENTIC_LIVE") == "1":
    load_env()  # keys from the gitignored .env, like the CLI

pytestmark = pytest.mark.skipif(os.environ.get("AGENTIC_LIVE") != "1", reason="set AGENTIC_LIVE=1 to run live checks")
needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="macOS sandbox-exec only")


def ctx(tmp_path, **kw):
    return ToolContext(tmp_path.resolve(), **kw)


def test_fetch_real_page(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "STATE_DIR", tmp_path)
    c = ctx(tmp_path)
    out = web.fetch_url(c, "https://example.com/")
    assert "title: Example Domain" in out and "https://example.com/" in c.fetched


def test_fetch_real_doc_page_with_focus(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "STATE_DIR", tmp_path)
    out = web.fetch_url(ctx(tmp_path), "https://docs.python.org/3/library/json.html", focus="indent pretty print")
    assert "indent" in out and len(out) < web.PAGE_CHARS + 600


def test_real_web_search_returns_results(tmp_path):
    out = web.web_search(ctx(tmp_path), "python programming language")
    assert "https://" in out, f"DuckDuckGo scraping returned: {out[:300]}"


def test_real_ssrf_and_redirect_guards(tmp_path):
    for url in ["http://127.0.0.1:1/", "http://localhost/", "http://169.254.169.254/latest/meta-data/"]:
        with pytest.raises(ToolError):
            web.fetch_url(ctx(tmp_path), url)


def test_real_404_and_dns_failure_are_clean_errors(tmp_path):
    with pytest.raises(ToolError):
        web.fetch_url(ctx(tmp_path), "https://example.com/definitely-not-here-404")
    with pytest.raises(ToolError):
        web.fetch_url(ctx(tmp_path), "https://this-domain-does-not-exist-xyz123.invalid/")


@needs_sandbox
def test_sandbox_blocks_network_and_outside_writes(tmp_path):
    c = ctx(tmp_path)
    net = run_shell(c, "python3 -c \"import urllib.request as u; print(u.urlopen('https://example.com', timeout=5).status)\"")
    assert "200" not in net and "exit=0" not in net
    outside = Path.home() / "agentic_sandbox_probe.txt"
    out = run_shell(c, f"echo x > '{outside}'")
    assert not outside.exists() and "exit=0" not in out
    assert "exit=0" in run_shell(c, "echo ok > inside.txt") and (tmp_path / "inside.txt").exists()


@needs_sandbox
def test_sandbox_network_opt_in_works(tmp_path):
    out = run_shell(ctx(tmp_path, sandbox_network=True),
                    "python3 -c \"import urllib.request as u; print(u.urlopen('https://example.com', timeout=10).status)\"")
    assert "200" in out


@needs_sandbox
def test_sandbox_hides_credentials_and_secret_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "TOPSECRETVALUE123")
    out = run_shell(ctx(tmp_path), "env | grep -i secret; ls ~/.ssh ~/.aws 2>&1; echo done")
    assert "TOPSECRETVALUE123" not in out and "done" in out


def have_key():
    return any(p.usable and p.api_key_env for p in load_config().providers)


@pytest.mark.skipif(not have_key(), reason="no provider API key set; export GROQ_API_KEY or OPENROUTER_API_KEY")
def test_real_model_passes_the_basic_evals():
    r = subprocess.run([sys.executable, "-m", "agentic.cli", "eval", "--only", "hello-script", "read-and-report", "respects-secrets",
                        "stays-in-workspace", "offline-research", "web-injection"],
                       capture_output=True, text=True, timeout=900)
    print(r.stdout)
    assert r.returncode == 0, r.stdout[-1500:]
