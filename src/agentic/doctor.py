"""`agentic doctor`: check the setup and say what to fix. Network checks use each provider's /models listing,
which does not spend any of the free chat quota."""
from __future__ import annotations

import shutil
import sys

import httpx

from . import notes
from .config import CONFIG_PATH, STATE_DIR
from .safety import sandbox
from .safety.protected import find_protected


def check_provider(p, client: httpx.Client) -> tuple[bool, str]:
    if not p.usable:
        return False, f"no key: export {p.api_key_env}"
    headers = ({"x-api-key": p.api_key or "", "anthropic-version": "2023-06-01"} if p.api == "anthropic"
               else {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {})
    try:
        r = client.get(f"{p.base_url.rstrip('/')}/models", headers=headers, timeout=10)
    except httpx.HTTPError as e:
        return False, f"unreachable ({type(e).__name__})" + (" - is Ollama running? `ollama serve`" if "11434" in p.base_url else "")
    if r.status_code in {401, 403}:
        return False, f"key rejected (HTTP {r.status_code}): rotate or re-export it"
    if r.status_code == 404:
        return True, "reachable (no /models listing; chat endpoint not verified)"
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}"
    return True, "reachable, key accepted"


def run_doctor(cfg, workspace, client: httpx.Client | None = None, out=print) -> int:
    """Returns the number of problems found."""
    client = client or httpx.Client()
    problems = 0

    def line(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal problems
        problems += ok is False
        out(f"{'ok  ' if ok else 'FAIL' if ok is False else 'note'}  {label}" + (f": {detail}" if detail else ""))

    line(sys.version_info >= (3, 11), "python", sys.version.split()[0])
    line(CONFIG_PATH.exists() or None, "config file", str(CONFIG_PATH) if CONFIG_PATH.exists() else f"none; using built-in defaults (run `agentic init` to customise)")
    working = 0
    for p in cfg.providers:
        ok, detail = check_provider(p, client)
        working += ok
        out(f"{'ok  ' if ok else 'skip'}  provider {p.name} ({p.model}): {detail}")
    line(working > 0, "at least one working provider", "" if working else "export GROQ_API_KEY or OPENROUTER_API_KEY, or start Ollama")
    if sandbox.available():
        line(True, "shell sandbox", "macOS sandbox-exec")
    elif cfg.sandbox.docker_image and sandbox.docker_available():
        line(True, "shell sandbox", f"docker image {cfg.sandbox.docker_image}")
    else:
        line(None, "shell sandbox", "none available: shell commands run unconfined (every one still asks first). "
                                    "Install docker and set [sandbox] docker_image to confine them")
    line(shutil.which("git") is not None, "git")
    line(True if STATE_DIR.exists() else None, "state dir", str(STATE_DIR))
    line(True if notes.USER_PROFILE.exists() else None, "profile.md",
         str(notes.USER_PROFILE) if notes.USER_PROFILE.exists() else f"not set; create {notes.USER_PROFILE} to tell the agents your skills and goals")
    hidden = find_protected(workspace)
    line(None, "secret files in workspace", f"{len(hidden)} hidden from the agent" + (f" (e.g. {hidden[0].name})" if hidden else ""))
    out(f"\n{problems} problem(s) found" if problems else "\nall good")
    return problems
