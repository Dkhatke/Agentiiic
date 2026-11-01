from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .audit import Audit
from .config import CONFIG_PATH, SAMPLE_CONFIG, load_config
from .loop import run
from .permissions import Gate, Mode
from .orchestrator import select_profile
from .profiles import all_profiles, load_profile
from .embeddings import Embedder
from .router import Router, RouterError
from .store import Store
from .tools import ToolContext, default_registry


def _ask(summary: str) -> bool:
    return input(f"\nAllow? {summary}\n[y/N] ").strip().lower() in {"y", "yes"}


def cmd_run(args) -> int:
    cfg = load_config()
    profile = select_profile(args.task) if args.profile == "auto" else load_profile(args.profile)
    print(f"profile: {profile.name}")
    mode = Mode(args.mode or profile.mode or "ask-on-write")
    store = Store()
    router = Router(cfg.providers, store=store)
    task_id, messages = args.task_id, args.messages
    if task_id is None:
        task_id = store.create_task(args.task, profile.name, str(Path(args.workspace).resolve()))
    print(f"task #{task_id}  (resume later with: agentic resume {task_id})")
    ctx = ToolContext(Path(args.workspace).resolve(), cfg.limits.shell_timeout,
                      cfg.sandbox.enabled, cfg.sandbox.network,
                      embedder=Embedder(cfg.embeddings) if cfg.embeddings.usable else None, store=store)
    profile.max_steps = min(profile.max_steps, cfg.limits.max_steps)
    try:
        out = run(args.task, profile, router, default_registry(), Gate(mode, _ask), ctx,
                  Audit(), cfg.limits.max_model_calls, on_event=lambda e: print(f"  · {e}"),
                  store=store, task_id=task_id, messages=messages)
    except RouterError as e:
        print(f"error: {e}")
        return 1
    print(f"\n{out}")
    return 0


def cmd_resume(args) -> int:
    t = Store().load_task(args.id)
    if not t:
        print(f"no task #{args.id}")
        return 1
    args.task, args.profile, args.workspace = t["task"], t["profile"], t["workspace"]
    args.task_id, args.messages = t["id"], t["messages"]
    return cmd_run(args)


def cmd_tasks(_) -> int:
    for tid, status, profile, task in Store().list_tasks():
        print(f"#{tid:<4} {status:8} {profile:10} {task[:70]}")
    return 0


def cmd_usage(_) -> int:
    rows = Store().usage_today()
    for name, calls, tokens in rows:
        print(f"{name:12} {calls:5} calls  {tokens:8} tokens")
    if not rows:
        print("no usage today")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(args.port)
    return 0


def cmd_eval(args) -> int:
    from .evals import load_tasks, run_task
    cfg = load_config()
    router = Router(cfg.providers, store=Store())
    tasks = load_tasks(Path(args.dir))
    if args.only:
        tasks = [t for t in tasks if t["name"] in args.only]
    passed = 0
    for t in tasks:
        try:
            r = run_task(t, load_profile(t.get("profile", args.profile)), router)
        except RouterError as e:
            print(f"ERR   {t['name']}: {e}")
            continue
        passed += r.passed
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name:22} {r.model_calls:3} model calls  " + "; ".join(r.failures))
    print(f"\n{passed}/{len(tasks)} passed")
    return 0 if passed == len(tasks) else 1


def cmd_mcp(args) -> int:
    from .mcp_server import serve
    serve(Path(args.workspace))
    return 0


def cmd_models(_) -> int:
    import httpx
    try:
        data = httpx.get("https://openrouter.ai/api/v1/models", timeout=20).json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"could not fetch model list: {e}")
        return 1
    free = [m for m in data if m["pricing"].get("prompt") == "0" and m["pricing"].get("completion") == "0"
            and (m.get("architecture") or {}).get("output_modalities") == ["text"]
            and "tools" in (m.get("supported_parameters") or [])]
    for m in sorted(free, key=lambda m: -m.get("context_length", 0)):
        print(f"{m['id']:52} {m.get('context_length', 0):>8} ctx")
    print(f"\n{len(free)} free tool-calling chat models (put an id in `model =` in {CONFIG_PATH})")
    return 0


def cmd_providers(_) -> int:
    for p in load_config().providers:
        print(f"{'ok ' if p.usable else 'no '} {p.name:12} {p.model}  ({p.base_url})"
              + ("" if p.usable else f"  [set {p.api_key_env}]"))
    return 0


def cmd_init(_) -> int:
    if CONFIG_PATH.exists():
        print(f"already exists: {CONFIG_PATH}")
        return 1
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(SAMPLE_CONFIG)
    print(f"wrote {CONFIG_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentic", description="Permission-gated personal build agent")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a task")
    r.add_argument("task")
    r.add_argument("--profile", default="auto", help="auto, or one of: " + ", ".join(p.name for p in all_profiles()))
    r.add_argument("--workspace", default=".")
    r.add_argument("--mode", default=None, choices=[m.value for m in Mode])
    r.set_defaults(fn=cmd_run, task_id=None, messages=None)
    rs = sub.add_parser("resume", help="continue a saved task")
    rs.add_argument("id", type=int)
    rs.add_argument("--mode", default=None, choices=[m.value for m in Mode])
    rs.set_defaults(fn=cmd_resume)
    sv = sub.add_parser("serve", help="start the local web UI and API")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(fn=cmd_serve)
    ev = sub.add_parser("eval", help="run the eval tasks against the configured providers")
    ev.add_argument("--dir", default="evals")
    ev.add_argument("--profile", default="builder")
    ev.add_argument("--only", nargs="*")
    ev.set_defaults(fn=cmd_eval)
    mc = sub.add_parser("mcp", help="MCP server (stdio, read-only tools) for editors")
    mc.add_argument("--workspace", default=".")
    mc.set_defaults(fn=cmd_mcp)
    sub.add_parser("tasks", help="list recent tasks").set_defaults(fn=cmd_tasks)
    sub.add_parser("usage", help="today's provider usage").set_defaults(fn=cmd_usage)
    sub.add_parser("models", help="list free tool-calling models on OpenRouter").set_defaults(fn=cmd_models)
    sub.add_parser("providers", help="show configured providers").set_defaults(fn=cmd_providers)
    sub.add_parser("init", help="write a sample config").set_defaults(fn=cmd_init)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
