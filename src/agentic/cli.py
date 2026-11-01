from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .audit import Audit
from .config import CONFIG_PATH, SAMPLE_CONFIG, load_config
from .loop import run
from .permissions import Gate, Mode
from .profiles import load_profile
from .router import Router, RouterError
from .tools import ToolContext, default_registry


def _ask(summary: str) -> bool:
    return input(f"\nAllow? {summary}\n[y/N] ").strip().lower() in {"y", "yes"}


def cmd_run(args) -> int:
    cfg = load_config()
    profile = load_profile(args.profile)
    router = Router(cfg.providers)
    ctx = ToolContext(Path(args.workspace).resolve(), cfg.limits.shell_timeout,
                      cfg.sandbox.enabled, cfg.sandbox.network)
    profile.max_steps = min(profile.max_steps, cfg.limits.max_steps)
    try:
        out = run(args.task, profile, router, default_registry(), Gate(Mode(args.mode), _ask), ctx,
                  Audit(), cfg.limits.max_model_calls, on_event=lambda e: print(f"  · {e}"))
    except RouterError as e:
        print(f"error: {e}")
        return 1
    print(f"\n{out}")
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
    r.add_argument("--profile", default="builder")
    r.add_argument("--workspace", default=".")
    r.add_argument("--mode", default="ask-on-write", choices=[m.value for m in Mode])
    r.set_defaults(fn=cmd_run)
    sub.add_parser("providers", help="show configured providers").set_defaults(fn=cmd_providers)
    sub.add_parser("init", help="write a sample config").set_defaults(fn=cmd_init)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
