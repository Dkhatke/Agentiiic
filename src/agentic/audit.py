"""Append-only JSONL audit log of every tool call, approval and denial."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import STATE_DIR


class Audit:
    def __init__(self, path: Path | None = None):
        self.path = path or STATE_DIR / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **data) -> None:
        line = json.dumps({"ts": time.time(), "event": event, **data}, default=str)
        with self.path.open("a") as f:
            f.write(line + "\n")
