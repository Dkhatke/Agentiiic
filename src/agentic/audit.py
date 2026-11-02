"""Append-only JSONL audit log of every tool call, approval and denial. Secrets are scrubbed from every field
(including the task text you typed), and the file rotates so it cannot grow without bound."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import STATE_DIR
from .safety.redact import redact

MAX_BYTES = 5_000_000
KEEP_ROTATED = 3


def scrub(obj):
    """Redact every string inside nested data (redacting the dumped JSON text could break its quoting)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


class Audit:
    def __init__(self, path: Path | None = None, max_bytes: int = MAX_BYTES):
        self.path = path or STATE_DIR / "audit.jsonl"
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate(self) -> None:
        for i in range(KEEP_ROTATED, 0, -1):  # audit.jsonl.2 -> .3, .1 -> .2, current -> .1
            src = self.path if i == 1 else self.path.with_name(f"{self.path.name}.{i - 1}")
            if src.exists():
                src.replace(self.path.with_name(f"{self.path.name}.{i}"))

    def record(self, event: str, **data) -> None:
        line = json.dumps({"ts": time.time(), "event": event, **scrub(data)}, default=str)
        if self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
            self._rotate()
        with self.path.open("a") as f:
            f.write(line + "\n")
