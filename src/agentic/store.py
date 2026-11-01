"""SQLite store: per-provider usage (daily budgets) and resumable task history."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .config import STATE_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (day TEXT, provider TEXT, calls INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0,
                                  PRIMARY KEY (day, provider));
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, profile TEXT, workspace TEXT,
                                  status TEXT, messages TEXT, created REAL, updated REAL);
"""


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class Store:
    def __init__(self, path: Path | None = None):
        path = path or STATE_DIR / "agentic.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    # usage
    def add_usage(self, provider: str, calls: int = 0, tokens: int = 0) -> None:
        self.db.execute(
            "INSERT INTO usage (day, provider, calls, tokens) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(day, provider) DO UPDATE SET calls = calls + ?, tokens = tokens + ?",
            (_today(), provider, calls, tokens, calls, tokens))
        self.db.commit()

    def calls_today(self, provider: str) -> int:
        row = self.db.execute("SELECT calls FROM usage WHERE day = ? AND provider = ?", (_today(), provider)).fetchone()
        return row[0] if row else 0

    def usage_today(self) -> list[tuple[str, int, int]]:
        return self.db.execute("SELECT provider, calls, tokens FROM usage WHERE day = ? ORDER BY provider",
                               (_today(),)).fetchall()

    # tasks
    def create_task(self, task: str, profile: str, workspace: str) -> int:
        now = time.time()
        cur = self.db.execute("INSERT INTO tasks (task, profile, workspace, status, messages, created, updated) "
                              "VALUES (?, ?, ?, 'running', '[]', ?, ?)", (task, profile, workspace, now, now))
        self.db.commit()
        return cur.lastrowid

    def save_task(self, task_id: int, messages: list[dict], status: str) -> None:
        self.db.execute("UPDATE tasks SET messages = ?, status = ?, updated = ? WHERE id = ?",
                        (json.dumps(messages), status, time.time(), task_id))
        self.db.commit()

    def load_task(self, task_id: int) -> dict | None:
        row = self.db.execute("SELECT id, task, profile, workspace, status, messages FROM tasks WHERE id = ?",
                              (task_id,)).fetchone()
        if not row:
            return None
        return dict(zip(("id", "task", "profile", "workspace", "status"), row[:5]), messages=json.loads(row[5]))

    def list_tasks(self, limit: int = 15) -> list[tuple]:
        return self.db.execute("SELECT id, status, profile, task FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
