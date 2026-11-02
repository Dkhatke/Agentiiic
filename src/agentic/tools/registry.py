"""Tool registry: schema + risk level per tool. The model only ever sees registered tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    root: Path
    shell_timeout: int = 60
    sandbox: bool = True
    sandbox_network: bool = False
    allow_main_commit: bool = False
    http: object | None = None  # injectable httpx.Client (tests)
    embedder: object | None = None  # embeddings.Embedder, set when configured
    store: object | None = None
    fetched: set = field(default_factory=set)  # URLs actually read this run (for citation checks)
    judge: object | None = None  # callable(prompt) -> str backed by a separate model (judge_ideas)
    searxng_url: str = ""  # self-hosted SearXNG instance; replaces DuckDuckGo scraping when set
    sandbox_image: str = ""  # docker image for the shell sandbox on non-macOS hosts
    allow_hosts: set = field(default_factory=set)  # hosts exempt from the SSRF check and the page cache (SearXNG, offline evals)

    def resolve(self, path: str) -> Path:
        """Resolve a path and refuse anything outside the workspace root."""
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes workspace: {path}")
        return p


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    risk: Risk
    func: Callable[..., str]
    risk_fn: Callable[[dict], Risk] | None = None  # per-call override (e.g. safe shell commands)
    untrusted: bool = False  # output may carry attacker-controlled text (files, web pages)

    def risk_for(self, args: dict) -> Risk:
        return self.risk_fn(args) if self.risk_fn else self.risk

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class Registry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self, allowed: list[str] | None = None) -> list[dict]:
        return [t.schema() for n, t in self._tools.items() if allowed is None or n in allowed]
