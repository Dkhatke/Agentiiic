"""Config: providers (any OpenAI-compatible endpoint), limits, default permission mode."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("AGENTIC_CONFIG", Path.home() / ".config/agentic/config.toml"))
STATE_DIR = Path(os.environ.get("AGENTIC_STATE", Path.home() / ".local/state/agentic"))

SAMPLE_CONFIG = """\
# Providers are tried in order. Any OpenAI-compatible endpoint works.
# Keys are read from environment variables, never stored in this file.

[[providers]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "openrouter/free"          # change to any ':free' model from openrouter.ai/models

[[providers]]
name = "ollama"
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:3b"         # any model you've pulled; no key needed
tool_mode = "text"                 # small models often can't do native tool-calling

[limits]
max_steps = 25
max_model_calls = 60
shell_timeout = 60

[sandbox]                          # macOS: shell can only write in the workspace
enabled = true
network = false                    # true lets shell commands reach the internet
"""


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    tool_mode: str = "native"  # "native" = API tool-calling; "text" = works with any model

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def usable(self) -> bool:
        return self.api_key_env is None or bool(self.api_key)


@dataclass
class Limits:
    max_steps: int = 25
    max_model_calls: int = 60
    shell_timeout: int = 60


@dataclass
class Sandbox:
    enabled: bool = True
    network: bool = False  # set true if shell commands must reach the internet (pip/npm install)


@dataclass
class Config:
    providers: list[Provider] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)
    sandbox: Sandbox = field(default_factory=Sandbox)


def load_config(path: Path = CONFIG_PATH) -> Config:
    data = tomllib.loads(path.read_text()) if path.exists() else tomllib.loads(SAMPLE_CONFIG)
    return Config(
        providers=[Provider(**p) for p in data.get("providers", [])],
        limits=Limits(**data.get("limits", {})),
        sandbox=Sandbox(**data.get("sandbox", {})),
    )
