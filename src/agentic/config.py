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
# One OpenRouter key serves every entry below; free models rate-limit often, so the chain falls through.
# Free model IDs change: list current ones with `agentic models`.

[[providers]]
name = "groq-gptoss"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
model = "openai/gpt-oss-120b"   # fast, strong tool use

[[providers]]
name = "groq-qwen"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
model = "qwen/qwen3.8-27b"   # fallback
max_input_tokens = 7000            # Groq free tier ITPM limit for this model

[[providers]]
name = "or-coder"
quota = "openrouter"
daily_calls = 45                   # OpenRouter free tier: 50 requests/day per account
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "cohere/north-mini-code:free"   # coding-agent model, 256K context

[[providers]]
name = "or-gemma"
quota = "openrouter"
daily_calls = 45                   # OpenRouter free tier: 50 requests/day per account
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "google/gemma-4-31b-it:free"   # strong general model, native function calling

[[providers]]
name = "or-nemotron"
quota = "openrouter"
daily_calls = 45                   # OpenRouter free tier: 50 requests/day per account
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "nvidia/nemotron-3.5-lightning:free"   # fast, 1M context

[[providers]]
name = "or-auto"
quota = "openrouter"
daily_calls = 45                   # OpenRouter free tier: 50 requests/day per account
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "openrouter/free"   # OpenRouter picks any available free model

[[providers]]
name = "ollama"
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:3b"         # any model you've pulled; no key needed
tool_mode = "text"                 # small models often can't do native tool-calling

[limits]
max_steps = 25
max_model_calls = 60
shell_timeout = 60

[embeddings]                       # optional semantic search; sends file text to this provider
model = "nvidia/nemotron-3.5-embed-1b:free"
enabled = true

[sandbox]                          # macOS: shell can only write in the workspace
enabled = true
network = false                    # true lets shell commands reach the internet
# docker_image = "python:3.12-slim"  # Linux/Windows: run shell commands in this container (needs docker)

# [judge]                          # which provider scores the ideator's ideas (default: not the chain's first)
# provider = "groq-qwen"

# [search]
# searxng_url = "http://localhost:8080"   # self-hosted SearXNG replaces DuckDuckGo scraping

# Native Anthropic API (paid; key in ANTHROPIC_API_KEY). Put it first or last in the chain:
# [[providers]]
# name = "claude"
# api = "anthropic"
# base_url = "https://api.anthropic.com/v1"
# api_key_env = "ANTHROPIC_API_KEY"
# model = "claude-haiku-5-5"
"""


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    daily_calls: int | None = None  # skip this provider once its quota group has been called this many times today
    max_input_tokens: int | None = None  # provider's per-request input limit; the loop shrinks context to fit
    quota: str | None = None        # shared quota group (e.g. all OpenRouter free models share one account limit)
    tool_mode: str = "native"  # "native" = API tool-calling; "text" = works with any model
    api: str = "openai"        # "openai" = any OpenAI-compatible endpoint; "anthropic" = native Messages API

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def usable(self) -> bool:
        return self.api_key_env is None or bool(self.api_key)


@dataclass
class EmbeddingConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "nvidia/nemotron-3.5-embed-1b:free"
    api_key_env: str | None = "OPENROUTER_API_KEY"
    enabled: bool = True

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def usable(self) -> bool:
        return self.enabled and (self.api_key_env is None or bool(self.api_key))


@dataclass
class Limits:
    max_steps: int = 25
    max_model_calls: int = 60
    shell_timeout: int = 60


@dataclass
class Sandbox:
    enabled: bool = True
    network: bool = False  # set true if shell commands must reach the internet (pip/npm install)
    docker_image: str = ""  # non-macOS: run shell commands in this container


@dataclass
class Judge:
    provider: str = ""  # name of the provider that scores ideas; empty = a different one from the chain's first


@dataclass
class Search:
    searxng_url: str = ""


@dataclass
class Config:
    providers: list[Provider] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)
    sandbox: Sandbox = field(default_factory=Sandbox)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: Search = field(default_factory=Search)
    judge: Judge = field(default_factory=Judge)


ENV_FILES = [Path(__file__).resolve().parents[2] / ".env", CONFIG_PATH.parent / ".env"]  # the agentic repo, then ~/.config/agentic


def load_env(files: list[Path] | None = None) -> list[Path]:
    """Put KEY=VALUE lines from the agentic .env into this process's environment. Variables already set win, and the
    workspace's own .env is never read here (it holds the app's secrets, not the agent's). Returns the files used."""
    used = []
    for f in files if files is not None else ENV_FILES:
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        used.append(f)
        for line in lines:
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if sep and key.isidentifier() and value and key not in os.environ:
                os.environ[key] = value
    return used


def load_config(path: Path = CONFIG_PATH) -> Config:
    data = tomllib.loads(path.read_text()) if path.exists() else tomllib.loads(SAMPLE_CONFIG)
    return Config(
        providers=[Provider(**p) for p in data.get("providers", [])],
        limits=Limits(**data.get("limits", {})),
        sandbox=Sandbox(**data.get("sandbox", {})),
        embeddings=EmbeddingConfig(**data.get("embeddings", {})),
        search=Search(**data.get("search", {})),
        judge=Judge(**data.get("judge", {})),
    )
