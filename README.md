# agentic

Lightweight, permission-gated personal agent for building projects. See [PLAN.md](PLAN.md).

```
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/agentic init        # writes ~/.config/agentic/config.toml
.venv/bin/agentic providers   # shows which providers are usable
export OPENROUTER_API_KEY=... # or run Ollama locally
.venv/bin/agentic run "add a hello script" --workspace ./some-project
.venv/bin/pytest
```

Any OpenAI-compatible endpoint works as a provider (OpenRouter, Ollama, Groq, Gemini's compat API...).
Writes and shell commands ask for approval; every call is logged to `~/.local/state/agentic/audit.jsonl`.
