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

Any OpenAI-compatible endpoint works as a provider (OpenRouter, Ollama, Groq, Gemini's compat API...). Anthropic's native API works too (`api = "anthropic"`). Optional config: `[search] searxng_url` for self-hosted search, `[sandbox] docker_image` for a container shell sandbox on non-macOS. See the commented examples in `agentic init`. Editors: `vscode-extension/` is a small launcher for VS Code/Antigravity (see its README). Ideas are scored by a separate judge model (`[judge]` in the config).
Writes and shell commands ask for approval; every call is logged to `~/.local/state/agentic/audit.jsonl`.

## Testing
```
.venv/bin/python -m pytest -q                      # ~1900 tests, no network, no keys (30 s)
AGENTIC_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -v   # real internet + real sandbox (+ a real model if a key is set)
.venv/bin/agentic doctor                              # is my setup working? which provider is down?
.venv/bin/agentic eval                                # task suite against your real providers (needs a key)
cd vscode-extension && node --test test/             # editor launcher
```
