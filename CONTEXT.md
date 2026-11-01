# CONTEXT: living build log

Updated after every phase. Read this first when resuming. The roadmap is in [PLAN.md](PLAN.md); this file records what was actually done, with what, and why.

## Project
Personal, lightweight, permission-gated agent for building projects (code, research, ideas, college work). Not a thin LLM wrapper: it has a tool registry, permission gate, sandbox, redaction, audit log and provider router. Repo: https://github.com/Dkhatke/Agentiiic (the user allowed pushing directly to `main` for this repo).

## Stack and why
| Choice | Why |
|---|---|
| Python 3.12 via `uv` (system Python is 3.9) | stdlib `tomllib`, modern typing |
| `httpx` is the only runtime dependency | keep install and idle footprint small |
| TOML for config and profiles | stdlib parser, no PyYAML |
| OpenAI-compatible HTTP API for all providers | one client covers OpenRouter, Ollama, Groq, Gemini compat |
| SQLite and JSONL (no Postgres, Redis, Docker, vector DB) | lightweight; deferred until needed |
| macOS `sandbox-exec` for shell | confines writes and network with no Docker VM |
| Machine: Apple Silicon arm64, 16 GB RAM, Ollama not installed yet | a 3-4B local model is comfortable |

## Decisions
- Agents are **profiles** (TOML) over one engine, not separate programs.
- Keys come from env vars only. The user adds an OpenRouter key and/or Ollama later; the agent was built first.
- `tool_mode = "text"` per provider lets weak or free models work without native tool-calling (idea from mini-swe-agent).
- The ChatGPT "NEXUS" design was reviewed. Kept: manifests, state persistence, branch per task, data-classification idea. Deferred: Postgres/pgvector, Redis, Docker, Next.js, LangGraph, multi-user auth.
- Never commit to `main` from the agent's `git_commit` tool (this repo's `main` pushes are the user's explicit exception).

## Phase log
| Phase | Status | What was built |
|---|---|---|
| 0 Skeleton | done | package layout, config, CLI (`init`, `providers`, `run`), tests |
| 1 Loop and tools | done | agent loop with step and model-call caps; read, list, write, edit, shell tools; permission gate (ask-all, ask-on-write, readonly); JSONL audit log |
| 2 Router | mostly done | ordered fallback, cooldown on 401/403/429/5xx/unreachable, text tool mode. **Pending:** per-key daily budgets and a usage view |
| 3 Safety | done | secret redaction (output and audit), protected files (.env, keys), untrusted-data wrapping and taint that disables auto-approved shell, macOS sandbox (writes confined to the workspace, no network by default) |
| 4 Profiles and orchestrator | done | profiles builder, researcher, ideator, general, security; keyword orchestrator (`--profile auto`); `repo_map`, `git_branch`, `git_commit` (refuses main/master); read-only profiles default to `readonly` mode |
| 5 Researcher web tools | done | `web_search` (DuckDuckGo HTML, no key) and `fetch_url`; NETWORK risk so always asks; output untrusted and taints the gate; SSRF guard (public http(s) only, redirects re-checked); 1 MB cap; 24h disk cache. Verified live. Researcher profile now `ask-on-write` mode (readonly would deny network) |
| 6 Builder polish and memory | next | per-key daily budgets and usage view, task memory (SQLite), checkpoints and resume |

## Where things live
- `src/agentic/loop.py`: the agent loop and `Profile`
- `src/agentic/router.py`, `textmode.py`: providers, fallback, text tool protocol
- `src/agentic/permissions.py`: the gate
- `src/agentic/tools/`: registry plus built-in, dev (`dev.py`) and web (`web.py`) tools
- `src/agentic/safety/`: redaction and sandbox
- `src/agentic/profiles/*.toml`: agent profiles
- `src/agentic/orchestrator.py`: profile selection
- `tests/test_core.py`: 18 tests (run `.venv/bin/pytest`)

## Known gaps and notes
- Not yet run against a real model (no key or Ollama yet). Everything is covered by mocked tests.
- `openrouter/free` as the default model name is unverified; change it to a real `:free` model.
- The sandbox restricts writes and network, not reads. Each shell command still asks for approval.
- The sandbox only works on macOS.
- `web_search` scrapes DuckDuckGo's HTML page, which can change layout or rate-limit; swap in SearXNG or a keyed API later if it breaks.
- Orchestrator is keyword-based; an LLM classifier is an optional later upgrade.

## How to run
```
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/agentic init && .venv/bin/agentic providers
export OPENROUTER_API_KEY=...        # or run Ollama
.venv/bin/agentic run "task" --workspace ./project
```
