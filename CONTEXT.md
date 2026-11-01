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
| 2 Router | done | ordered fallback, cooldown on 401/403/429/5xx/unreachable, text tool mode. **Budgets and usage view added in phase 6 |
| 3 Safety | done | secret redaction (output and audit), protected files (.env, keys), untrusted-data wrapping and taint that disables auto-approved shell, macOS sandbox (writes confined to the workspace, no network by default) |
| 4 Profiles and orchestrator | done | profiles builder, researcher, ideator, general, security; keyword orchestrator (`--profile auto`); `repo_map`, `git_branch`, `git_commit` (refuses main/master); read-only profiles default to `readonly` mode |
| 5 Researcher web tools | done | `web_search` (DuckDuckGo HTML, no key) and `fetch_url`; NETWORK risk so always asks; output untrusted and taints the gate; SSRF guard (public http(s) only, redirects re-checked); 1 MB cap; 24h disk cache. Verified live. Researcher profile now `ask-on-write` mode (readonly would deny network) |
| 6 Budgets, memory, resume | done | `store.py` (SQLite at `~/.local/state/agentic/agentic.db`): per-provider `daily_calls` cap (router skips a provider once spent, falls through to the next), token usage, `agentic usage`; every task saved after each step, `agentic tasks`, `agentic resume <id>` (drops a half-finished tool turn). Loop now defaults reply role to assistant |
| 7 Daemon and web UI | done | `server.py` (stdlib `http.server`, no new deps): `agentic serve` binds 127.0.0.1 only, random token (file mode 600, also in the URL hash), Host-header check against DNS rebinding, JSON API + long-poll events (not SSE), approvals via the UI's Approve/Deny buttons (10 min timeout = deny); single-file UI at `ui/index.html` (dark/light, mobile-friendly). Tested end to end with a scripted router, plus a live smoke test |
| 8 MCP and evals | done | `mcp_server.py`: stdio MCP server (`agentic mcp --workspace .`) exposing only read-only tools (read_file, list_dir, repo_map), redacted and confined, so VS Code or Antigravity can pull project context but never write; `evals.py` + `evals/*.toml` (4 tasks: create script, fix bug, read and report, refuse to leak `.env`) with `agentic eval`; runs auto-approved in a temp workspace |
| 9 Playbooks and deploy | done | `load_playbook` tool plus `playbooks/*.md` (deploy, new-project, college-report); `deploy` profile in `ask-all` mode that follows the deploy playbook (inspect, pre-flight, dry run, build, release, verify, rollback note); builder and general load playbooks on demand |
| 9b Semantic search | done (unverified live) | `embeddings.py`: OpenAI-compatible `/embeddings` client (default model `nvidia/nemotron-3.5-embed-1b:free` on OpenRouter, from the user's paste) + SQLite vector index (JSON vectors, pure-Python cosine, incremental by mtime, pruned on delete). `semantic_search` tool on builder and researcher: NETWORK risk so it asks first; skips `.env`/keys/`node_modules`; caps 300 files / 1500 chunks. Embedding models cannot chat, so this is NOT in the providers list |
| 10 Real-model validation | next (needs the user's key or Ollama) | run `agentic eval` against a real model, tune prompts and defaults, verify `openrouter/free` model name |

Optional later ideas: native Anthropic provider adapter, model-based profile routing, sandboxed read confinement, SearXNG search, VS Code extension, Docker sandbox for non-macOS.

## Where things live
- `src/agentic/loop.py`: the agent loop and `Profile`
- `src/agentic/router.py`, `textmode.py`: providers, fallback, text tool protocol
- `src/agentic/permissions.py`: the gate
- `src/agentic/tools/`: registry plus built-in, dev (`dev.py`) and web (`web.py`) tools
- `src/agentic/safety/`: redaction and sandbox
- `src/agentic/profiles/*.toml`: agent profiles (builder, researcher, ideator, general, security, deploy)
- `src/agentic/playbooks/*.md`: recipes loaded on demand
- `src/agentic/orchestrator.py`: profile selection
- `src/agentic/server.py`, `ui/index.html`: local daemon and web UI
- `src/agentic/store.py`: SQLite usage and tasks
- `src/agentic/mcp_server.py`: read-only MCP server for editors
- `src/agentic/evals.py`, `evals/`: eval harness and tasks
- `tests/test_core.py`: 33 tests (`test_core.py`, `test_server.py`, `test_evals_mcp.py`, `test_embeddings.py`) (run `.venv/bin/pytest`)

## Known gaps and notes
- The evals have only been run with scripted fake models; real pass rates are unknown until a real model is connected.
- The embeddings endpoint and model id were taken from the user's OpenRouter snippet and are untested live (no key yet). Indexed file text is sent to that provider, hence the approval prompt; set `enabled = false` under `[embeddings]` to turn it off.
- Not yet run against a real model (no key or Ollama yet). Everything is covered by mocked tests.
- Default provider chain (verified against OpenRouter's public model list on 2026-11-01): `cohere/north-mini-code:free`, `google/gemma-4-31b-it:free`, `nvidia/nemotron-3.5-lightning:free`, `openrouter/free`, then local Ollama. Free IDs change and some are retired (a few listed "going away"), so `agentic models` lists current free tool-calling models. One OpenRouter key serves the whole chain; the free-tier rate limit is account-wide, so per-provider `daily_calls` counters don't add up across entries.
- The sandbox restricts writes and network, not reads. Each shell command still asks for approval.
- The sandbox only works on macOS.
- `web_search` scrapes DuckDuckGo's HTML page, which can change layout or rate-limit; swap in SearXNG or a keyed API later if it breaks.
- Orchestrator is keyword-based; an LLM classifier is an optional later upgrade.

## Using it with VS Code / Antigravity
No integration is required: run `agentic` in the editor terminal, or open the web UI beside it; edits land on disk and the editor reloads them. For context inside the editor's own AI, add an MCP server with command `/abs/path/.venv/bin/agentic` and args `mcp --workspace /abs/path/to/project` (read-only; check your editor's MCP settings for the exact file).

## How to run
```
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/agentic init && .venv/bin/agentic providers
export OPENROUTER_API_KEY=...        # or run Ollama
.venv/bin/agentic serve              # web UI at the printed URL
.venv/bin/agentic run "task" --workspace ./project
```
