# Personal Project-Builder Agent: Plan

## Goal
A lightweight, secure, free-to-run agent for **building projects** (code, research, ideas) that also handles random tasks like college work. It asks before acting. It is built in a way that teaches how agents work.

## Decisions (change any of these)
- Language: Python 3.12, with plain provider SDKs or `httpx`. No LangChain.
- Interface: headless local core with thin clients: CLI first, then a small web UI, then optional editor extension. All clients talk to the same core (see "Clients and editor coexistence").
- Profiles are TOML (stdlib, no extra dependency).
- Models: free tiers behind a router (Gemini, Groq, Cerebras, OpenRouter `:free`) with Ollama as the offline fallback. You supply the keys.
- State: SQLite plus plain files. No vector DB at the start.
- Git: always a feature branch and a PR, never commit to main.

## Architecture
```
CLI -> Orchestrator -> Agent loop <-> Model router -> providers
                          |
              Permission gate -> Tool registry -> Sandbox
                          |
              Memory (SQLite) + Audit log + Checkpoints
```
Everything is one process. Agents are **profiles** (YAML: prompt, allowed tools, model tier, permission mode), not separate programs.

| Profile | Job |
|---|---|
| builder | scaffold, code, run tests, iterate, open a PR |
| researcher | search, fetch, cross-check and cite; finds models, datasets and libraries |
| ideator | project ideas, scoped to your skills, time and free-tier limits |
| general | college work, writing, explanations |
| security | read-only audit: secrets, dependencies, config |
| deploy | dry run first, then approval at each stage (later) |

A small router step picks the profile from your request, and you can override it with `--profile`.

## Clients and editor coexistence
The core runs as a local daemon bound to `127.0.0.1` only, with a random local token (the "API key" is just this local token, and it is stored hashed). It exposes HTTP plus server-sent events for streaming and approvals.
- **CLI**: thin client over the daemon. Works inside the VS Code or Antigravity terminal.
- **Web UI**: small static page (no heavy framework) showing chat, plan, live tool calls and approval buttons.
- **Works with any editor with no integration**: the agent edits files on disk in your project, and editors reload changes. Git diff is the shared review surface.
- **Optional integrations (later)**: expose tools as an MCP server, or a thin VS Code extension (Antigravity is a VS Code fork, so it should reuse it; verify extension-marketplace support when we get there).

## Compute budget (what keeps it feasible)
The agent itself is I/O-bound glue code; the heavy compute is the model, and we push that off your machine.
- **Inference**: free cloud tiers by default, so local CPU/RAM use is near zero. A local model is an optional fallback only: small (about 3-4B), loaded on demand and unloaded after idle, and never while your project is building.
- **Idle cost**: a single Python process, SQLite, no Docker, Postgres, Redis or vector DB running in the background.
- **Sandbox**: subprocess with a restricted cwd, timeouts, output caps and (on macOS) `sandbox-exec`. Docker only on demand for risky runs.
- **Concurrency**: 1 task and 1 model call at a time by default. Sub-agents only for parallel research, using the cheapest model, because multi-agent multiplies token use.
- **Separate budgets** for model calls, tokens, steps, shell time, browser actions and disk. A token quota alone doesn't stop a runaway loop.
- **Context**: repo map via ripgrep/tree-sitter, read only relevant files, compact history, cache research.
- **Privacy tiers**: files tagged public/private; private code and secrets never go to cloud models unless you allow it per project.

## Ideas adopted from the ChatGPT "NEXUS" design
- Agent manifests validated against a strict schema (permissions, limits, approval-required actions).
- Tasks as a dependency graph, so independent steps can run in parallel later.
- Persisted task state, with pause, resume and cancel.
- Agent factory builds temporary agents from approved templates only, never from generated code.
- Branch per task, no auto-push, commit only on approval.
- Data-classification control over what may leave the machine.
Deliberately deferred: PostgreSQL/pgvector, Redis, always-on Docker, Next.js, LangGraph, 15 API groups, multi-user auth. We can revisit each when a real need appears.

## Core components
1. **Agent loop**: plan, choose a tool, ask permission, execute, observe, repeat. It has hard caps on steps, tokens and time.
2. **Tool registry**: each tool has a schema and a risk level (`read`, `write`, `network`, `destructive`).
3. **Permission gate** with three modes:
   - `ask-all`
   - `ask-on-write` (default)
   - `auto-readonly`
   Destructive actions always ask and show a diff or preview.
4. **Model router**: a fallback chain, backoff on 429, per-key daily budgets, and small models for cheap steps. Keys live in the OS keychain.
5. **Sandbox**: a restricted working directory, a command allowlist and network off by default. Fetched content is treated as untrusted data and can't trigger tools on its own.
6. **Safety layer**: secret and PII redaction (reuse your redactor) and an append-only audit log.
7. **Memory**: a task log, your preferences, a research cache and lessons learned.
8. **Checkpoints**: a git snapshot before writes, plus resume after a crash or rate limit.

## Milestones
Each one is small enough to finish in a few sessions, and each has a "learn" note so you understand what you built.

| # | Milestone | Done when | You learn |
|---|---|---|---|
| 0 | DONE: skeleton, config, tests | `agentic --help` works | project layout |
| 1 | DONE: loop, read/list/write/edit/shell tools, permission gate | it edits a file only after you approve | what an agent loop really is |
| 2 | PARTLY DONE: ordered fallback + cooldown + call cap done; per-key daily budgets and usage view pending | it survives a rate limit by switching provider | free-tier engineering |
| 3 | DONE: sandbox-exec shell, redaction, protected files, untrusted-data tainting | a malicious file can't make it run a command unasked | agent security |
| 4 | DONE: profiles (builder, researcher, ideator, general, security), keyword orchestrator, repo_map, git tools | the same engine behaves as builder or researcher | agents as config |
| 5 | DONE: web_search, fetch_url, SSRF guard, cache | researcher profile can search and cite | tool use and RAG basics |
| 6 | Builder: scaffold, test, branch, PR, undo | builds a small project end to end | the coding-agent workflow |
| 7 | Ideator and general profiles, playbooks (markdown recipes) | gives scoped ideas and handles a college task | prompting and context management |
| 8 | Memory, checkpoints, resume | a killed task resumes where it stopped | state and durability |
| 9 | Eval harness: about 10 repeatable tasks with scores | you can tell if a change helped | how to measure agents |
| 10 | MCP client, security and deploy profiles | uses an MCP server, runs a dry-run deploy | the ecosystem |

**MVP is milestones 0 to 4.** That is a safe, working builder that you can already use for projects.

## Constraints
- Idle footprint: under ~100 MB RAM and a small install (no local model required).
- Token discipline: summarize and truncate context, cache research, and use the cheapest adequate model per step.
- Default caps: 25 steps and a daily token budget per provider (tune in config).
- No account auto-creation or key rotation tricks. Only keys you supplied, within each provider's terms.

## Proposed folder structure
```
agent/
  cli.py
  loop.py
  router/        providers, budgets, fallback
  tools/         registry, fs, shell, web, git
  safety/        permissions, sandbox, redact, audit
  profiles/      builder.yaml, researcher.yaml, ...
  memory/        sqlite store, checkpoints
  playbooks/     deploy.md, new-project.md, ...
evals/
tests/
```

## Risks
- Free-tier limits and model quality vary, so the router and evals are what keep it reliable.
- Prompt injection from the web is the main security risk, which is why milestone 3 comes before research and builder.
- Scope creep: don't start on UI, vector DB or multi-agent swarms until the MVP is used for real work.

## Open items
- Confirm Python and CLI-first.
- Which free provider keys you already have.
- Whether the repo gets a name and a GitHub remote now.

## Design inspiration from existing repos
- **mini-swe-agent** (SWE-agent team): ~100-line loop, linear message history, each shell command is a fresh `subprocess.run`, execution environment is swappable (local to Docker/bubblewrap), and it can drive any model through plain text. Adopted: text tool protocol (`tool_mode = "text"`) so small and free models work, plus the swappable-environment idea for the sandbox milestone.
- **Aider**: git-native edits and a repo map. Planned for the builder milestone (commit per approved change, ripgrep/tree-sitter map).
- **OpenCode**: many providers behind one config, LSP awareness. We already have provider-agnostic config; LSP is a later idea.
