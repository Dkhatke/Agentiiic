# Agentic Launcher (VS Code / Antigravity)

A thin launcher, not an integration. Each command opens the integrated terminal and runs the `agentic` CLI there, so the
agent's permission prompts appear in the terminal and nothing is auto-approved. The extension reads and edits no files.

Commands (Ctrl/Cmd+Shift+P): **Agentic: Run Task**, **Run Task About Selection** (also in the editor right-click menu),
**Security Audit This Workspace**, **Research a Question**, **Open Web UI** (starts `agentic serve` and opens it with the token).

## Install
No build step.
1. `cd vscode-extension && npx @vscode/vsce package --allow-missing-repository` produces `agentic-launcher-0.1.0.vsix`.
2. In VS Code or Antigravity: Extensions panel, "..." menu, **Install from VSIX...**.
   (Or symlink this folder into `~/.vscode/extensions/` and restart.)
3. If `agentic` is not on your PATH, set **Agentic: Command** to the absolute path, e.g. `/Users/you/Agentic Project/.venv/bin/agentic`.

Tests: `node --test test/` (uses a fake `vscode` object; the extension has not been loaded into a real editor in development).
