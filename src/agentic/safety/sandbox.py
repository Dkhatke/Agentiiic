"""macOS sandbox-exec wrapper: shell commands can only write inside the workspace (and temp dirs),
and have no network unless explicitly allowed. Other platforms fall back to unwrapped execution."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def profile(root: Path, network: bool) -> str:
    lines = ["(version 1)", "(allow default)"]
    if not network:
        lines.append("(deny network*)")
    writable = " ".join(f'(subpath "{p}")' for p in (root, "/private/tmp", "/private/var/folders", "/dev"))
    lines.append(f"(deny file-write* (require-not (require-any {writable})))")
    return "\n".join(lines)


def wrap_command(command: str, root: Path, network: bool = False) -> list[str] | None:
    """Return argv that runs `command` sandboxed, or None if sandboxing is unavailable."""
    if not available():
        return None
    return ["sandbox-exec", "-p", profile(root, network), "/bin/sh", "-c", command]
