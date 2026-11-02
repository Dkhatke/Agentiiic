"""Files the agent must never read: .env files, private keys, certificates, credentials.

Tool-level checks match names; the shell sandboxes hide the files the OS way (find_protected lists the ones that exist),
so obfuscation like `cat .e*` or `cat $(printf '.e%s' nv)` cannot reach them."""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

PROTECTED = re.compile(r"(^|/)(\.env(\..*)?|id_rsa.*|id_ed25519.*|.*\.pem|.*\.p12|credentials(\.json)?|\.netrc)$")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".agentic", ".tox", "dist", "build"}
MAX_FILES = 200
GLOB_CHARS = re.compile(r"[*?\[]")


def find_protected(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if PROTECTED.search(f):
                found.append(Path(dirpath, f))
                if len(found) >= MAX_FILES:
                    return found
    return found


def expands_to_protected(part: str, root: Path) -> bool:
    """True if a shell word containing glob characters would match a protected file in the workspace."""
    if not GLOB_CHARS.search(part):
        return False
    return any(PROTECTED.search(m) for m in glob.glob(part, root_dir=root, recursive=True))
