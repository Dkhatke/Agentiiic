"""Files the agent must never read: .env files, private keys, certificates, credentials.

Tool-level checks match names; the shell sandboxes hide the files the OS way (find_protected lists the ones that exist),
so obfuscation like `cat .e*` or `cat $(printf '.e%s' nv)` cannot reach them."""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

PROTECTED = re.compile(
    r"(^|/)(\.env(\..*)?|id_(rsa|dsa|ecdsa|ed25519).*|.*\.(pem|p12|pfx|key|jks|keystore)|credentials(\.json)?|\.netrc|\.npmrc|"
    r"\.pypirc|\.git-credentials|\.pgpass|\.htpasswd|service[-_]?account.*\.json)$", re.I)  # re.I: macOS volumes ignore case
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".agentic", ".tox", "dist", "build", ".pytest_cache"}
MAX_FILES = 200
GLOB_CHARS = re.compile(r"[*?\[]")


def walk_files(root: Path, base: Path | None = None, include_protected: bool = False):
    """Regular files under `base`, sorted. Skips vendor dirs, secret files (unless asked) and every symlink, so a link
    inside the workspace can never lead a tool to a file outside it."""
    for dirpath, dirs, files in os.walk(base or root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not Path(dirpath, d).is_symlink())
        for name in sorted(files):
            f = Path(dirpath, name)
            if not f.is_symlink() and (include_protected or not PROTECTED.search(name)):
                yield f


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
    try:
        return any(PROTECTED.search(m) for m in glob.glob(part, root_dir=root, recursive=True))
    except (ValueError, OSError):  # e.g. a NUL byte: the shell could not use such a word either
        return False
