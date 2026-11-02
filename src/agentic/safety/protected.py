"""Files the agent must never read: .env files, private keys, certificates, credentials.

Tool-level checks match names; the shell sandboxes hide the files the OS way (find_protected lists the ones that exist),
so obfuscation like `cat .e*` or `cat $(printf '.e%s' nv)` cannot reach them."""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from .redact import PATTERNS as TOKEN_PATTERNS

PROTECTED = re.compile(
    r"(^|/)(\.env(\..*)?|id_(rsa|dsa|ecdsa|ed25519).*|.*\.(pem|p12|pfx|key|jks|keystore)|credentials(\.json)?|\.netrc|\.npmrc|"
    r"\.pypirc|\.git-credentials|\.pgpass|\.htpasswd|service[-_]?account.*\.json)$", re.I)  # re.I: macOS volumes ignore case
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".agentic", ".tox", "dist", "build", ".pytest_cache"}
MAX_FILES = 200
SNIFF_BYTES = 16_384
SNIFF_MAX_SIZE = 1_000_000
GLOB_CHARS = re.compile(r"[*?\[]")


def secret_reason(p: Path, sniff: bool = True) -> str | None:
    """Why a file must not be read or sent anywhere, or None. Beyond the name check this catches the two disguises a name
    cannot: a hard link (its name is ours but its data may be a file outside the workspace) and a file whose contents are
    a private key or a known token format under an innocent name (notes.txt, backup.dat). Direct reads pass sniff=False:
    they are redacted already, so such a file stays readable but is hidden from the shell, scans, search and commits."""
    if PROTECTED.search(str(p)):
        return "secret file name"
    try:
        if p.is_symlink() or not p.is_file():
            return None
        st = p.stat()
        if st.st_nlink > 1:
            return "hard link (its data may be a file outside the workspace)"
        if not sniff or st.st_size > SNIFF_MAX_SIZE:
            return None
        with open(p, "rb") as fh:
            head = fh.read(SNIFF_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    if any(rx.search(head) for rx in TOKEN_PATTERNS):
        return "contents look like a private key or access token"
    return None


def walk_files(root: Path, base: Path | None = None, include_protected: bool = False):
    """Regular files under `base`, sorted. Skips vendor dirs, secret files (unless asked) and every symlink, so a link
    inside the workspace can never lead a tool to a file outside it."""
    for dirpath, dirs, files in os.walk(base or root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not Path(dirpath, d).is_symlink())
        for name in sorted(files):
            f = Path(dirpath, name)
            if not f.is_symlink() and (include_protected or secret_reason(f) is None):
                yield f


def find_protected(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if secret_reason(Path(dirpath, f)):
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
