"""Shell sandbox. macOS: sandbox-exec lets commands write only inside the workspace (TMPDIR is pointed inside it),
hides credential directories from reads, and blocks the network unless allowed. Elsewhere (or on request): a docker
container that sees only the workspace. With neither available, commands run unwrapped."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# credential locations a shell command has no business reading (relative to $HOME)
SECRET_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config/gcloud", ".config/agentic", ".azure")
SECRET_FILES = (".netrc", ".npmrc", ".pypirc", ".git-credentials")


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def docker_available() -> bool:
    return shutil.which("docker") is not None


def profile(root: Path, network: bool, home: Path | None = None) -> str:
    home = home or Path.home()
    lines = ["(version 1)", "(allow default)"]
    if not network:
        lines.append("(deny network*)")
    writable = " ".join(f'(subpath "{p}")' for p in (root, "/dev"))  # temp lives inside root via TMPDIR
    lines.append(f"(deny file-write* (require-not (require-any {writable})))")
    hidden = [f'(subpath "{home / d}")' for d in SECRET_DIRS] + [f'(literal "{home / f}")' for f in SECRET_FILES]
    lines.append(f"(deny file-read* {' '.join(hidden)})")
    return "\n".join(lines)


def docker_argv(command: str, root: Path, network: bool, image: str) -> list[str]:
    argv = ["docker", "run", "--rm", "-v", f"{root}:/work", "-w", "/work",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
    if hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]  # files created inside belong to you, not root
    if not network:
        argv += ["--network", "none"]
    return argv + [image, "/bin/sh", "-c", command]


def wrap_command(command: str, root: Path, network: bool = False, image: str = "") -> list[str] | None:
    """Return argv that runs `command` sandboxed, or None if sandboxing is unavailable."""
    if image and docker_available():
        return docker_argv(command, root, network, image)
    if not available():
        return None
    return ["sandbox-exec", "-p", profile(root, network), "/bin/sh", "-c", command]
