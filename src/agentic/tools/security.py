"""Security review tools: deterministic candidate finders. The model's job is to confirm or dismiss each
candidate against the surrounding code (scanner first, LLM filter second), not to guess from scratch."""
from __future__ import annotations

import re
import subprocess

from ..safety.redact import ASSIGNMENT, PATTERNS
from .builtin import PROTECTED, _S, _clip, _obj
from ..safety.protected import walk_files
from .registry import Registry, Risk, Tool, ToolContext

MAX_HITS = 80
SECRET_LABELS = ["private key block", "API key (sk-/pk- style)", "GitHub token", "AWS access key id",
                 "Google API key", "Groq key", "Bearer token"]
CODE_RULES = [  # (id, severity, regex, why it matters)
    ("eval-exec", "high", r"\b(eval|exec)\s*\(", "runs arbitrary code if any part is attacker-controlled"),
    ("shell-true", "high", r"subprocess\.\w+\(.*shell\s*=\s*True", "shell injection if the command includes user input"),
    ("os-system", "high", r"\bos\.(system|popen)\(", "shell injection if the command includes user input"),
    ("pickle-load", "high", r"\bpickle\.loads?\(", "unpickling untrusted data executes code"),
    ("yaml-load", "medium", r"\byaml\.load\((?!.*Loader\s*=)", "yaml.load without SafeLoader can construct arbitrary objects"),
    ("sql-concat", "high", r"(execute|query)\(\s*(f[\"']|[\"'][^\"']*[\"']\s*(\+|%))\s*.*\b(SELECT|INSERT|UPDATE|DELETE)\b|(SELECT|INSERT|UPDATE|DELETE)\b[^\n]*\{[^}]+\}", "SQL built from strings: injection"),
    ("innerhtml", "medium", r"\.innerHTML\s*=|dangerouslySetInnerHTML", "XSS if the value includes user input"),
    ("tls-verify-off", "medium", r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false", "disables certificate checking"),
    ("weak-hash", "medium", r"\b(md5|sha1)\s*\(|hashlib\.(md5|sha1)\b", "weak for passwords or signatures"),
    ("debug-on", "medium", r"\bdebug\s*=\s*True", "debug mode can expose internals in production"),
    ("hardcoded-secret", "high", r"(?i)\b(password|passwd|secret|api_key)\s*=\s*[\"'][^\"']{4,}[\"']", "credential committed in code"),
    ("cors-wildcard", "medium", r"Access-Control-Allow-Origin[^\n]*\*|allow_origins\s*=\s*\[\s*[\"']\*[\"']", "any website can call this API"),
]
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".php", ".rs", ".html", ".yml", ".yaml", ".sh"}


def _files(ctx: ToolContext, include_protected: bool):
    for f in walk_files(ctx.root, include_protected=include_protected):
        if f.stat().st_size <= 500_000:
            yield f


def scan_secrets(ctx: ToolContext) -> str:
    """Find secret-looking values. Reports file:line, type and length only; the value is never printed."""
    hits = []
    for f in _files(ctx, include_protected=True):
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(f.relative_to(ctx.root))
        for n, line in enumerate(lines, 1):
            for label, rx in zip(SECRET_LABELS, PATTERNS):
                if m := rx.search(line):
                    hits.append(f"{rel}:{n}: {label} (length {len(m.group())})")
            if m := ASSIGNMENT.search(line):
                hits.append(f"{rel}:{n}: value assigned to {m.group(1)} (length {len(m.group(4))})")
    tracked = _tracked_secret_files(ctx)
    out = [f"TRACKED IN GIT (high risk): {t}" for t in tracked] + hits[:MAX_HITS]
    if len(hits) > MAX_HITS:
        out.append(f"...[{len(hits) - MAX_HITS} more]")
    return "\n".join(out) or "no secret-looking values found"


def _tracked_secret_files(ctx: ToolContext) -> list[str]:
    if not (ctx.root / ".git").exists():
        return []
    try:
        r = subprocess.run(["git", "ls-files"], cwd=ctx.root, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [f for f in r.stdout.splitlines() if PROTECTED.search(f) and not f.endswith((".example", ".sample"))]


def scan_code(ctx: ToolContext) -> str:
    """Pattern candidates for common vulnerabilities. These are leads, not verdicts: check each in context."""
    compiled = [(i, sev, re.compile(rx), why) for i, sev, rx, why in CODE_RULES]
    hits = []
    for f in _files(ctx, include_protected=False):
        if f.suffix not in CODE_SUFFIXES:
            continue
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(f.relative_to(ctx.root))
        in_tests = bool(re.search(r"(^|/)(tests?|__tests__|spec)(/|$)|test_|\.test\.|_test\.", rel))
        for n, line in enumerate(lines, 1):
            for rule, sev, rx, why in compiled:
                if rx.search(line):
                    tag = " (in tests, likely benign)" if in_tests else ""
                    hits.append(f"{rel}:{n} [{sev}] {rule}{tag}: {line.strip()[:120]}   <- {why}")
                    if len(hits) >= MAX_HITS:
                        return "\n".join(hits) + f"\n...[stopped at {MAX_HITS}; narrow the scope]"
    return "\n".join(hits) or "no pattern candidates found (this does not prove the code is safe)"


def preflight(ctx: ToolContext) -> str:
    """Deploy readiness facts: git state, tracked secrets, stack, tests, big files."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ctx.root, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    lines = []
    if (ctx.root / ".git").exists():
        dirty = git("status", "--porcelain")
        lines.append(f"git branch: {git('symbolic-ref', '--short', 'HEAD') or '(none)'}")
        lines.append("working tree: " + ("CLEAN" if not dirty else f"UNCOMMITTED CHANGES ({len(dirty.splitlines())} files)"))
        lines += [f"SECRET FILE TRACKED IN GIT: {t}" for t in _tracked_secret_files(ctx)]
    else:
        lines.append("not a git repository (no version history to roll back to)")
    markers = {"package.json": "Node", "pyproject.toml": "Python", "requirements.txt": "Python", "Dockerfile": "Docker",
               "go.mod": "Go", "Cargo.toml": "Rust", "index.html": "static site", "vercel.json": "Vercel config",
               "netlify.toml": "Netlify config", "render.yaml": "Render config"}
    found = sorted({v for k, v in markers.items() if (ctx.root / k).exists()})
    lines.append("stack hints: " + (", ".join(found) or "none detected"))
    has_tests = any(ctx.root.glob("tests")) or any(ctx.root.rglob("test_*.py")) or any(ctx.root.rglob("*.test.*"))
    lines.append("tests: " + ("found" if has_tests else "NONE FOUND"))
    lines.append("README: " + ("yes" if (ctx.root / "README.md").exists() else "MISSING"))
    lines.append(".gitignore: " + ("yes" if (ctx.root / ".gitignore").exists() else "MISSING"))
    big = [f"{f.relative_to(ctx.root)} ({f.stat().st_size // 1_000_000} MB)" for f in _files(ctx, True) if f.stat().st_size > 5_000_000]
    lines += [f"large file: {b}" for b in big[:5]]
    return "\n".join(lines)


def register(reg: Registry) -> None:
    reg.register(Tool("scan_secrets", "Find committed secrets (reports location and type only, never the value) "
                      "and secret files tracked by git.", _obj({}, []), Risk.READ, scan_secrets, untrusted=True))
    reg.register(Tool("scan_code", "Find pattern candidates for common vulnerabilities (eval, shell injection, SQL "
                      "concatenation, weak hashes...). Each hit is a lead to verify in context, not a verdict.",
                      _obj({}, []), Risk.READ, scan_code, untrusted=True))
    reg.register(Tool("preflight", "Deploy readiness facts: git state, tracked secrets, stack, tests, README, big files.",
                      _obj({}, []), Risk.READ, preflight))
