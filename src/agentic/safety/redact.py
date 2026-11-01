"""Scrub secrets from anything that goes to a model or a log."""
from __future__ import annotations

import re

PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}"),            # OpenAI/OpenRouter-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),                   # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                           # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),                      # Google API keys
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),                         # Groq
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"),
]
ASSIGNMENT = re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*)(\s*[=:]\s*)(['\"]?)([^\s'\"]{6,})\3")


def redact(text: str) -> str:
    for p in PATTERNS:
        text = p.sub("[REDACTED]", text)
    return ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
