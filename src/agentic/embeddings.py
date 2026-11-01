"""Optional semantic search: OpenAI-compatible /embeddings client + a small SQLite vector index.

Text sent for embedding leaves the machine, so the tool is NETWORK-risk (asks first) and skips
protected/secret files. Vectors are stored as JSON and compared in pure Python: fine for small repos.
"""
from __future__ import annotations

import math
from pathlib import Path

import httpx

from .config import EmbeddingConfig
from .tools.builtin import PROTECTED
from .tools.dev import SKIP_DIRS
from .tools.registry import ToolError

SUFFIXES = {".py", ".js", ".ts", ".tsx", ".md", ".txt", ".toml", ".json", ".html", ".css", ".java", ".go", ".rs"}
MAX_FILES, MAX_CHUNKS, MAX_FILE_BYTES, CHUNK_LINES, BATCH = 300, 1500, 100_000, 40, 16


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, client: httpx.Client | None = None):
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=60)

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            try:
                r = self.client.post(f"{self.cfg.base_url.rstrip('/')}/embeddings", headers=headers,
                                     json={"model": self.cfg.model, "input": texts[i:i + BATCH]})
            except httpx.HTTPError as e:
                raise ToolError(f"embeddings request failed ({type(e).__name__})") from e
            if r.status_code >= 400:
                raise ToolError(f"embeddings HTTP {r.status_code}: {r.text[:150]}")
            try:
                out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
            except (KeyError, TypeError, ValueError) as e:
                raise ToolError("malformed embeddings response") from e
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _chunks(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(errors="replace").splitlines()
    return [(i + 1, "\n".join(lines[i:i + CHUNK_LINES])) for i in range(0, len(lines), CHUNK_LINES) if lines[i:i + CHUNK_LINES]]


def index_workspace(root: Path, embedder: Embedder, store) -> int:
    """Embed new or changed files; returns number of chunks embedded this run."""
    files = [f for f in sorted(root.rglob("*"))
             if f.is_file() and f.suffix in SUFFIXES and f.stat().st_size < MAX_FILE_BYTES
             and not any(p in SKIP_DIRS for p in f.relative_to(root).parts) and not PROTECTED.search(str(f))][:MAX_FILES]
    store.prune_chunks(str(root), {str(f.relative_to(root)) for f in files})
    embedded, budget = 0, MAX_CHUNKS
    for f in files:
        rel, mtime = str(f.relative_to(root)), f.stat().st_mtime
        if store.chunk_mtime(str(root), rel) == mtime:
            budget -= store.chunk_count(str(root), rel)
            continue
        chunks = _chunks(f)[:max(budget, 0)]
        if not chunks:
            continue
        vecs = embedder.embed([t for _, t in chunks])
        store.replace_chunks(str(root), rel, mtime, [(ln, t, v) for (ln, t), v in zip(chunks, vecs)])
        embedded += len(chunks)
        budget -= len(chunks)
    return embedded


def search(root: Path, query: str, embedder: Embedder, store, k: int = 5) -> str:
    index_workspace(root, embedder, store)
    qv = embedder.embed([query])[0]
    scored = sorted(((_cosine(qv, v), p, ln, t) for p, ln, t, v in store.all_chunks(str(root))), reverse=True)[:k]
    if not scored:
        return "index is empty (no indexable files)"
    return "\n\n".join(f"{p}:{ln}  (score {s:.2f})\n{t[:600]}" for s, p, ln, t in scored)
