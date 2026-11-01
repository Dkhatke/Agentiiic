import json

import httpx
import pytest

from agentic.config import EmbeddingConfig
from agentic.embeddings import Embedder, index_workspace, search
from agentic.store import Store
from agentic.tools import ToolContext, ToolError
from agentic.tools.dev import semantic_search

VOCAB = ["database", "login", "image", "password", "render"]


def fake_embedding_client(calls):
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body["input"])
        data = [{"index": i, "embedding": [float(t.lower().count(w)) for w in VOCAB]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data[::-1]})  # out of order on purpose
    return httpx.Client(transport=httpx.MockTransport(handler))


def setup(tmp_path):
    (tmp_path / "auth.py").write_text("def login(user, password):\n    # check password against database\n    pass\n")
    (tmp_path / "gfx.py").write_text("def render(image):\n    # draw the image\n    pass\n")
    (tmp_path / ".env").write_text("password image database SECRET")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("login login login")
    calls = []
    return Embedder(EmbeddingConfig(api_key_env=None), fake_embedding_client(calls)), Store(tmp_path / "db.sqlite"), calls


def test_search_ranks_by_meaning_and_skips_secrets(tmp_path):
    emb, store, calls = setup(tmp_path)
    out = search(tmp_path.resolve(), "how does password login work", emb, store)
    assert out.splitlines()[0].startswith("auth.py")
    sent = " ".join(" ".join(c) for c in calls)
    assert "SECRET" not in sent and "login login login" not in sent  # .env and node_modules never embedded


def test_index_is_incremental_and_prunes(tmp_path):
    emb, store, calls = setup(tmp_path)
    root = tmp_path.resolve()
    assert index_workspace(root, emb, store) == 2
    assert index_workspace(root, emb, store) == 0            # unchanged: nothing re-embedded
    (tmp_path / "gfx.py").write_text("def render(image):\n    return image\n")
    import os, time
    os.utime(tmp_path / "gfx.py", (time.time() + 5, time.time() + 5))
    assert index_workspace(root, emb, store) == 1
    (tmp_path / "auth.py").unlink()
    index_workspace(root, emb, store)
    assert {p for p, *_ in store.all_chunks(str(root))} == {"gfx.py"}


def test_tool_requires_configuration_and_handles_errors(tmp_path):
    with pytest.raises(ToolError):
        semantic_search(ToolContext(tmp_path.resolve()), "x")
    bad = Embedder(EmbeddingConfig(api_key_env=None), httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401))))
    with pytest.raises(ToolError):
        bad.embed(["x"])
