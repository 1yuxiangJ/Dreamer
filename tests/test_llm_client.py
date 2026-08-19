"""Pure-unit tests for LLM client construction."""
from __future__ import annotations

import importlib


def test_dashscope_embedder_keeps_raw_string_input(monkeypatch):
    """DashScope compatible embeddings require raw string input, not token arrays."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("EMBED_API_KEY", "test-embed-key")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://test@localhost:5432/dreamer"
    )

    import dreamer.config
    import dreamer.llm.client

    importlib.reload(dreamer.config)
    importlib.reload(dreamer.llm.client)

    embedder = dreamer.llm.client.get_embedder()

    assert embedder.check_embedding_ctx_length is False


async def test_embed_text_reuses_cached_vector(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("EMBED_API_KEY", "test-embed-key")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://test@localhost:5432/dreamer"
    )

    import dreamer.config
    import dreamer.llm.client

    importlib.reload(dreamer.config)
    importlib.reload(dreamer.llm.client)

    calls = 0

    class FakeEmbedder:
        async def aembed_query(self, text: str) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0, 2.0, 3.0]

    monkeypatch.setattr(dreamer.llm.client, "get_embedder", lambda: FakeEmbedder())
    dreamer.llm.client.clear_embedding_cache()

    first = await dreamer.llm.client.embed_text("same text")
    second = await dreamer.llm.client.embed_text("same text")

    assert first == [1.0, 2.0, 3.0]
    assert second == [1.0, 2.0, 3.0]
    assert calls == 1
