from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any

import pytest

from dreamer.awake import tools


class _SessionContext(AbstractAsyncContextManager[object]):
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


def _sessionmaker() -> Any:
    return lambda: _SessionContext()


@pytest.mark.asyncio
async def test_duplicate_search_does_not_increment_usage(monkeypatch):
    async def fake_search(_session: object, _query: str, limit: int):
        assert limit == 5
        return [SimpleNamespace(
            id=7,
            content="Existing fact.",
            tags=["test"],
            confidence=3,
            stability="long_term",
            salience=3,
            distance=0.01,
        )]

    monkeypatch.setattr(tools, "session_factory", _sessionmaker)
    monkeypatch.setattr(tools, "semantic_search_archival", fake_search)
    result = await tools.find_archival_duplicates.ainvoke({"query": "Existing fact."})

    assert result["results"][0]["id"] == 7


@pytest.mark.asyncio
async def test_recall_search_only_returns_candidates(monkeypatch):
    async def fake_search(_session: object, _query: str, limit: int):
        return [SimpleNamespace(
            id=8,
            content="Relevant fact.",
            tags=["test"],
            confidence=3,
            stability="long_term",
            salience=2,
            distance=0.05,
        )]

    monkeypatch.setattr(tools, "session_factory", _sessionmaker)
    monkeypatch.setattr(tools, "semantic_search_archival", fake_search)

    result = await tools.search_archival.ainvoke({"query": "Relevant fact.", "limit": 3})

    assert result["results"][0]["id"] == 8
