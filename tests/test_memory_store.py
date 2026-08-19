"""Integration tests for dreamer.memory.store.

Requires PG + pgvector + mneme_test database. Run with:

    pytest --run-integration -m integration

Day 03+: flesh out fixtures + assertions when environment is available.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from dreamer.db.models import ArchivalFact, CoreBlock, MemoryOpsLog
from dreamer.memory import store

# Mark the whole module as integration — skipped by default.
pytestmark = pytest.mark.integration


def _embedding_for(text: str) -> list[float]:
    vec = [0.0] * 1024
    lowered = text.lower()
    if "java" in lowered:
        vec[1] = 1.0
    else:
        vec[0] = 1.0
    return vec


@pytest.fixture
def fake_embedding(monkeypatch):
    async def fake_embed_text(text: str) -> list[float]:
        return _embedding_for(text)

    monkeypatch.setattr(store, "embed_text", fake_embed_text)


@pytest.mark.asyncio
async def test_insert_and_search_archival(integration_session, fake_embedding):
    """Insert a fact, semantic-search for it, expect a hit with low distance."""
    fact_id = await store.insert_archival(
        integration_session,
        content="User prefers FastAPI for Python backend services.",
        tags=["preference", "backend"],
        confidence=3,
        source="test",
        actor="awake_agent",
        reason="integration test",
    )

    results = await store.semantic_search_archival(
        integration_session,
        query="FastAPI backend framework",
    )

    assert [r.id for r in results] == [fact_id]
    assert results[0].content == "User prefers FastAPI for Python backend services."
    assert results[0].distance == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_awake_cannot_write_core(integration_session):
    """write_core_block(actor='awake_agent', ...) must raise PermissionError
    and log policy_violation. This is the most important security guarantee.
    """
    with pytest.raises(PermissionError):
        await store.write_core_block(
            integration_session,
            label="background",
            new_value="Awake should not be able to write this.",
            actor="awake_agent",
            reason="negative permission test",
        )

    core = await integration_session.get(CoreBlock, "background")
    logs = (
        await integration_session.execute(
            select(MemoryOpsLog).where(MemoryOpsLog.op_type == "policy_violation")
        )
    ).scalars().all()

    assert core is not None
    assert core.value == ""
    assert len(logs) == 1
    assert logs[0].actor == "awake_agent"
    assert logs[0].target_kind == "core"
    assert logs[0].target_id == "background"


@pytest.mark.asyncio
async def test_sleep_can_write_core(integration_session):
    """write_core_block(actor='sleep_agent', ...) updates the block and logs."""
    await store.write_core_block(
        integration_session,
        label="skills",
        new_value="Java backend, PostgreSQL, async Python.",
        actor="sleep_agent",
        reason="promote stable skill summary",
    )

    core = await integration_session.get(CoreBlock, "skills")
    logs = (
        await integration_session.execute(
            select(MemoryOpsLog).where(MemoryOpsLog.op_type == "sleep_promote")
        )
    ).scalars().all()

    assert core is not None
    assert core.value == "Java backend, PostgreSQL, async Python."
    assert core.version == 2
    assert core.last_writer == "sleep_agent"
    assert len(logs) == 1
    assert logs[0].before_value == ""
    assert logs[0].after_value == "Java backend, PostgreSQL, async Python."


@pytest.mark.asyncio
async def test_soft_delete_archival(integration_session, fake_embedding):
    """Forget marks is_deleted=True and excludes from future search."""
    fact_id = await store.insert_archival(
        integration_session,
        content="User is learning vector database indexing.",
        tags=["study"],
        confidence=2,
        source="test",
        actor="awake_agent",
    )

    outcome = await store.soft_delete_archival(
        integration_session,
        fact_id=fact_id,
        reason="user asked to forget",
        actor="awake_agent",
    )

    fact = await integration_session.get(ArchivalFact, fact_id)
    results = await store.semantic_search_archival(
        integration_session,
        query="vector database indexing",
    )

    assert outcome == "deleted"
    assert fact is not None
    assert fact.is_deleted is True
    assert results == []

    second_outcome = await store.soft_delete_archival(
        integration_session,
        fact_id=fact_id,
        reason="duplicate retry",
        actor="awake_agent",
    )
    missing_outcome = await store.soft_delete_archival(
        integration_session,
        fact_id=fact_id + 100_000,
        reason="missing fact",
        actor="awake_agent",
    )
    forget_logs = (
        await integration_session.execute(
            select(MemoryOpsLog).where(MemoryOpsLog.op_type == "forget")
        )
    ).scalars().all()

    assert second_outcome == "already_deleted"
    assert missing_outcome == "not_found"
    assert len(forget_logs) == 1


@pytest.mark.asyncio
async def test_mark_archival_used_no_log(integration_session, fake_embedding):
    """Marking facts as used updates counters without writing memory_ops_log."""
    fact_id = await store.insert_archival(
        integration_session,
        content="User wants concise engineering explanations.",
        tags=["preference"],
        confidence=3,
        source="test",
        actor="awake_agent",
    )
    log_count_before = (
        await integration_session.execute(select(MemoryOpsLog))
    ).scalars().all()

    await store.mark_archival_used(integration_session, [fact_id])

    fact = await integration_session.get(ArchivalFact, fact_id)
    log_count_after = (
        await integration_session.execute(select(MemoryOpsLog))
    ).scalars().all()

    assert fact is not None
    assert fact.use_count == 1
    assert fact.last_used_at is not None
    assert len(log_count_after) == len(log_count_before)
