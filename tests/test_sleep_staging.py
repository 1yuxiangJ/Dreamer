"""Tests for sleep.staging snapshot + atomic swap.

Requires PG + pgvector + mneme_test database. Run with:

    pytest --run-integration -m integration
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from dreamer.sleep.staging import atomic_swap, cleanup_staging, snapshot_to_staging
from dreamer.sleep.tools import (
    find_consolidation_clusters,
    get_promote_candidates,
    get_stale_candidates,
    summarize_state,
)

pytestmark = pytest.mark.integration


def _vector_literal(axis: int = 0) -> str:
    vec = [0.0] * 1024
    vec[axis] = 1.0
    return "[" + ",".join(str(v) for v in vec) + "]"


async def _insert_archival(session, content: str, axis: int = 0) -> int:
    return int((await session.execute(text(
        """
        INSERT INTO archival_facts (content, tags, confidence, source, embedding)
        VALUES (:content, ARRAY['test'], 3, 'test', CAST(:embedding AS vector))
        RETURNING id
        """
    ), {"content": content, "embedding": _vector_literal(axis)})).scalar_one())


async def _insert_signal_archival(
    session,
    content: str,
    confidence: int,
    stability: str,
    salience: int,
    use_count: int,
) -> int:
    return int((await session.execute(text(
        """
        INSERT INTO archival_facts (
            content, tags, confidence, stability, salience, source, embedding,
            use_count
        )
        VALUES (
            :content, ARRAY['test'], :confidence, :stability, :salience, 'test',
            CAST(:embedding AS vector), :use_count
        )
        RETURNING id
        """
    ), {
        "content": content,
        "confidence": confidence,
        "stability": stability,
        "salience": salience,
        "use_count": use_count,
        "embedding": _vector_literal(),
    })).scalar_one())


async def _count_table(session, table: str) -> int:
    return int((await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one())


@pytest.mark.asyncio
async def test_snapshot_creates_staging_tables(integration_session):
    """After snapshot_to_staging, *_staging tables exist with same rows as main."""
    await _insert_archival(integration_session, "Fact copied into staging.")
    await integration_session.commit()

    await snapshot_to_staging(integration_session)

    core_staging_exists = (await integration_session.execute(
        text("SELECT to_regclass('public.core_blocks_staging')")
    )).scalar_one()
    archival_staging_exists = (await integration_session.execute(
        text("SELECT to_regclass('public.archival_facts_staging')")
    )).scalar_one()

    assert core_staging_exists == "core_blocks_staging"
    assert archival_staging_exists == "archival_facts_staging"
    assert await _count_table(integration_session, "core_blocks_staging") == 5
    assert await _count_table(integration_session, "archival_facts_staging") == 1


@pytest.mark.asyncio
async def test_find_consolidation_clusters_uses_pgvector_distance(integration_session):
    """Sleep consolidation can compare vector params inside raw SQL."""
    first_id = await _insert_archival(
        integration_session,
        "User prefers concise technical answers.",
        axis=0,
    )
    second_id = await _insert_archival(
        integration_session,
        "User likes direct engineering explanations.",
        axis=0,
    )
    await integration_session.commit()
    await snapshot_to_staging(integration_session)

    clusters = await find_consolidation_clusters(
        integration_session,
        distance_threshold=0.01,
    )

    assert len(clusters) == 1
    assert {item["id"] for item in clusters[0]} == {first_id, second_id}


@pytest.mark.asyncio
async def test_promote_candidates_require_long_term_salient_explicit_memory(
    integration_session,
):
    """Promotion candidates must be explicit, durable, highly salient, and frequently used."""
    stable_id = await _insert_signal_archival(
        integration_session,
        "User prefers direct engineering explanations.",
        confidence=3,
        stability="long_term",
        salience=3,
        use_count=6,
    )
    await _insert_signal_archival(
        integration_session,
        "User likes a specific restaurant's fries.",
        confidence=3,
        stability="long_term",
        salience=2,
        use_count=12,
    )
    await _insert_signal_archival(
        integration_session,
        "User currently mainly plays CS2.",
        confidence=3,
        stability="stage",
        salience=2,
        use_count=9,
    )
    await _insert_signal_archival(
        integration_session,
        "User's local Dreamer path is /Users/mac/dream.",
        confidence=3,
        stability="long_term",
        salience=1,
        use_count=9,
    )
    await integration_session.commit()
    await snapshot_to_staging(integration_session)

    candidates = await get_promote_candidates(integration_session)

    assert [item["id"] for item in candidates] == [stable_id]
    assert candidates[0]["stability"] == "long_term"
    assert candidates[0]["salience"] == 3


@pytest.mark.asyncio
async def test_demote_staleness_uses_creation_time_for_never_used_facts(
    integration_session,
):
    """Never-used facts receive the same age window as previously used facts."""
    rows = (await integration_session.execute(text(
        """
        INSERT INTO archival_facts (
            content, tags, confidence, stability, salience, source,
            created_at, last_used_at
        ) VALUES
            (
                'New and never used.', ARRAY['test'], 1, 'temporary', 1, 'test',
                now() - INTERVAL '1 day', NULL
            ),
            (
                'Old and never used.', ARRAY['test'], 1, 'temporary', 1, 'test',
                now() - INTERVAL '100 days', NULL
            ),
            (
                'Old but recently used.', ARRAY['test'], 1, 'temporary', 1, 'test',
                now() - INTERVAL '100 days', now() - INTERVAL '1 day'
            ),
            (
                'Old and last used long ago.', ARRAY['test'], 1, 'temporary', 1, 'test',
                now() - INTERVAL '200 days', now() - INTERVAL '100 days'
            ),
            (
                'Old but high signal.', ARRAY['test'], 3, 'long_term', 3, 'test',
                now() - INTERVAL '100 days', NULL
            )
        RETURNING id, content
        """
    ))).all()
    ids = {row.content: row.id for row in rows}
    await integration_session.commit()

    summary = await summarize_state(integration_session, last_cycle_ts=None)
    await snapshot_to_staging(integration_session)
    candidates = await get_stale_candidates(integration_session)

    assert summary.old_unused_count == 3
    assert {item["id"] for item in candidates} == {
        ids["Old and never used."],
        ids["Old and last used long ago."],
    }


@pytest.mark.asyncio
async def test_atomic_swap_replaces_main(integration_session):
    """After swap, queries against main return rows that were written to staging."""
    snapshot_ts = await snapshot_to_staging(integration_session)
    await integration_session.execute(text(
        """
        UPDATE core_blocks_staging
        SET value = 'Sleep rewrote this block.',
            version = version + 1,
            last_writer = 'sleep_agent'
        WHERE label = 'background'
        """
    ))
    await integration_session.commit()

    await atomic_swap(integration_session, snapshot_ts)

    value = (await integration_session.execute(text(
        "SELECT value FROM core_blocks WHERE label = 'background'"
    ))).scalar_one()

    assert value == "Sleep rewrote this block."
    assert await _count_table(integration_session, "core_blocks_staging") == 0


@pytest.mark.asyncio
async def test_atomic_swap_merges_new_archival_during_cycle(integration_session):
    """Archival rows inserted by Awake during the cycle (created_at > snapshot_ts)
    are merged into the new main after swap."""
    snapshot_ts = await snapshot_to_staging(integration_session)
    inserted_id = await _insert_archival(
        integration_session,
        "Awake inserted this while Sleep was working.",
    )
    await integration_session.commit()

    await atomic_swap(integration_session, snapshot_ts)

    content = (await integration_session.execute(text(
        "SELECT content FROM archival_facts WHERE id = :id"
    ), {"id": inserted_id})).scalar_one_or_none()

    assert content == "Awake inserted this while Sleep was working."


@pytest.mark.asyncio
async def test_atomic_swap_merges_existing_archival_fields_by_owner(
    integration_session,
):
    """Sleep semantic edits and Awake usage/deletion updates must both survive."""
    recalled_id = await _insert_signal_archival(
        integration_session,
        "Original wording.",
        confidence=3,
        stability="long_term",
        salience=3,
        use_count=5,
    )
    forgotten_id = await _insert_signal_archival(
        integration_session,
        "Awake will forget this during Sleep.",
        confidence=3,
        stability="long_term",
        salience=2,
        use_count=1,
    )
    await integration_session.commit()

    snapshot_ts = await snapshot_to_staging(integration_session)

    # Sleep owns semantic consolidation and demotion in staging.
    await integration_session.execute(text(
        """
        UPDATE archival_facts_staging
        SET content = 'Sleep consolidated wording.', is_deleted = TRUE
        WHERE id = :id
        """
    ), {"id": recalled_id})

    # Awake owns live usage signals and can forget an existing main-table fact.
    await integration_session.execute(text(
        """
        UPDATE archival_facts
        SET use_count = use_count + 1,
            last_used_at = TIMESTAMPTZ '2026-07-11 12:00:00+08'
        WHERE id = :id
        """
    ), {"id": recalled_id})
    await integration_session.execute(text(
        "UPDATE archival_facts SET is_deleted = TRUE WHERE id = :id"
    ), {"id": forgotten_id})
    await integration_session.commit()

    await atomic_swap(integration_session, snapshot_ts)

    recalled = (await integration_session.execute(text(
        """
        SELECT content, use_count, last_used_at, is_deleted
        FROM archival_facts WHERE id = :id
        """
    ), {"id": recalled_id})).one()
    forgotten = (await integration_session.execute(text(
        "SELECT is_deleted FROM archival_facts WHERE id = :id"
    ), {"id": forgotten_id})).scalar_one()

    assert recalled.content == "Sleep consolidated wording."
    assert recalled.use_count == 6
    assert recalled.last_used_at is not None
    assert recalled.is_deleted is True
    assert forgotten is True


@pytest.mark.asyncio
async def test_sleep_logs_are_pending_until_swap_commits(integration_session):
    """Sleep phase logs should only enter memory_ops_log after swap succeeds."""
    from dreamer.sleep.tools import apply_resolutions

    snapshot_ts = await snapshot_to_staging(integration_session)

    pending_ops = await apply_resolutions(integration_session, [{
        "fix_block": "preferences",
        "new_block_value": "User prefers direct, concrete engineering explanations.",
        "reason": "remove contradiction with habits block",
    }])

    before_swap = (await integration_session.execute(text(
        "SELECT count(*) FROM memory_ops_log"
    ))).scalar_one()

    await atomic_swap(integration_session, snapshot_ts, pending_ops=pending_ops)

    op_type = (await integration_session.execute(text(
        "SELECT op_type FROM memory_ops_log WHERE target_id = 'preferences'"
    ))).scalar_one()

    assert before_swap == 0
    assert op_type == "sleep_resolve"


@pytest.mark.asyncio
async def test_snapshot_repairs_missing_archival_id_sequence(integration_session):
    """Sleep should recover if a previous swap left archival id default missing."""
    await integration_session.execute(text(
        "ALTER TABLE archival_facts ALTER COLUMN id DROP DEFAULT"
    ))
    await integration_session.execute(text(
        "DROP SEQUENCE IF EXISTS archival_facts_id_seq CASCADE"
    ))
    await integration_session.execute(text(
        """
        INSERT INTO archival_facts (id, content, tags, confidence, source, embedding)
        VALUES (42, 'Existing fact with explicit id.', ARRAY['test'], 3, 'test',
                CAST(:embedding AS vector))
        """
    ), {"embedding": _vector_literal()})
    await integration_session.commit()

    snapshot_ts = await snapshot_to_staging(integration_session)
    await atomic_swap(integration_session, snapshot_ts)

    new_id = await _insert_archival(
        integration_session,
        "Insert after repaired swap should get generated id.",
    )

    assert new_id > 42


@pytest.mark.asyncio
async def test_cleanup_staging_drops_tables(integration_session):
    """After cleanup, *_staging tables no longer exist."""
    await snapshot_to_staging(integration_session)

    await cleanup_staging(integration_session)

    core_staging_exists = (await integration_session.execute(
        text("SELECT to_regclass('public.core_blocks_staging')")
    )).scalar_one()
    archival_staging_exists = (await integration_session.execute(
        text("SELECT to_regclass('public.archival_facts_staging')")
    )).scalar_one()

    assert core_staging_exists is None
    assert archival_staging_exists is None


@pytest.mark.asyncio
async def test_sequence_survives_cleanup_after_atomic_swap(integration_session):
    """Dropping old-main staging after swap must not delete the live ID sequence."""
    await _insert_archival(integration_session, "Fact before swap.")
    await integration_session.commit()
    snapshot_ts = await snapshot_to_staging(integration_session)
    await atomic_swap(integration_session, snapshot_ts)

    await cleanup_staging(integration_session)
    inserted_id = await _insert_archival(integration_session, "Fact after cleanup.")

    assert inserted_id >= 2
