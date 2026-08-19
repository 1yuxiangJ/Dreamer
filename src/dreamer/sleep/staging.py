"""Staging snapshot + atomic swap for Sleep agent.

Concurrency model (Letta-paper-inspired, MVP-simplified):

  1. snapshot_to_staging()
     - DROP + CREATE  *_staging  (LIKE main INCLUDING ALL)
     - INSERT INTO   *_staging   SELECT * FROM main
     - Records snapshot_ts.

  2. Sleep agent works only on `*_staging` tables.

  3. atomic_swap()
     - In a single transaction:
         a) Briefly block archival writers while still allowing readers
         b) Move any new archival rows from main → staging
            (created_at > snapshot_ts, skip if already present)
         c) Merge Awake-owned usage fields from main into existing staging rows
         d) Use a three-step RENAME for each main ↔ staging pair
         e) Truncate the now-staging (old main).
     - Awake's next read picks up the new main.

Field ownership during the merge:
  - Sleep semantic fields (content/tags/confidence/etc.) stay in staging.
  - Awake usage fields (use_count/last_used_at) take the freshest value.
  - is_deleted is monotonic: a delete by either agent remains deleted.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from dreamer.config import settings
from dreamer.db.models import MemoryOpsLog
from dreamer.memory.store import MemoryOpDraft

# Tables that participate in staging.
_STAGED_TABLES = ("core_blocks", "archival_facts")
_ARCHIVAL_ID_SEQUENCE = "archival_facts_id_seq"


async def _set_archival_id_default(session: AsyncSession, table_name: str) -> None:
    await session.execute(text(
        f"""
        ALTER TABLE {table_name}
        ALTER COLUMN id SET DEFAULT nextval('{_ARCHIVAL_ID_SEQUENCE}'::regclass)
        """
    ))


async def ensure_archival_id_sequence(session: AsyncSession) -> None:
    """Ensure archival_facts.id has a stable sequence across table swaps.

    `CREATE TABLE ... LIKE ... INCLUDING ALL` plus rename-based swaps can leave
    the live table without the original BIGSERIAL default in older local DBs.
    Sleep must repair that before creating staging, otherwise snapshot fails
    before any ops_log entry can be written.
    """
    await session.execute(text(
        f"CREATE SEQUENCE IF NOT EXISTS {_ARCHIVAL_ID_SEQUENCE}"
    ))
    await session.execute(text(
        f"""
        SELECT setval(
            '{_ARCHIVAL_ID_SEQUENCE}',
            GREATEST((SELECT COALESCE(MAX(id), 0) FROM archival_facts), 1),
            true
        )
        """
    ))
    await _set_archival_id_default(session, "archival_facts")
    await session.execute(text(
        f"ALTER SEQUENCE {_ARCHIVAL_ID_SEQUENCE} OWNED BY archival_facts.id"
    ))


async def snapshot_to_staging(session: AsyncSession) -> datetime:
    """Snapshot the main tables into *_staging counterparts.

    Drops + recreates the staging tables (idempotent).

    Returns the snapshot timestamp (used by atomic_swap to merge newer rows).
    """
    snapshot_ts = datetime.now(UTC)

    await ensure_archival_id_sequence(session)

    for tbl in _STAGED_TABLES:
        staging = f"{tbl}_staging"
        await session.execute(text(f"DROP TABLE IF EXISTS {staging} CASCADE"))
        await session.execute(
            text(f"CREATE TABLE {staging} (LIKE {tbl} INCLUDING ALL)")
        )
        if tbl == "archival_facts":
            await _set_archival_id_default(session, "archival_facts_staging")
        await session.execute(
            text(f"INSERT INTO {staging} SELECT * FROM {tbl}")
        )

    await session.commit()
    return snapshot_ts


async def atomic_swap(
    session: AsyncSession,
    snapshot_ts: datetime,
    pending_ops: list[MemoryOpDraft] | None = None,
) -> None:
    """Atomically swap staging ↔ main.

    Args:
        snapshot_ts: timestamp returned by snapshot_to_staging().
        pending_ops: Sleep audit log rows to flush only if the swap commits.
    """
    # All of this in one transaction.
    await session.execute(
        text("SELECT set_config('lock_timeout', :lock_timeout, true)"),
        {"lock_timeout": f"{settings.sleep_swap_lock_timeout_ms}ms"},
    )

    # Freeze archival writes across merge + swap. Normal SELECTs remain available;
    # ALTER TABLE below briefly upgrades this to ACCESS EXCLUSIVE for the rename.
    await session.execute(text(
        "LOCK TABLE archival_facts IN SHARE ROW EXCLUSIVE MODE"
    ))

    # Step b: merge new archival rows from main → staging (Sleep didn't see them).
    await session.execute(text(
        """
        INSERT INTO archival_facts_staging
            SELECT * FROM archival_facts
            WHERE created_at > :snapshot_ts
        ON CONFLICT (id) DO NOTHING
        """
    ), {"snapshot_ts": snapshot_ts})

    # Step c: preserve Awake changes to rows that existed at snapshot time.
    # Semantic fields intentionally stay untouched in staging because Sleep owns
    # them. Deletion is monotonic, so a forget/demote from either side wins.
    await session.execute(text(
        """
        UPDATE archival_facts_staging AS staging
        SET use_count = GREATEST(staging.use_count, main.use_count),
            last_used_at = CASE
                WHEN staging.last_used_at IS NULL THEN main.last_used_at
                WHEN main.last_used_at IS NULL THEN staging.last_used_at
                ELSE GREATEST(staging.last_used_at, main.last_used_at)
            END,
            is_deleted = staging.is_deleted OR main.is_deleted
        FROM archival_facts AS main
        WHERE staging.id = main.id
          AND (
              main.use_count > staging.use_count
              OR (
                  main.last_used_at IS NOT NULL
                  AND (
                      staging.last_used_at IS NULL
                      OR main.last_used_at > staging.last_used_at
                  )
              )
              OR (main.is_deleted AND NOT staging.is_deleted)
          )
        """
    ))

    # Step d: swap names using a tmp suffix.
    # Use double rename via tmp to ensure atomicity within the transaction.
    for tbl in _STAGED_TABLES:
        staging = f"{tbl}_staging"
        tmp = f"{tbl}_tmp_swap"
        await session.execute(text(f"ALTER TABLE {tbl} RENAME TO {tmp}"))
        await session.execute(text(f"ALTER TABLE {staging} RENAME TO {tbl}"))
        await session.execute(text(f"ALTER TABLE {tmp} RENAME TO {staging}"))

    # Sequence ownership follows the old main table object when it is renamed
    # to staging. Rebind it to the new live main before staging can be dropped.
    await _set_archival_id_default(session, "archival_facts")
    await session.execute(text(
        f"ALTER SEQUENCE {_ARCHIVAL_ID_SEQUENCE} OWNED BY archival_facts.id"
    ))

    # Step e: truncate the now-staging (which was the old main).
    for tbl in _STAGED_TABLES:
        staging = f"{tbl}_staging"
        await session.execute(text(f"TRUNCATE {staging}"))

    if pending_ops:
        session.add_all(MemoryOpsLog(**op) for op in pending_ops)

    await session.commit()


async def cleanup_staging(session: AsyncSession) -> None:
    """Drop staging tables (after successful swap or if cycle aborted)."""
    for tbl in _STAGED_TABLES:
        await session.execute(text(f"DROP TABLE IF EXISTS {tbl}_staging CASCADE"))
    await ensure_archival_id_sequence(session)
    await session.commit()
