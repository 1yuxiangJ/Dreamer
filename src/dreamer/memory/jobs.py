"""Durable PostgreSQL-backed queue for write-side memory jobs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dreamer.config import settings
from dreamer.db.models import MemoryWriteJob, get_sessionmaker

_RETRY_BACKOFF_SECONDS = (5, 30, 120)


@dataclass(frozen=True)
class EnqueuedMemoryWriteJob:
    id: int
    status: str
    dedupe_key: str


@dataclass(frozen=True)
class MemoryWriteJobSnapshot:
    id: int
    operation: str
    command: str
    payload: dict[str, Any]
    attempt_count: int


def make_dedupe_key(operation: str, command: str) -> str:
    raw = f"{operation}\0{command}".encode()
    return hashlib.sha256(raw).hexdigest()


async def enqueue_awake_write(
    operation: str,
    command: str,
    payload: dict[str, Any] | None = None,
) -> EnqueuedMemoryWriteJob:
    """Persist a write job before returning accepted to the MCP caller."""
    dedupe_key = make_dedupe_key(operation, command)
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        stmt = (
            pg_insert(MemoryWriteJob)
            .values(
                operation=operation,
                command=command,
                payload=payload or {},
                dedupe_key=dedupe_key,
                status="pending",
                max_attempts=settings.memory_write_worker_max_attempts,
            )
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
            .returning(
                MemoryWriteJob.id,
                MemoryWriteJob.status,
                MemoryWriteJob.dedupe_key,
            )
        )
        inserted = (await session.execute(stmt)).one_or_none()
        if inserted is not None:
            await session.commit()
            return EnqueuedMemoryWriteJob(
                id=int(inserted.id),
                status=str(inserted.status),
                dedupe_key=str(inserted.dedupe_key),
            )

        res = await session.execute(
            select(
                MemoryWriteJob.id,
                MemoryWriteJob.status,
                MemoryWriteJob.dedupe_key,
            ).where(MemoryWriteJob.dedupe_key == dedupe_key)
        )
        existing = res.one()
        await session.commit()
        return EnqueuedMemoryWriteJob(
            id=int(existing.id),
            status=str(existing.status),
            dedupe_key=str(existing.dedupe_key),
        )


def _snapshot(job: MemoryWriteJob) -> MemoryWriteJobSnapshot:
    return MemoryWriteJobSnapshot(
        id=job.id,
        operation=job.operation,
        command=job.command,
        payload=dict(job.payload or {}),
        attempt_count=job.attempt_count,
    )


async def requeue_stale_running_jobs() -> int:
    """Return jobs left running by a crashed worker back to pending."""
    stale_before = datetime.now(UTC) - timedelta(
        seconds=settings.memory_write_job_stale_seconds
    )
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        stmt = (
            update(MemoryWriteJob)
            .where(MemoryWriteJob.status == "running")
            .where(MemoryWriteJob.locked_at < stale_before)
            .values(
                status="pending",
                locked_at=None,
                updated_at=datetime.now(UTC),
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return int(getattr(result, "rowcount", 0) or 0)


async def claim_next_write_job() -> MemoryWriteJobSnapshot | None:
    """Claim one available pending job for this worker process."""
    session_maker = get_sessionmaker()
    now = datetime.now(UTC)
    async with session_maker() as session, session.begin():
        res = await session.execute(
            select(MemoryWriteJob)
            .where(MemoryWriteJob.status == "pending")
            .where(
                or_(
                    MemoryWriteJob.available_at.is_(None),
                    MemoryWriteJob.available_at <= now,
                )
            )
            .order_by(MemoryWriteJob.created_at, MemoryWriteJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = res.scalar_one_or_none()
        if job is None:
            return None
        job.status = "running"
        job.attempt_count += 1
        job.locked_at = now
        job.updated_at = now
        return _snapshot(job)


async def mark_write_job_succeeded(
    job_id: int,
    result: dict[str, Any],
) -> None:
    session_maker = get_sessionmaker()
    now = datetime.now(UTC)
    async with session_maker() as session:
        job = await session.get(MemoryWriteJob, job_id)
        if job is None:
            return
        job.status = "succeeded"
        job.completed_at = now
        job.locked_at = None
        job.last_error = None
        job.result = result
        job.updated_at = now
        await session.commit()


async def mark_write_job_failed(job_id: int, error: str) -> None:
    session_maker = get_sessionmaker()
    now = datetime.now(UTC)
    async with session_maker() as session:
        job = await session.get(MemoryWriteJob, job_id)
        if job is None:
            return
        clean_error = error[:2000]
        if job.attempt_count >= job.max_attempts:
            job.status = "failed"
            job.available_at = now
        else:
            backoff_index = max(0, min(job.attempt_count - 1, len(_RETRY_BACKOFF_SECONDS) - 1))
            job.status = "pending"
            job.available_at = now + timedelta(
                seconds=_RETRY_BACKOFF_SECONDS[backoff_index]
            )
        job.locked_at = None
        job.last_error = clean_error
        job.updated_at = now
        await session.commit()


def result_to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize Awake output before storing it in JSONB."""
    parsed = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    return cast(dict[str, Any], parsed)
