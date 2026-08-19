"""CLI helpers for inspecting durable memory write jobs."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from sqlalchemy import desc, select

from dreamer.db.models import MemoryWriteJob, get_sessionmaker


def _dt(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


async def snapshot(limit: int | None = 20) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        stmt = select(MemoryWriteJob).order_by(desc(MemoryWriteJob.id))
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = (
            await session.execute(stmt)
        ).scalars()
        jobs = [
            {
                "id": job.id,
                "operation": job.operation,
                "status": job.status,
                "attempt_count": job.attempt_count,
                "max_attempts": job.max_attempts,
                "available_at": _dt(job.available_at),
                "locked_at": _dt(job.locked_at),
                "completed_at": _dt(job.completed_at),
                "last_error": job.last_error,
                "payload": job.payload,
                "created_at": _dt(job.created_at),
                "updated_at": _dt(job.updated_at),
            }
            for job in rows
        ]
    return {"status": "ok", "limit": limit, "jobs": jobs}


async def run(limit: int) -> int:
    print(json.dumps(await snapshot(limit), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect durable Dreamer memory write jobs.",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    return asyncio.run(run(args.limit))
