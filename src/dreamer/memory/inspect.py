"""Read-only memory inspection helpers for local debugging."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from dreamer.db.models import ArchivalFact, CoreBlock, MemoryOpsLog, get_sessionmaker


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _preview(value: str | None, max_chars: int = 240) -> str | None:
    if value is None or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


async def collect_snapshot(
    limit: int = 10,
    include_deleted: bool = False,
) -> dict[str, Any]:
    """Collect a small read-only snapshot of current memory state."""
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        core_rows = (
            await session.execute(select(CoreBlock).order_by(CoreBlock.label))
        ).scalars()
        archival_stmt = select(ArchivalFact).order_by(ArchivalFact.id.desc()).limit(limit)
        if not include_deleted:
            archival_stmt = archival_stmt.where(ArchivalFact.is_deleted.is_(False))
        archival_rows = (await session.execute(archival_stmt)).scalars()

        active_count = int((
            await session.execute(
                select(func.count(ArchivalFact.id)).where(
                    ArchivalFact.is_deleted.is_(False)
                )
            )
        ).scalar_one())
        deleted_count = int((
            await session.execute(
                select(func.count(ArchivalFact.id)).where(
                    ArchivalFact.is_deleted.is_(True)
                )
            )
        ).scalar_one())
        op_rows = (
            await session.execute(
                select(MemoryOpsLog)
                .order_by(MemoryOpsLog.ts.desc(), MemoryOpsLog.id.desc())
                .limit(limit)
            )
        ).scalars()

        return {
            "counts": {
                "active_archival_facts": active_count,
                "deleted_archival_facts": deleted_count,
            },
            "core_blocks": [
                {
                    "label": row.label,
                    "value": row.value,
                    "version": row.version,
                    "last_writer": row.last_writer,
                    "updated_at": _dt(row.updated_at),
                }
                for row in core_rows
            ],
            "archival_facts": [
                {
                    "id": row.id,
                    "content": row.content,
                    "tags": row.tags or [],
                    "confidence": row.confidence,
                    "stability": row.stability,
                    "salience": row.salience,
                    "source": row.source,
                    "is_deleted": row.is_deleted,
                    "use_count": row.use_count,
                    "created_at": _dt(row.created_at),
                    "last_used_at": _dt(row.last_used_at),
                }
                for row in archival_rows
            ],
            "recent_ops": [
                {
                    "id": row.id,
                    "op_type": row.op_type,
                    "actor": row.actor,
                    "target_kind": row.target_kind,
                    "target_id": row.target_id,
                    "before_value_preview": _preview(row.before_value),
                    "after_value_preview": _preview(row.after_value),
                    "reason": row.reason,
                    "ts": _dt(row.ts),
                }
                for row in op_rows
            ],
        }


def format_snapshot(snapshot: dict[str, Any]) -> str:
    """Format a snapshot as stable JSON for terminal output."""
    return json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)


async def run(limit: int = 10, include_deleted: bool = False) -> int:
    snapshot = await collect_snapshot(limit=limit, include_deleted=include_deleted)
    print(format_snapshot(snapshot))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Dreamer memory tables.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum archival facts and recent ops to print.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include soft-deleted archival facts in the archival_facts output.",
    )
    args = parser.parse_args()
    return asyncio.run(run(limit=args.limit, include_deleted=args.include_deleted))
