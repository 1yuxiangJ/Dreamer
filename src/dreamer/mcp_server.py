"""MCP server: 4 tools exposed via streamable-http transport.

`remember` uses the durable queue because deduplication, embedding, and Awake
ReAct can be slow. `recall` awaits Awake because the caller needs its semantic
result. `list_memory` and `forget` use deterministic direct-DB paths so they do
not depend on the LLM provider.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from dreamer.awake.agent import run_awake as _run_awake
from dreamer.awake.agent import run_recall as _run_recall
from dreamer.config import settings
from dreamer.db.models import get_sessionmaker
from dreamer.memory.jobs import enqueue_awake_write
from dreamer.memory.store import (
    get_memory_overview,
    list_archival_facts,
    soft_delete_archival,
)
from dreamer.sleep.scheduler import mark_awake_activity

logger = logging.getLogger("dreamer.mcp")


async def run_awake(command: str) -> dict[str, Any]:
    """Wrap Awake invocation to mark activity for the Sleep idle scheduler.

    Every MCP tool entry goes through this so idle detection works correctly.
    """
    mark_awake_activity()
    return await _run_awake(command)


async def run_recall(command: str) -> dict[str, Any]:
    """Run semantic Recall while participating in idle-time tracking."""
    mark_awake_activity()
    return await _run_recall(command)


async def _enqueue_awake_write(
    command: str,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist write-side Awake work and return immediately.

    Remember is eventually consistent but durable: accepted means the job has
    been committed to PostgreSQL and can survive process restart before the
    background worker executes it.
    """
    mark_awake_activity()
    job = await enqueue_awake_write(operation, command, payload)
    return {
        "status": "accepted",
        "mode": "durable_async",
        "operation": operation,
        "job_id": job.id,
        "job_status": job.status,
        "message": (
            f"{operation} request accepted; durable job {job.id} will be "
            "processed in the background."
        ),
    }


mcp = FastMCP(
    "dreamer",
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
)


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@mcp.tool()
async def remember(
    content: str,
    tags: list[str] | None = None,
    confidence: int = 2,
    stability: str = "long_term",
    salience: int = 2,
) -> dict[str, Any]:
    """Store a fact about the user.

    Call for long-term, cross-conversation user facts: identity, goals,
    skills, communication preferences, work/study habits, lifestyle habits,
    hobbies, entertainment preferences, relaxation patterns, product tastes,
    and stable likes/dislikes.

    Good lifestyle examples:
    - User likes football.
    - User plays CS2 and often relaxes with games.
    - User watches Bilibili/Douyin as a common leisure habit.
    - User used to exercise regularly and values fitness.

    Do NOT call for temporary state, one-off events, today's plan, short-term
    mood, or project-specific facts. If a fact sounds recent/temporary, ask a
    follow-up and only store it when the user confirms it is a stable pattern.
    Project-specific facts (architecture, library choices, project conventions)
    belong in CLAUDE.md or Claude Code's per-project auto memory instead.

    Memory signal policy:
    - confidence = factual certainty: 3=user explicitly said it,
      2=reasonable but not fully confirmed, 1=inferred/tentative.
- stability = time horizon: "long_term", "stage" (stage-specific), or
  "temporary".
    - salience = future usefulness: 3=strongly affects future collaboration,
      2=useful in related contexts, 1=minor/passive reference.

    If a user message mixes stable long-term facts with temporary details,
    split them into separate memories with different stability/salience values,
    or skip the temporary detail. Do not package the whole message as a single
    high-salience long_term memory.

    Args:
        content: The fact about the user.
        tags: Topical tags, e.g. ["preference", "code-style", "hobby",
            "lifestyle", "entertainment"].
        confidence: 1=tentative, 2=partly confirmed, 3=explicitly stated.
        stability: "long_term", "stage", or "temporary".
        salience: 1=low, 2=medium, 3=high future usefulness.
    """
    tag_str = ", ".join(tags) if tags else "(none)"
    cmd = (
        "remember this fact about the user:\n"
        f"  content: {content}\n"
        f"  tags: {tag_str}\n"
        f"  confidence: {confidence}\n"
        f"  stability: {stability}\n"
        f"  salience: {salience}\n"
        "First check for near-duplicates via find_archival_duplicates, then insert."
    )
    return await _enqueue_awake_write(
        cmd,
        "remember",
        {
            "content": content,
            "tags": tags or [],
            "confidence": confidence,
            "stability": stability,
            "salience": salience,
        },
    )


@mcp.tool()
async def recall(query: str, limit: int = 5) -> dict[str, Any]:
    """Semantic search over the user's memory.

    Returns matching archival facts + core block context.

    Args:
        query: Natural-language query (e.g. "what are my code-style preferences?").
        limit: Max archival results (default 5).
    """
    cmd = (
        "recall: search the user's memory for facts relevant to:\n"
        f"  query: {query}\n"
        f"  limit: {limit}\n"
        "Include relevant core block context. Your final response must be one "
        "JSON object with keys answer, core_blocks, archival, and used_fact_ids. "
        "used_fact_ids must contain only IDs of archival facts that you actually "
        "used in answer, not every candidate returned by search."
    )
    return await run_recall(cmd)


@mcp.tool()
async def list_memory() -> dict[str, Any]:
    """Overview of who the user is (all core blocks + archival total).

    Recommended at the start of a new conversation.
    """
    mark_awake_activity()
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        overview = await get_memory_overview(session)
        archival_facts = await list_archival_facts(session, limit=20)

    return {
        "status": "ok",
        "mode": "direct_db",
        "core_blocks": [
            {
                "label": block.label,
                "value": block.value,
                "value_preview": block.value[:240],
                "char_limit": block.char_limit,
                "version": block.version,
                "updated_at": _dt(block.updated_at),
            }
            for block in overview.core_blocks
        ],
        "archival_total": overview.archival_count,
        "archival_facts_limit": 20,
        "archival_facts": [
            {
                "id": fact.id,
                "content": fact.content,
                "tags": fact.tags,
                "confidence": fact.confidence,
                "stability": fact.stability,
                "salience": fact.salience,
                "source": fact.source,
                "created_at": _dt(fact.created_at),
                "last_used_at": _dt(fact.last_used_at),
                "use_count": fact.use_count,
            }
            for fact in archival_facts
        ],
    }


@mcp.tool()
async def forget(fact_id: int, reason: str) -> dict[str, Any]:
    """Soft-delete a fact discovered to be wrong/outdated.

    Args:
        fact_id: ID from a prior `recall` result.
        reason: Why we're forgetting it.
    """
    mark_awake_activity()
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        outcome = await soft_delete_archival(
            session,
            fact_id=fact_id,
            reason=reason,
            actor="awake_agent",
        )

    return {
        "status": "ok" if outcome != "not_found" else "not_found",
        "mode": "direct_db",
        "operation": "forget",
        "fact_id": fact_id,
        "outcome": outcome,
        "message": {
            "deleted": f"archival fact {fact_id} was forgotten",
            "already_deleted": f"archival fact {fact_id} was already forgotten",
            "not_found": f"archival fact {fact_id} was not found",
        }[outcome],
    }
