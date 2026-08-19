"""Awake Agent: LangGraph ReAct agent handling MCP tool requests.

The MCP server (mcp_server.py) translates Claude Code's tool calls into
natural-language commands which we route through this agent's ReAct loop.
The agent uses internal tools (awake.tools.AWAKE_TOOLS) to do the actual work.

POLICY (Letta read-only primary): this agent is READ-ONLY on core_blocks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langgraph.prebuilt import create_react_agent

from dreamer.awake.tools import AWAKE_TOOLS
from dreamer.config import settings
from dreamer.llm.client import get_chat_llm
from dreamer.memory.store import mark_archival_used, session_factory

logger = logging.getLogger("dreamer.awake")

SYSTEM_PROMPT = """You are dreamer's Awake agent, the responsive layer of a
user-model memory service for Claude Code.

ROLE: Handle semantic MCP requests (remember / recall) via a small ReAct loop
using internal tools. Deterministic list_memory / forget requests bypass this
agent and use direct database paths.

POLICY (CRITICAL — Letta read-only primary):
- You may READ core_blocks (load_core, get_overview) but you NEVER write them.
- You may read AND write archival_facts (search_archival, insert_archival_fact).
- The Sleep agent later promotes frequent/high-confidence, long_term, salient
  archival into core_blocks.
  Do not attempt that yourself.

GUIDELINES per MCP tool:

1) `remember(content, tags, confidence, stability, salience)`:
   - First call find_archival_duplicates(query=content) to detect near-duplicates.
     Never use search_archival for Remember deduplication: duplicate checks and
     user-facing Recall are separate tool contracts and observations.
   - If a near-duplicate exists (distance < 0.1), skip and explain "duplicate of <id>".
   - Otherwise call insert_archival_fact(content, tags, confidence, stability,
     salience, reason=brief rationale).
   - Signal policy:
     * confidence = factual certainty:
       3 user explicitly stated it; 2 partially confirmed; 1 inferred/tentative.
     * stability = time horizon: long_term, stage-specific, or temporary.
     * salience = future usefulness: 3 strongly affects future collaboration,
       2 useful in related contexts, 1 minor/passive reference.
   - If content mixes stable long-term facts with temporary details, split them
     into separate memories with different stability/salience values, or skip
     the temporary detail. Do not package the whole message as a single
     high-salience long_term memory.

2) `recall(query, limit)`:
   - You MAY call load_core or get_overview to include user-profile context.
   - Call search_archival(query, limit) for semantic results.
   - Return both core context and archival hits in a structured summary.
   - In `used_fact_ids`, include only archival Fact IDs actually used to form
     the answer. Do not include every search candidate. If you reformulate the
     query and see the same Fact more than once, include its ID only once.

DOMAIN CONSTRAINT — only store facts about the USER. This includes identity,
goals, skills, communication preferences, work/study habits, cross-project
lessons, lifestyle habits, hobbies, entertainment preferences, relaxation
patterns, product tastes, and stable likes/dislikes.

Do NOT store project-specific facts; those belong in Claude Code's CLAUDE.md /
per-project auto memory. Do NOT store temporary state, one-off events, today's
plan, or short-term mood. If a fact sounds recent/temporary, ask a follow-up and
only store it when the user confirms it is a stable pattern.

Be concise. Always return a structured summary of what you did.
"""

_agent: Any = None


def get_awake_agent() -> Any:
    """Lazy singleton — build the LangGraph ReAct agent once."""
    global _agent
    if _agent is None:
        llm = get_chat_llm(
            temperature=0.0,
            timeout=settings.awake_llm_timeout_seconds,
            max_retries=settings.awake_llm_max_retries,
        )
        _agent = create_react_agent(llm, AWAKE_TOOLS, prompt=SYSTEM_PROMPT)
    return _agent


async def run_awake(command: str) -> dict[str, Any]:
    """Run the Awake agent with a natural-language command.

    Args:
        command: e.g. "remember this fact about the user: ..."

    Returns dict with final_message and step_count.
    """
    outcome = await _invoke_awake(command)
    if outcome["status"] != "ok":
        return outcome
    messages = outcome.pop("messages")
    final = messages[-1]
    return {
        **outcome,
        "final_message": getattr(final, "content", str(final)),
        "step_count": len(messages),
    }


async def run_recall(command: str) -> dict[str, Any]:
    """Run Recall and record usage only for facts selected in the final answer."""
    outcome = await _invoke_awake(command)
    if outcome["status"] != "ok":
        return outcome

    messages = outcome.pop("messages")
    final = messages[-1]
    final_text = _content_to_text(getattr(final, "content", str(final)))
    payload = _parse_json_object(final_text)
    candidates = _search_candidates(messages)
    allowed_ids = set(candidates)
    requested_ids = payload.get("used_fact_ids", []) if payload else []
    if not isinstance(requested_ids, list):
        requested_ids = []
    used_fact_ids = sorted({
        fact_id
        for raw_id in requested_ids
        if isinstance(raw_id, (int, str))
        and str(raw_id).isdigit()
        and (fact_id := int(raw_id)) in allowed_ids
    })

    if used_fact_ids:
        session_maker = session_factory()
        async with session_maker() as session:
            await mark_archival_used(session, used_fact_ids)

    answer = payload.get("answer") if payload else None
    archival = payload.get("archival") if payload else None
    core_blocks = payload.get("core_blocks") if payload else None
    if not isinstance(archival, list):
        archival = [candidates[fact_id] for fact_id in used_fact_ids]
    if not isinstance(core_blocks, list):
        core_blocks = []

    return {
        **outcome,
        "final_message": answer if isinstance(answer, str) else final_text,
        "core_blocks": core_blocks,
        "archival": archival,
        "used_fact_ids": used_fact_ids,
        "step_count": len(messages),
    }


async def _invoke_awake(command: str) -> dict[str, Any]:
    agent = get_awake_agent()
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [("user", command)]},
                config={"recursion_limit": settings.awake_react_recursion_limit},
            ),
            timeout=settings.awake_overall_timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "awake command timed out after %.1fs: %s",
            settings.awake_overall_timeout_seconds,
            command[:200],
        )
        return {
            "status": "timeout",
            "final_message": (
                "Awake agent timed out before completing the request; "
                "please retry or inspect service logs."
            ),
            "step_count": 0,
        }
    return {
        "status": "ok",
        "messages": result["messages"],
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Recall final response was not valid JSON; usage was not recorded")
        return None
    return parsed if isinstance(parsed, dict) else None


def _search_candidates(messages: list[Any]) -> dict[int, dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for message in messages:
        if getattr(message, "name", None) != "search_archival":
            continue
        raw = getattr(message, "content", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            continue
        for result in raw["results"]:
            if not isinstance(result, dict):
                continue
            raw_id = result.get("id")
            if isinstance(raw_id, int):
                candidates[raw_id] = result
    return candidates
