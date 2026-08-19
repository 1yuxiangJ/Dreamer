"""Sleep Agent: LangGraph StateGraph orchestrating sleep-time consolidation.

Per Letta sleep-time compute (arxiv 2504.13171), this agent is the SOLE
writer of core_blocks. The Awake agent only writes archival; Sleep promotes
and consolidates.

A cycle is modeled with 10 node types; repair is visited only when needed:
  1. snapshot    — clone main → *_staging
  2. plan        — LLM decides which subsequent phases to run
  3. consolidate — merge near-duplicates (staging only)
  4. promote     — lift archival → core_blocks (only path for new Core facts)
  5. demote      — soft-delete stale low-confidence archival (staging)
  6. resolve     — fix internal contradictions in core_blocks (staging)
  7. validate    — quality-gate Core changes made in this cycle
  8. repair      — bounded Core repair, then loop back to validate
  9. reflect     — write "about user" snapshot to memory_ops_log
 10. swap        — atomic_swap staging → main, or cleanup on abort

Budget enforcement:
  - max_wall_time (settings.sleep_max_wall_time_seconds, default 300s)
  - per-phase deadline check; skip phase if exceeded
  - max_tokens NOT enforced in MVP (Day 05+ improvement)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text as sql_text

from dreamer.config import settings
from dreamer.db.models import get_sessionmaker
from dreamer.llm.client import get_chat_llm
from dreamer.memory.store import MemoryOpDraft
from dreamer.sleep import prompts, staging, tools

logger = logging.getLogger("dreamer.sleep")

# In-process bookkeeping. For production, persist to DB.
_last_cycle_ts: datetime | None = None


class SleepState(TypedDict, total=False):
    snapshot_ts: datetime
    deadline_ts: float
    plan: list[str]
    consolidate_actions: list[Any]
    promote_actions: list[Any]
    demote_actions: list[Any]
    contradictions: list[Any]
    core_validation_status: str
    core_validation_issues: list[dict[str, Any]]
    core_validation_attempts: int
    core_repair_attempts: int
    core_changed_blocks: list[str]
    reflection_text: str
    pending_ops: list[MemoryOpDraft]
    aborted: bool
    abort_reason: str | None


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


async def _llm_json(prompt: str) -> dict[str, Any]:
    """Call chat LLM and parse JSON response (tolerant of code fences)."""
    llm = get_chat_llm(temperature=0.0)
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = _content_to_text(resp.content if hasattr(resp, "content") else resp)
    return _safe_parse_json(raw)


def _safe_parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return {"_raw": parsed}
    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed; raw[:500]=%r", raw[:500])
        return {"_parse_error": str(exc), "_raw": raw[:1000]}


def _budget_ok(state: SleepState) -> bool:
    return state.get("deadline_ts", float("inf")) > time.monotonic()


def _append_pending_ops(
    state: SleepState,
    pending_ops: list[MemoryOpDraft],
) -> SleepState:
    if not pending_ops:
        return state
    return {
        **state,
        "pending_ops": [*state.get("pending_ops", []), *pending_ops],
    }


def _count_decision(actions: list[Any] | None, decision: str) -> int:
    expected = decision.upper()
    count = 0
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        actual = str(action.get("decision", "")).upper()
        if actual == expected:
            count += 1
    return count


def _validation_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# ---------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------


async def node_snapshot(state: SleepState) -> SleepState:
    if not _budget_ok(state):
        return {**state, "aborted": True, "abort_reason": "deadline before snapshot"}
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        ts = await staging.snapshot_to_staging(session)
    logger.info("snapshot_to_staging @ %s", ts)
    return {**state, "snapshot_ts": ts}


async def node_plan(state: SleepState) -> SleepState:
    if state.get("aborted") or not _budget_ok(state):
        return state
    global _last_cycle_ts
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        summary = await tools.summarize_state(session, _last_cycle_ts)

    rendered = prompts.PLAN_PROMPT.format(
        state_summary=json.dumps({
            "core_blocks": summary.core_blocks,
            "archival_count": summary.archival_count,
            "new_since_last_cycle": summary.new_archival_since_last_cycle,
            "old_unused_count": summary.old_unused_count,
            "frequent_fact_count": summary.frequent_fact_count,
        }, indent=2, default=str),
        min_archival=settings.sleep_min_archival_count,
    )
    decision = await _llm_json(rendered)
    requested = decision.get("phases", ["reflect"])
    if not isinstance(requested, list):
        requested = []
    allowed = ["consolidate", "promote", "demote", "resolve", "reflect"]
    phases = [phase for phase in requested if phase in allowed]
    if summary.archival_count >= settings.sleep_min_archival_count:
        required: list[str] = []
        if summary.new_archival_since_last_cycle > 0:
            required.append("consolidate")
        if summary.frequent_fact_count > 0:
            required.append("promote")
        if summary.old_unused_count > 0:
            required.append("demote")
        for phase in required:
            if phase not in phases:
                insert_at = phases.index("reflect") if "reflect" in phases else len(phases)
                phases.insert(insert_at, phase)
    logger.info("plan: phases=%s reason=%s", phases, decision.get("reason"))
    return {**state, "plan": phases}


async def node_consolidate(state: SleepState) -> SleepState:
    if (
        state.get("aborted")
        or "consolidate" not in state.get("plan", [])
        or not _budget_ok(state)
    ):
        return state
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        clusters = await tools.find_consolidation_clusters(session)
        if not clusters:
            logger.info("consolidate: no clusters found")
            return {**state, "consolidate_actions": []}
        rendered = prompts.CONSOLIDATE_PROMPT.format(
            clusters_json=json.dumps(clusters, indent=2, default=str),
        )
        decision = await _llm_json(rendered)
        actions = decision.get("actions", [])
        pending_ops = await tools.apply_consolidation(session, actions)
    logger.info("consolidate: %d actions", len(actions))
    return _append_pending_ops(
        {**state, "consolidate_actions": actions},
        pending_ops,
    )


async def node_promote(state: SleepState) -> SleepState:
    if (
        state.get("aborted")
        or "promote" not in state.get("plan", [])
        or not _budget_ok(state)
    ):
        return state
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        candidates = await tools.get_promote_candidates(session)
        if not candidates:
            logger.info("promote: no candidates")
            return {**state, "promote_actions": []}
        summary = await tools.summarize_state(
            session,
            None,
            core_table="core_blocks_staging",
        )
        rendered = prompts.PROMOTE_PROMPT.format(
            core_blocks_json=json.dumps(summary.core_blocks, indent=2, default=str),
            candidates_json=json.dumps(candidates, indent=2, default=str),
        )
        decision = await _llm_json(rendered)
        actions = decision.get("actions", [])
        pending_ops = await tools.apply_promotions(session, actions)
    logger.info("promote: %d actions", len(actions))
    return _append_pending_ops(
        {**state, "promote_actions": actions},
        pending_ops,
    )


async def node_demote(state: SleepState) -> SleepState:
    if (
        state.get("aborted")
        or "demote" not in state.get("plan", [])
        or not _budget_ok(state)
    ):
        return state
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        stale = await tools.get_stale_candidates(session)
        if not stale:
            logger.info("demote: no candidates")
            return {**state, "demote_actions": []}
        rendered = prompts.DEMOTE_PROMPT.format(
            stale_json=json.dumps(stale, indent=2, default=str),
        )
        decision = await _llm_json(rendered)
        actions = decision.get("actions", [])
        pending_ops = await tools.apply_demotions(session, actions)
    logger.info("demote: %d actions", len(actions))
    return _append_pending_ops(
        {**state, "demote_actions": actions},
        pending_ops,
    )


async def node_resolve(state: SleepState) -> SleepState:
    if (
        state.get("aborted")
        or "resolve" not in state.get("plan", [])
        or not _budget_ok(state)
    ):
        return state
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        summary = await tools.summarize_state(
            session,
            None,
            core_table="core_blocks_staging",
        )
        recent = (await session.execute(sql_text(
            "SELECT op_type, actor, target_kind, target_id, "
            "before_value, after_value, reason, ts "
            "FROM memory_ops_log "
            "WHERE target_kind = 'core' "
            "AND op_type IN ('sleep_promote', 'sleep_resolve', 'sleep_core_repair') "
            "ORDER BY id DESC LIMIT 20"
        ))).all()
        historical_ops = [
            {
                "op_type": r.op_type, "actor": r.actor,
                "target_kind": r.target_kind, "target_id": r.target_id,
                "before_value": r.before_value, "after_value": r.after_value,
                "reason": r.reason,
                "ts": r.ts.isoformat() if r.ts else None,
            }
            for r in recent
        ]
        current_cycle_ops = [
            dict(op)
            for op in state.get("pending_ops", [])
            if op.get("target_kind") == "core"
            and op.get("op_type") in {
                "sleep_promote", "sleep_resolve", "sleep_core_repair",
            }
        ]
        rendered = prompts.RESOLVE_PROMPT.format(
            core_blocks_json=json.dumps(summary.core_blocks, indent=2, default=str),
            current_cycle_core_ops_json=json.dumps(
                current_cycle_ops, indent=2, default=str,
            ),
            historical_core_ops_json=json.dumps(
                historical_ops, indent=2, default=str,
            ),
        )
        decision = await _llm_json(rendered)
        contradictions = decision.get("contradictions", [])
        pending_ops = await tools.apply_resolutions(session, contradictions)
    logger.info("resolve: %d contradictions", len(contradictions))
    return _append_pending_ops(
        {**state, "contradictions": contradictions},
        pending_ops,
    )


async def node_validate_core(state: SleepState) -> SleepState:
    """Validate only Core blocks changed by this Sleep cycle."""
    if state.get("aborted"):
        return state

    session_maker = get_sessionmaker()
    async with session_maker() as session:
        context = await tools.get_core_validation_context(
            session,
            state.get("pending_ops", []),
            state.get("promote_actions", []),
        )

    changed_blocks = [
        str(block["label"])
        for block in context.get("changed_blocks", [])
        if block.get("label")
    ]
    if not changed_blocks:
        logger.info("validate_core: no Core changes; semantic validation skipped")
        return {
            **state,
            "core_validation_status": "PASS",
            "core_validation_issues": [],
            "core_changed_blocks": [],
        }

    if not _budget_ok(state):
        return {
            **state,
            "aborted": True,
            "abort_reason": "deadline before Core validation",
            "core_validation_status": "FATAL",
            "core_changed_blocks": changed_blocks,
        }

    deterministic_issues = tools.get_core_constraint_issues(context)
    rendered = prompts.VALIDATE_CORE_PROMPT.format(
        validation_context_json=json.dumps(context, indent=2, default=str),
        deterministic_issues_json=json.dumps(
            deterministic_issues,
            indent=2,
            default=str,
        ),
    )
    decision = await _llm_json(rendered)
    status = str(decision.get("status", "FATAL")).upper()
    semantic_issues = _validation_issues(decision.get("issues"))
    issues = [*deterministic_issues, *semantic_issues]

    if any(issue.get("severity") == "FATAL" for issue in deterministic_issues):
        status = "FATAL"
    elif issues and status == "PASS":
        status = "REPAIRABLE"
    if status not in {"PASS", "REPAIRABLE", "FATAL"}:
        status = "FATAL"
        issues.append({
            "code": "invalid_validation_output",
            "message": "Validator returned an unsupported status.",
        })

    validation_attempts = state.get("core_validation_attempts", 0) + 1
    next_state: SleepState = {
        **state,
        "core_validation_status": status,
        "core_validation_issues": issues,
        "core_validation_attempts": validation_attempts,
        "core_changed_blocks": changed_blocks,
    }
    repair_attempts = state.get("core_repair_attempts", 0)

    if status == "PASS":
        validation_op = tools.draft_core_validation_log(
            changed_blocks,
            validation_attempts,
            repair_attempts,
        )
        logger.info(
            "validate_core: PASS blocks=%s validations=%d repairs=%d",
            changed_blocks,
            validation_attempts,
            repair_attempts,
        )
        return _append_pending_ops(next_state, [validation_op])

    if status == "FATAL" or repair_attempts >= settings.sleep_core_repair_max_attempts:
        reason = (
            "Core validation returned FATAL"
            if status == "FATAL"
            else "Core repair retry limit exceeded"
        )
        logger.warning("validate_core: aborting; %s", reason)
        return {**next_state, "aborted": True, "abort_reason": reason}

    logger.info("validate_core: REPAIRABLE issues=%d", len(issues))
    return next_state


async def node_repair_core(state: SleepState) -> SleepState:
    """Repair rejected Core changes and return to validation."""
    if state.get("aborted") or state.get("core_validation_status") != "REPAIRABLE":
        return state
    repair_attempts = state.get("core_repair_attempts", 0)
    if (
        repair_attempts >= settings.sleep_core_repair_max_attempts
        or not _budget_ok(state)
    ):
        return {
            **state,
            "aborted": True,
            "abort_reason": "Core repair budget exhausted",
            "core_validation_status": "FATAL",
        }

    session_maker = get_sessionmaker()
    async with session_maker() as session:
        context = await tools.get_core_validation_context(
            session,
            state.get("pending_ops", []),
            state.get("promote_actions", []),
        )
        rendered = prompts.REPAIR_CORE_PROMPT.format(
            validation_context_json=json.dumps(context, indent=2, default=str),
            issues_json=json.dumps(
                state.get("core_validation_issues", []),
                indent=2,
                default=str,
            ),
        )
        decision = await _llm_json(rendered)
        repairs = decision.get("repairs", [])
        if not isinstance(repairs, list):
            repairs = []
        pending_ops = await tools.apply_core_repairs(
            session,
            [repair for repair in repairs if isinstance(repair, dict)],
            set(state.get("core_changed_blocks", [])),
        )

    next_attempt = repair_attempts + 1
    if not pending_ops:
        logger.warning("repair_core: no applicable repair; aborting")
        return {
            **state,
            "core_repair_attempts": next_attempt,
            "core_validation_status": "FATAL",
            "aborted": True,
            "abort_reason": "Core repair produced no applicable changes",
        }
    logger.info("repair_core: applied=%d attempt=%d", len(pending_ops), next_attempt)
    return _append_pending_ops(
        {
            **state,
            "core_repair_attempts": next_attempt,
        },
        pending_ops,
    )


async def node_reflect(state: SleepState) -> SleepState:
    if (
        state.get("aborted")
        or not _budget_ok(state)
    ):
        return state
    reflected_ops = [
        dict(op)
        for op in state.get("pending_ops", [])
        if op.get("op_type") != "sleep_reflect"
    ]
    if not reflected_ops:
        logger.info("reflect: no pending operations; skipped")
        return state
    rendered = prompts.REFLECT_PROMPT.format(
        pending_ops_json=json.dumps(reflected_ops, indent=2, default=str),
    )
    llm = get_chat_llm(temperature=0.3)
    resp = await llm.ainvoke([HumanMessage(content=rendered)])
    reflection_text = _content_to_text(
        resp.content if hasattr(resp, "content") else resp
    )
    pending_ops = [tools.draft_reflection_log(reflection_text)]
    logger.info("reflect: operation summary logged")
    return _append_pending_ops(
        {**state, "reflection_text": reflection_text},
        pending_ops,
    )


async def node_swap(state: SleepState) -> SleepState:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        if state.get("aborted"):
            logger.warning("swap aborted: %s; cleaning up staging",
                           state.get("abort_reason"))
            await staging.cleanup_staging(session)
            return state
        ts = state.get("snapshot_ts")
        if ts is None:
            logger.warning("no snapshot ts; cleaning up staging")
            await staging.cleanup_staging(session)
            return state
        await staging.atomic_swap(session, ts, pending_ops=state.get("pending_ops", []))
        logger.info("atomic_swap complete")
    return state


# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------


def route_after_core_validation(state: SleepState) -> str:
    if state.get("aborted"):
        return "swap"
    if state.get("core_validation_status") == "REPAIRABLE":
        return "repair_core"
    return "reflect"


def build_sleep_graph() -> Any:
    g = StateGraph(SleepState)
    g.add_node("snapshot", node_snapshot)
    g.add_node("plan", node_plan)
    g.add_node("consolidate", node_consolidate)
    g.add_node("promote", node_promote)
    g.add_node("demote", node_demote)
    g.add_node("resolve", node_resolve)
    g.add_node("validate_core", node_validate_core)
    g.add_node("repair_core", node_repair_core)
    g.add_node("reflect", node_reflect)
    g.add_node("swap", node_swap)

    g.add_edge(START, "snapshot")
    g.add_edge("snapshot", "plan")
    g.add_edge("plan", "consolidate")
    g.add_edge("consolidate", "promote")
    g.add_edge("promote", "demote")
    g.add_edge("demote", "resolve")
    g.add_edge("resolve", "validate_core")
    g.add_conditional_edges(
        "validate_core",
        route_after_core_validation,
        {
            "repair_core": "repair_core",
            "reflect": "reflect",
            "swap": "swap",
        },
    )
    g.add_edge("repair_core", "validate_core")
    g.add_edge("reflect", "swap")
    g.add_edge("swap", END)
    return g.compile()


_graph: Any = None


def get_sleep_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_sleep_graph()
    return _graph


# ---------------------------------------------------------------
# Public entrypoint (called by scheduler)
# ---------------------------------------------------------------


async def run_sleep_cycle() -> dict[str, Any]:
    """Run one full Sleep cycle.

    Returns a summary dict for logging / observability.
    """
    global _last_cycle_ts
    graph = get_sleep_graph()
    deadline = time.monotonic() + settings.sleep_max_wall_time_seconds
    init_state: SleepState = {
        "deadline_ts": deadline,
        "aborted": False,
        "abort_reason": None,
    }
    logger.info("sleep cycle starting (budget=%ds)",
                settings.sleep_max_wall_time_seconds)
    try:
        final_state = await graph.ainvoke(init_state)
    except Exception as exc:
        logger.exception("sleep cycle failed: %s", exc)
        try:
            session_maker = get_sessionmaker()
            async with session_maker() as session:
                await staging.cleanup_staging(session)
        except Exception:
            pass
        return {"status": "error", "error": str(exc)}

    if not final_state.get("aborted"):
        _last_cycle_ts = datetime.now(UTC)

    promote_actions = final_state.get("promote_actions") or []

    return {
        "status": "aborted" if final_state.get("aborted") else "ok",
        "abort_reason": final_state.get("abort_reason"),
        "plan": final_state.get("plan", []),
        "consolidate_count": len(final_state.get("consolidate_actions") or []),
        "promote_candidate_count": len(promote_actions),
        "promote_count": _count_decision(promote_actions, "PROMOTE"),
        "demote_count": len(final_state.get("demote_actions") or []),
        "contradictions_count": len(final_state.get("contradictions") or []),
        "core_validation_status": final_state.get("core_validation_status"),
        "core_validation_attempts": final_state.get(
            "core_validation_attempts",
            0,
        ),
        "core_repair_attempts": final_state.get("core_repair_attempts", 0),
        "core_changed_blocks": final_state.get("core_changed_blocks", []),
        "reflection_preview": (final_state.get("reflection_text") or "")[:200],
    }
