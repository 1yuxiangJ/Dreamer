"""APScheduler-based trigger for Sleep cycles.

Two triggers:
  1. Idle detection — periodic 60s check; if no Awake activity for
     `sleep_idle_threshold_seconds`, fire a cycle.
  2. Daily cron — at `sleep_daily_cron_hour` (default 03:00).

Activity tracking: `mark_awake_activity()` is called from the MCP server
on every tool invocation. The scheduler reads this to detect idleness.

Single-flight: only one cycle at a time. Concurrent triggers are skipped.
"""
from __future__ import annotations

import logging
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from dreamer.config import settings

logger = logging.getLogger("dreamer.sleep.scheduler")

# Module-level state (in-process; MVP single-user).
_last_awake_activity_monotonic: float = time.monotonic()
_cycle_running: bool = False
_run_sleep_cycle_fn = None  # late-bound to avoid circular import


def mark_awake_activity() -> None:
    """Called by MCP server tools to reset the idle clock."""
    global _last_awake_activity_monotonic
    _last_awake_activity_monotonic = time.monotonic()


def _idle_seconds() -> float:
    return time.monotonic() - _last_awake_activity_monotonic


async def _try_run_cycle(reason: str) -> None:
    global _cycle_running
    if _cycle_running:
        logger.info("sleep trigger=%s skipped (already running)", reason)
        return
    _cycle_running = True
    try:
        if _run_sleep_cycle_fn is None:
            logger.error("sleep cycle fn not injected yet")
            return
        result = await _run_sleep_cycle_fn()
        logger.info("sleep cycle (trigger=%s) result=%s", reason, result)
    finally:
        _cycle_running = False


async def _idle_tick() -> None:
    if _cycle_running:
        return
    if _idle_seconds() >= settings.sleep_idle_threshold_seconds:
        await _try_run_cycle("idle")
        # Reset to avoid immediate re-trigger.
        mark_awake_activity()


async def _cron_tick() -> None:
    await _try_run_cycle("cron")


def start_sleep_scheduler() -> AsyncIOScheduler | None:
    """Start the scheduler. Call from FastAPI/Starlette lifespan startup."""
    if not settings.sleep_scheduler_enabled:
        logger.info("sleep scheduler disabled; set SLEEP_SCHEDULER_ENABLED=true to enable")
        return None

    # Late import to break circular dep on dreamer.sleep.agent.
    from dreamer.sleep.agent import run_sleep_cycle

    global _run_sleep_cycle_fn
    _run_sleep_cycle_fn = run_sleep_cycle

    sched = AsyncIOScheduler()
    sched.add_job(
        _idle_tick,
        trigger=IntervalTrigger(seconds=60),
        id="dreamer-idle-check",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    sched.add_job(
        _cron_tick,
        trigger=CronTrigger(hour=settings.sleep_daily_cron_hour),
        id="dreamer-daily-cron",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    sched.start()
    logger.info(
        "sleep scheduler started: idle_threshold=%ds, daily_cron=%02d:00",
        settings.sleep_idle_threshold_seconds,
        settings.sleep_daily_cron_hour,
    )
    return sched
