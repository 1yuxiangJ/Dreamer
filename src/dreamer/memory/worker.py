"""Background worker that drains durable memory write jobs."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from dreamer.awake.agent import run_awake as _run_awake
from dreamer.config import settings
from dreamer.memory.jobs import (
    MemoryWriteJobSnapshot,
    claim_next_write_job,
    mark_write_job_failed,
    mark_write_job_succeeded,
    requeue_stale_running_jobs,
    result_to_jsonable,
)
from dreamer.sleep.scheduler import mark_awake_activity

logger = logging.getLogger("dreamer.memory.worker")

AwakeRunner = Callable[[str], Awaitable[dict[str, Any]]]


async def process_one_job(
    run_awake: AwakeRunner = _run_awake,
) -> bool:
    """Process one durable write job if available."""
    job = await claim_next_write_job()
    if job is None:
        return False

    try:
        mark_awake_activity()
        result = await run_awake(job.command)
        if result.get("status") not in (None, "ok"):
            raise RuntimeError(f"Awake returned non-ok status: {result}")
    except Exception as exc:
        logger.exception(
            "memory write job failed (id=%s operation=%s)",
            job.id,
            job.operation,
        )
        await mark_write_job_failed(job.id, str(exc))
        return True

    await mark_write_job_succeeded(job.id, result_to_jsonable(result))
    logger.info(
        "memory write job succeeded (id=%s operation=%s)",
        job.id,
        job.operation,
    )
    return True


async def drain_available_jobs(
    run_awake: AwakeRunner = _run_awake,
) -> int:
    """Process currently available jobs until the queue is empty."""
    processed = 0
    while await process_one_job(run_awake):
        processed += 1
    return processed


async def worker_loop(
    run_awake: AwakeRunner = _run_awake,
    poll_seconds: float | None = None,
) -> None:
    """Long-running worker loop started by the app lifespan."""
    delay = (
        settings.memory_write_worker_poll_seconds
        if poll_seconds is None
        else poll_seconds
    )
    while True:
        await asyncio.sleep(delay)
        try:
            stale_count = await requeue_stale_running_jobs()
            if stale_count:
                logger.warning("requeued %d stale memory write jobs", stale_count)
            processed = await drain_available_jobs(run_awake)
            if processed:
                logger.info("processed %d memory write jobs", processed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("memory write worker loop failed")


def start_memory_write_worker() -> asyncio.Task[None] | None:
    """Start the in-process durable write worker if enabled."""
    if not settings.memory_write_worker_enabled:
        logger.info("memory write worker disabled")
        return None
    logger.info("memory write worker starting")
    return asyncio.create_task(worker_loop(), name="dreamer-memory-write-worker")


async def stop_memory_write_worker(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


__all__ = [
    "MemoryWriteJobSnapshot",
    "drain_available_jobs",
    "process_one_job",
    "start_memory_write_worker",
    "stop_memory_write_worker",
    "worker_loop",
]
