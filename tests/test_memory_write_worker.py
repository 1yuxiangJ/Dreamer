from __future__ import annotations

import pytest

from dreamer.memory import worker
from dreamer.memory.jobs import MemoryWriteJobSnapshot


@pytest.mark.asyncio
async def test_process_one_job_runs_awake_and_marks_succeeded(monkeypatch):
    job = MemoryWriteJobSnapshot(
        id=7,
        operation="remember",
        command="remember content",
        payload={"content": "remember content"},
        attempt_count=1,
    )
    completed: list[tuple[int, dict[str, object]]] = []

    async def claim():
        return job

    async def run_awake(command: str) -> dict[str, object]:
        assert command == "remember content"
        return {"status": "ok"}

    async def mark_succeeded(job_id: int, result: dict[str, object]) -> None:
        completed.append((job_id, result))

    async def mark_failed(job_id: int, error: str) -> None:
        raise AssertionError(f"job should not fail: {job_id} {error}")

    monkeypatch.setattr(worker, "claim_next_write_job", claim)
    monkeypatch.setattr(worker, "mark_write_job_succeeded", mark_succeeded)
    monkeypatch.setattr(worker, "mark_write_job_failed", mark_failed)

    processed = await worker.process_one_job(run_awake)

    assert processed is True
    assert completed == [(7, {"status": "ok"})]


@pytest.mark.asyncio
async def test_process_one_job_marks_failed_on_awake_error(monkeypatch):
    job = MemoryWriteJobSnapshot(
        id=8,
        operation="remember",
        command="remember content that triggers a provider timeout",
        payload={"content": "remember content that triggers a provider timeout"},
        attempt_count=1,
    )
    failures: list[tuple[int, str]] = []

    async def claim():
        return job

    async def run_awake(command: str) -> dict[str, object]:
        raise RuntimeError("provider timeout")

    async def mark_succeeded(job_id: int, result: dict[str, object]) -> None:
        raise AssertionError(f"job should not succeed: {job_id} {result}")

    async def mark_failed(job_id: int, error: str) -> None:
        failures.append((job_id, error))

    monkeypatch.setattr(worker, "claim_next_write_job", claim)
    monkeypatch.setattr(worker, "mark_write_job_succeeded", mark_succeeded)
    monkeypatch.setattr(worker, "mark_write_job_failed", mark_failed)

    processed = await worker.process_one_job(run_awake)

    assert processed is True
    assert failures == [(8, "provider timeout")]
