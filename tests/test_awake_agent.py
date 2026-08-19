from __future__ import annotations

import asyncio
import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest

from dreamer.awake import agent as awake_agent


class CapturingAgent:
    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None

    async def ainvoke(
        self,
        payload: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.config = config
        return {"messages": [type("Msg", (), {"content": "ok"})()]}


@pytest.mark.asyncio
async def test_run_awake_passes_recursion_limit(monkeypatch):
    fake = CapturingAgent()
    monkeypatch.setattr(awake_agent, "get_awake_agent", lambda: fake)
    monkeypatch.setattr(awake_agent.settings, "awake_react_recursion_limit", 8)
    monkeypatch.setattr(awake_agent.settings, "awake_overall_timeout_seconds", 45.0)

    result = await awake_agent.run_awake("list memory")

    assert result["final_message"] == "ok"
    assert fake.config == {"recursion_limit": 8}


@pytest.mark.asyncio
async def test_run_awake_returns_timeout_status(monkeypatch):
    class SlowAgent:
        async def ainvoke(
            self,
            payload: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            await asyncio.sleep(1)
            return {"messages": []}

    monkeypatch.setattr(awake_agent, "get_awake_agent", lambda: SlowAgent())
    monkeypatch.setattr(awake_agent.settings, "awake_react_recursion_limit", 8)
    monkeypatch.setattr(awake_agent.settings, "awake_overall_timeout_seconds", 0.01)

    result = await awake_agent.run_awake("remember something")

    assert result["status"] == "timeout"
    assert "timed out" in result["final_message"]


@pytest.mark.asyncio
async def test_run_recall_marks_only_final_selected_facts(monkeypatch):
    class RecallAgent:
        async def ainvoke(
            self,
            payload: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            tool_result = type("ToolResult", (), {
                "name": "search_archival",
                "content": {
                    "results": [
                        {"id": 11, "content": "used"},
                        {"id": 12, "content": "candidate only"},
                    ],
                },
            })()
            final = type("Final", (), {
                "content": json.dumps({
                    "answer": "answer from fact 11",
                    "core_blocks": [],
                    "archival": [{"id": 11, "content": "used"}],
                    "used_fact_ids": [11, 11, 999],
                }),
            })()
            return {"messages": [tool_result, final]}

    class SessionContext(AbstractAsyncContextManager[object]):
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    marked: list[int] = []

    async def mark_used(_session: object, fact_ids: list[int]) -> None:
        marked.extend(fact_ids)

    monkeypatch.setattr(awake_agent, "get_awake_agent", lambda: RecallAgent())
    monkeypatch.setattr(awake_agent, "session_factory", lambda: lambda: SessionContext())
    monkeypatch.setattr(awake_agent, "mark_archival_used", mark_used)

    result = await awake_agent.run_recall("recall")

    assert result["used_fact_ids"] == [11]
    assert marked == [11]
