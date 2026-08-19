from __future__ import annotations

import pytest

from dreamer.sleep import agent


class _FakeGraph:
    async def ainvoke(self, _state):
        return {
            "aborted": False,
            "plan": ["promote", "reflect"],
            "consolidate_actions": [],
            "promote_actions": [
                {"decision": "PROMOTE", "fact_id": 1},
                {"decision": "SKIP", "fact_id": 2},
                {"decision": "SKIP", "fact_id": 3},
            ],
            "demote_actions": [],
            "contradictions": [],
            "reflection_text": "user prefers concrete answers",
        }


@pytest.mark.asyncio
async def test_sleep_summary_separates_promote_candidates_from_applied(
    monkeypatch,
):
    monkeypatch.setattr(agent, "get_sleep_graph", lambda: _FakeGraph())

    summary = await agent.run_sleep_cycle()

    assert summary["promote_candidate_count"] == 3
    assert summary["promote_count"] == 1
