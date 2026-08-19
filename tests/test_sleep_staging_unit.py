from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from dreamer.sleep import staging


class RecordingSession:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: list[dict[str, Any] | None] = []
        self.committed = False

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> None:
        self.statements.append(str(statement))
        self.params.append(params)

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_atomic_swap_sets_local_lock_timeout(monkeypatch):
    session = RecordingSession()
    monkeypatch.setattr(staging.settings, "sleep_swap_lock_timeout_ms", 500)

    await staging.atomic_swap(session, datetime.now(UTC))  # type: ignore[arg-type]

    assert "set_config('lock_timeout'" in session.statements[0]
    assert session.params[0] == {"lock_timeout": "500ms"}
    assert "LOCK TABLE archival_facts IN SHARE ROW EXCLUSIVE MODE" in session.statements[1]
    assert "INSERT INTO archival_facts_staging" in session.statements[2]
    assert "UPDATE archival_facts_staging AS staging" in session.statements[3]
    assert session.committed is True
