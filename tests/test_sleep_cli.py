from __future__ import annotations

import json

import pytest

from dreamer.sleep import cli


@pytest.mark.asyncio
async def test_run_once_prints_json_and_returns_zero(monkeypatch, capsys):
    async def fake_cycle():
        return {
            "status": "ok",
            "plan": ["reflect"],
            "reflection_preview": "user prefers concrete answers",
        }

    monkeypatch.setattr(cli, "run_sleep_cycle", fake_cycle)

    exit_code = await cli.run_once()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["plan"] == ["reflect"]


@pytest.mark.asyncio
async def test_run_once_can_override_min_archival_count(monkeypatch, capsys):
    original = cli.settings.sleep_min_archival_count

    async def fake_cycle():
        return {
            "status": "ok",
            "min_archival_count": cli.settings.sleep_min_archival_count,
        }

    monkeypatch.setattr(cli, "run_sleep_cycle", fake_cycle)

    try:
        exit_code = await cli.run_once(min_archival_count=0)
    finally:
        cli.settings.sleep_min_archival_count = original

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["min_archival_count"] == 0


@pytest.mark.asyncio
async def test_run_once_returns_one_on_error_status(monkeypatch, capsys):
    async def fake_cycle():
        return {"status": "error", "error": "boom"}

    monkeypatch.setattr(cli, "run_sleep_cycle", fake_cycle)

    exit_code = await cli.run_once()

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "boom"
