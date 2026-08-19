from __future__ import annotations

import pytest

from scripts import relabel_memory_signals
from scripts.relabel_memory_signals import parse_relabel_response, validate_labels


def test_parse_relabel_response_accepts_fenced_json():
    parsed = parse_relabel_response(
        """```json
        [
          {"id": 7, "stability": "stage", "salience": 2, "reason": "current goal"}
        ]
        ```"""
    )

    assert parsed == [
        {
            "id": 7,
            "stability": "stage",
            "salience": 2,
            "reason": "current goal",
        }
    ]


def test_validate_labels_rejects_unknown_ids_and_bad_values():
    rows = [{"id": 1, "content": "User prefers direct answers."}]
    labels = [
        {"id": 1, "stability": "long_term", "salience": 3, "reason": "explicit"},
        {"id": 2, "stability": "forever", "salience": 9, "reason": "bad"},
    ]

    assert validate_labels(rows, labels) == [
        {"id": 1, "stability": "long_term", "salience": 3, "reason": "explicit"}
    ]


@pytest.mark.asyncio
async def test_run_reports_model_error_as_json(monkeypatch, capsys):
    async def fake_load_rows(limit: int):
        return [{"id": 1, "content": "User prefers direct answers."}]

    async def fake_relabel(rows):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(relabel_memory_signals, "load_rows", fake_load_rows)
    monkeypatch.setattr(relabel_memory_signals, "relabel", fake_relabel)

    exit_code = await relabel_memory_signals.run(limit=1, apply=False)

    rendered = capsys.readouterr().out
    assert exit_code == 1
    assert '"status": "error"' in rendered
    assert "provider unavailable" in rendered


@pytest.mark.asyncio
async def test_run_reports_database_error_as_json(monkeypatch, capsys):
    async def fake_load_rows(limit: int):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(relabel_memory_signals, "load_rows", fake_load_rows)

    exit_code = await relabel_memory_signals.run(limit=1, apply=False)

    rendered = capsys.readouterr().out
    assert exit_code == 1
    assert '"status": "error"' in rendered
    assert '"stage": "load_rows"' in rendered
    assert "database unavailable" in rendered
