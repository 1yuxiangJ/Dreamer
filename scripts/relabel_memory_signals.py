#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage
from sqlalchemy import text

from dreamer.db.models import get_sessionmaker
from dreamer.llm.client import get_chat_llm

VALID_STABILITY = {"long_term", "stage", "temporary"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_relabel_response(raw: str) -> list[dict[str, Any]]:
    """Parse model JSON, accepting plain JSON or fenced ```json blocks."""
    text_value = raw.strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text_value = "\n".join(lines).strip()
    parsed = json.loads(text_value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array.")
    return [item for item in parsed if isinstance(item, dict)]


def validate_labels(
    rows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only labels that point to known ids and valid signal values."""
    valid_ids = {int(row["id"]) for row in rows}
    accepted: list[dict[str, Any]] = []
    for item in labels:
        try:
            fact_id = int(item["id"])
            stability = str(item["stability"])
            salience = int(item["salience"])
        except (KeyError, TypeError, ValueError):
            continue
        if fact_id not in valid_ids:
            continue
        if stability not in VALID_STABILITY:
            continue
        if salience not in {1, 2, 3}:
            continue
        accepted.append({
            "id": fact_id,
            "stability": stability,
            "salience": salience,
            "reason": str(item.get("reason", "")),
        })
    return accepted


def build_prompt(rows: list[dict[str, Any]]) -> str:
    return f"""Relabel Dreamer archival memories with two memory signals.

Definitions:
- stability: "long_term" for durable identity/preferences/habits; "stage" for
  current career/project/tooling/life-stage facts; "temporary" for short-lived
  plans, moods, locations, or one-off events.
- salience: 3 if the fact strongly affects future collaboration; 2 if useful in
  related contexts; 1 if minor/passive reference.

Keep confidence unchanged. Return ONLY JSON array:
[
  {{"id": 1, "stability": "long_term", "salience": 3, "reason": "..."}}
]

Facts:
{json.dumps(rows, ensure_ascii=False, indent=2, default=str)}
"""


async def load_rows(limit: int) -> list[dict[str, Any]]:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        result = await session.execute(text(
            """
            SELECT id, content, tags, confidence, stability, salience, source,
                   use_count, created_at, last_used_at
            FROM archival_facts
            WHERE is_deleted = FALSE
            ORDER BY id
            LIMIT :limit
            """
        ), {"limit": limit})
        return [dict(row) for row in result.mappings().all()]


async def relabel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    llm = get_chat_llm(temperature=0.0)
    response = await llm.ainvoke([HumanMessage(content=build_prompt(rows))])
    raw = _content_to_text(response.content if hasattr(response, "content") else response)
    return validate_labels(rows, parse_relabel_response(raw))


async def apply_labels(labels: list[dict[str, Any]]) -> None:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        for item in labels:
            await session.execute(text(
                """
                UPDATE archival_facts
                SET stability = :stability,
                    salience = :salience
                WHERE id = :id
                """
            ), item)
        await session.commit()


async def run(limit: int, apply: bool) -> int:
    try:
        rows = await load_rows(limit)
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "stage": "load_rows",
            "mode": "apply" if apply else "dry-run",
            "loaded_count": 0,
            "error": str(exc),
            "hint": (
                "No database writes were made. Check DATABASE_URL/PostgreSQL "
                "availability, then retry."
            ),
        }, ensure_ascii=False, indent=2))
        return 1
    try:
        labels = await relabel(rows)
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "stage": "relabel",
            "mode": "apply" if apply else "dry-run",
            "loaded_count": len(rows),
            "error": str(exc),
            "hint": (
                "No database writes were made. Check network/proxy/API provider "
                "availability, then retry."
            ),
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": "ok",
        "mode": "apply" if apply else "dry-run",
        "loaded_count": len(rows),
        "accepted_count": len(labels),
        "labels": labels,
    }, ensure_ascii=False, indent=2))
    if apply and labels:
        await apply_labels(labels)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Relabel existing archival facts with stability/salience."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write accepted labels to the database. Default is dry-run only.",
    )
    args = parser.parse_args()
    return asyncio.run(run(limit=args.limit, apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
