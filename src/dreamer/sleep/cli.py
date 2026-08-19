"""CLI helpers for manually triggering Sleep cycles."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dreamer.config import settings
from dreamer.sleep.agent import run_sleep_cycle


async def run_once(min_archival_count: int | None = None) -> int:
    """Run one Sleep cycle and print a JSON summary."""
    if min_archival_count is not None:
        settings.sleep_min_archival_count = min_archival_count
    result: dict[str, Any] = await run_sleep_cycle()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result.get("status") == "error" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Dreamer Sleep cycle.")
    parser.add_argument(
        "--min-archival-count",
        type=int,
        default=None,
        help=(
            "Temporarily override SLEEP_MIN_ARCHIVAL_COUNT for this run. "
            "Use 0 for demo runs with a small local memory set."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(run_once(min_archival_count=args.min_archival_count))
