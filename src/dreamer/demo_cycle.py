"""One-shot demo cycle runner."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dreamer.config import settings
from dreamer.demo_seed import seed_demo_memory
from dreamer.memory.inspect import collect_snapshot
from dreamer.sleep.agent import run_sleep_cycle


async def run_demo_cycle(seed: bool = False) -> dict[str, Any]:
    """Optionally seed demo facts, run Sleep with demo-friendly threshold, inspect."""
    seed_result: dict[str, Any] | None = None
    if seed:
        seed_result = await seed_demo_memory()

    previous_min_archival_count = settings.sleep_min_archival_count
    settings.sleep_min_archival_count = 0
    try:
        sleep_result = await run_sleep_cycle()
    finally:
        settings.sleep_min_archival_count = previous_min_archival_count

    snapshot = await collect_snapshot(limit=10)
    return {
        "status": "ok" if sleep_result.get("status") != "error" else "error",
        "seed": seed_result,
        "sleep": sleep_result,
        "snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Dreamer demo cycle.")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Insert demo-tagged facts before running Sleep.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --seed to confirm database writes.",
    )
    args = parser.parse_args()
    if args.seed and not args.yes:
        print("Refusing to seed demo memory without --yes.")
        return 2

    result = asyncio.run(run_demo_cycle(seed=args.seed))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result["status"] == "error" else 0
