"""CLI helper for manually draining durable memory write jobs."""
from __future__ import annotations

import argparse
import asyncio
import json

from dreamer.memory.worker import process_one_job


async def run(limit: int | None = None) -> int:
    processed = 0
    while limit is None or processed < limit:
        did_process = await process_one_job()
        if not did_process:
            break
        processed += 1
    print(json.dumps({"status": "ok", "processed": processed}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually drain Dreamer durable memory write jobs.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(run(args.limit))
