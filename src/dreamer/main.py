"""dreamer service entrypoint.

Starlette app mounting the MCP streamable-http transport at `/mcp` plus a
`/health` endpoint. Sleep agent scheduler integration is Day 03+.

Run with:
    uv run python -m dreamer            # via __main__.py
    uv run uvicorn dreamer.main:app     # direct
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from dreamer.config import settings
from dreamer.console import routes as console_routes
from dreamer.db.models import dispose_engine
from dreamer.mcp_server import mcp

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dreamer")


async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "dreamer"})


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncIterator[None]:
    """Startup + shutdown hooks."""
    logger.info("dreamer startup; mcp at %s", settings.mcp_server_path)
    from dreamer.memory.worker import start_memory_write_worker, stop_memory_write_worker
    from dreamer.sleep.scheduler import start_sleep_scheduler

    scheduler = start_sleep_scheduler()
    memory_worker = start_memory_write_worker()
    _app.state.memory_worker = memory_worker
    async with mcp.session_manager.run():
        try:
            yield
        finally:
            logger.info("dreamer shutdown")
            await stop_memory_write_worker(
                getattr(_app.state, "memory_worker", memory_worker)
            )
            _app.state.memory_worker = None
            if scheduler is not None:
                scheduler.shutdown(wait=False)
            await dispose_engine()


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health),
        *console_routes,
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)


def main() -> None:
    """Entry point for `python -m dreamer`."""
    import uvicorn

    uvicorn.run(
        "dreamer.main:app",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
