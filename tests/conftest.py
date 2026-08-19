"""Shared pytest fixtures + integration test gating.

Pure unit tests need no fixtures here.

Integration tests (require running PG + pgvector + `mneme_test` database) are
gated behind `--run-integration`:

    pytest                              # unit only
    pytest --run-integration            # unit + integration
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "src" / "dreamer" / "db" / "schema.sql"


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires local PG + pgvector + mneme_test db).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @integration items unless --run-integration set."""
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="--run-integration not set")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def integration_db_url() -> str:
    return os.getenv(
        "DATABASE_URL_TEST",
        "postgresql+asyncpg://mac@localhost:5432/mneme_test",
    )


def _schema_statements() -> list[str]:
    uncommented = "\n".join(
        line for line in SCHEMA_SQL.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    return [
        stmt.strip()
        for stmt in uncommented.split(";")
        if stmt.strip()
    ]


@pytest.fixture
async def integration_session(integration_db_url: str):
    engine = create_async_engine(integration_db_url, echo=False)

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS core_blocks_staging CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS archival_facts_staging CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS memory_ops_log CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS archival_facts CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS core_blocks CASCADE"))
        for stmt in _schema_statements():
            await conn.exec_driver_sql(stmt)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()
