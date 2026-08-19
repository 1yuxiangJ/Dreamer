# Quick Start

## Prerequisites

- Docker and Docker Compose, or PostgreSQL 17+ with the pgvector extension.
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) for native development.
- A chat-model API key and an embedding-model API key. The default configuration uses DeepSeek chat and DashScope `text-embedding-v3`.

## Option A: Docker Compose

```bash
git clone https://github.com/1yuxiangJ/Dreamer.git
cd Dreamer
cp .env.example .env
```

Edit `.env` and fill these values:

```dotenv
DEEPSEEK_API_KEY=...
EMBED_API_KEY=...
```

Start the database and service:

```bash
docker compose up --build
```

Verify the service in another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"dreamer"}
```

Stop the stack with:

```bash
docker compose down
```

Use `docker compose down -v` only when you intentionally want to remove local database data.

## Option B: Native development

Create a PostgreSQL database with pgvector, then apply the schema:

```bash
createdb dreamer
psql dreamer -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql dreamer -f src/dreamer/db/schema.sql
```

Create `.env` from the example. Set `DATABASE_URL` for your local PostgreSQL role, then install dependencies and start the service:

```bash
cp .env.example .env
uv sync --all-extras
uv run python -m dreamer
```

## Connect an MCP host

Dreamer exposes streamable HTTP MCP at `http://127.0.0.1:8000/mcp`.

Example configuration:

```json
{
  "mcpServers": {
    "dreamer": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

For Claude Code:

```bash
claude mcp add --transport http dreamer --scope user http://127.0.0.1:8000/mcp
claude mcp list
```

The host chooses when to call `remember`, `recall`, `list_memory`, and `forget`. To encourage proactive memory writes, add host-specific instructions that define which stable user facts should be remembered.

## Run tests

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
```

For PostgreSQL integration tests, configure an isolated test database:

```bash
DATABASE_URL_TEST=postgresql+asyncpg://USER@localhost:5432/dreamer_test \
uv run pytest --run-integration
```

## Enable automatic Sleep cycles

Dreamer keeps automatic Sleep disabled by default. Set this in `.env` when you want idle-time and daily scheduling:

```dotenv
SLEEP_SCHEDULER_ENABLED=true
SLEEP_IDLE_THRESHOLD_SECONDS=1800
SLEEP_DAILY_CRON_HOUR=3
```

You can manually trigger a cycle during development:

```bash
uv run python scripts/run_sleep_once.py
```
