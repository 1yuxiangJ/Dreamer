#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ and re-run."
    exit 1
fi

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Add DEEPSEEK_API_KEY and EMBED_API_KEY."
fi

uv sync --all-extras

echo
echo "Dependencies are ready. Start PostgreSQL + pgvector with:"
echo "  docker compose up -d postgres"
echo "Then start Dreamer with:"
echo "  uv run python -m dreamer"
