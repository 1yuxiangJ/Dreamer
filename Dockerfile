FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8000

CMD ["python", "-m", "dreamer"]
