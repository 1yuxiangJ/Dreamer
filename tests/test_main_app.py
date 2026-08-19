"""Pure-unit tests for the Starlette app routing."""
from __future__ import annotations

from starlette.testclient import TestClient

from dreamer.main import app


def test_mcp_endpoint_is_mounted_at_configured_path():
    """Claude Code expects the configured /mcp URL to reach FastMCP, not 404."""
    with TestClient(app) as client:
        response = client.get("/mcp/")

    assert response.status_code != 404
