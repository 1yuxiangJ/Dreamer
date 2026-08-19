from __future__ import annotations

from dreamer import mcp_server
from dreamer.awake import agent as awake_agent


def test_remember_tool_description_includes_lifestyle_memory_policy():
    doc = mcp_server.remember.__doc__ or ""

    assert "hobbies" in doc
    assert "entertainment" in doc
    assert "lifestyle" in doc
    assert "temporary" in doc
    assert "stable long-term" in doc
    assert "stage-specific" in doc
    assert "split" in doc
    assert "stability" in doc
    assert "salience" in doc


def test_awake_domain_constraint_includes_lifestyle_memory_policy():
    prompt = awake_agent.SYSTEM_PROMPT

    assert "hobbies" in prompt
    assert "entertainment" in prompt
    assert "lifestyle" in prompt
    assert "temporary" in prompt
    assert "stable long-term" in prompt
    assert "stage-specific" in prompt
    assert "split" in prompt
    assert "stability" in prompt
    assert "salience" in prompt
