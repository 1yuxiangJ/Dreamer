"""Memory store: CRUD + semantic search.

Access policy (Letta read-only primary):
  Awake agent  → read core_blocks, read/write archival_facts
  Sleep agent  → full write on both (sole writer of core_blocks)
"""
