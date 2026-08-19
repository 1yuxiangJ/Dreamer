"""Sleep agent prompt templates (Letta sleep-time compute).

Each prompt is a stage in the Sleep cycle. The cycle is orchestrated by
sleep/agent.py:run_sleep_cycle as a LangGraph StateGraph; each node fills
in the variables and calls the chat LLM with the rendered prompt.

CRITICAL POLICY (encoded into every prompt):
- The Sleep agent is the ONLY actor allowed to modify core_blocks.
- Awake agent only writes archival; you (Sleep) promote / consolidate.
- Be conservative — when in doubt, do nothing.
- Always log reason for every change (returned in tool calls / final output).

These prompts will need iteration once real LLM traffic flows. Day 03
locks the structure; Day 05+ tunes the wording based on observed behavior.
"""
from __future__ import annotations

# =====================================================================
# Phase 0: PLAN
# Determine which phases to run this cycle.
# =====================================================================

PLAN_PROMPT = """You are dreamer's Sleep agent at the start of an idle-time consolidation cycle.

Below is a summary of the current memory state. Decide which phases are worth
running this cycle. Skip phases that have nothing to do.

Memory state:
{state_summary}

Phases available:
1. consolidate — merge near-duplicate archival facts (cosine > 0.85)
2. promote    — lift frequent, explicit, durable, useful archival into core blocks
3. demote     — soft-delete archival that's stale and low-signal
4. resolve    — detect & fix contradictions within core blocks
5. reflect    — summarize this cycle's applied pending operations

Output strictly this JSON (no commentary):
{{
  "phases": ["consolidate", "promote", "reflect"],
  "reason": "<one-line rationale per phase, semicolon-separated>"
}}

Constraints:
- If archival_count < {min_archival}: skip everything except reflect.
- If no archival created since last cycle: skip consolidate.
- If frequent_fact_count > 0: include promote. The Promote phase applies its
  stricter confidence, stability, and salience filters.
- If old_unused_count > 0: include demote. The runtime also enforces this inspection;
  the Demote phase still decides conservatively between FORGET and KEEP.
- Include reflect for a potentially productive cycle. Runtime skips it without
  calling the LLM when no pending operation was actually applied.
"""


# =====================================================================
# Phase 1: CONSOLIDATE
# Merge near-duplicate archival facts.
# =====================================================================

CONSOLIDATE_PROMPT = """You are dreamer's Sleep agent in the CONSOLIDATE phase.

Below are clusters of archival facts that the embedding similarity flagged as
near-duplicates (cosine distance < 0.15). Decide for each cluster:

  - MERGE: pick the best wording, mark others as superseded.
  - KEEP_ALL: clusters are not actually duplicates (different nuance).

Clusters:
{clusters_json}

Output strictly this JSON:
{{
  "actions": [
    {{
      "cluster_index": 0,
      "decision": "MERGE",
      "kept_id": 42,
      "discarded_ids": [37, 51],
      "merged_content": "<final wording>",
      "reason": "<why merged>"
    }},
    ...
  ]
}}

Be conservative: if any cluster member has confidence=3 and others don't,
PREFER keeping the high-confidence wording. Preserve stability/salience semantics
in the merged wording; do not turn temporary details into long-term traits.
"""


# =====================================================================
# Phase 2: PROMOTE
# Lift archival → core_blocks.
# =====================================================================

PROMOTE_PROMPT = """You are dreamer's Sleep agent in the PROMOTE phase.

You are the ONLY actor allowed to modify core_blocks. Decide whether to lift
each candidate archival fact into an appropriate core block.

Current core blocks (do not duplicate existing content):
{core_blocks_json}

Candidate archival facts (high use_count + confidence=3 + stability=long_term
+ salience=3):
{candidates_json}

The 5 core blocks: background, preferences, habits, skills, lessons_learned.

For each candidate, output:

{{
  "actions": [
    {{
      "fact_id": 123,
      "decision": "PROMOTE",
      "target_block": "preferences",
      "new_block_value": "<full block value; keep existing content plus this fact>",
      "reason": "<why this belongs in core>"
    }},
    {{
      "fact_id": 456,
      "decision": "SKIP",
      "reason": "<why not — duplicate of existing core / not generalizable / too specific>"
    }},
    ...
  ]
}}

Rules:
- new_block_value MUST be the COMPLETE new value, not a diff.
- Keep block under char_limit (default 2000).
- PROMOTE only if the fact is general, durable, and useful across future
  conversations.
- Treat stability and salience as hard safety signals: never promote temporary
  facts or medium/low-salience trivia even if phrased confidently.
- Route core updates by semantics:
  preferences = likes/dislikes, values, priorities, and choice tendencies;
  habits = repeated behaviors, routines, rhythms, and ways the user spends time.
- Keep core values generalized. Do not promote fine-grained lifestyle details
  such as a specific food, game mode, device location, or one-off context unless
  they reveal a broader high-salience pattern.
- If unsure, SKIP.
"""


# =====================================================================
# Phase 3: DEMOTE
# Soft-delete stale low-confidence archival.
# =====================================================================

DEMOTE_PROMPT = """You are dreamer's Sleep agent in the DEMOTE phase.

Below are stale archival facts with low signal: low confidence, temporary
stability, or low salience. A fact is stale when its last use was over 90 days
ago, or, if it has never been used, when it was created over 90 days ago.
Decide whether each can be safely forgotten.

Stale candidates:
{stale_json}

Output:
{{
  "actions": [
    {{"fact_id": 12, "decision": "FORGET", "reason": "..."}},
    {{"fact_id": 18, "decision": "KEEP", "reason": "may still be relevant"}}
  ]
}}

Rules:
- NEVER forget facts with confidence=3.
- Be extra conservative with stability=long_term and salience>=2.
- NEVER forget facts that contradict / clarify core blocks (those need resolve).
- When in doubt, KEEP.
"""


# =====================================================================
# Phase 4: RESOLVE
# Detect and fix contradictions within core blocks.
# =====================================================================

RESOLVE_PROMPT = """You are dreamer's Sleep agent in the RESOLVE phase.

Below is the full current state of the 5 core blocks. Detect internal
contradictions (between blocks, or within one block) and propose fixes.

Core blocks:
{core_blocks_json}

Current-cycle Core operations already applied to staging:
{current_cycle_core_ops_json}

Recent historical Core mutations (last 20 relevant operations):
{historical_core_ops_json}

Output:
{{
  "contradictions": [
    {{
      "blocks_involved": ["preferences", "habits"],
      "description": "...",
      "fix_block": "preferences",
      "new_block_value": "<entire updated block value>",
      "reason": "..."
    }}
  ]
}}

If no contradictions are detected, output {{"contradictions": []}}.
Be very conservative — only flag genuine logical conflicts, not stylistic differences.
"""


# =====================================================================
# Phase 5: VALIDATE CORE
# Verify only Core blocks changed during this Sleep cycle.
# =====================================================================

VALIDATE_CORE_PROMPT = """You are dreamer's Core quality evaluator.

Review ONLY the Core blocks changed by PROMOTE or RESOLVE in the current Sleep
cycle. This is a pre-commit quality gate, not a general Core refresh. Do not
mark unchanged or merely old-looking information for deletion.

Validation context:
{validation_context_json}

Deterministic issues already found by code:
{deterministic_issues_json}

Check the changed values for:
1. unsupported claims not present in the original Core or source facts;
2. accidental loss of useful original information;
3. logical contradictions or obvious repetition;
4. content placed in the wrong Core block;
5. over-generalization from a narrow source fact;
6. deterministic constraints reported above.

Output strictly this JSON:
{{
  "status": "PASS",
  "issues": []
}}

Or, when a bounded rewrite can fix the result:
{{
  "status": "REPAIRABLE",
  "issues": [
    {{
      "block": "preferences",
      "code": "unsupported_claim",
      "message": "<what is wrong>",
      "repair_instruction": "<specific bounded correction>"
    }}
  ]
}}

Use status FATAL only when the candidate Core is too damaged or ambiguous to
repair safely. Be conservative: uncertainty about a changed claim is a reason
to remove that claim during repair, not a reason to invent supporting facts.
"""


# =====================================================================
# Phase 6: REPAIR CORE
# Repair only blocks rejected by the quality evaluator.
# =====================================================================

REPAIR_CORE_PROMPT = """You are dreamer's Core repair agent.

The current Sleep cycle changed a small set of Core blocks, and validation
found specific issues. Produce bounded repairs for those blocks only.

Validation context:
{validation_context_json}

Issues:
{issues_json}

Output strictly this JSON:
{{
  "repairs": [
    {{
      "block": "preferences",
      "new_block_value": "<complete repaired block value>",
      "reason": "<which validation issues this fixes>"
    }}
  ]
}}

Rules:
- Only modify blocks present in validation_context.changed_blocks.
- new_block_value is the COMPLETE block value, not a diff.
- Preserve useful original information unless an issue explicitly rejects it.
- Use only the original Core and listed source facts as evidence.
- Do not add new facts, clean up unrelated old content, or modify unchanged blocks.
- Keep each value within its char_limit.
"""


# =====================================================================
# Phase 7: REFLECT
# Summarize this cycle's operations.
# =====================================================================

REFLECT_PROMPT = """You are dreamer's Sleep agent in the REFLECT phase.

Summarize only the memory changes already applied to staging in this Sleep
cycle. The user (or developer) will read this in memory_ops_log to understand
what the cycle changed and why.

Pending operations:
{pending_ops_json}

Output a single string of 2-4 sentences. No JSON, no markdown. Just plain text.
Tone: factual and concise. Mention only operations present above; do not infer
or restate a general user profile.

Example:
"Merged two overlapping development-tool preferences and promoted one
frequently used communication preference into Core. The changed Core blocks
passed validation without repair."
"""


# =====================================================================
# Mapping for the StateGraph
# =====================================================================

PROMPTS = {
    "plan": PLAN_PROMPT,
    "consolidate": CONSOLIDATE_PROMPT,
    "promote": PROMOTE_PROMPT,
    "demote": DEMOTE_PROMPT,
    "resolve": RESOLVE_PROMPT,
    "validate_core": VALIDATE_CORE_PROMPT,
    "repair_core": REPAIR_CORE_PROMPT,
    "reflect": REFLECT_PROMPT,
}
