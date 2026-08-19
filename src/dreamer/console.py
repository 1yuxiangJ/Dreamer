"""Local web console for inspecting Dreamer memory state."""
# ruff: noqa: E501
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from dreamer.config import settings
from dreamer.db.models import get_sessionmaker
from dreamer.memory.inspect import collect_snapshot
from dreamer.memory.job_inspect import snapshot as collect_job_snapshot
from dreamer.memory.jobs import enqueue_awake_write
from dreamer.memory.worker import start_memory_write_worker, stop_memory_write_worker
from dreamer.sleep.agent import run_sleep_cycle
from dreamer.sleep.staging import ensure_archival_id_sequence

BULK_MEMORY_SEED_PATH = Path(__file__).parents[2] / "data" / "bulk_memory_seed_100.jsonl"
VALID_STABILITY = {"temporary", "stage", "long_term"}
CORE_BLOCK_LABELS = (
    "background",
    "preferences",
    "habits",
    "skills",
    "lessons_learned",
)


def build_remember_command(item: dict[str, Any]) -> str:
    tag_str = ", ".join(item["tags"]) if item["tags"] else "(none)"
    return (
        "remember this fact about the user:\n"
        f"  content: {item['content']}\n"
        f"  tags: {tag_str}\n"
        f"  confidence: {item['confidence']}\n"
        f"  stability: {item['stability']}\n"
        f"  salience: {item['salience']}\n"
        "First check for near-duplicates via find_archival_duplicates, then insert."
    )


def _validate_seed_item(raw: dict[str, Any], line_no: int) -> dict[str, Any]:
    content = raw.get("content")
    tags = raw.get("tags")
    confidence = raw.get("confidence")
    stability = raw.get("stability")
    salience = raw.get("salience")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"seed line {line_no}: content must be a non-empty string")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"seed line {line_no}: tags must be a list of strings")
    if confidence not in {1, 2, 3}:
        raise ValueError(f"seed line {line_no}: confidence must be 1, 2, or 3")
    if stability not in VALID_STABILITY:
        raise ValueError(f"seed line {line_no}: stability must be temporary, stage, or long_term")
    if salience not in {1, 2, 3}:
        raise ValueError(f"seed line {line_no}: salience must be 1, 2, or 3")
    return {
        "content": content.strip(),
        "tags": tags,
        "confidence": confidence,
        "stability": stability,
        "salience": salience,
    }


def load_bulk_memory_seed(path: Path | None = None) -> list[dict[str, Any]]:
    seed_path = path or BULK_MEMORY_SEED_PATH
    items: list[dict[str, Any]] = []
    with seed_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"seed line {line_no}: expected JSON object")
            items.append(_validate_seed_item(raw, line_no))
    return items


async def enqueue_bulk_memory_seed() -> dict[str, Any]:
    items = load_bulk_memory_seed()
    jobs: list[dict[str, Any]] = []
    for item in items:
        job = await enqueue_awake_write("remember", build_remember_command(item), item)
        jobs.append({
            "id": job.id,
            "status": job.status,
            "content": item["content"],
        })
    accepted_count = sum(1 for job in jobs if job["status"] in {"pending", "running", "succeeded"})
    return {
        "status": "ok",
        "mode": "durable_async",
        "dataset": str(BULK_MEMORY_SEED_PATH),
        "dataset_count": len(items),
        "accepted_count": accepted_count,
        "jobs": jobs,
    }


async def clear_all_memory_data() -> dict[str, Any]:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        before = {}
        for table_name in (
            "archival_facts",
            "memory_ops_log",
            "memory_write_jobs",
        ):
            count = (await session.execute(
                text(f"SELECT count(*) FROM {table_name}")
            )).scalar_one()
            before[table_name] = int(count)

        for table_name in (
            "core_blocks_staging",
            "archival_facts_staging",
        ):
            await session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))

        await ensure_archival_id_sequence(session)

        await session.execute(text(
            """
            TRUNCATE archival_facts, memory_ops_log, memory_write_jobs
            RESTART IDENTITY
            """
        ))
        await session.execute(text("DELETE FROM core_blocks"))
        for label in CORE_BLOCK_LABELS:
            await session.execute(
                text(
                    """
                    INSERT INTO core_blocks
                        (label, value, char_limit, version, last_writer, updated_at)
                    VALUES
                        (:label, '', 2000, 1, 'sleep_agent', now())
                    """
                ),
                {"label": label},
            )
        await session.execute(text(
            "ALTER SEQUENCE IF EXISTS archival_facts_id_seq RESTART WITH 1"
        ))
        await session.commit()

    return {
        "status": "ok",
        "cleared": before,
        "core_blocks_reset": len(CORE_BLOCK_LABELS),
        "staging_tables_dropped": [
            "core_blocks_staging",
            "archival_facts_staging",
        ],
    }


async def collect_console_snapshot() -> dict[str, Any]:
    """Collect dashboard data without invoking LLM-backed agents."""
    memory = await collect_snapshot(limit=40, include_deleted=False)
    jobs = await collect_job_snapshot(limit=None)
    return {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "memory": memory,
        "jobs": jobs,
    }


async def console_page(_request: Request) -> HTMLResponse:
    return HTMLResponse(CONSOLE_HTML)


async def console_snapshot(_request: Request) -> JSONResponse:
    return JSONResponse(await collect_console_snapshot())


async def console_run_sleep(_request: Request) -> JSONResponse:
    summary = await run_sleep_cycle()
    return JSONResponse({
        "status": "ok" if summary.get("status") == "ok" else "error",
        "summary": summary,
    })


async def console_clear_all(request: Request) -> JSONResponse:
    existing_worker = getattr(request.app.state, "memory_worker", None)
    await stop_memory_write_worker(existing_worker)
    request.app.state.memory_worker = None
    try:
        summary = await clear_all_memory_data()
    finally:
        if settings.memory_write_worker_enabled:
            request.app.state.memory_worker = start_memory_write_worker()
    return JSONResponse({"status": "ok", "summary": summary})


async def console_bulk_remember(_request: Request) -> JSONResponse:
    if not BULK_MEMORY_SEED_PATH.exists():
        return JSONResponse(
            {
                "status": "unavailable",
                "detail": "The local bulk seed dataset is not bundled with this repository.",
            },
            status_code=404,
        )
    return JSONResponse(await enqueue_bulk_memory_seed())


routes = [
    Route("/console", console_page, methods=["GET"]),
    Route("/api/console/snapshot", console_snapshot, methods=["GET"]),
    Route("/api/console/sleep/run", console_run_sleep, methods=["POST"]),
    Route("/api/console/clear/run", console_clear_all, methods=["POST"]),
    Route("/api/console/bulk-remember/run", console_bulk_remember, methods=["POST"]),
]


CONSOLE_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dreamer Console</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf7;
      --panel-2: #ebe6dc;
      --ink: #20231f;
      --muted: #6d6a62;
      --line: #d6d0c3;
      --green: #2f6f59;
      --green-soft: #dcebe4;
      --amber: #9a651f;
      --amber-soft: #f3e4c8;
      --red: #a94335;
      --red-soft: #f1d9d4;
      --steel: #405a63;
      --shadow: 0 18px 45px rgba(39, 35, 28, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(32,35,31,0.045) 1px, transparent 1px),
        linear-gradient(0deg, rgba(32,35,31,0.035) 1px, transparent 1px),
        radial-gradient(circle at top left, rgba(47,111,89,0.14), transparent 34rem),
        var(--bg);
      background-size: 34px 34px, 34px 34px, auto, auto;
      font-family: "Avenir Next", "Gill Sans", ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0;
    }

    button, input, select { font: inherit; }

    .shell {
      display: grid;
      grid-template-columns: 248px minmax(0, 1fr);
      min-height: 100vh;
    }

    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 24px 18px;
      background: rgba(255, 253, 247, 0.82);
      border-right: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }

    .brand {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 30px;
      line-height: 1;
      margin: 0 0 8px;
    }

    .subtitle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 28px;
    }

    nav {
      display: grid;
      gap: 7px;
    }

    nav a {
      color: var(--ink);
      text-decoration: none;
      padding: 10px 12px;
      border: 1px solid transparent;
      border-radius: 8px;
      font-size: 14px;
    }

    nav a:hover {
      background: var(--green-soft);
      border-color: #b8d2c7;
    }

    .main {
      padding: 28px;
      max-width: 1500px;
      width: 100%;
    }

    .topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(34px, 4vw, 58px);
      line-height: 1;
      letter-spacing: 0;
    }

    .dek {
      color: var(--muted);
      max-width: 720px;
      margin-top: 10px;
      line-height: 1.5;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .btn {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: #fffdf7;
      border-radius: 8px;
      padding: 10px 13px;
      cursor: pointer;
      min-height: 40px;
    }

    .btn.secondary {
      background: var(--panel);
      color: var(--ink);
      border-color: var(--line);
    }

    .btn.danger {
      background: var(--red);
      color: #fffdf7;
      border-color: var(--red);
    }

    .btn:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .stat, section {
      background: rgba(255, 253, 247, 0.88);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .stat {
      padding: 16px;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .value {
      font-size: 31px;
      line-height: 1.1;
      margin-top: 8px;
      font-family: Georgia, "Times New Roman", serif;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
      gap: 14px;
      align-items: start;
    }

    section {
      overflow: hidden;
      margin-bottom: 14px;
    }

    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(235, 230, 220, 0.55);
    }

    h2 {
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
    }

    .section-body {
      padding: 14px 16px 16px;
    }

    .core-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .core-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-height: 132px;
    }

    .core-card h3 {
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--green);
      letter-spacing: 0;
    }

    .core-card p, .empty {
      color: var(--muted);
      line-height: 1.45;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(244, 241, 234, 0.8);
    }

    td {
      line-height: 1.45;
    }

    .panel-scroll {
      overflow: auto;
      max-height: 420px;
    }

    .panel-scroll.compact {
      max-height: 300px;
    }

    .timeline-scroll {
      padding: 14px 16px 16px;
    }

    .panel-scroll th {
      position: sticky;
      top: 0;
      z-index: 1;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--panel-2);
      color: var(--steel);
      font-size: 12px;
      margin: 0 4px 4px 0;
      white-space: nowrap;
    }

    .chip.ok { color: var(--green); background: var(--green-soft); }
    .chip.warn { color: var(--amber); background: var(--amber-soft); }
    .chip.bad { color: var(--red); background: var(--red-soft); }

    .timeline {
      display: grid;
      gap: 10px;
    }

    .op {
      border-left: 3px solid var(--green);
      padding: 0 0 0 11px;
    }

    .op strong {
      display: block;
      font-size: 13px;
    }

    .op p {
      margin: 4px 0 0;
      color: var(--muted);
      line-height: 1.4;
      word-break: break-word;
    }

    .status-line {
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
      min-height: 20px;
    }

    .mono {
      font-family: "SFMono-Regular", Consolas, ui-monospace, monospace;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #24251f;
      color: #f8f2e7;
      padding: 14px;
      border-radius: 8px;
      max-height: 360px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.5;
    }

    @media (max-width: 1000px) {
      .shell { grid-template-columns: 1fr; }
      aside {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      nav { grid-template-columns: repeat(5, minmax(0, 1fr)); }
      nav a { text-align: center; padding: 9px 6px; }
      .topbar { flex-direction: column; }
      .actions { justify-content: flex-start; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 640px) {
      .main { padding: 18px; }
      nav { grid-template-columns: 1fr 1fr; }
      .stats { grid-template-columns: 1fr; }
      .core-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1 class="brand">Dreamer</h1>
      <div class="subtitle">Claude Code 的本地记忆控制台。集中查看 Durable Writes、Archival Facts、Sleep Ops 和 Core Blocks。</div>
      <nav>
        <a href="#overview">概览</a>
        <a href="#core">Core</a>
        <a href="#facts">Facts</a>
        <a href="#jobs">Jobs</a>
        <a href="#ops">Ops</a>
      </nav>
    </aside>
    <main class="main">
      <div class="topbar">
        <div>
          <h1>Dreamer Console</h1>
          <div class="dek">这是 Dreamer 的本地观察面，用来查看已经记住的内容、异步写入是否完成，以及 Sleep 对记忆做了哪些整理。</div>
          <div id="status" class="status-line">正在加载快照...</div>
        </div>
        <div class="actions">
          <button class="btn secondary" id="refreshBtn">刷新</button>
          <button class="btn secondary" id="seedBtn">批量 remember 100 条</button>
          <button class="btn danger" id="clearBtn">清空所有数据</button>
          <button class="btn" id="sleepBtn">运行 Sleep</button>
        </div>
      </div>

      <section id="overview">
        <div class="section-head"><h2>概览</h2><span id="generatedAt" class="label"></span></div>
        <div class="section-body stats" id="stats"></div>
      </section>

      <div class="grid">
        <div>
          <section id="core">
            <div class="section-head"><h2>Core Blocks</h2><span class="label">Sleep 管理</span></div>
            <div class="section-body"><div class="core-grid" id="coreBlocks"></div></div>
          </section>

          <section id="facts">
            <div class="section-head"><h2>Archival Facts</h2><span class="label">最新有效 facts</span></div>
            <div class="panel-scroll"><table id="factsTable"></table></div>
          </section>
        </div>

        <div>
          <section id="jobs">
            <div class="section-head"><h2>Write Jobs</h2><span class="label">remember</span></div>
            <div class="panel-scroll compact"><table id="jobsTable"></table></div>
          </section>

          <section id="ops">
            <div class="section-head"><h2>Ops Log</h2><span class="label">最近变更</span></div>
            <div class="panel-scroll compact timeline-scroll"><div class="timeline" id="opsTimeline"></div></div>
          </section>

          <section>
            <div class="section-head"><h2>操作结果</h2><span class="label">Sleep / bulk remember 输出</span></div>
            <div class="section-body"><pre id="sleepOutput">{}</pre></div>
          </section>
        </div>
      </div>
    </main>
  </div>

  <script>
    const els = {
      status: document.getElementById("status"),
      generatedAt: document.getElementById("generatedAt"),
      stats: document.getElementById("stats"),
      coreBlocks: document.getElementById("coreBlocks"),
      factsTable: document.getElementById("factsTable"),
      jobsTable: document.getElementById("jobsTable"),
      opsTimeline: document.getElementById("opsTimeline"),
      sleepOutput: document.getElementById("sleepOutput"),
      refreshBtn: document.getElementById("refreshBtn"),
      seedBtn: document.getElementById("seedBtn"),
      clearBtn: document.getElementById("clearBtn"),
      sleepBtn: document.getElementById("sleepBtn"),
    };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[ch]);
    }

    function fmtTime(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function chip(text, tone = "") {
      return `<span class="chip ${tone}">${esc(text)}</span>`;
    }

    function statusTone(status) {
      if (status === "succeeded" || status === "ok") return "ok";
      if (status === "failed" || status === "error") return "bad";
      if (status === "pending" || status === "running") return "warn";
      return "";
    }

    function renderStats(data) {
      const memory = data.memory || {};
      const counts = memory.counts || {};
      const jobs = ((data.jobs || {}).jobs || []);
      const failedJobs = jobs.filter((job) => job.status === "failed").length;
      const pendingJobs = jobs.filter((job) => job.status === "pending" || job.status === "running").length;
      const recentOps = (memory.recent_ops || []).length;
      const items = [
        ["有效 Facts", counts.active_archival_facts ?? 0],
        ["已删除 Facts", counts.deleted_archival_facts ?? 0],
        ["未完成 Jobs", pendingJobs],
        ["最近 Ops", recentOps],
      ];
      if (failedJobs > 0) items[2][1] = `${pendingJobs} 未完成 / ${failedJobs} failed`;
      els.stats.innerHTML = items.map(([label, value]) => `
        <div class="stat"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>
      `).join("");
    }

    function renderCore(blocks) {
      if (!blocks || blocks.length === 0) {
        els.coreBlocks.innerHTML = `<p class="empty">没有 Core Blocks。</p>`;
        return;
      }
      els.coreBlocks.innerHTML = blocks.map((block) => `
        <article class="core-card">
          <h3>${esc(block.label)} <span class="chip">v${esc(block.version)}</span></h3>
          <p>${esc(block.value || "空")}</p>
        </article>
      `).join("");
    }

    function renderFacts(facts) {
      if (!facts || facts.length === 0) {
        els.factsTable.innerHTML = `<tbody><tr><td class="empty">没有有效 Archival Facts。</td></tr></tbody>`;
        return;
      }
      els.factsTable.innerHTML = `
        <thead><tr><th>ID</th><th>内容</th><th>Signals</th><th>使用次数</th><th>创建时间</th></tr></thead>
        <tbody>
        ${facts.map((fact) => `
          <tr>
            <td class="mono">${esc(fact.id)}</td>
            <td>${esc(fact.content)}<div>${(fact.tags || []).map((tag) => chip(tag)).join("")}</div></td>
            <td>${chip("c" + fact.confidence)}${chip(fact.stability)}${chip("s" + fact.salience)}</td>
            <td class="mono">${esc(fact.use_count)}</td>
            <td>${esc(fmtTime(fact.created_at))}</td>
          </tr>
        `).join("")}
        </tbody>`;
    }

    function renderJobs(jobs) {
      if (!jobs || jobs.length === 0) {
        els.jobsTable.innerHTML = `<tbody><tr><td class="empty">没有 Write Jobs。</td></tr></tbody>`;
        return;
      }
      els.jobsTable.innerHTML = `
        <thead><tr><th>ID</th><th>Op</th><th>状态</th><th>尝试次数</th><th>更新时间</th></tr></thead>
        <tbody>
        ${jobs.map((job) => `
          <tr>
            <td class="mono">${esc(job.id)}</td>
            <td>${esc(job.operation)}</td>
            <td>${chip(job.status, statusTone(job.status))}${job.last_error ? `<p>${esc(job.last_error)}</p>` : ""}</td>
            <td class="mono">${esc(job.attempt_count)} / ${esc(job.max_attempts)}</td>
            <td>${esc(fmtTime(job.updated_at))}</td>
          </tr>
        `).join("")}
        </tbody>`;
    }

    function renderOps(ops) {
      if (!ops || ops.length === 0) {
        els.opsTimeline.innerHTML = `<p class="empty">没有最近 Ops。</p>`;
        return;
      }
      els.opsTimeline.innerHTML = ops.map((op) => `
        <div class="op">
          <strong>${esc(op.op_type)} ${chip(op.actor)} ${op.target_id ? chip(op.target_id) : ""}</strong>
          <p>${esc(op.reason || op.after_value_preview || op.before_value_preview || "没有记录 reason。")}</p>
          <p class="mono">${esc(fmtTime(op.ts))}</p>
        </div>
      `).join("");
    }

    function render(data) {
      els.generatedAt.textContent = data.generated_at ? `生成于 ${fmtTime(data.generated_at)}` : "";
      renderStats(data);
      renderCore((data.memory || {}).core_blocks || []);
      renderFacts((data.memory || {}).archival_facts || []);
      renderJobs(((data.jobs || {}).jobs || []));
      renderOps((data.memory || {}).recent_ops || []);
    }

    async function refresh() {
      els.status.textContent = "正在加载快照...";
      els.refreshBtn.disabled = true;
      try {
        const response = await fetch("/api/console/snapshot");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        render(data);
        els.status.textContent = "快照加载完成。";
      } catch (error) {
        els.status.textContent = `快照加载失败: ${error.message}`;
      } finally {
        els.refreshBtn.disabled = false;
      }
    }

    async function runSleep() {
      els.sleepBtn.disabled = true;
      els.sleepOutput.textContent = "Sleep 运行中...";
      try {
        const response = await fetch("/api/console/sleep/run", { method: "POST" });
        const data = await response.json();
        els.sleepOutput.textContent = JSON.stringify(data.summary || data, null, 2);
        await refresh();
      } catch (error) {
        els.sleepOutput.textContent = `Sleep 运行失败: ${error.message}`;
      } finally {
        els.sleepBtn.disabled = false;
      }
    }

    async function runBulkRemember() {
      els.seedBtn.disabled = true;
      els.sleepOutput.textContent = "正在批量入队 remember jobs...";
      try {
        const response = await fetch("/api/console/bulk-remember/run", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        els.sleepOutput.textContent = JSON.stringify(data, null, 2);
        await refresh();
      } catch (error) {
        els.sleepOutput.textContent = `批量 remember 失败: ${error.message}`;
      } finally {
        els.seedBtn.disabled = false;
      }
    }

    async function clearAllData() {
      const ok = window.confirm("确定清空所有 Dreamer 数据吗？这会删除 Facts、Ops Log、Write Jobs，并重置 Core Blocks。");
      if (!ok) return;
      els.clearBtn.disabled = true;
      els.sleepOutput.textContent = "正在清空所有数据...";
      try {
        const response = await fetch("/api/console/clear/run", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        els.sleepOutput.textContent = JSON.stringify(data.summary || data, null, 2);
        await refresh();
      } catch (error) {
        els.sleepOutput.textContent = `清空失败: ${error.message}`;
      } finally {
        els.clearBtn.disabled = false;
      }
    }

    els.refreshBtn.addEventListener("click", refresh);
    els.seedBtn.addEventListener("click", runBulkRemember);
    els.clearBtn.addEventListener("click", clearAllData);
    els.sleepBtn.addEventListener("click", runSleep);
    refresh();
  </script>
</body>
</html>
"""
