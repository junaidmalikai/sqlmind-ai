"""Autonomous Learning — planner improves from historical experiences."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PlannerExperienceStore:
    """Store successful/failed plans, agent sequences, SQL & chart strategies."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        ensure_dirs(Path(db_path).parent)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    tenant_id TEXT,
                    database_name TEXT,
                    question TEXT,
                    goal_summary TEXT,
                    success INTEGER,
                    agent_sequence_json TEXT,
                    sql_strategy TEXT,
                    chart_strategy TEXT,
                    plan_json TEXT,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pe_success ON planner_experiences(success)"
            )

    def record(
        self,
        *,
        question: str,
        goal_summary: str = "",
        success: bool,
        agent_sequence: list[str] | None = None,
        sql_strategy: str = "",
        chart_strategy: str = "",
        plan: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        session_id: str = "",
        tenant_id: str = "default",
        database_name: str = "",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO planner_experiences
                (session_id, tenant_id, database_name, question, goal_summary, success,
                 agent_sequence_json, sql_strategy, chart_strategy, plan_json, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tenant_id,
                    database_name,
                    question,
                    goal_summary,
                    1 if success else 0,
                    json.dumps(agent_sequence or []),
                    sql_strategy,
                    chart_strategy,
                    json.dumps(plan or {}, default=str),
                    json.dumps(metrics or {}, default=str),
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def record_from_result(
        self, result: dict[str, Any], *, tenant_id: str = "default"
    ) -> int:
        route = list(result.get("route_history") or [])
        plan = result.get("adaptive_plan") or {}
        success = bool(result.get("query_success")) or (
            bool(result.get("final_response")) and not result.get("error")
        )
        return self.record(
            question=result.get("question") or "",
            goal_summary=(result.get("goal_spec") or {}).get("goal")
            or plan.get("goal_summary")
            or "",
            success=success,
            agent_sequence=route,
            sql_strategy=(result.get("sql_explanation") or "")[:500],
            chart_strategy=str(result.get("chart_type") or ""),
            plan=plan if isinstance(plan, dict) else {},
            metrics={
                "row_count": result.get("row_count"),
                "retry_count": result.get("retry_count"),
                "reflection_count": result.get("reflection_count"),
                "replan_count": (plan or {}).get("replan_count"),
            },
            session_id=result.get("session_id") or "",
            tenant_id=tenant_id,
            database_name=result.get("database_name") or "",
        )

    def best_agent_sequences(self, *, top_k: int = 5) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_sequence_json, success FROM planner_experiences
                WHERE agent_sequence_json IS NOT NULL AND agent_sequence_json != '[]'
                ORDER BY id DESC LIMIT 500
                """
            ).fetchall()
        counter: Counter[str] = Counter()
        success_map: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            seq = row["agent_sequence_json"]
            counter[seq] += 1
            success_map[seq].append(int(row["success"]))
        ranked = []
        for seq, count in counter.most_common(top_k * 3):
            succ = success_map[seq]
            rate = sum(succ) / max(len(succ), 1)
            ranked.append(
                {
                    "sequence": json.loads(seq),
                    "count": count,
                    "success_rate": round(rate, 3),
                }
            )
        ranked.sort(key=lambda x: (x["success_rate"], x["count"]), reverse=True)
        return ranked[:top_k]

    def best_sql_strategies(self, *, top_k: int = 5) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sql_strategy FROM planner_experiences
                WHERE success = 1 AND sql_strategy != ''
                ORDER BY id DESC LIMIT 200
                """
            ).fetchall()
        seen: list[str] = []
        for r in rows:
            s = (r["sql_strategy"] or "").strip()
            if s and s not in seen:
                seen.append(s)
            if len(seen) >= top_k:
                break
        return seen

    def best_chart_strategies(self, *, top_k: int = 5) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chart_strategy, COUNT(*) as c FROM planner_experiences
                WHERE success = 1 AND chart_strategy != '' AND chart_strategy != 'none'
                GROUP BY chart_strategy ORDER BY c DESC LIMIT ?
                """,
                (top_k,),
            ).fetchall()
        return [r["chart_strategy"] for r in rows]

    def format_for_planner(self, question: str = "", *, max_chars: int = 2000) -> str:
        """Hints injected into Planner context before planning."""
        parts = ["## Planner learning hints"]
        seqs = self.best_agent_sequences(top_k=3)
        if seqs:
            parts.append("Best agent sequences:")
            for s in seqs:
                parts.append(
                    f"- rate={s['success_rate']} n={s['count']}: {' → '.join(s['sequence'][:8])}"
                )
        sqls = self.best_sql_strategies(top_k=3)
        if sqls:
            parts.append("Successful SQL strategies:")
            for s in sqls:
                parts.append(f"- {s[:200]}")
        charts = self.best_chart_strategies(top_k=3)
        if charts:
            parts.append(f"Preferred charts: {', '.join(charts)}")
        text = "\n".join(parts)
        return text[:max_chars] if len(parts) > 1 else ""
