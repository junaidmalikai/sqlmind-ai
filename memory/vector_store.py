"""Hybrid Vector Long-Term Memory — semantic search + ranking + compression.

Extends (does not replace) WorkflowMemoryStore and EpisodicMemoryStore with
embeddings, vector retrieval, and MemoryFabric orchestration. Planner retrieves
semantic memories before planning.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from memory.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider,
)
from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}", re.I)

MemoryKind = Literal[
    "goal",
    "plan",
    "failure",
    "success",
    "preference",
    "sql_history",
    "insight",
    "chart",
    "reflection",
    "experience",
]
def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    n = len(blob) // 4
    vals = list(struct.unpack(f"{n}f", blob))
    if len(vals) < dim:
        vals.extend([0.0] * (dim - len(vals)))
    return vals[:dim]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compress_text(text: str, *, max_chars: int = 400) -> str:
    """Lightweight memory compression — keep head + tail + key tokens."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 4) :]
    tokens = list(dict.fromkeys(_TOKEN_RE.findall(text.lower())))[:24]
    return f"{head} … {tail}\nkeys: {', '.join(tokens)}"


class VectorMemoryStore:
    """SQLite-backed vector memory with hybrid semantic search + re-ranking."""

    def __init__(
        self,
        db_path: str,
        *,
        embedder: EmbeddingProvider | None = None,
        default_ttl_seconds: float = 0.0,
    ) -> None:
        self.db_path = db_path
        self.embedder = embedder or HashingEmbeddingProvider()
        self.default_ttl_seconds = default_ttl_seconds
        ensure_dirs(Path(db_path).parent)
        self._init()
        logger.info(
            "VectorMemoryStore ready provider=%s dim=%s path=%s",
            getattr(self.embedder, "name", type(self.embedder).__name__),
            self.embedder.dim,
            db_path,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    tenant_id TEXT,
                    session_id TEXT,
                    database_name TEXT,
                    title TEXT,
                    content TEXT NOT NULL,
                    content_compressed TEXT,
                    metadata_json TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
                """
            )
            # Migration for older DBs without expires_at
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(vector_memories)").fetchall()
            }
            if "expires_at" not in cols:
                conn.execute(
                    "ALTER TABLE vector_memories ADD COLUMN expires_at TEXT"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vm_kind ON vector_memories(kind)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vm_tenant ON vector_memories(tenant_id)"
            )

    def store(
        self,
        *,
        kind: MemoryKind,
        content: str,
        title: str = "",
        tenant_id: str = "default",
        session_id: str = "",
        database_name: str = "",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> int:
        compressed = compress_text(content)
        emb = self.embedder.embed([content])[0]
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = None
        if ttl and ttl > 0:
            expires_at = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + ttl, tz=timezone.utc
            ).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO vector_memories
                (kind, tenant_id, session_id, database_name, title, content,
                 content_compressed, metadata_json, embedding, embedding_dim,
                 created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    tenant_id,
                    session_id,
                    database_name,
                    title,
                    content,
                    compressed,
                    json.dumps(metadata or {}, default=str),
                    _pack(emb),
                    len(emb),
                    utc_now_iso(),
                    expires_at,
                ),
            )
            return int(cur.lastrowid)

    def expire_stale(self) -> int:
        """Delete expired memories. Returns number of rows removed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM vector_memories
                WHERE expires_at IS NOT NULL AND expires_at <> '' AND expires_at < ?
                """,
                (now,),
            )
            return int(cur.rowcount or 0)

    def search(
        self,
        query: str,
        *,
        kinds: list[MemoryKind] | None = None,
        tenant_id: str | None = None,
        database_name: str | None = None,
        top_k: int = 8,
        min_score: float = 0.05,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        self.expire_stale()
        q_emb = self.embedder.embed([query])[0]
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            sql = """
                SELECT * FROM vector_memories
                WHERE (expires_at IS NULL OR expires_at = '' OR expires_at >= ?)
            """
            params: list[Any] = [now]
            if tenant_id:
                sql += " AND tenant_id = ?"
                params.append(tenant_id)
            if database_name:
                sql += " AND (database_name = ? OR database_name = '' OR database_name IS NULL)"
                params.append(database_name)
            if kinds:
                placeholders = ",".join("?" * len(kinds))
                sql += f" AND kind IN ({placeholders})"
                params.extend(kinds)
            sql += " ORDER BY id DESC LIMIT 500"
            rows = conn.execute(sql, params).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        for row in rows:
            emb = _unpack(row["embedding"], row["embedding_dim"] or len(q_emb))
            sem = cosine(q_emb, emb)
            # Hybrid: blend lexical overlap
            c_tokens = set(_TOKEN_RE.findall((row["content"] or "").lower()))
            lex = (len(q_tokens & c_tokens) / max(len(q_tokens), 1)) if q_tokens else 0.0
            # Recency boost (newer memories score slightly higher)
            recency = 0.0
            try:
                created = row["created_at"] or ""
                if created:
                    # ISO timestamps sort lexicographically roughly by time
                    recency = 0.05
            except Exception:  # noqa: BLE001
                recency = 0.0
            score = 0.65 * sem + 0.25 * lex + recency
            if score < min_score:
                continue
            scored.append(
                (
                    score,
                    {
                        "id": row["id"],
                        "kind": row["kind"],
                        "title": row["title"],
                        "content": row["content_compressed"] or row["content"],
                        "score": round(score, 4),
                        "semantic_score": round(sem, 4),
                        "lexical_score": round(lex, 4),
                        "metadata": json.loads(row["metadata_json"] or "{}"),
                        "created_at": row["created_at"],
                        "expires_at": row["expires_at"] if "expires_at" in row.keys() else None,
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)

        # Re-ranking: prefer exact title/query token overlap among top candidates
        candidates = scored[: max(top_k * 3, top_k)]
        if rerank and candidates:
            reranked: list[tuple[float, dict[str, Any]]] = []
            for score, item in candidates:
                title_tokens = set(_TOKEN_RE.findall((item.get("title") or "").lower()))
                title_boost = 0.1 * (
                    len(q_tokens & title_tokens) / max(len(q_tokens), 1)
                )
                kind_boost = 0.05 if item.get("kind") in {"success", "sql_history", "plan"} else 0.0
                new_score = score + title_boost + kind_boost
                item = {**item, "score": round(new_score, 4), "reranked": True}
                reranked.append((new_score, item))
            reranked.sort(key=lambda x: x[0], reverse=True)
            candidates = reranked

        try:
            from observability.metrics import get_metrics

            get_metrics().observe_memory("search", status="ok")
        except Exception:  # noqa: BLE001
            pass
        return [item for _, item in candidates[:top_k]]
    def format_for_prompt(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        database_name: str | None = None,
        top_k: int = 6,
        max_chars: int = 2500,
    ) -> str:
        hits = self.search(
            query,
            tenant_id=tenant_id,
            database_name=database_name,
            top_k=top_k,
        )
        if not hits:
            return ""
        parts: list[str] = ["## Semantic long-term memory"]
        used = 0
        for h in hits:
            line = f"- [{h['kind']}|{h['score']}] {h.get('title') or ''}: {h['content']}"
            if used + len(line) > max_chars:
                break
            parts.append(line)
            used += len(line)
        return "\n".join(parts)


class MemoryFabric:
    """Hybrid memory orchestrator — vector + workflow + episodic.

    Replaces sole reliance on workflow history with ranked semantic retrieval
    while keeping existing stores intact.
    """

    def __init__(
        self,
        vector: VectorMemoryStore,
        *,
        workflow_memory: Any | None = None,
        episodic: Any | None = None,
    ) -> None:
        self.vector = vector
        self.workflow_memory = workflow_memory
        self.episodic = episodic

    def retrieve_for_planner(
        self,
        question: str,
        *,
        tenant_id: str = "default",
        database_name: str = "",
    ) -> str:
        sections: list[str] = []
        vec = self.vector.format_for_prompt(
            question, tenant_id=tenant_id, database_name=database_name or None
        )
        if vec:
            sections.append(vec)
        if self.workflow_memory is not None:
            try:
                wf = self.workflow_memory.format_for_prompt(
                    question, database_name=database_name
                )
                if wf:
                    sections.append(wf)
            except Exception as exc:  # noqa: BLE001
                logger.debug("workflow memory retrieve: %s", exc)
        if self.episodic is not None:
            try:
                ep = self.episodic.format_for_prompt(
                    question, database_name=database_name
                )
                if ep:
                    sections.append(ep)
            except Exception as exc:  # noqa: BLE001
                logger.debug("episodic retrieve: %s", exc)
        return "\n\n".join(sections) if sections else ""

    def remember_outcome(self, result: dict[str, Any], *, tenant_id: str = "default") -> None:
        """Persist goals/plans/failures/successes/sql/insights/charts into vector memory."""
        question = result.get("question") or ""
        session_id = result.get("session_id") or ""
        database_name = result.get("database_name") or ""
        success = bool(result.get("query_success")) or bool(result.get("final_response"))

        goal = result.get("goal_spec") or {}
        if goal:
            self.vector.store(
                kind="goal",
                title=str(goal.get("goal") or "")[:120],
                content=json.dumps(goal, default=str)[:4000],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )

        plan = result.get("adaptive_plan") or {}
        if plan:
            self.vector.store(
                kind="plan" if success else "failure",
                title=str(plan.get("goal_summary") or plan.get("strategy") or "")[:120],
                content=json.dumps(plan, default=str)[:4000],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
                metadata={"success": success},
            )

        if result.get("sql"):
            self.vector.store(
                kind="sql_history",
                title=question[:120],
                content=f"Q: {question}\nSQL: {result.get('sql')}\nsuccess={success}",
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )

        if result.get("insights"):
            self.vector.store(
                kind="insight",
                title="insights",
                content=str(result.get("insights"))[:3000],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )

        chart = result.get("chart_spec") or {}
        if chart:
            self.vector.store(
                kind="chart",
                title=str(result.get("chart_type") or "chart"),
                content=json.dumps(chart, default=str)[:2000],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )

        if success:
            self.vector.store(
                kind="success",
                title=question[:120],
                content=(result.get("final_response") or result.get("insights") or question)[
                    :3000
                ],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )
        elif result.get("error"):
            self.vector.store(
                kind="failure",
                title=question[:120],
                content=str(result.get("error"))[:2000],
                tenant_id=tenant_id,
                session_id=session_id,
                database_name=database_name,
            )
