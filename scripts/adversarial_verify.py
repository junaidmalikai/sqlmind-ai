"""Adversarial runtime verification of agentic claims — produces JSON evidence.

Run: uv run python scripts/adversarial_verify.py
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

# Prefer a tools-capable local model present on this machine
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "qwen3:8b")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from agents.nodes import make_retry_agent, make_supervisor_agent
from database.connector import DatabaseConfig, DatabaseConnector
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaInspector
from graph.state import empty_state
from graph.workflow import SQLMindOrchestrator, build_sqlmind_graph, route_next
from sample_data.seed import create_sample_database
from services.llm_service import LLMService
from tools.routing_tools import ROUTE_TOOL_TO_NODE, build_routing_tools
from utils.security import SQLSecurityGuard

OUT = Path("data/verification_evidence.json")
EVIDENCE: dict[str, Any] = {"tests": {}, "meta": {}}


def _log(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def setup_db(tmp: Path):
    db = create_sample_database(tmp / "retail.db")
    config = DatabaseConfig(dialect="sqlite", sqlite_path=db, database="retail")
    connector = DatabaseConnector(config)
    engine = connector.connect()
    schema = SchemaInspector(engine, "sqlite", "retail").discover()
    security = SQLSecurityGuard(
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
        dialect="sqlite",
        max_rows=200,
    )
    executor = QueryExecutor(engine, security=security, database_label="retail")
    return connector, schema, executor, security


# ---------------------------------------------------------------------------
# Test 1a — Live supervisor: analytics vs non-SQL question
# ---------------------------------------------------------------------------

def test1_live_routing(llm: LLMService, schema, executor, security) -> dict[str, Any]:
    """Live Ollama: supervisor first tool-call only (not full multi-agent graph).

    Full-graph live runs with local 8B models often hang for many minutes; the
    claim under test is that the *first* routing decision is LLM tool-call driven
    and differs by question type.
    """
    result: dict[str, Any] = {"name": "Test1_live_routing", "runs": [], "mode": "supervisor_only"}
    agent = make_supervisor_agent(llm)

    questions = [
        (
            "analytics",
            "Show total number of customers by city — I need a SQL query result",
            {"prefer": {"sql_agent", "schema_agent"}},
        ),
        (
            "non_sql",
            "Just summarize what this database is about for an executive. "
            "Do NOT generate or run SQL. Prefer a database summary.",
            {"prefer": {"summary_agent", "schema_agent", "finalize", "clarify"}},
        ),
    ]

    for label, question, meta in questions:
        state = empty_state(question, session_id=f"t1-{label}")
        state.update(
            {
                "schema_text": schema.to_prompt_text()[:3000],
                "database_name": schema.database_name,
                "dialect": schema.dialect,
                "supervisor_visit": 0,
            }
        )
        t0 = time.time()
        try:
            out = agent(state)
            elapsed = round(time.time() - t0, 2)
            next_agent = out.get("next_agent")
            log = (out.get("agent_logs") or [{}])[0]
            result["runs"].append(
                {
                    "label": label,
                    "question": question,
                    "elapsed_s": elapsed,
                    "next_agent": next_agent,
                    "supervisor_log": log.get("message"),
                    "reasoning": (out.get("supervisor_reasoning") or "")[:400],
                    "status": out.get("status"),
                    "in_preferred_set": next_agent in meta["prefer"],
                    "preferred_set": sorted(meta["prefer"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            result["runs"].append(
                {
                    "label": label,
                    "question": question,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-1200:],
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )

    if len(result["runs"]) == 2 and all("next_agent" in r for r in result["runs"]):
        result["routing_differed"] = (
            result["runs"][0]["next_agent"] != result["runs"][1]["next_agent"]
        )
    else:
        result["routing_differed"] = False
    return result


# ---------------------------------------------------------------------------
# Test 1b — Malformed tool-call → fail path (no DEFAULT_PLANS)
# ---------------------------------------------------------------------------

def test1_malformed_no_fallback(schema, executor, security) -> dict[str, Any]:
    class BrokenLLM:
        """Always returns plain text — no tool_calls."""

        def history_messages(self, history, limit: int = 12):  # noqa: ANN001
            return []

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001
            def _invoke(messages):  # noqa: ANN001
                return AIMessage(content="I think we should run SQL next.")

            return RunnableLambda(_invoke)

        def invoke_structured(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("should not be used")

    llm = BrokenLLM()
    app = build_sqlmind_graph(
        schema, executor, llm=llm, security=security, checkpointer=MemorySaver()  # type: ignore[arg-type]
    )
    initial = empty_state("show sales", session_id="t1-break")
    initial["schema_text"] = schema.to_prompt_text()
    initial["database_name"] = "retail"
    initial["dialect"] = "sqlite"
    initial["max_supervisor_visits"] = 3

    result = app.invoke(initial, config={"configurable": {"thread_id": "t1-break"}})
    # Check no DEFAULT_PLANS in codebase for silent inject
    from graph import state as state_mod

    has_default_plans_const = hasattr(state_mod, "DEFAULT_PLANS")

    return {
        "name": "Test1_malformed_tool_call",
        "status": result.get("status"),
        "next_agent_final_path": result.get("status"),
        "final_response_snippet": (result.get("final_response") or "")[:400],
        "error_field": result.get("error"),
        "agent_logs": result.get("agent_logs"),
        "took_fail_or_failed_status": result.get("status") == "failed"
        or any(
            (l.get("agent") == "supervisor" and "No tool" in (l.get("message") or ""))
            for l in (result.get("agent_logs") or [])
        )
        or "fail" in str(result.get("route_history") or []).lower()
        or result.get("status") == "failed",
        "DEFAULT_PLANS_constant_still_defined": has_default_plans_const,
        "normalize_plan_empty_returns": __import__(
            "graph.state", fromlist=["normalize_plan"]
        ).normalize_plan([], "query"),
    }


# ---------------------------------------------------------------------------
# Test 1c — Instrument: tool_call name must equal routed next_agent
# ---------------------------------------------------------------------------

def test1_toolcall_drives_route() -> dict[str, Any]:
    traces = []

    class ScriptedRouteLLM:
        def __init__(self, tool_name: str) -> None:
            self.tool_name = tool_name
            self.bound_tools = None

        def history_messages(self, history, limit: int = 12):  # noqa: ANN001
            return []

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001
            self.bound_tools = [t.name for t in tools]
            name = self.tool_name

            def _invoke(messages):  # noqa: ANN001
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "x",
                            "name": name,
                            "args": {"reasoning": f"forced:{name}"},
                        }
                    ],
                )

            return RunnableLambda(_invoke)

    results = []
    for tool_name, expected_node in [
        ("route_to_summary_agent", "summary_agent"),
        ("route_to_sql_agent", "sql_agent"),
        ("route_to_finalize", "finalize"),
        ("not_a_real_tool", "fail"),
    ]:
        llm = ScriptedRouteLLM(tool_name)
        agent = make_supervisor_agent(llm)  # type: ignore[arg-type]
        state = empty_state("q")
        state["database_name"] = "retail"
        state["dialect"] = "sqlite"
        out = agent(state)
        mapped = ROUTE_TOOL_TO_NODE.get(tool_name)
        results.append(
            {
                "emitted_tool_call": tool_name,
                "ROUTE_TOOL_TO_NODE_map": mapped,
                "state_next_agent": out.get("next_agent"),
                "matches_map": out.get("next_agent") == (mapped or "fail"),
                "supervisor_log": (out.get("agent_logs") or [{}])[0].get("message"),
                "bound_tool_names_sample": (llm.bound_tools or [])[:5],
            }
        )
        traces.append(out)

    return {
        "name": "Test1_toolcall_drives_next_agent",
        "cases": results,
        "all_match": all(c["matches_map"] for c in results),
    }


# ---------------------------------------------------------------------------
# Test 2 — Retry diagnostic: different failures → different SQL attempts
# ---------------------------------------------------------------------------

def test2_retry_diagnostic(llm: LLMService, schema, executor, security) -> dict[str, Any]:
    """Two failure contexts: capture diagnosis + regenerated SQL side by side."""
    from models.structured import RetryDecisionModel, SQLResponseModel
    from prompts.templates import retry_prompt, sql_prompt

    cases = []
    failures = [
        {
            "label": "validation_mutation",
            "sql": "DELETE FROM customers",
            "errors": "Forbidden statement type / mutation not allowed",
            "question": "remove all customers",
        },
        {
            "label": "unknown_column",
            "sql": "SELECT customer_names, cities FROM customers LIMIT 5",
            "errors": "no such column: customer_names",
            "question": "list customer names and cities",
        },
    ]

    for f in failures:
        # Live LLM diagnosis
        decision: RetryDecisionModel = llm.invoke_structured(
            retry_prompt(),
            RetryDecisionModel,
            {
                "dialect": "sqlite",
                "schema_text": schema.to_prompt_text()[:4000],
                "question": f["question"],
                "sql": f["sql"],
                "errors": f["errors"],
                "retry_count": 0,
                "max_retries": 3,
            },
        )
        # Live LLM regenerate with that diagnosis
        regenerated: SQLResponseModel = llm.invoke_structured(
            sql_prompt(),
            SQLResponseModel,
            {
                "history": [],
                "dialect": "sqlite",
                "schema_text": schema.to_prompt_text()[:4000],
                "question": f["question"],
                "memory_summary": "(none)",
                "prior_sql": f["sql"],
                "fix_hint": f"{decision.fix_hint} | {decision.revised_approach}",
                "prior_error": f["errors"],
            },
        )
        cases.append(
            {
                "label": f["label"],
                "attempt1_sql": f["sql"],
                "error": f["errors"],
                "diagnosis_reasoning": decision.reasoning,
                "failure_class": decision.failure_class,
                "next_action": decision.next_action,
                "fix_hint": decision.fix_hint,
                "revised_approach": decision.revised_approach,
                "attempt2_sql": regenerated.sql,
                "attempt2_differs": regenerated.sql.strip().upper()
                != f["sql"].strip().upper(),
                "attempt2_is_readonly": "DELETE" not in regenerated.sql.upper()
                and "DROP" not in regenerated.sql.upper(),
            }
        )

    # Also run retry_agent node path for both
    agent = make_retry_agent(llm)
    node_outs = []
    for f in failures:
        st = empty_state(f["question"])
        st.update(
            {
                "sql": f["sql"],
                "sql_error": f["errors"],
                "schema_text": schema.to_prompt_text()[:4000],
                "dialect": "sqlite",
                "retry_count": 0,
                "max_retries": 3,
            }
        )
        out = agent(st)
        node_outs.append(
            {
                "label": f["label"],
                "retry_diagnosis": out.get("retry_diagnosis"),
                "fix_hint": out.get("fix_hint"),
                "next_agent": out.get("next_agent"),
                "retry_next_action": out.get("retry_next_action"),
            }
        )

    hints_differ = (
        len(node_outs) == 2 and node_outs[0]["fix_hint"] != node_outs[1]["fix_hint"]
    )
    sqls_differ = all(c["attempt2_differs"] for c in cases)

    return {
        "name": "Test2_retry_diagnostic",
        "cases": cases,
        "retry_agent_node_outs": node_outs,
        "fix_hints_differ_across_failure_types": hints_differ,
        "attempt2_sql_differs_from_attempt1_all_cases": sqls_differ,
        "diagnoses_differ": cases[0]["diagnosis_reasoning"] != cases[1]["diagnosis_reasoning"]
        if len(cases) == 2
        else False,
    }


# ---------------------------------------------------------------------------
# Test 3 — Supervisor routing changes with context
# ---------------------------------------------------------------------------

def test3_supervisor_context(llm: LLMService) -> dict[str, Any]:
    agent = make_supervisor_agent(llm)
    runs = []

    # Fresh: no results — expect schema or sql or summary depending on question
    fresh = empty_state("Show revenue by city")
    fresh.update(
        {
            "database_name": "retail",
            "dialect": "sqlite",
            "schema_text": "",
            "query_success": False,
            "row_count": 0,
            "insights": "",
            "chart_spec": {},
            "conversation_history": [],
            "memory_summary": "",
            "episodic_context": "(none)",
        }
    )
    out_fresh = agent(fresh)
    runs.append(
        {
            "label": "fresh_no_results",
            "next_agent": out_fresh.get("next_agent"),
            "log": (out_fresh.get("agent_logs") or [{}])[0].get("message"),
            "reasoning": (out_fresh.get("supervisor_reasoning") or "")[:300],
        }
    )

    # After successful query with insights already present — should lean finalize/export/reflect
    after = empty_state("Show revenue by city")
    after.update(
        {
            "database_name": "retail",
            "dialect": "sqlite",
            "schema_text": "TABLE orders(...)",
            "query_success": True,
            "row_count": 42,
            "sql": "SELECT city, SUM(amount) FROM orders GROUP BY city",
            "insights": "City A leads revenue.",
            "chart_spec": {"chart_type": "bar"},
            "chart_type": "bar",
            "reflection_verdict": "",
            "conversation_history": [
                {"role": "user", "content": "Show revenue by city"},
                {"role": "assistant", "content": "City A leads."},
            ],
            "memory_summary": "User asked revenue by city; query succeeded.",
            "episodic_context": "- Q: Show revenue by city\n  SQL: SELECT ...\n  Outcome: City A leads",
            "route_history": ["sql_agent", "visualization_agent", "insight_agent"],
        }
    )
    out_after = agent(after)
    runs.append(
        {
            "label": "after_success_with_insights",
            "next_agent": out_after.get("next_agent"),
            "log": (out_after.get("agent_logs") or [{}])[0].get("message"),
            "reasoning": (out_after.get("supervisor_reasoning") or "")[:300],
        }
    )

    return {
        "name": "Test3_supervisor_context",
        "runs": runs,
        "routing_differed": runs[0]["next_agent"] != runs[1]["next_agent"],
    }


# ---------------------------------------------------------------------------
# Test 4 — Streaming timestamps (provider-level)
# ---------------------------------------------------------------------------

def test4_streaming(llm: LLMService) -> dict[str, Any]:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a short sentence about retail analytics."),
            ("human", "One sentence only."),
        ]
    )
    chunks: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    err = None
    try:
        for i, chunk in enumerate(llm.stream_text(prompt, {})):
            chunks.append(
                {
                    "i": i,
                    "t_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "len": len(chunk),
                    "preview": chunk[:40],
                }
            )
            if i >= 40:
                break
    except Exception as exc:  # noqa: BLE001
        err = str(exc)

    # Check UI does not contain fake typewriter helper
    chat_src = Path("ui/pages/chat.py").read_text(encoding="utf-8")
    has_token_chunks = "_token_chunks" in chat_src
    has_write_stream = "write_stream" in chat_src
    has_stream_answer = "stream_answer_tokens" in chat_src

    multi_chunk = len(chunks) >= 2
    staggered = False
    if multi_chunk:
        staggered = chunks[-1]["t_ms"] > chunks[0]["t_ms"]

    return {
        "name": "Test4_streaming",
        "error": err,
        "n_chunks": len(chunks),
        "chunks_sample": chunks[:12],
        "multiple_chunks_received": multi_chunk,
        "timestamps_increase": staggered,
        "ui_has_fake__token_chunks": has_token_chunks,
        "ui_uses_write_stream": has_write_stream,
        "ui_calls_stream_answer_tokens": has_stream_answer,
        "provider": llm.settings.llm_provider,
        "model": llm.settings.resolve_model(),
    }


# ---------------------------------------------------------------------------
# Test 5 — Spot checks
# ---------------------------------------------------------------------------

def test5_spotchecks(llm: LLMService, schema, executor, security) -> dict[str, Any]:
    from agents.nodes import make_reflection_agent, make_schema_agent
    from models.structured import ReflectionModel
    from memory.episodic import EpisodicMemoryStore
    from tools.sqlmind_tools import build_toolbelt, tools_by_name

    checks: dict[str, Any] = {}

    # Reflection changes path
    ref_llm_accept = llm
    # Live reflection
    agent = make_reflection_agent(llm)
    st = empty_state("customers by city")
    st.update(
        {
            "sql": "SELECT city, COUNT(*) n FROM customers GROUP BY city",
            "query_success": True,
            "row_count": 5,
            "insights": "Lahore has the most customers.",
            "chart_type": "bar",
            "dataframe_records": [{"city": "Lahore", "n": 3}],
            "reflection_count": 0,
            "max_reflections": 2,
        }
    )
    try:
        out = agent(st)
        checks["reflection"] = {
            "verdict": out.get("reflection_verdict"),
            "next_agent": out.get("next_agent"),
            "notes": (out.get("reflection_notes") or "")[:300],
            "ok": out.get("reflection_verdict") in {
                "accept",
                "retry_sql",
                "improve_insights",
                "replan",
                "clarify",
            },
        }
    except Exception as exc:  # noqa: BLE001
        checks["reflection"] = {"error": str(exc), "ok": False}

    # Episodic retrieve injected
    store = EpisodicMemoryStore("data/verify_episodes.sqlite3")
    store.record(
        question="Top customers by spend",
        sql="SELECT name, SUM(total) FROM orders GROUP BY name",
        outcome="Alice leads",
        success=True,
        database_name="retail",
    )
    ctx = store.format_for_prompt("customer spend leaders", database_name="retail")
    checks["episodic"] = {
        "context_preview": ctx[:400],
        "ok": "SQL:" in ctx and "(no prior episodes)" not in ctx,
    }

    # Schema bind_tools — live
    tool_map = tools_by_name(
        build_toolbelt(
            schema=schema,
            executor=executor,
            security=security,
            export_dir="exports",
        )
    )
    schema_agent = make_schema_agent(schema, llm, tool_map)
    try:
        sout = schema_agent(empty_state("what tables exist"))
        logs = sout.get("agent_logs") or []
        checks["schema_bind_tools"] = {
            "logs": logs,
            "suggestions_n": len(sout.get("suggested_questions") or []),
            "tool_call_logged": any(
                "tool_call:schema_tool" in (l.get("message") or "") for l in logs
            ),
            "ok": bool(sout.get("schema_text")),
        }
    except Exception as exc:  # noqa: BLE001
        checks["schema_bind_tools"] = {"error": str(exc), "ok": False}

    # Parallel Send still present
    from graph.workflow import route_after_execution

    s = empty_state("q")
    s["query_success"] = True
    fan = route_after_execution(s)
    checks["parallel_send"] = {
        "type": type(fan).__name__,
        "is_list": isinstance(fan, list),
        "len": len(fan) if isinstance(fan, list) else None,
        "ok": isinstance(fan, list) and len(fan) == 2,
    }

    # Export/finalize naming
    nodes_src = Path("agents/nodes.py").read_text(encoding="utf-8")
    checks["export_finalize_labeling"] = {
        "has_make_export_node": "def make_export_node" in nodes_src,
        "has_finalize_node": "def finalize_node" in nodes_src,
        "doc_says_not_ai": "not an AI agent" in nodes_src.lower()
        or "Deterministic" in nodes_src,
        "ok": "def make_export_node" in nodes_src and "def finalize_node" in nodes_src,
    }

    return {"name": "Test5_spotchecks", "checks": checks}


def main() -> None:
    tmp = Path("data/verify_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    connector, schema, executor, security = setup_db(tmp)

    # Point settings at qwen3:8b for this process
    from config.settings import get_settings

    get_settings.cache_clear()
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "qwen3:8b"
    get_settings.cache_clear()
    llm = LLMService(get_settings())
    EVIDENCE["meta"] = {
        "provider": llm.settings.llm_provider,
        "model": llm.settings.resolve_model(),
        "ollama_base": llm.settings.ollama_base_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _log(f"Using provider={EVIDENCE['meta']['provider']} model={EVIDENCE['meta']['model']}")

    try:
        _log("=== Test 1c toolcall drives route (instrumented) ===")
        EVIDENCE["tests"]["t1_instrumented"] = test1_toolcall_drives_route()
        _log(json.dumps(EVIDENCE["tests"]["t1_instrumented"], indent=2)[:2000])

        _log("=== Test 1b malformed -> fail ===")
        EVIDENCE["tests"]["t1_malformed"] = test1_malformed_no_fallback(
            schema, executor, security
        )
        _log(json.dumps(EVIDENCE["tests"]["t1_malformed"], indent=2)[:2000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")

        _log("=== Test 1a live routing (Ollama) ===")
        EVIDENCE["tests"]["t1_live"] = test1_live_routing(llm, schema, executor, security)
        _log(json.dumps(EVIDENCE["tests"]["t1_live"], indent=2)[:4000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")

        _log("=== Test 2 retry diagnostic (Ollama) ===")
        EVIDENCE["tests"]["t2"] = test2_retry_diagnostic(llm, schema, executor, security)
        _log(json.dumps(EVIDENCE["tests"]["t2"], indent=2)[:4000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")

        _log("=== Test 3 supervisor context (Ollama) ===")
        EVIDENCE["tests"]["t3"] = test3_supervisor_context(llm)
        _log(json.dumps(EVIDENCE["tests"]["t3"], indent=2)[:3000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")

        _log("=== Test 4 streaming ===")
        EVIDENCE["tests"]["t4"] = test4_streaming(llm)
        _log(json.dumps(EVIDENCE["tests"]["t4"], indent=2)[:3000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")

        _log("=== Test 5 spot checks ===")
        EVIDENCE["tests"]["t5"] = test5_spotchecks(llm, schema, executor, security)
        _log(json.dumps(EVIDENCE["tests"]["t5"], indent=2)[:4000])
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        EVIDENCE["fatal_error"] = str(exc)
        EVIDENCE["fatal_traceback"] = traceback.format_exc()
        _log(f"FATAL: {exc}")
        _log(traceback.format_exc()[-2000:])
    finally:
        connector.dispose()
        EVIDENCE["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(EVIDENCE, indent=2, default=str), encoding="utf-8")
        _log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
