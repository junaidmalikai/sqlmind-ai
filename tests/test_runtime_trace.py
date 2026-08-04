"""Tests for the Runtime Execution Trace System."""

from __future__ import annotations

from pathlib import Path

from observability.runtime_trace import (
    TRACE_FILENAMES,
    ExecutionTraceSession,
    VERIFICATION_CHECKS,
    execution_trace_session,
    export_execution_trace,
    finalize_and_export,
    get_active_trace,
    get_last_completed_trace,
    safe_trace,
    verify_architecture,
    wrap_trace_node,
)


def test_execution_trace_session_context() -> None:
    assert get_active_trace() is None
    with execution_trace_session(
        session_id="sess-1",
        user_id="user-1",
        tenant_id="tenant-1",
        llm_provider="ollama",
        model_name="qwen3:8b",
        question="how many orders?",
    ) as session:
        assert get_active_trace() is session
        assert session.execution_id
        safe_trace("node_enter", node_name="goal_understanding", state={"session_id": "sess-1"})
        safe_trace(
            "node_exit",
            node_name="goal_understanding",
            state_patch={"next_agent": "planner", "goal_spec": {"goal_id": "g1", "confidence": 0.9}},
            status="ok",
        )
        safe_trace("iam_check", action="agent_invoke", resource="agent.planner", allowed=True)
        safe_trace(
            "message_bus",
            bus_event="Publish",
            sender="goal_understanding",
            receiver="planner",
            msg_type="goal",
        )
        safe_trace("memory_event", kind="Hybrid Search", memories="past sql…")
        safe_trace("plugin_event", kind="Plugin Discovery", detail={"count": 1})
        safe_trace(
            "reliability_event",
            kind="Circuit Breaker",
            node_name="sql_agent",
            status="allow",
        )
        safe_trace(
            "node_enter",
            node_name="memory_graph",
            state={"session_id": "sess-1"},
        )
        safe_trace(
            "node_exit",
            node_name="memory_graph",
            state_patch={"next_agent": "planning_graph"},
            status="ok",
        )
        assert len(session.all_events()) >= 3
    assert get_active_trace() is None
    assert get_last_completed_trace() is not None


def test_export_writes_all_artifacts(tmp_path: Path) -> None:
    session = ExecutionTraceSession(
        session_id="s",
        user_id="u",
        tenant_id="t",
        llm_provider="ollama",
        model_name="m",
    )
    session.mark_start(question="q")
    session.node_enter("planner", state={"session_id": "s"})
    session.node_exit("planner", state_patch={"next_agent": "task_decomposition"})
    session.iam_check(action="agent_invoke", resource="agent.planner", allowed=True)
    session.message_bus(
        bus_event="Publish", sender="planner", receiver="coordinator", msg_type="plan"
    )
    session.memory_event(kind="Memory Retrieval", memories="ctx")
    session.plugin_event(kind="Plugin Discovery", detail={"count": 0})
    session.reliability_event(kind="Circuit Breaker", status="allow")
    session.tool_event(tool_name="schema_tool", arguments={"x": 1}, result="ok")
    paths = finalize_and_export(
        session,
        output_dir=tmp_path,
        success=True,
        final_state={"session_id": "s", "final_response": "done", "query_success": True},
        print_summary=False,
    )
    assert set(TRACE_FILENAMES) <= set(paths.keys())
    for name in TRACE_FILENAMES:
        assert (tmp_path / name).exists()
        assert (tmp_path / session.execution_id / name).exists()
    summary = (tmp_path / "execution_summary.md").read_text(encoding="utf-8")
    assert "Execution ID" in summary
    assert "Total AI Agents Executed" in summary


def test_architecture_verification_keys() -> None:
    session = ExecutionTraceSession(session_id="s")
    session.mark_start(question="q")
    session.node_enter("goal_understanding")
    session.node_exit("goal_understanding", state_patch={"next_agent": "planner"})
    session.node_enter("planner")
    session.node_exit("planner")
    session.node_enter("task_decomposition")
    session.node_exit("task_decomposition")
    session.node_enter("execution_coordinator")
    session.node_exit("execution_coordinator")
    session.node_enter("memory_graph")
    session.node_exit("memory_graph")
    session.node_enter("planning_graph")
    session.node_exit("planning_graph")
    session.node_enter("execution_graph")
    session.node_exit("execution_graph")
    session.node_enter("reflection_graph")
    session.node_exit("reflection_graph")
    session.node_enter("recovery_graph")
    session.node_exit("recovery_graph")
    session.node_enter("export_graph")
    session.node_exit("export_graph")
    session.node_enter("reflection_agent")
    session.node_exit("reflection_agent")
    session.node_enter("replan_agent")
    session.node_exit("replan_agent")
    session.iam_check(action="a", resource="r", allowed=True)
    session.message_bus(bus_event="Publish", sender="a", receiver="b")
    session.memory_event(kind="Memory Retrieval")
    session.plugin_event(kind="Plugin Execution", plugin_id="echo")
    session.reliability_event(kind="Circuit Breaker", status="allow")
    session.mark_end(success=True)
    checks = verify_architecture(session)
    assert list(checks.keys()) == list(VERIFICATION_CHECKS)
    assert checks["Goal Understanding executed"]["passed"]
    assert checks["Observability active"]["passed"]
    assert checks["IAM enforced"]["passed"]
    assert checks["Message Bus used"]["passed"]


def test_wrap_trace_node_preserves_result() -> None:
    def node(state: dict) -> dict:
        return {"next_agent": "finalize", "ok": True}

    with execution_trace_session(session_id="s", question="q"):
        wrapped = wrap_trace_node("summary_agent", node)
        out = wrapped({"session_id": "s", "question": "q"})
        assert out["next_agent"] == "finalize"
        session = get_active_trace()
        assert session is not None
        assert "summary_agent" in session.nodes_executed()


def test_safe_trace_noop_without_session() -> None:
    # Must never raise when no active session
    safe_trace("node_enter", node_name="supervisor", state={})
    safe_trace("iam_check", action="x", resource="y", allowed=True)


def test_export_execution_trace_idempotent(tmp_path: Path) -> None:
    session = ExecutionTraceSession(session_id="s")
    session.mark_start(question="q")
    session.mark_end(success=True)
    paths1 = export_execution_trace(session, output_dir=tmp_path, print_summary=False)
    paths2 = export_execution_trace(session, output_dir=tmp_path, print_summary=False)
    assert paths1.keys() == paths2.keys()
