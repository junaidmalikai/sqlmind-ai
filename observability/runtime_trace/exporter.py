"""Export runtime traces to enterprise audit artifact files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from observability.runtime_trace.collector import ExecutionTraceSession
from observability.runtime_trace.models import TraceCategory, truncate
from observability.runtime_trace.verifier import verify_architecture
from utils.helpers import ensure_dirs
from utils.logging_config import get_logger

logger = get_logger(__name__)

TRACE_FILENAMES = (
    "execution_trace.json",
    "execution_trace.md",
    "execution_summary.md",
    "architecture_trace.json",
    "agent_trace.json",
    "graph_trace.json",
    "memory_trace.json",
    "security_trace.json",
    "plugin_trace.json",
    "tool_trace.json",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _timeline_rows(
    session: ExecutionTraceSession, *, category: TraceCategory | None = None
) -> list[dict[str, Any]]:
    events = session.all_events()
    if category is not None:
        events = [e for e in events if e.category == category]
    return [
        {
            "timestamp": e.timestamp,
            "event_kind": e.event_kind,
            "node_name": e.node_name or e.agent_name or e.subgraph_name,
            "duration_ms": e.duration_ms,
            "decision": e.decision,
            "status": e.status,
            "current_state": e.current_state,
            "next_state": e.next_state,
        }
        for e in events
    ]


def build_execution_trace_payload(session: ExecutionTraceSession) -> dict[str, Any]:
    verification = verify_architecture(session)
    summary = session.build_summary(
        architecture_verification={k: v["passed"] for k, v in verification.items()}
    )
    return {
        "execution_id": session.execution_id,
        "session_id": session.session_id,
        "user_id": session.user_id,
        "tenant_id": session.tenant_id,
        "workspace_id": session.workspace_id,
        "goal_id": session.goal_id,
        "llm_provider": session.llm_provider,
        "model_name": session.model_name,
        "summary": summary.model_dump(),
        "architecture_verification": verification,
        "subgraph_reports": [r.model_dump() for r in session.subgraph_reports()],
        "agent_reports": [r.model_dump() for r in session.agent_reports()],
        "timelines": {
            "execution": _timeline_rows(session),
            "agent": _timeline_rows(session, category=TraceCategory.AGENT)
            or [
                t
                for t in _timeline_rows(session)
                if t.get("node_name") in {a.agent_name for a in session.agent_reports()}
            ],
            "goal": [
                e.to_audit_row()
                for e in session.all_events()
                if e.node_name in {"goal_understanding", "goal_tracking"}
                or "goal" in (e.event_kind or "").lower()
            ],
            "task": [
                e.to_audit_row()
                for e in session.all_events()
                if e.task_id
                or e.node_name
                in {"task_decomposition", "execution_coordinator"}
            ],
            "memory": _timeline_rows(session, category=TraceCategory.MEMORY),
            "tool": _timeline_rows(session, category=TraceCategory.TOOL),
            "plugin": _timeline_rows(session, category=TraceCategory.PLUGIN),
            "iam": _timeline_rows(session, category=TraceCategory.IAM),
            "graph": _timeline_rows(session, category=TraceCategory.GRAPH)
            + _timeline_rows(session, category=TraceCategory.SUBGRAPH)
            + _timeline_rows(session, category=TraceCategory.LIFECYCLE),
            "export": [
                e.to_audit_row()
                for e in session.all_events()
                if e.node_name in {"export_node", "export_graph"}
                or "export" in (e.event_kind or "").lower()
            ],
        },
        "events": [e.to_audit_row() for e in session.all_events()],
    }


def render_execution_trace_md(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Runtime Execution Trace",
        "",
        f"- **Execution ID**: `{payload.get('execution_id')}`",
        f"- **Session ID**: `{payload.get('session_id')}`",
        f"- **User ID**: `{payload.get('user_id')}`",
        f"- **Tenant ID**: `{payload.get('tenant_id')}`",
        f"- **Workspace ID**: `{payload.get('workspace_id')}`",
        f"- **Goal ID**: `{payload.get('goal_id')}`",
        f"- **Provider / Model**: `{payload.get('llm_provider')}` / `{payload.get('model_name')}`",
        f"- **Overall Success**: `{summary.get('overall_success')}`",
        f"- **Total Duration (ms)**: `{summary.get('total_duration_ms')}`",
        "",
        "## Execution Timeline",
        "",
        "| Timestamp | Kind | Node | Duration ms | Decision | Status |",
        "|---|---|---|---:|---|---|",
    ]
    for row in (payload.get("timelines") or {}).get("execution") or []:
        lines.append(
            f"| {row.get('timestamp','')} | {row.get('event_kind','')} | "
            f"{row.get('node_name','')} | {row.get('duration_ms','')} | "
            f"{row.get('decision','')} | {row.get('status','')} |"
        )
    lines.extend(["", "## Subgraph Reports", ""])
    for sg in payload.get("subgraph_reports") or []:
        lines.append(
            f"- `{sg.get('name')}`: started={sg.get('started')} completed={sg.get('completed')} "
            f"skipped={sg.get('skipped')} failed={sg.get('failed')} "
            f"duration_ms={sg.get('duration_ms')} reason={sg.get('reason')}"
        )
    lines.extend(["", "## Architecture Verification", ""])
    for name, detail in (payload.get("architecture_verification") or {}).items():
        mark = "✓" if detail.get("passed") else "✗"
        lines.append(f"- {mark} {name} — {detail.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def render_execution_summary_md(payload: dict[str, Any]) -> str:
    s = payload.get("summary") or {}
    verification = payload.get("architecture_verification") or {}
    lines = [
        "# Execution Summary",
        "",
        "====================================================",
        "",
        "Execution Summary",
        "",
        "====================================================",
        "",
        f"Execution ID: {s.get('execution_id')}",
        f"Total Duration: {s.get('total_duration_ms')} ms",
        f"Total AI Agents Executed: {s.get('total_ai_agents_executed')}",
        f"Total LangGraph Nodes Executed: {s.get('total_langgraph_nodes_executed')}",
        f"Total Subgraphs Executed: {s.get('total_subgraphs_executed')}",
        f"Total Tools Called: {s.get('total_tools_called')}",
        f"Total Memory Retrievals: {s.get('total_memory_retrievals')}",
        f"Total Plugin Calls: {s.get('total_plugin_calls')}",
        f"Total IAM Checks: {s.get('total_iam_checks')}",
        f"Total Messages Sent: {s.get('total_messages_sent')}",
        f"Total Messages Received: {s.get('total_messages_received')}",
        f"Total Retries: {s.get('total_retries')}",
        f"Total Reflections: {s.get('total_reflections')}",
        f"Total Replans: {s.get('total_replans')}",
        f"Total Checkpoints: {s.get('total_checkpoints')}",
        f"Total Parallel Executions: {s.get('total_parallel_executions')}",
        f"Overall Success: {s.get('overall_success')}",
        "",
        "====================================================",
        "",
        "## Architecture Verification",
        "",
    ]
    for name, detail in verification.items():
        mark = "✓" if detail.get("passed") else "✗"
        lines.append(f"{mark} {name}")
    lines.append("")
    return "\n".join(lines)


def format_console_summary(payload: dict[str, Any]) -> str:
    s = payload.get("summary") or {}
    verification = payload.get("architecture_verification") or {}
    lines = [
        "",
        "====================================================",
        "Execution Summary",
        "====================================================",
        f"Execution ID: {s.get('execution_id')}",
        f"Total Duration: {s.get('total_duration_ms')} ms",
        f"Total AI Agents Executed: {s.get('total_ai_agents_executed')}",
        f"Total LangGraph Nodes Executed: {s.get('total_langgraph_nodes_executed')}",
        f"Total Subgraphs Executed: {s.get('total_subgraphs_executed')}",
        f"Total Tools Called: {s.get('total_tools_called')}",
        f"Total Memory Retrievals: {s.get('total_memory_retrievals')}",
        f"Total Plugin Calls: {s.get('total_plugin_calls')}",
        f"Total IAM Checks: {s.get('total_iam_checks')}",
        f"Total Messages Sent: {s.get('total_messages_sent')}",
        f"Total Messages Received: {s.get('total_messages_received')}",
        f"Total Retries: {s.get('total_retries')}",
        f"Total Reflections: {s.get('total_reflections')}",
        f"Total Replans: {s.get('total_replans')}",
        f"Total Checkpoints: {s.get('total_checkpoints')}",
        f"Total Parallel Executions: {s.get('total_parallel_executions')}",
        f"Overall Success: {s.get('overall_success')}",
        "====================================================",
        "Architecture Verification",
        "====================================================",
    ]
    for name, detail in verification.items():
        mark = "✓" if detail.get("passed") else "✗"
        lines.append(f"{mark} {name}")
    lines.append("====================================================")
    lines.append("")
    return "\n".join(lines)


def export_execution_trace(
    session: ExecutionTraceSession,
    *,
    output_dir: str | Path,
    print_summary: bool = True,
) -> dict[str, str]:
    """Write all audit artifacts for one execution. Returns path map."""
    out = Path(output_dir)
    # Per-execution subdirectory keeps concurrent runs from clobbering
    run_dir = out / session.execution_id
    ensure_dirs(run_dir)
    # Also write "latest" copies at the root for convenient audit pickup
    ensure_dirs(out)

    payload = build_execution_trace_payload(session)
    paths: dict[str, str] = {}

    files: dict[str, Any] = {
        "execution_trace.json": payload,
        "architecture_trace.json": {
            "execution_id": session.execution_id,
            "verification": payload["architecture_verification"],
            "subgraph_reports": payload["subgraph_reports"],
            "lifecycle_seen": sorted(session.lifecycle_seen()),
            "nodes_executed": sorted(session.nodes_executed()),
            "timelines": {
                "graph": payload["timelines"]["graph"],
                "execution": payload["timelines"]["execution"],
            },
        },
        "agent_trace.json": {
            "execution_id": session.execution_id,
            "agents": payload["agent_reports"],
            "timeline": payload["timelines"]["agent"],
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.agent_name or e.category == TraceCategory.AGENT
            ],
        },
        "graph_trace.json": {
            "execution_id": session.execution_id,
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.category
                in {
                    TraceCategory.GRAPH,
                    TraceCategory.SUBGRAPH,
                    TraceCategory.LIFECYCLE,
                }
            ],
            "timeline": payload["timelines"]["graph"],
            "subgraphs": payload["subgraph_reports"],
        },
        "memory_trace.json": {
            "execution_id": session.execution_id,
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.category == TraceCategory.MEMORY
                or e.node_name in {"memory_agent", "memory_graph"}
            ],
            "timeline": payload["timelines"]["memory"],
        },
        "security_trace.json": {
            "execution_id": session.execution_id,
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.category == TraceCategory.IAM
            ],
            "timeline": payload["timelines"]["iam"],
        },
        "plugin_trace.json": {
            "execution_id": session.execution_id,
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.category == TraceCategory.PLUGIN
            ],
            "timeline": payload["timelines"]["plugin"],
        },
        "tool_trace.json": {
            "execution_id": session.execution_id,
            "events": [
                e.to_audit_row()
                for e in session.all_events()
                if e.category == TraceCategory.TOOL
            ],
            "timeline": payload["timelines"]["tool"],
        },
    }

    for name, content in files.items():
        target = run_dir / name
        _write_json(target, content)
        latest = out / name
        _write_json(latest, content)
        paths[name] = str(target)

    md_trace = render_execution_trace_md(payload)
    md_summary = render_execution_summary_md(payload)
    for name, text in (
        ("execution_trace.md", md_trace),
        ("execution_summary.md", md_summary),
    ):
        (run_dir / name).write_text(text, encoding="utf-8")
        (out / name).write_text(text, encoding="utf-8")
        paths[name] = str(run_dir / name)

    session._export_paths = paths  # noqa: SLF001 — intentional for callers
    if print_summary:
        text = format_console_summary(payload)
        # Prefer logger so Streamlit / uvicorn capture it; also print for CLI audits
        logger.info("\n%s", text)
        try:
            print(text)
        except Exception:  # noqa: BLE001
            pass
    return paths


def finalize_and_export(
    session: ExecutionTraceSession,
    *,
    output_dir: str | Path,
    success: bool,
    final_state: dict[str, Any] | None = None,
    print_summary: bool = True,
) -> dict[str, str]:
    """Mark END, then write all audit files."""
    session.bind_ids_from_state(final_state)
    session.infer_subgraphs_from_leaves()
    session.mark_end(
        success=success,
        output_summary=truncate(
            {
                "query_success": (final_state or {}).get("query_success"),
                "error": (final_state or {}).get("error"),
                "next_agent": (final_state or {}).get("next_agent"),
                "final_response": truncate(
                    (final_state or {}).get("final_response"), 300
                ),
            }
        ),
    )
    # Soft checkpoint evidence when a session completed with a thread id
    if (final_state or {}).get("session_id"):
        session.checkpoint(
            thread_id=str((final_state or {}).get("session_id")),
            detail="run boundary checkpoint",
        )
    return export_execution_trace(
        session, output_dir=output_dir, print_summary=print_summary
    )
