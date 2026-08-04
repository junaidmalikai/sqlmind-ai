"""Runtime Execution Trace System — enterprise audit instrumentation.

Captures the complete execution lifecycle without modifying business logic.
Public surface used by orchestrator, enterprise wrappers, bus, IAM, and plugins.
"""

from __future__ import annotations

from observability.runtime_trace.collector import (
    ExecutionTraceSession,
    execution_trace_session,
    get_active_trace,
    get_last_completed_trace,
    new_execution_id,
    reset_active_trace,
    safe_trace,
    set_active_trace,
)
from observability.runtime_trace.exporter import (
    TRACE_FILENAMES,
    export_execution_trace,
    finalize_and_export,
    format_console_summary,
)
from observability.runtime_trace.models import (
    AGENT_NODE_NAMES,
    SUBGRAPH_NAMES,
    AgentReport,
    ExecutionSummary,
    GraphEventKind,
    SubgraphReport,
    TraceCategory,
    TraceEvent,
    truncate,
)
from observability.runtime_trace.verifier import VERIFICATION_CHECKS, verify_architecture
from observability.runtime_trace.wrap import wrap_trace_node

__all__ = [
    "AGENT_NODE_NAMES",
    "SUBGRAPH_NAMES",
    "TRACE_FILENAMES",
    "VERIFICATION_CHECKS",
    "AgentReport",
    "ExecutionSummary",
    "ExecutionTraceSession",
    "GraphEventKind",
    "SubgraphReport",
    "TraceCategory",
    "TraceEvent",
    "execution_trace_session",
    "export_execution_trace",
    "finalize_and_export",
    "format_console_summary",
    "get_active_trace",
    "get_last_completed_trace",
    "new_execution_id",
    "reset_active_trace",
    "safe_trace",
    "set_active_trace",
    "truncate",
    "verify_architecture",
    "wrap_trace_node",
]
