"""Runtime execution trace schemas — audit-ready event and timeline models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id() -> str:
    return uuid4().hex[:16]


class TraceCategory(str, Enum):
    GRAPH = "graph"
    SUBGRAPH = "subgraph"
    AGENT = "agent"
    MESSAGE_BUS = "message_bus"
    IAM = "iam"
    MEMORY = "memory"
    PLUGIN = "plugin"
    TOOL = "tool"
    RELIABILITY = "reliability"
    LIFECYCLE = "lifecycle"
    LLM = "llm"


class GraphEventKind(str, Enum):
    START = "START"
    NODE_ENTER = "Node Enter"
    NODE_EXIT = "Node Exit"
    CONDITIONAL_EDGE = "Conditional Edge"
    PARALLEL_SEND = "Parallel Send"
    LOOP = "Loop"
    RETRY = "Retry"
    REFLECTION = "Reflection"
    REPLAN = "Replan"
    INTERRUPT = "Interrupt"
    RESUME = "Resume"
    CHECKPOINT = "Checkpoint"
    SUBGRAPH_ENTER = "Subgraph Enter"
    SUBGRAPH_EXIT = "Subgraph Exit"
    END = "END"


SUBGRAPH_NAMES: frozenset[str] = frozenset(
    {
        "memory_graph",
        "planning_graph",
        "execution_graph",
        "analytics_graph",
        "reflection_graph",
        "recovery_graph",
        "export_graph",
    }
)

AGENT_NODE_NAMES: frozenset[str] = frozenset(
    {
        "memory_agent",
        "goal_understanding",
        "planner",
        "task_decomposition",
        "execution_coordinator",
        "replan_agent",
        "goal_tracking",
        "supervisor",
        "schema_agent",
        "sql_agent",
        "validation_node",
        "execution_node",
        "visualization_agent",
        "insight_agent",
        "optimization_agent",
        "dashboard_agent",
        "summary_agent",
        "export_node",
        "reflection_agent",
        "retry_agent",
        "clarify",
        "approval_gate",
        "finalize",
        "fail",
        "join_post_query",
    }
)

PLANNING_LIFECYCLE_NODES: dict[str, str] = {
    "goal_understanding": "Goal Understanding",
    "planner": "Planning",
    "task_decomposition": "Task Decomposition",
    "execution_coordinator": "Execution Coordination",
    "reflection_agent": "Reflection",
    "replan_agent": "Replanning",
    "memory_agent": "Memory",
    "export_node": "Export",
}


class TraceEvent(BaseModel):
    """Single timestamped runtime event — the atomic unit of an execution audit."""

    event_id: str = Field(default_factory=_event_id)
    execution_id: str = ""
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = "default"
    workspace_id: str = "default"
    goal_id: str = ""
    task_id: str = ""
    timestamp: str = Field(default_factory=_utc_now)
    duration_ms: float | None = None
    category: TraceCategory = TraceCategory.LIFECYCLE
    event_kind: str = ""
    node_name: str = ""
    subgraph_name: str = ""
    agent_name: str = ""
    parent_agent: str = ""
    current_state: str = ""
    next_state: str = ""
    input_summary: Any = None
    output_summary: Any = None
    decision: str = ""
    confidence: float | None = None
    reasoning_summary: str = ""
    llm_provider: str = ""
    model_name: str = ""
    tokens: dict[str, Any] | None = None
    latency_ms: float | None = None
    cost: float | None = None
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "category": self.category.value if isinstance(self.category, TraceCategory) else self.category,
            "event_kind": self.event_kind,
            "node_name": self.node_name,
            "subgraph_name": self.subgraph_name,
            "agent_name": self.agent_name,
            "parent_agent": self.parent_agent,
            "current_state": self.current_state,
            "next_state": self.next_state,
            "input": self.input_summary,
            "output": self.output_summary,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "status": self.status,
            "payload": self.payload,
        }


class SubgraphReport(BaseModel):
    name: str
    started: bool = False
    completed: bool = False
    skipped: bool = True
    failed: bool = False
    duration_ms: float | None = None
    reason: str = "not observed in this execution"


class AgentReport(BaseModel):
    agent_name: str
    started: bool = False
    finished: bool = False
    execution_time_ms: float | None = None
    system_prompt_version: str = ""
    model: str = ""
    input_summary: Any = None
    output_summary: Any = None
    decision: str = ""
    confidence: float | None = None
    selected_tools: list[str] = Field(default_factory=list)
    memory_used: bool = False
    messages_sent: int = 0
    messages_received: int = 0
    status: str = "not_executed"


class ExecutionSummary(BaseModel):
    execution_id: str
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = "default"
    workspace_id: str = "default"
    started_at: str = ""
    ended_at: str = ""
    total_duration_ms: float = 0.0
    total_ai_agents_executed: int = 0
    total_langgraph_nodes_executed: int = 0
    total_subgraphs_executed: int = 0
    total_tools_called: int = 0
    total_memory_retrievals: int = 0
    total_plugin_calls: int = 0
    total_iam_checks: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_retries: int = 0
    total_reflections: int = 0
    total_replans: int = 0
    total_checkpoints: int = 0
    total_parallel_executions: int = 0
    overall_success: bool = False
    architecture_verification: dict[str, bool] = Field(default_factory=dict)
    verification_details: dict[str, str] = Field(default_factory=dict)


TimelineName = Literal[
    "execution",
    "agent",
    "goal",
    "task",
    "memory",
    "tool",
    "plugin",
    "iam",
    "graph",
    "export",
]


def truncate(value: Any, limit: int = 500) -> Any:
    """Cheap summary truncation for audit payloads — never raise."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 24:
                out["…"] = f"+{len(value) - 24} keys"
                break
            out[str(k)] = truncate(v, max(64, limit // 4))
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)[:12]
        summarized = [truncate(x, max(64, limit // 4)) for x in items]
        if len(value) > 12:
            summarized.append(f"…+{len(value) - 12} more")
        return summarized
    try:
        text = str(value)
    except Exception:  # noqa: BLE001
        return "<unrepr>"
    return truncate(text, limit)
