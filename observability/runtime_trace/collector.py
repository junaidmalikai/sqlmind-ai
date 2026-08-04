"""In-memory runtime execution trace collector — minimal overhead, context-bound."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator
from uuid import uuid4

from observability.runtime_trace.models import (
    AGENT_NODE_NAMES,
    PLANNING_LIFECYCLE_NODES,
    SUBGRAPH_NAMES,
    AgentReport,
    ExecutionSummary,
    GraphEventKind,
    SubgraphReport,
    TraceCategory,
    TraceEvent,
    truncate,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

_ACTIVE: ContextVar["ExecutionTraceSession | None"] = ContextVar(
    "sqlmind_runtime_trace_session", default=None
)
_GLOBAL_LOCK = threading.RLock()
_LAST_COMPLETED: "ExecutionTraceSession | None" = None


def new_execution_id() -> str:
    return uuid4().hex


class ExecutionTraceSession:
    """Collects every runtime event for one graph execution."""

    def __init__(
        self,
        *,
        execution_id: str | None = None,
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "default",
        workspace_id: str = "default",
        goal_id: str = "",
        llm_provider: str = "",
        model_name: str = "",
        enabled: bool = True,
    ) -> None:
        self.execution_id = execution_id or new_execution_id()
        self.session_id = session_id
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.workspace_id = workspace_id
        self.goal_id = goal_id
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.enabled = enabled
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.overall_success = False
        self._lock = threading.RLock()
        self.events: list[TraceEvent] = []
        self._node_enter_ts: dict[str, float] = {}
        self._subgraph_enter_ts: dict[str, float] = {}
        self._agent_reports: dict[str, AgentReport] = {}
        self._subgraph_reports: dict[str, SubgraphReport] = {
            name: SubgraphReport(name=name, skipped=True, reason="not observed")
            for name in sorted(SUBGRAPH_NAMES)
        }
        self._counters: dict[str, int] = {
            "nodes": 0,
            "agents": 0,
            "subgraphs": 0,
            "tools": 0,
            "memory": 0,
            "plugins": 0,
            "iam": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "retries": 0,
            "reflections": 0,
            "replans": 0,
            "checkpoints": 0,
            "parallel": 0,
        }
        self._lifecycle_seen: set[str] = set()
        self._nodes_executed: set[str] = set()
        self._export_paths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    def bind_ids_from_state(self, state: dict[str, Any] | None) -> None:
        if not state or not isinstance(state, dict):
            return
        self.session_id = str(state.get("session_id") or self.session_id)
        self.tenant_id = str(state.get("tenant_id") or self.tenant_id)
        self.workspace_id = str(state.get("workspace_id") or self.workspace_id)
        iam = state.get("iam_session") or {}
        if isinstance(iam, dict):
            self.user_id = str(
                iam.get("principal_id") or iam.get("actor") or self.user_id
            )
            if iam.get("tenant_id"):
                self.tenant_id = str(iam["tenant_id"])
            if iam.get("workspace_id"):
                self.workspace_id = str(iam["workspace_id"])
        goal = state.get("goal_spec") or {}
        if isinstance(goal, dict) and goal.get("goal_id"):
            self.goal_id = str(goal["goal_id"])
        elif state.get("goal_id"):
            self.goal_id = str(state["goal_id"])

    def _base_kwargs(self, **extra: Any) -> dict[str, Any]:
        task_id = extra.pop("task_id", "") or ""
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "goal_id": self.goal_id,
            "task_id": task_id,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            **extra,
        }

    def record(self, **kwargs: Any) -> TraceEvent | None:
        if not self.enabled:
            return None
        try:
            event = TraceEvent(**self._base_kwargs(**kwargs))
        except Exception as exc:  # noqa: BLE001
            logger.debug("trace event build failed: %s", exc)
            return None
        with self._lock:
            self.events.append(event)
        return event

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def mark_start(self, *, question: str = "", input_summary: Any = None) -> None:
        self.record(
            category=TraceCategory.LIFECYCLE,
            event_kind=GraphEventKind.START.value,
            current_state="START",
            next_state="Goal Understanding",
            input_summary=truncate(input_summary or {"question": question}),
            reasoning_summary="Execution started — architecture lifecycle begins",
        )

    def mark_end(self, *, success: bool, output_summary: Any = None) -> None:
        self.ended_at = time.time()
        self.overall_success = success
        duration = (self.ended_at - self.started_at) * 1000.0
        self.record(
            category=TraceCategory.LIFECYCLE,
            event_kind=GraphEventKind.END.value,
            current_state="Export" if "export_node" in self._nodes_executed or "export_graph" in self._nodes_executed else "END",
            next_state="END",
            duration_ms=round(duration, 2),
            output_summary=truncate(output_summary),
            status="ok" if success else "error",
            reasoning_summary="Execution ended",
        )

    # ------------------------------------------------------------------
    # Graph / node / subgraph
    # ------------------------------------------------------------------

    def node_enter(
        self,
        node_name: str,
        *,
        state: dict[str, Any] | None = None,
        parent_agent: str = "",
        input_summary: Any = None,
    ) -> None:
        self.bind_ids_from_state(state)
        self._node_enter_ts[node_name] = time.perf_counter()
        is_subgraph = node_name in SUBGRAPH_NAMES
        category = TraceCategory.SUBGRAPH if is_subgraph else TraceCategory.GRAPH
        kind = (
            GraphEventKind.SUBGRAPH_ENTER.value
            if is_subgraph
            else GraphEventKind.NODE_ENTER.value
        )
        if is_subgraph:
            report = self._subgraph_reports[node_name]
            report.started = True
            report.skipped = False
            report.reason = "started"
            self._subgraph_enter_ts[node_name] = time.perf_counter()

        lifecycle = PLANNING_LIFECYCLE_NODES.get(node_name)
        if lifecycle:
            self._lifecycle_seen.add(lifecycle)

        task_id = ""
        if state:
            progress = state.get("execution_progress") or {}
            if isinstance(progress, dict):
                task_id = str(progress.get("current_task_id") or "")
            active = state.get("active_task") or {}
            if isinstance(active, dict) and active.get("task_id"):
                task_id = str(active["task_id"])

        self.record(
            category=category,
            event_kind=kind,
            node_name=node_name,
            subgraph_name=node_name if is_subgraph else "",
            agent_name=node_name if node_name in AGENT_NODE_NAMES else "",
            parent_agent=parent_agent or str((state or {}).get("parent_agent") or ""),
            current_state=lifecycle or node_name,
            next_state="",
            input_summary=truncate(
                input_summary
                or _state_input_summary(state)
            ),
            task_id=task_id,
            status="started",
        )

        if node_name in AGENT_NODE_NAMES:
            report = self._agent_reports.setdefault(
                node_name, AgentReport(agent_name=node_name)
            )
            report.started = True
            report.status = "running"
            report.model = self.model_name
            report.input_summary = truncate(
                input_summary or _state_input_summary(state)
            )

    def node_exit(
        self,
        node_name: str,
        *,
        state_patch: dict[str, Any] | None = None,
        status: str = "ok",
        decision: str = "",
        confidence: float | None = None,
        reasoning_summary: str = "",
        output_summary: Any = None,
        error: str = "",
    ) -> None:
        started = self._node_enter_ts.pop(node_name, None)
        duration = (
            round((time.perf_counter() - started) * 1000.0, 2)
            if started is not None
            else None
        )
        is_subgraph = node_name in SUBGRAPH_NAMES
        category = TraceCategory.SUBGRAPH if is_subgraph else TraceCategory.GRAPH
        kind = (
            GraphEventKind.SUBGRAPH_EXIT.value
            if is_subgraph
            else GraphEventKind.NODE_EXIT.value
        )
        patch = state_patch or {}
        next_agent = str(patch.get("next_agent") or "")
        lifecycle = PLANNING_LIFECYCLE_NODES.get(node_name)

        if is_subgraph:
            report = self._subgraph_reports[node_name]
            report.completed = status == "ok"
            report.failed = status not in {"ok", "skipped"}
            report.skipped = False
            report.duration_ms = duration
            report.reason = error or status
            if status == "ok":
                with self._lock:
                    self._counters["subgraphs"] += 1

        with self._lock:
            self._nodes_executed.add(node_name)
            self._counters["nodes"] += 1
            if node_name in AGENT_NODE_NAMES:
                self._counters["agents"] += 1

        conf = confidence
        if conf is None and isinstance(patch.get("goal_spec"), dict):
            try:
                conf = float(patch["goal_spec"].get("confidence"))
            except (TypeError, ValueError):
                conf = None

        decision_val = decision or next_agent or str(patch.get("intent") or "")
        reasoning = reasoning_summary or str(
            patch.get("supervisor_reasoning")
            or patch.get("sql_explanation")
            or (patch.get("auton_decisions") or [""])[-1:][0]
            or ""
        )
        if isinstance(reasoning, dict):
            reasoning = str(reasoning.get("reason") or reasoning)[:500]

        self.record(
            category=category,
            event_kind=kind,
            node_name=node_name,
            subgraph_name=node_name if is_subgraph else "",
            agent_name=node_name if node_name in AGENT_NODE_NAMES else "",
            current_state=lifecycle or node_name,
            next_state=next_agent,
            duration_ms=duration,
            latency_ms=duration,
            output_summary=truncate(output_summary or _state_output_summary(patch)),
            decision=str(decision_val)[:300],
            confidence=conf,
            reasoning_summary=truncate(str(reasoning), 400) if reasoning else "",
            status="error" if error or status == "error" else status,
            payload={"error": error} if error else {},
        )

        # Specialized graph kinds
        if node_name == "reflection_agent" or "reflection" in node_name:
            with self._lock:
                self._counters["reflections"] += 1
            self.record(
                category=TraceCategory.GRAPH,
                event_kind=GraphEventKind.REFLECTION.value,
                node_name=node_name,
                agent_name=node_name,
                duration_ms=duration,
                output_summary=truncate(output_summary or _state_output_summary(patch)),
            )
            self._lifecycle_seen.add("Reflection")

        if node_name == "replan_agent" or patch.get("replan_decision"):
            with self._lock:
                self._counters["replans"] += 1
            self.record(
                category=TraceCategory.GRAPH,
                event_kind=GraphEventKind.REPLAN.value,
                node_name=node_name,
                agent_name=node_name,
                decision=str(patch.get("replan_decision") or decision_val)[:300],
                duration_ms=duration,
            )
            self._lifecycle_seen.add("Replanning")

        if node_name == "retry_agent" or patch.get("should_retry"):
            with self._lock:
                self._counters["retries"] += 1
            self.record(
                category=TraceCategory.GRAPH,
                event_kind=GraphEventKind.RETRY.value,
                node_name=node_name,
                agent_name=node_name,
                decision=str(patch.get("retry_next_action") or "")[:300],
                duration_ms=duration,
            )

        if node_name in AGENT_NODE_NAMES:
            report = self._agent_reports.setdefault(
                node_name, AgentReport(agent_name=node_name)
            )
            report.finished = True
            report.execution_time_ms = duration
            report.output_summary = truncate(
                output_summary or _state_output_summary(patch)
            )
            report.decision = str(decision_val)[:300]
            report.confidence = conf
            report.status = "error" if error or status == "error" else "completed"
            report.model = self.model_name
            if patch.get("vector_memory_context") or patch.get("episodic_context"):
                report.memory_used = True
            tools = patch.get("selected_tools") or patch.get("tools_used")
            if isinstance(tools, list):
                report.selected_tools = [str(t) for t in tools[:20]]

    def conditional_edge(self, source: str, dest: str, *, reason: str = "") -> None:
        self.record(
            category=TraceCategory.GRAPH,
            event_kind=GraphEventKind.CONDITIONAL_EDGE.value,
            node_name=source,
            current_state=source,
            next_state=dest,
            decision=dest,
            reasoning_summary=reason or f"{source} → {dest}",
        )

    def parallel_send(self, targets: list[str], *, source: str = "") -> None:
        with self._lock:
            self._counters["parallel"] += 1
        self.record(
            category=TraceCategory.GRAPH,
            event_kind=GraphEventKind.PARALLEL_SEND.value,
            node_name=source,
            current_state=source,
            next_state=",".join(targets),
            decision="parallel",
            output_summary={"targets": targets},
            reasoning_summary=f"Fan-out to {len(targets)} workers",
        )

    def checkpoint(self, *, thread_id: str = "", detail: str = "") -> None:
        with self._lock:
            self._counters["checkpoints"] += 1
        self.record(
            category=TraceCategory.GRAPH,
            event_kind=GraphEventKind.CHECKPOINT.value,
            decision="checkpoint",
            reasoning_summary=detail or "LangGraph checkpoint",
            payload={"thread_id": thread_id},
        )

    def interrupt(self, *, node_name: str = "", reason: str = "") -> None:
        self.record(
            category=TraceCategory.GRAPH,
            event_kind=GraphEventKind.INTERRUPT.value,
            node_name=node_name,
            reasoning_summary=reason or "HITL interrupt",
            status="interrupted",
        )

    def resume(self, *, resume_value: Any = None) -> None:
        self.record(
            category=TraceCategory.GRAPH,
            event_kind=GraphEventKind.RESUME.value,
            input_summary=truncate(resume_value),
            reasoning_summary="HITL resume",
        )

    # ------------------------------------------------------------------
    # Domain traces
    # ------------------------------------------------------------------

    def iam_check(
        self,
        *,
        action: str,
        resource: str,
        allowed: bool,
        decision: str = "",
        approval_required: bool = False,
        approval_granted: bool | None = None,
        auth_type: str = "authorization",
        detail: Any = None,
    ) -> None:
        with self._lock:
            self._counters["iam"] += 1
        self.record(
            category=TraceCategory.IAM,
            event_kind=auth_type,
            decision=decision or ("allowed" if allowed else "denied"),
            status="ok" if allowed else "denied",
            payload={
                "action": action,
                "resource": resource,
                "allowed": allowed,
                "denied": not allowed,
                "approval_required": approval_required,
                "approval_granted": approval_granted,
                "rbac": True,
                "abac": True,
                "detail": truncate(detail, 300),
            },
            reasoning_summary=f"IAM {action} on {resource}: {'ALLOW' if allowed else 'DENY'}",
        )

    def message_bus(
        self,
        *,
        bus_event: str,
        sender: str = "",
        receiver: str = "",
        msg_type: str = "",
        payload_summary: Any = None,
        direction: str = "sent",
    ) -> None:
        # Meta/status events are audit evidence but should not inflate traffic counters
        countable = bus_event not in {
            "Message Queue",
            "Status",
            "Subscribe",
        }
        with self._lock:
            if countable:
                if direction == "sent":
                    self._counters["messages_sent"] += 1
                else:
                    self._counters["messages_received"] += 1
                if sender in self._agent_reports:
                    if direction == "sent":
                        self._agent_reports[sender].messages_sent += 1
                if receiver in self._agent_reports and direction != "sent":
                    self._agent_reports[receiver].messages_received += 1
        self.record(
            category=TraceCategory.MESSAGE_BUS,
            event_kind=bus_event,
            agent_name=sender,
            parent_agent=receiver,
            input_summary=truncate(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "type": msg_type,
                    "payload": payload_summary,
                }
            ),
            decision=bus_event,
            reasoning_summary=f"MessageBus {bus_event}: {sender} → {receiver}",
        )

    def memory_event(
        self,
        *,
        kind: str,
        memories: Any = None,
        score: float | None = None,
        detail: Any = None,
    ) -> None:
        with self._lock:
            self._counters["memory"] += 1
        self._lifecycle_seen.add("Memory")
        self.record(
            category=TraceCategory.MEMORY,
            event_kind=kind,
            confidence=score,
            output_summary=truncate(memories),
            payload={"detail": truncate(detail, 300)},
            reasoning_summary=f"Memory: {kind}",
        )

    def plugin_event(
        self,
        *,
        kind: str,
        plugin_id: str = "",
        status: str = "ok",
        output: Any = None,
        detail: Any = None,
    ) -> None:
        if kind in {"Plugin Execution", "plugin_execute", "execute"}:
            with self._lock:
                self._counters["plugins"] += 1
        self.record(
            category=TraceCategory.PLUGIN,
            event_kind=kind,
            agent_name=plugin_id,
            status=status,
            output_summary=truncate(output),
            payload={"plugin_id": plugin_id, "detail": truncate(detail, 300)},
            reasoning_summary=f"Plugin {kind}: {plugin_id}",
        )

    def tool_event(
        self,
        *,
        tool_name: str,
        arguments: Any = None,
        result: Any = None,
        status: str = "ok",
        duration_ms: float | None = None,
        agent_name: str = "",
    ) -> None:
        with self._lock:
            self._counters["tools"] += 1
            if agent_name and agent_name in self._agent_reports:
                if tool_name not in self._agent_reports[agent_name].selected_tools:
                    self._agent_reports[agent_name].selected_tools.append(tool_name)
        self.record(
            category=TraceCategory.TOOL,
            event_kind="Tool Invocation",
            agent_name=agent_name,
            node_name=tool_name,
            input_summary=truncate({"tool": tool_name, "arguments": arguments}),
            output_summary=truncate(result),
            duration_ms=duration_ms,
            latency_ms=duration_ms,
            status=status,
            reasoning_summary=f"Tool {tool_name}: {status}",
        )

    def reliability_event(
        self,
        *,
        kind: str,
        node_name: str = "",
        status: str = "ok",
        detail: Any = None,
    ) -> None:
        self.record(
            category=TraceCategory.RELIABILITY,
            event_kind=kind,
            node_name=node_name,
            status=status,
            payload={"detail": truncate(detail, 400)},
            reasoning_summary=f"Reliability: {kind}",
        )

    def llm_event(
        self,
        *,
        agent_name: str = "",
        provider: str = "",
        model: str = "",
        tokens: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        cost: float | None = None,
        decision: str = "",
        confidence: float | None = None,
        reasoning_summary: str = "",
    ) -> None:
        self.record(
            category=TraceCategory.LLM,
            event_kind="LLM Call",
            agent_name=agent_name,
            llm_provider=provider or self.llm_provider,
            model_name=model or self.model_name,
            tokens=tokens,
            latency_ms=latency_ms,
            cost=cost,
            decision=decision,
            confidence=confidence,
            reasoning_summary=reasoning_summary or "LLM invocation",
        )

    def observe_stream_update(self, node_name: str, update: dict[str, Any]) -> None:
        """Observe LangGraph stream_mode=updates without altering workflow.

        When nodes are not wrapped (e.g. inner subgraph nodes already wrapped),
        stream observation still records enter/exit pairs for audit completeness.
        """
        if not self.enabled:
            return
        if isinstance(update, dict):
            self.bind_ids_from_state(update)
        # Prefer wrapper-emitted enter/exit; only synthesize if never entered
        if node_name and node_name not in self._node_enter_ts and node_name not in self._nodes_executed:
            self.node_enter(node_name, state=update if isinstance(update, dict) else None)
            self.node_exit(
                node_name,
                state_patch=update if isinstance(update, dict) else None,
                status="ok" if not (isinstance(update, dict) and update.get("error")) else "error",
            )
        elif node_name and node_name in self._node_enter_ts:
            # Wrapper entered but stream delivers the patch — exit will come from wrapper
            pass
        if isinstance(update, dict) and update.get("parallel_job") and node_name in {
            "visualization_agent",
            "insight_agent",
        }:
            # Soft evidence of parallel path; explicit Parallel Send is recorded by wrapper/orchestrator
            pass

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def infer_subgraphs_from_leaves(self) -> None:
        """Mark subgraphs completed when equivalent leaf nodes ran (invoke path)."""
        leaf_map = {
            "memory_graph": {"memory_agent"},
            "planning_graph": {"goal_understanding", "planner", "task_decomposition"},
            "execution_graph": {
                "execution_coordinator",
                "supervisor",
                "sql_agent",
                "execution_node",
                "schema_agent",
                "validation_node",
            },
            "analytics_graph": {
                "visualization_agent",
                "insight_agent",
                "optimization_agent",
                "dashboard_agent",
                "summary_agent",
            },
            "reflection_graph": {"reflection_agent"},
            "recovery_graph": {"retry_agent", "replan_agent"},
            "export_graph": {"export_node"},
        }
        nodes = self.nodes_executed()
        for sg_name, leaves in leaf_map.items():
            report = self._subgraph_reports[sg_name]
            if report.started or report.completed:
                continue
            hit = nodes & leaves
            if not hit:
                continue
            report.started = True
            report.completed = True
            report.skipped = False
            report.reason = f"inferred from leaf nodes: {sorted(hit)}"
            with self._lock:
                self._counters["subgraphs"] += 1
                self._nodes_executed.add(sg_name)
            self.record(
                category=TraceCategory.SUBGRAPH,
                event_kind=GraphEventKind.SUBGRAPH_EXIT.value,
                node_name=sg_name,
                subgraph_name=sg_name,
                status="ok",
                reasoning_summary=report.reason,
                decision="inferred",
            )

    def build_summary(
        self, *, architecture_verification: dict[str, bool] | None = None
    ) -> ExecutionSummary:
        ended = self.ended_at or time.time()
        duration = (ended - self.started_at) * 1000.0
        from datetime import datetime, timezone

        parallel_count = self._counters["parallel"]
        try:
            from observability.parallel_metrics import get_parallel_metrics

            snap = get_parallel_metrics().snapshot(session_id=self.session_id or None)
            # Prefer accurate collector when routing recorded waves
            if snap.get("total_parallel_executions", 0) > parallel_count:
                parallel_count = int(snap["total_parallel_executions"])
        except Exception:  # noqa: BLE001
            pass

        return ExecutionSummary(
            execution_id=self.execution_id,
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            started_at=datetime.fromtimestamp(
                self.started_at, tz=timezone.utc
            ).isoformat(),
            ended_at=datetime.fromtimestamp(ended, tz=timezone.utc).isoformat(),
            total_duration_ms=round(duration, 2),
            total_ai_agents_executed=self._counters["agents"],
            total_langgraph_nodes_executed=self._counters["nodes"],
            total_subgraphs_executed=self._counters["subgraphs"],
            total_tools_called=self._counters["tools"],
            total_memory_retrievals=self._counters["memory"],
            total_plugin_calls=self._counters["plugins"],
            total_iam_checks=self._counters["iam"],
            total_messages_sent=self._counters["messages_sent"],
            total_messages_received=self._counters["messages_received"],
            total_retries=self._counters["retries"],
            total_reflections=self._counters["reflections"],
            total_replans=self._counters["replans"],
            total_checkpoints=self._counters["checkpoints"],
            total_parallel_executions=parallel_count,
            overall_success=self.overall_success,
            architecture_verification=architecture_verification or {},
        )

    def subgraph_reports(self) -> list[SubgraphReport]:
        return [self._subgraph_reports[n] for n in sorted(SUBGRAPH_NAMES)]

    def agent_reports(self) -> list[AgentReport]:
        return list(self._agent_reports.values())

    def events_by_category(self, category: TraceCategory) -> list[TraceEvent]:
        with self._lock:
            return [e for e in self.events if e.category == category]

    def all_events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self.events)

    def lifecycle_seen(self) -> set[str]:
        return set(self._lifecycle_seen)

    def nodes_executed(self) -> set[str]:
        return set(self._nodes_executed)


def _state_input_summary(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    keys = (
        "question",
        "next_agent",
        "intent",
        "sql",
        "retry_count",
        "plan_active",
        "goal_id",
    )
    return {k: truncate(state.get(k), 200) for k in keys if state.get(k) not in (None, "", [], {})}


def _state_output_summary(patch: dict[str, Any] | None) -> dict[str, Any]:
    if not patch:
        return {}
    keys = (
        "next_agent",
        "intent",
        "sql",
        "sql_valid",
        "query_success",
        "error",
        "status",
        "insights",
        "final_response",
        "row_count",
        "needs_approval",
        "approval_decision",
    )
    out = {k: truncate(patch.get(k), 200) for k in keys if k in patch}
    if "goal_spec" in patch and isinstance(patch["goal_spec"], dict):
        out["goal_id"] = patch["goal_spec"].get("goal_id")
        out["goal_confidence"] = patch["goal_spec"].get("confidence")
    if "adaptive_plan" in patch and isinstance(patch["adaptive_plan"], dict):
        out["plan_id"] = patch["adaptive_plan"].get("plan_id")
    return out


# ------------------------------------------------------------------
# Context accessors
# ------------------------------------------------------------------

def get_active_trace() -> ExecutionTraceSession | None:
    return _ACTIVE.get()


def set_active_trace(session: ExecutionTraceSession | None) -> Token:
    return _ACTIVE.set(session)


def reset_active_trace(token: Token) -> None:
    _ACTIVE.reset(token)


def get_last_completed_trace() -> ExecutionTraceSession | None:
    with _GLOBAL_LOCK:
        return _LAST_COMPLETED


def _store_completed(session: ExecutionTraceSession) -> None:
    global _LAST_COMPLETED
    with _GLOBAL_LOCK:
        _LAST_COMPLETED = session


@contextmanager
def execution_trace_session(
    *,
    session_id: str = "",
    user_id: str = "",
    tenant_id: str = "default",
    workspace_id: str = "default",
    llm_provider: str = "",
    model_name: str = "",
    enabled: bool = True,
    question: str = "",
) -> Iterator[ExecutionTraceSession]:
    """Bind a trace session for the duration of one orchestrator run."""
    session = ExecutionTraceSession(
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        llm_provider=llm_provider,
        model_name=model_name,
        enabled=enabled,
    )
    token = set_active_trace(session)
    try:
        session.mark_start(question=question)
        yield session
    finally:
        _store_completed(session)
        reset_active_trace(token)


def safe_trace(fn_name: str, **kwargs: Any) -> None:
    """Call a method on the active trace if present — never raise to callers."""
    session = get_active_trace()
    if session is None or not session.enabled:
        return
    try:
        method = getattr(session, fn_name, None)
        if callable(method):
            method(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("runtime trace %s failed: %s", fn_name, exc)
