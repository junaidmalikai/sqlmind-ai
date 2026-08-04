"""Phase 2 planner domain models — Goal, Task, Plan, Replan (structured outputs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Priority = Literal["low", "medium", "high", "critical"]
TaskStatus = Literal["pending", "ready", "running", "completed", "failed", "skipped"]
PlanStatus = Literal["draft", "ready", "executing", "replanning", "completed", "failed", "clarifying"]
ExecutionMode = Literal["sequential", "parallel", "mixed"]
ReplanStrategy = Literal[
    "revise_tasks",
    "swap_agent",
    "narrow_scope",
    "ask_clarify",
    "abort",
    "continue_with_supervisor",
]


class GoalConstraint(BaseModel):
    """A hard or soft constraint attached to a Goal."""

    name: str
    value: str
    soft: bool = False


class GoalSpec(BaseModel):
    """Structured understanding of the user's intent (Goal Understanding Agent)."""

    goal: str = Field(description="Primary goal in one clear sentence")
    objectives: list[str] = Field(
        default_factory=list,
        description="Concrete objectives derived from the goal",
    )
    constraints: list[GoalConstraint] = Field(default_factory=list)
    expected_output: str = Field(
        default="",
        description="What the user expects back (table, chart, dashboard, export, …)",
    )
    priority: Priority = "medium"
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    ambiguity_flags: list[str] = Field(
        default_factory=list,
        description="Aspects that are unclear or underspecified",
    )
    needs_clarification: bool = False
    clarification_question: str | None = None
    rewritten_question: str = Field(
        default="",
        description="Standalone, context-resolved question for downstream agents",
    )
    simple: bool = Field(
        default=False,
        description="True for single-step requests — use fast path via Supervisor",
    )
    intent_label: str = Field(
        default="query",
        description="Coarse intent label (query, schema, dashboard, export, …)",
    )
    success_criteria: list[str] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TaskSpec(BaseModel):
    """One unit of work produced by Task Decomposition."""

    id: str = Field(description="Stable task id, e.g. t1")
    title: str
    description: str = ""
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills the Agent Registry must match (no hardcoded agent names)",
    )
    preferred_tags: list[str] = Field(default_factory=list)
    provides_any: list[str] = Field(
        default_factory=list,
        description="Artifacts this task should produce (sql, insights, chart, …)",
    )
    responsible_agent_id: str | None = Field(
        default=None,
        description="Filled by selection — capability id from registry",
    )
    responsible_graph_node: str | None = Field(
        default=None,
        description="LangGraph node resolved from registry",
    )
    required_tools: list[str] = Field(default_factory=list)
    expected_output: str = ""
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task ids that must complete first",
    )
    parallel_safe: bool = False
    status: TaskStatus = "pending"
    estimated_cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative cost units (tokens/latency heuristic)",
    )
    notes: str = ""


class TaskGraph(BaseModel):
    """Dependency graph of tasks for a goal."""

    tasks: list[TaskSpec] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list)

    def task_map(self) -> dict[str, TaskSpec]:
        return {t.id: t for t in self.tasks}

    def ready_tasks(self) -> list[TaskSpec]:
        done = {t.id for t in self.tasks if t.status in {"completed", "skipped"}}
        ready: list[TaskSpec] = []
        for t in self.tasks:
            if t.status not in {"pending", "ready"}:
                continue
            if all(dep in done for dep in t.dependencies):
                ready.append(t)
        return ready

    def all_done(self) -> bool:
        if not self.tasks:
            return True
        return all(t.status in {"completed", "skipped", "failed"} for t in self.tasks)

    def mark(self, task_id: str, status: TaskStatus) -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = status
                return

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: dict[str, Any] | None) -> TaskGraph | None:
        if not data:
            return None
        return cls.model_validate(data)


class PlanStep(BaseModel):
    """Executable step in an AdaptivePlan (ordered / parallel groups)."""

    step_id: str
    task_ids: list[str] = Field(default_factory=list)
    mode: Literal["sequential", "parallel"] = "sequential"
    rationale: str = ""


class AdaptivePlan(BaseModel):
    """Compiled execution strategy consumed by the Execution Coordinator."""

    plan_id: str = Field(default="plan-1")
    goal_summary: str = ""
    strategy: str = Field(default="", description="High-level execution strategy")
    steps: list[PlanStep] = Field(default_factory=list)
    execution_mode: ExecutionMode = "sequential"
    estimated_cost: float = 0.0
    agent_ids: list[str] = Field(
        default_factory=list,
        description="Capability ids selected for this plan",
    )
    use_supervisor_fallback: bool = Field(
        default=False,
        description="If True, hand soft routing to existing Supervisor after plan preamble",
    )
    status: PlanStatus = "draft"
    current_step_index: int = 0
    max_replans: int = 2
    replan_count: int = 0
    reasoning: str = ""
    memory_hints_used: list[str] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: dict[str, Any] | None) -> AdaptivePlan | None:
        if not data:
            return None
        return cls.model_validate(data)


class PlannerOutput(BaseModel):
    """Structured output from the Planner Agent."""

    strategy: str
    execution_mode: ExecutionMode = "sequential"
    estimated_cost: float = Field(ge=0.0, default=1.0)
    required_skills_sequence: list[list[str]] = Field(
        default_factory=list,
        description="Ordered groups of skills; inner list = parallel-capable skills",
    )
    use_supervisor_fallback: bool = False
    reasoning: str = ""
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Optional explicit parallel skill groups",
    )


class DecompositionOutput(BaseModel):
    """Structured output from the Task Decomposition Agent."""

    tasks: list[TaskSpec]
    reasoning: str = ""

    @field_validator("tasks")
    @classmethod
    def _non_empty_ids(cls, tasks: list[TaskSpec]) -> list[TaskSpec]:
        seen: set[str] = set()
        for i, t in enumerate(tasks):
            if not t.id:
                t.id = f"t{i + 1}"
            if t.id in seen:
                t.id = f"{t.id}_{i + 1}"
            seen.add(t.id)
        return tasks


class ReplanDecision(BaseModel):
    """Structured output from Reflection & Replanning (true replan, not bare retry)."""

    strategy: ReplanStrategy
    reasoning: str
    failure_analysis: str = ""
    updated_goal_notes: str = ""
    drop_task_ids: list[str] = Field(default_factory=list)
    add_objectives: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class AutonDecision(BaseModel):
    """Autonomous meta-decision recorded for observability."""

    decision_type: Literal[
        "agent_selection",
        "tool_selection",
        "retry",
        "reflect",
        "replan",
        "clarify",
        "parallel",
        "finalize",
        "supervisor_handoff",
    ]
    chosen: str
    alternatives: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class ExecutionProgress(BaseModel):
    """Coordinator progress snapshot."""

    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    active_task_ids: list[str] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[AutonDecision] = Field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: dict[str, Any] | None) -> ExecutionProgress:
        if not data:
            return cls()
        return cls.model_validate(data)
