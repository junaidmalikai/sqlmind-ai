"""Goal Tracking domain models — lifecycle, progress, and status updates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


GoalLifecycleStatus = Literal[
    "created",
    "planning",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
]

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {"planning", "cancelled"},
    "planning": {"running", "waiting", "failed", "cancelled"},
    "running": {"waiting", "completed", "failed", "cancelled", "planning"},
    "waiting": {"running", "planning", "cancelled", "failed"},
    "completed": set(),
    "failed": {"planning"},  # replan may revive
    "cancelled": set(),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _goal_id() -> str:
    return f"goal-{uuid4().hex[:12]}"


class GoalObjectiveProgress(BaseModel):
    """Progress against a single objective within a tracked goal."""

    objective: str
    status: Literal["pending", "in_progress", "completed", "blocked", "failed"] = "pending"
    evidence: str = ""
    updated_at: datetime = Field(default_factory=_utc_now)


class GoalTrackingRecord(BaseModel):
    """First-class tracked goal with full lifecycle state."""

    goal_id: str = Field(default_factory=_goal_id)
    session_id: str = ""
    tenant_id: str = "default"
    workspace_id: str = "default"
    actor: str = ""
    title: str = ""
    description: str = ""
    status: GoalLifecycleStatus = "created"
    objectives: list[GoalObjectiveProgress] = Field(default_factory=list)
    completed_objectives: int = 0
    total_objectives: int = 0
    completion_pct: float = Field(ge=0.0, le=1.0, default=0.0)
    blocked: bool = False
    block_reason: str = ""
    failure_reason: str = ""
    plan_id: str = ""
    related_task_ids: list[str] = Field(default_factory=list)
    notify_planner: bool = False
    notify_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None

    def estimate_completion(self) -> float:
        if self.total_objectives <= 0:
            if self.status == "completed":
                return 1.0
            if self.status in {"failed", "cancelled"}:
                return 0.0
            if self.status == "running":
                return 0.5
            if self.status == "planning":
                return 0.15
            return 0.0
        done = sum(1 for o in self.objectives if o.status == "completed")
        self.completed_objectives = done
        self.completion_pct = round(done / max(self.total_objectives, 1), 4)
        return self.completion_pct

    def can_transition(self, new_status: GoalLifecycleStatus) -> bool:
        return new_status in ALLOWED_TRANSITIONS.get(self.status, set())

    def transition(self, new_status: GoalLifecycleStatus, *, reason: str = "") -> None:
        if new_status == self.status:
            self.updated_at = _utc_now()
            return
        if not self.can_transition(new_status):
            raise ValueError(
                f"Invalid goal transition {self.status!r} → {new_status!r} "
                f"(goal_id={self.goal_id})"
            )
        self.status = new_status
        self.updated_at = _utc_now()
        if reason:
            if new_status == "failed":
                self.failure_reason = reason
            elif new_status == "waiting":
                self.blocked = True
                self.block_reason = reason
        if new_status == "running":
            self.blocked = False
            self.block_reason = ""
        if new_status == "completed":
            self.completion_pct = 1.0
            self.completed_at = _utc_now()
            self.blocked = False

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: dict[str, Any] | None) -> GoalTrackingRecord | None:
        if not data:
            return None
        return cls.model_validate(data)


class GoalStatusUpdate(BaseModel):
    """Delta emitted by the Goal Tracking Agent for Planner / Observability."""

    goal_id: str
    previous_status: GoalLifecycleStatus
    new_status: GoalLifecycleStatus
    completion_pct: float = 0.0
    blocked: bool = False
    notify_planner: bool = False
    reason: str = ""
    timestamp: datetime = Field(default_factory=_utc_now)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
