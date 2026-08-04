"""Phase 2+ — Goal Planner, TaskGraph, AdaptivePlan, Execution Coordinator, Goal Tracking.

Builds on the Phase 1 Autonomy Kernel. Existing specialist agents are unchanged.
"""

from __future__ import annotations

from planner.agents import (
    make_goal_understanding_agent,
    make_memory_agent,
    make_planner_agent,
    make_replan_agent,
    make_task_decomposition_agent,
    persist_workflow_outcome,
)
from planner.coordinator import (
    ExecutionCoordinator,
    make_execution_coordinator,
    route_coordinator_dispatch,
)
from planner.goal_models import GoalStatusUpdate, GoalTrackingRecord
from planner.goal_store import GoalStore
from planner.goal_tracker import make_goal_tracking_agent
from planner.messages import AgentMessage, AgentMessageBus, get_message_bus
from planner.models import (
    AdaptivePlan,
    AutonDecision,
    DecompositionOutput,
    ExecutionProgress,
    GoalSpec,
    PlannerOutput,
    ReplanDecision,
    TaskGraph,
    TaskSpec,
)
from planner.registry import AgentRegistration, AgentRegistry
from planner.selection import select_agent_for_task

__all__ = [
    "AdaptivePlan",
    "AgentMessage",
    "AgentMessageBus",
    "AgentRegistration",
    "AgentRegistry",
    "AutonDecision",
    "DecompositionOutput",
    "ExecutionCoordinator",
    "ExecutionProgress",
    "GoalSpec",
    "GoalStatusUpdate",
    "GoalStore",
    "GoalTrackingRecord",
    "PlannerOutput",
    "ReplanDecision",
    "TaskGraph",
    "TaskSpec",
    "get_message_bus",
    "make_execution_coordinator",
    "make_goal_tracking_agent",
    "make_goal_understanding_agent",
    "make_memory_agent",
    "make_planner_agent",
    "make_replan_agent",
    "make_task_decomposition_agent",
    "persist_workflow_outcome",
    "route_coordinator_dispatch",
    "select_agent_for_task",
]
