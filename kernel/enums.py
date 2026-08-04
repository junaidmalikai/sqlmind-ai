"""Capability taxonomy for the autonomy kernel."""

from __future__ import annotations

from enum import Enum


class CapabilityKind(str, Enum):
    """What a registered capability is."""

    AGENT = "agent"
    TOOL = "tool"
    MODEL = "model"
    MEMORY = "memory"
    POLICY = "policy"
    REASONER = "reasoner"
    VERIFIER = "verifier"
    RECOVERY = "recovery"
    WORKFLOW = "workflow"
    SKILL = "skill"


class RiskClass(str, Enum):
    """Risk classification — used by policy (never bypassed by AI)."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelTier(str, Enum):
    """Preferred model tier hint for the future Model Router."""

    LOCAL = "local"
    FAST = "fast"
    STANDARD = "standard"
    REASONING = "reasoning"
    SQL_OPTIMIZED = "sql_optimized"


class ExecutionStrategy(str, Enum):
    """How a capability prefers to run (hints for planner / runtime)."""

    SEQUENTIAL = "sequential"
    PARALLEL_SAFE = "parallel_safe"
    SECURITY_GATED = "security_gated"
    HITL = "hitl"
    DETERMINISTIC = "deterministic"
