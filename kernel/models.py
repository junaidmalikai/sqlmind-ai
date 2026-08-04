"""Typed domain models for the autonomy kernel (DDD value objects)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from kernel.enums import CapabilityKind, ExecutionStrategy, ModelTier, RiskClass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CapabilityStats(BaseModel):
    """Runtime performance counters — fed by later execution/learning phases."""

    invocations: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.invocations <= 0:
            return 0.0
        return self.successes / self.invocations

    @property
    def avg_latency_ms(self) -> float:
        if self.invocations <= 0:
            return 0.0
        return self.total_latency_ms / self.invocations

    def record(
        self,
        *,
        success: bool,
        latency_ms: float = 0.0,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.invocations += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_latency_ms += max(0.0, latency_ms)
        self.total_tokens += max(0, tokens)
        self.total_cost_usd += max(0.0, cost_usd)


class CapabilityDescriptor(BaseModel):
    """Immutable-ish capability metadata discoverable by planners and matchers.

    Existing LangGraph nodes and LangChain tools are *registered* as capabilities.
    The descriptor is the catalog entry; the callable/node remains elsewhere.
    """

    id: str = Field(description="Stable capability id, e.g. agent.sql")
    kind: CapabilityKind
    name: str = Field(description="Human-readable name")
    description: str = Field(description="What this capability does — used for matching")
    version: str = "1.0.0"
    tags: frozenset[str] = Field(default_factory=frozenset)
    skills: frozenset[str] = Field(
        default_factory=frozenset,
        description="Capability skills for matching (nl2sql, visualize, …)",
    )
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    risk_class: RiskClass = RiskClass.LOW
    preferred_model_tier: ModelTier = ModelTier.STANDARD
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    # Graph / routing integration
    graph_node: str | None = Field(
        default=None,
        description="LangGraph node name when this is an agent/utility node",
    )
    route_tool_name: str | None = Field(
        default=None,
        description="Supervisor bind_tools name that selects this capability",
    )
    route_tool_description: str | None = None
    # Governance
    ai_routable: bool = Field(
        default=True,
        description="If False, AI/planner cannot select this (security gate nodes)",
    )
    system_protected: bool = Field(
        default=False,
        description="Cannot be unregistered or overwritten at runtime",
    )
    requires_tools: frozenset[str] = Field(default_factory=frozenset)
    provides: frozenset[str] = Field(
        default_factory=frozenset,
        description="Effects/artifacts this capability produces (sql, chart, …)",
    )
    # Plugin metadata
    plugin_id: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stats: CapabilityStats = Field(default_factory=CapabilityStats)

    @field_validator("tags", "skills", "requires_tools", "provides", mode="before")
    @classmethod
    def _to_frozenset(cls, v: Any) -> frozenset[str]:
        if v is None:
            return frozenset()
        if isinstance(v, frozenset):
            return v
        return frozenset(v)

    def touch(self) -> CapabilityDescriptor:
        """Return a copy with updated_at refreshed."""
        return self.model_copy(update={"updated_at": _utc_now()})


class CapabilityMatch(BaseModel):
    """Scored match between a requirement and a registered capability."""

    capability_id: str
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    descriptor: CapabilityDescriptor


class CapabilityRequirement(BaseModel):
    """What the planner/supervisor needs — matched against the registry.

    Replaces hardcoded ``if task == SQL`` with declarative capability needs.
    """

    description: str = ""
    required_skills: frozenset[str] = Field(default_factory=frozenset)
    preferred_tags: frozenset[str] = Field(default_factory=frozenset)
    kind: CapabilityKind | None = None
    provides_any: frozenset[str] = Field(default_factory=frozenset)
    max_risk: RiskClass = RiskClass.HIGH
    ai_routable_only: bool = True
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("required_skills", "preferred_tags", "provides_any", mode="before")
    @classmethod
    def _to_frozenset(cls, v: Any) -> frozenset[str]:
        if v is None:
            return frozenset()
        if isinstance(v, frozenset):
            return v
        return frozenset(v)


class RegistrySnapshot(BaseModel):
    """Point-in-time catalog export for prompts, audits, and observability."""

    version: int
    generated_at: datetime = Field(default_factory=_utc_now)
    capabilities: list[CapabilityDescriptor]

    def routable_agents(self) -> list[CapabilityDescriptor]:
        return [
            c
            for c in self.capabilities
            if c.kind == CapabilityKind.AGENT
            and c.ai_routable
            and c.enabled
            and c.graph_node
        ]

    def catalog_text(self, *, routable_only: bool = True, max_chars: int = 4000) -> str:
        """Compact text for LLM prompts (Phase 2 planner will consume this)."""
        lines: list[str] = []
        for c in self.capabilities:
            if routable_only and (not c.ai_routable or not c.enabled):
                continue
            skills = ", ".join(sorted(c.skills)) or "-"
            lines.append(
                f"- {c.id} [{c.kind.value}] node={c.graph_node or '-'} "
                f"skills=[{skills}] :: {c.description}"
            )
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n…(truncated)"
        return text
