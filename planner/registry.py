"""Agent Registry — agent-focused facade over the Phase 1 CapabilityRegistry.

Planner discovers agents dynamically from this registry. No hardcoded routing.
New agents registered in the CapabilityRegistry become usable automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from kernel.enums import CapabilityKind, RiskClass
from kernel.matching import CapabilityMatcher
from kernel.models import CapabilityDescriptor, CapabilityMatch, CapabilityRequirement
from kernel.registry import CapabilityRegistry


HealthStatus = Literal["healthy", "degraded", "unavailable"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentRegistration(BaseModel):
    """Agent-facing view of a capability catalog entry."""

    agent_name: str
    capability_id: str
    capabilities: list[str] = Field(default_factory=list)  # skills
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)  # tags + skills
    priority: int = Field(default=50, ge=0, le=100)
    health: HealthStatus = "healthy"
    availability: bool = True
    graph_node: str | None = None
    risk_class: str = RiskClass.LOW.value
    description: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_descriptor(cls, desc: CapabilityDescriptor) -> AgentRegistration:
        inputs = list(desc.input_schema.keys()) if desc.input_schema else []
        outputs = list(desc.output_schema.keys()) if desc.output_schema else list(desc.provides)
        # Priority: prefer high success rate + lower risk
        priority = 50
        if desc.stats.invocations >= 3:
            priority = int(40 + 40 * desc.stats.success_rate)
        risk_penalty = {
            RiskClass.NONE: 0,
            RiskClass.LOW: 0,
            RiskClass.MEDIUM: 5,
            RiskClass.HIGH: 15,
            RiskClass.CRITICAL: 30,
        }.get(desc.risk_class, 5)
        priority = max(0, min(100, priority - risk_penalty))
        health: HealthStatus = "healthy"
        if not desc.enabled:
            health = "unavailable"
        elif desc.stats.invocations >= 5 and desc.stats.success_rate < 0.4:
            health = "degraded"
        return cls(
            agent_name=desc.name,
            capability_id=desc.id,
            capabilities=sorted(desc.skills),
            input_types=inputs,
            output_types=outputs or sorted(desc.provides),
            supported_tasks=sorted(desc.tags | desc.skills),
            priority=priority,
            health=health,
            availability=bool(desc.enabled and desc.ai_routable),
            graph_node=desc.graph_node,
            risk_class=desc.risk_class.value,
            description=desc.description,
            stats={
                "invocations": desc.stats.invocations,
                "success_rate": desc.stats.success_rate,
                "avg_latency_ms": desc.stats.avg_latency_ms,
            },
        )


class AgentRegistry:
    """Discoverable agent catalog for the Planner / Coordinator.

    Wraps ``CapabilityRegistry`` — does not duplicate storage. Plugins that
    ``register()`` on the capability registry appear here automatically
    (capability discovery without code changes).
    """

    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        matcher: CapabilityMatcher | None = None,
    ) -> None:
        self._registry = capability_registry
        self._matcher = matcher or CapabilityMatcher()

    @property
    def version(self) -> int:
        return int(self._registry.version)

    @property
    def capability_registry(self) -> CapabilityRegistry:
        return self._registry

    @property
    def matcher(self) -> CapabilityMatcher:
        return self._matcher

    def list_agents(self, *, available_only: bool = True) -> list[AgentRegistration]:
        caps = self._registry.list(
            kind=CapabilityKind.AGENT,
            enabled_only=available_only,
            ai_routable_only=True,
        )
        regs = [AgentRegistration.from_descriptor(c) for c in caps]
        if available_only:
            regs = [r for r in regs if r.availability and r.health != "unavailable"]
        regs.sort(key=lambda r: r.priority, reverse=True)
        return regs

    def get(self, capability_id: str) -> AgentRegistration | None:
        try:
            desc = self._registry.get(capability_id)
        except Exception:  # noqa: BLE001
            return None
        if desc.kind != CapabilityKind.AGENT:
            return None
        return AgentRegistration.from_descriptor(desc)

    def discover(
        self,
        *,
        required_skills: frozenset[str] | set[str] | list[str] | None = None,
        preferred_tags: frozenset[str] | set[str] | list[str] | None = None,
        provides_any: frozenset[str] | set[str] | list[str] | None = None,
        description: str = "",
        limit: int = 5,
    ) -> list[CapabilityMatch]:
        """Dynamic agent selection — no if-task switches."""
        req = CapabilityRequirement(
            description=description,
            required_skills=frozenset(required_skills or ()),
            preferred_tags=frozenset(preferred_tags or ()),
            provides_any=frozenset(provides_any or ()),
            kind=CapabilityKind.AGENT,
            ai_routable_only=True,
            limit=limit,
        )
        return self._matcher.match(req, self._registry)

    def best_agent(
        self,
        *,
        required_skills: frozenset[str] | set[str] | list[str] | None = None,
        preferred_tags: frozenset[str] | set[str] | list[str] | None = None,
        provides_any: frozenset[str] | set[str] | list[str] | None = None,
        description: str = "",
    ) -> CapabilityMatch | None:
        matches = self.discover(
            required_skills=required_skills,
            preferred_tags=preferred_tags,
            provides_any=provides_any,
            description=description,
            limit=1,
        )
        return matches[0] if matches else None

    def catalog_text(self, *, max_chars: int = 3500) -> str:
        lines = [
            f"- {r.capability_id} node={r.graph_node} "
            f"skills={r.capabilities} health={r.health} priority={r.priority} "
            f":: {r.description}"
            for r in self.list_agents()
        ]
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[: max_chars - 20] + "\n…(truncated)"
        return text

    def register_agent(self, descriptor: CapabilityDescriptor, *, handler: Any = None) -> None:
        """Hot-register a new agent capability (becomes discoverable immediately)."""
        if descriptor.kind != CapabilityKind.AGENT:
            raise ValueError("register_agent requires CapabilityKind.AGENT")
        self._registry.register(descriptor, handler=handler, overwrite=True)
