"""Capability matching — declarative scoring, no task-type switch statements."""

from __future__ import annotations

from kernel.enums import RiskClass
from kernel.models import CapabilityDescriptor, CapabilityMatch, CapabilityRequirement
from kernel.protocols import CapabilityRegistryProtocol

_RISK_ORDER = {
    RiskClass.NONE: 0,
    RiskClass.LOW: 1,
    RiskClass.MEDIUM: 2,
    RiskClass.HIGH: 3,
    RiskClass.CRITICAL: 4,
}


class CapabilityMatcher:
    """Match requirements to registry entries via skills, tags, and provides.

    This is the replacement for hardcoded ``if task == SQL: SQLAgent()``.
    Scoring is deterministic and explainable for audits.
    """

    def match(
        self,
        requirement: CapabilityRequirement,
        registry: CapabilityRegistryProtocol,
    ) -> list[CapabilityMatch]:
        candidates = registry.list(
            kind=requirement.kind.value if requirement.kind else None,
            enabled_only=True,
            ai_routable_only=requirement.ai_routable_only,
        )
        scored: list[CapabilityMatch] = []
        for desc in candidates:
            match = self._score(requirement, desc)
            if match is not None:
                scored.append(match)
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: requirement.limit]

    def best(
        self,
        requirement: CapabilityRequirement,
        registry: CapabilityRegistryProtocol,
    ) -> CapabilityMatch | None:
        matches = self.match(requirement, registry)
        return matches[0] if matches else None

    def _score(
        self,
        req: CapabilityRequirement,
        desc: CapabilityDescriptor,
    ) -> CapabilityMatch | None:
        if _RISK_ORDER[desc.risk_class] > _RISK_ORDER[req.max_risk]:
            return None

        reasons: list[str] = []
        score = 0.0

        # Hard filter: all required skills must be present
        if req.required_skills:
            missing = req.required_skills - desc.skills
            if missing:
                return None
            score += 0.45
            reasons.append(f"skills={sorted(req.required_skills)}")

        # Soft: preferred tags
        if req.preferred_tags:
            overlap = req.preferred_tags & desc.tags
            if overlap:
                tag_score = len(overlap) / max(1, len(req.preferred_tags))
                score += 0.2 * tag_score
                reasons.append(f"tags={sorted(overlap)}")

        # Soft: provides any of requested artifacts
        if req.provides_any:
            overlap = req.provides_any & desc.provides
            if not overlap and req.required_skills:
                # skills already satisfied; provides is bonus
                pass
            elif not overlap and not req.required_skills:
                return None
            elif overlap:
                score += 0.25
                reasons.append(f"provides={sorted(overlap)}")

        # Soft: description token overlap (lightweight lexical boost)
        if req.description:
            desc_tokens = _tokens(req.description)
            cap_tokens = _tokens(f"{desc.name} {desc.description}")
            if desc_tokens and cap_tokens:
                inter = desc_tokens & cap_tokens
                if inter:
                    boost = min(0.15, 0.03 * len(inter))
                    score += boost
                    reasons.append(f"desc_overlap={len(inter)}")

        # Historical success rate boost (learning hook for later phases)
        if desc.stats.invocations >= 3:
            score += 0.1 * desc.stats.success_rate
            reasons.append(f"success_rate={desc.stats.success_rate:.2f}")

        if score <= 0.0 and not req.required_skills and not req.provides_any:
            # Open discovery: return weak matches for catalog browsing
            score = 0.05
            reasons.append("catalog_browse")

        if score <= 0.0:
            return None

        return CapabilityMatch(
            capability_id=desc.id,
            score=min(1.0, score),
            reasons=reasons,
            descriptor=desc,
        )


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) >= 3}
