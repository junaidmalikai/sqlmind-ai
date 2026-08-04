"""Pydantic structured-output models for all LLM interactions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IntentType = Literal[
    "query",
    "schema",
    "summary",
    "kpi",
    "dashboard",
    "export",
    "optimize",
    "clarify",
]

# Specialist / utility node names the Supervisor may route to via tool calls
RoutableNode = Literal[
    "schema_agent",
    "sql_agent",
    "visualization_agent",
    "insight_agent",
    "optimization_agent",
    "dashboard_agent",
    "export_node",
    "summary_agent",
    "reflection_agent",
    "finalize",
    "clarify",
]

AgentName = RoutableNode  # backward-compatible alias


class IntentModel(BaseModel):
    """Optional metadata from Supervisor (routing is via bind_tools, not this plan)."""

    intent: IntentType = Field(description="Primary user intent label")
    reasoning: str = Field(description="Brief rationale")
    needs_clarification: bool = Field(default=False)
    clarification_question: str | None = Field(default=None)
    rewritten_question: str | None = Field(
        default=None,
        description="Context-resolved standalone question for follow-ups",
    )


class SQLResponseModel(BaseModel):
    """Structured SQL generation output."""

    sql: str = Field(description="Single read-only SQL statement, no markdown")
    explanation: str = Field(description="What the query computes in plain English")
    tables_used: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class RetryDecisionModel(BaseModel):
    """LLM diagnosis after SQL validation/execution failure."""

    should_retry: bool
    reasoning: str = Field(description="Root-cause diagnosis of why the attempt failed")
    failure_class: Literal[
        "syntax",
        "unknown_object",
        "permission_or_safety",
        "empty_or_wrong_logic",
        "impossible",
        "other",
    ] = "other"
    next_action: Literal["regenerate_sql", "replan", "give_up"] = Field(
        description=(
            "regenerate_sql: try a different SQL approach; "
            "replan: return to Supervisor to choose a different specialist path; "
            "give_up: stop and explain to the user"
        )
    )
    fix_hint: str = Field(
        default="",
        description="Concrete, attempt-specific guidance so the next SQL differs from the last",
    )
    revised_approach: str = Field(
        default="",
        description="How the next attempt should differ (joins, filters, aggregations, etc.)",
    )
    give_up_message: str | None = Field(
        default=None,
        description="User-facing message when giving up",
    )


class ReflectionModel(BaseModel):
    """Second-pass critique of SQL/results/insights before finalize."""

    verdict: Literal[
        "accept",
        "retry_sql",
        "improve_insights",
        "replan",
        "clarify",
    ]
    reasoning: str
    issues: list[str] = Field(default_factory=list)
    improvement_hint: str = Field(
        default="",
        description="Actionable guidance for the next specialist pass",
    )
    clarification_question: str | None = None


class ChartRecommendationModel(BaseModel):
    """AI chart recommendation for Plotly rendering."""

    chart_type: Literal[
        "bar", "line", "pie", "scatter", "histogram", "area", "none", "table"
    ]
    x_axis: str | None = None
    y_axis: str | None = None
    color: str | None = None
    aggregation: str | None = Field(
        default=None,
        description="e.g. sum, avg, count — informational",
    )
    title: str = "Query Results"
    legend: bool = True
    rationale: str = ""


class InsightModel(BaseModel):
    """Business insights over query results."""

    summary: str = Field(description="2-4 sentence executive summary")
    bullets: list[str] = Field(default_factory=list, min_length=0)
    trends: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DatabaseSummaryModel(BaseModel):
    """AI executive summary of a connected database."""

    business_overview: str
    database_purpose: str
    important_tables: list[str] = Field(default_factory=list)
    relationships_summary: str = ""
    potential_kpis: list[str] = Field(default_factory=list)
    business_metrics: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(
        default_factory=list,
        description="High-value analytics questions for this schema",
    )


class SuggestedQuestionsModel(BaseModel):
    """Dynamically generated analytics questions."""

    questions: list[str] = Field(min_length=3, max_length=12)
    themes: list[str] = Field(default_factory=list)


class KPIRecommendation(BaseModel):
    label: str
    description: str
    suggested_sql: str | None = None
    priority: Literal["high", "medium", "low"] = "medium"


class ChartSpecLite(BaseModel):
    title: str
    chart_type: str
    description: str


class DashboardModel(BaseModel):
    """AI-designed KPI dashboard blueprint."""

    title: str
    kpis: list[KPIRecommendation] = Field(default_factory=list)
    charts: list[ChartSpecLite] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    narrative: str = ""


class OptimizationAdviceModel(BaseModel):
    """AI SQL optimization advice from EXPLAIN plans."""

    summary: str
    index_suggestions: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    optimized_sql: str | None = None
    estimated_impact: str = ""


class ConversationSummaryModel(BaseModel):
    """Compressed memory of a long conversation."""

    summary: str
    key_entities: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    last_sql: str | None = None
    last_findings: str | None = None
