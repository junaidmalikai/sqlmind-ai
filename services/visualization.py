"""Plotly rendering driven by AI ChartRecommendationModel (no heuristic chart picker)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models.structured import ChartRecommendationModel


def dataframe_profile(df: pd.DataFrame, sample_rows: int = 20) -> dict[str, Any]:
    """Build a compact data profile for the visualization LLM."""
    if df is None or df.empty:
        return {
            "columns": [],
            "dtypes": {},
            "statistics": {},
            "sample": [],
            "row_count": 0,
        }
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().to_dict() if not numeric.empty else {}
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "statistics": stats,
        "sample": json_safe_records(df.head(sample_rows)),
        "row_count": len(df),
    }


def json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    import json

    return json.loads(json.dumps(df.to_dict(orient="records"), default=str))


def build_figure_from_recommendation(
    df: pd.DataFrame,
    rec: ChartRecommendationModel | dict[str, Any] | None,
) -> go.Figure | None:
    """Render a Plotly figure from an AI chart recommendation."""
    if df is None or df.empty or rec is None:
        return None

    if isinstance(rec, dict):
        try:
            rec = ChartRecommendationModel.model_validate(rec)
        except Exception:  # noqa: BLE001
            return None

    if rec.chart_type in {"none", "table"}:
        return None

    data = df.copy()
    x = rec.x_axis if rec.x_axis in data.columns else None
    y = rec.y_axis if rec.y_axis in data.columns else None
    color = rec.color if rec.color in data.columns else None
    title = rec.title or "Query Results"
    template = "plotly_white"

    # Coerce likely datetime x
    if x and not pd.api.types.is_datetime64_any_dtype(data[x]):
        maybe = pd.to_datetime(data[x], errors="coerce", format="mixed")
        if maybe.notna().mean() >= 0.7:
            data[x] = maybe

    try:
        if rec.chart_type == "bar" and x and y:
            fig = px.bar(data, x=x, y=y, color=color, title=title, template=template)
        elif rec.chart_type == "line" and x and y:
            fig = px.line(
                data, x=x, y=y, color=color, title=title, template=template, markers=True
            )
        elif rec.chart_type == "area" and x and y:
            fig = px.area(data, x=x, y=y, color=color, title=title, template=template)
        elif rec.chart_type == "pie" and x and y:
            fig = px.pie(data, names=x, values=y, title=title, template=template)
        elif rec.chart_type == "scatter":
            x_col = x or data.columns[0]
            y_col = y or (data.columns[1] if len(data.columns) > 1 else data.columns[0])
            fig = px.scatter(
                data, x=x_col, y=y_col, color=color, title=title, template=template
            )
        elif rec.chart_type == "histogram":
            col = y or x or data.columns[0]
            fig = px.histogram(data, x=col, color=color, title=title, template=template)
        else:
            return None

        fig.update_layout(
            margin=dict(l=24, r=24, t=48, b=24),
            showlegend=rec.legend,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#0F172A"),
            colorway=["#0D9488", "#0284C7", "#14B8A6", "#0F766E", "#38BDF8", "#2DD4BF"],
        )
        return fig
    except Exception:  # noqa: BLE001
        return None


def build_figure(df: pd.DataFrame, spec: dict[str, Any] | None = None) -> go.Figure | None:
    """Render from AI spec dict (preferred) or fall back to table-only."""
    if spec:
        return build_figure_from_recommendation(df, spec)
    return None
