"""Shared chart selection from SQL result DataFrames (no AI text / no dummy data)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

ChartKind = Literal["bar", "hbar", "pie", "line"]


@dataclass
class ChartSpec:
    kind: ChartKind
    title: str
    label_col: str
    value_col: str
    data: pd.DataFrame  # two columns: label, value (already aggregated / limited)


def _is_datetime_like(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    n = str(name).lower()
    if any(k in n for k in ("date", "time", "month", "year", "day", "week")):
        parsed = pd.to_datetime(series, errors="coerce")
        return bool(parsed.notna().mean() >= 0.7)
    return False


def _cat_cols(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not _is_datetime_like(s, c):
            continue
        if _is_datetime_like(s, c):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        out.append(str(c))
    return out


def _num_cols(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_datetime_like(df[c], c)
    ]


def _date_cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if _is_datetime_like(df[c], c)]


def _grouped(df: pd.DataFrame, label: str, value: str, limit: int = 12) -> pd.DataFrame:
    work = df[[label, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna(subset=[value])
    if work.empty:
        return work
    if work[label].dtype == object or not pd.api.types.is_numeric_dtype(work[label]):
        grouped = (
            work.groupby(label, dropna=False)[value]
            .sum()
            .reset_index()
            .sort_values(value, ascending=False)
            .head(limit)
        )
    else:
        grouped = work.head(limit)
    grouped.columns = ["label", "value"]
    grouped["label"] = grouped["label"].astype(str)
    return grouped


def build_chart_specs(df: pd.DataFrame | None, *, max_charts: int = 4) -> list[ChartSpec]:
    """Derive bar / hbar / pie / line chart specs purely from SQL result values."""
    if df is None or df.empty or len(df.columns) < 1:
        return []

    data = df.head(500).copy()
    cats = _cat_cols(data)
    nums = _num_cols(data)
    dates = _date_cols(data)
    specs: list[ChartSpec] = []

    if cats and nums:
        label, value = cats[0], nums[0]
        grouped = _grouped(data, label, value, limit=12)
        if not grouped.empty and grouped["value"].sum() != 0:
            specs.append(
                ChartSpec("bar", f"{value} by {label}", label, value, grouped.copy())
            )
            specs.append(
                ChartSpec(
                    "hbar",
                    f"{value} ranking",
                    label,
                    value,
                    grouped.sort_values("value", ascending=True).copy(),
                )
            )
            pie_src = grouped.head(10)
            if 1 < len(pie_src) <= 10:
                specs.append(
                    ChartSpec("pie", f"Share of {value}", label, value, pie_src.copy())
                )

    if dates and nums:
        dcol, y = dates[0], nums[0]
        series = data[[dcol, y]].copy()
        series[dcol] = pd.to_datetime(series[dcol], errors="coerce")
        series[y] = pd.to_numeric(series[y], errors="coerce")
        series = series.dropna().sort_values(dcol)
        if len(series) >= 2:
            line_df = pd.DataFrame(
                {
                    "label": series[dcol].dt.strftime("%Y-%m-%d"),
                    "value": series[y].astype(float),
                }
            ).head(60)
            specs.append(ChartSpec("line", f"{y} over time", dcol, y, line_df))
    elif nums and not specs:
        y = nums[0]
        line_df = pd.DataFrame(
            {
                "label": [str(i) for i in range(len(data.head(40)))],
                "value": pd.to_numeric(data[y].head(40), errors="coerce").fillna(0),
            }
        )
        if line_df["value"].abs().sum() > 0:
            specs.append(ChartSpec("line", f"{y} sequence", "index", y, line_df))

    # Deduplicate by kind, preserve order
    seen: set[str] = set()
    unique: list[ChartSpec] = []
    for spec in specs:
        if spec.kind in seen:
            continue
        seen.add(spec.kind)
        unique.append(spec)
        if len(unique) >= max_charts:
            break
    return unique


def build_plotly_figures(df: pd.DataFrame | None) -> list[tuple[str, Any]]:
    """Plotly figures from SQL results for in-app display."""
    import plotly.express as px
    import plotly.graph_objects as go

    specs = build_chart_specs(df)
    figs: list[tuple[str, Any]] = []
    layout = dict(
        margin=dict(l=24, r=24, t=48, b=24),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#0F172A"),
    )
    teal = ["#0D9488", "#0284C7", "#14B8A6", "#0F766E", "#38BDF8", "#2DD4BF"]
    for spec in specs:
        try:
            if spec.kind == "bar":
                fig = px.bar(
                    spec.data,
                    x="label",
                    y="value",
                    title=spec.title,
                    template="plotly_white",
                    color_discrete_sequence=[teal[0]],
                )
            elif spec.kind == "hbar":
                fig = px.bar(
                    spec.data,
                    x="value",
                    y="label",
                    orientation="h",
                    title=spec.title,
                    template="plotly_white",
                    color_discrete_sequence=[teal[2]],
                )
            elif spec.kind == "pie":
                fig = px.pie(
                    spec.data,
                    names="label",
                    values="value",
                    title=spec.title,
                    template="plotly_white",
                    color_discrete_sequence=teal,
                )
            else:
                fig = px.line(
                    spec.data,
                    x="label",
                    y="value",
                    markers=True,
                    title=spec.title,
                    template="plotly_white",
                    color_discrete_sequence=[teal[1]],
                )
            fig.update_layout(**layout)
            figs.append((spec.title, fig))
        except Exception:  # noqa: BLE001
            continue
    return figs


def draw_reportlab_chart(spec: ChartSpec, width: float = 460, height: float = 220) -> Any:
    """Native ReportLab Drawing chart (vector) from SQL values — not a screenshot."""
    from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors

    drawing = Drawing(width, height)
    labels = [str(x)[:18] for x in spec.data["label"].tolist()]
    values = [float(v) for v in spec.data["value"].tolist()]
    if not values:
        return drawing

    teal = colors.HexColor("#0D9488")
    ink = colors.HexColor("#0F172A")
    drawing.add(String(8, height - 14, spec.title[:60], fontName="Helvetica-Bold", fontSize=9, fillColor=ink))

    if spec.kind == "pie":
        pie = Pie()
        pie.x = width * 0.28
        pie.y = 18
        pie.width = height - 50
        pie.height = height - 50
        pie.data = values
        pie.labels = labels
        pie.slices.strokeWidth = 0.5
        for i in range(len(values)):
            pie.slices[i].fillColor = colors.Color(
                0.05 + (i * 0.08) % 0.5,
                0.45 + (i * 0.07) % 0.4,
                0.45 + (i * 0.05) % 0.3,
            )
        drawing.add(pie)
        return drawing

    if spec.kind == "line":
        lp = LinePlot()
        lp.x = 45
        lp.y = 30
        lp.height = height - 60
        lp.width = width - 70
        lp.data = [[(i, v) for i, v in enumerate(values)]]
        lp.lines[0].strokeColor = teal
        lp.lines[0].strokeWidth = 2
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = max(len(values) - 1, 1)
        lp.yValueAxis.valueMin = min(0, min(values))
        lp.yValueAxis.valueMax = max(values) * 1.1 if max(values) else 1
        drawing.add(lp)
        return drawing

    if spec.kind == "hbar":
        bc = HorizontalBarChart()
        bc.x = 90
        bc.y = 25
        bc.height = height - 55
        bc.width = width - 120
        bc.data = [values]
        bc.categoryAxis.categoryNames = labels
        bc.bars[0].fillColor = teal
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(values) * 1.1 if max(values) else 1
        drawing.add(bc)
        return drawing

    bc = VerticalBarChart()
    bc.x = 40
    bc.y = 40
    bc.height = height - 70
    bc.width = width - 60
    bc.data = [values]
    bc.categoryAxis.categoryNames = labels
    bc.bars[0].fillColor = teal
    bc.categoryAxis.labels.angle = 35
    bc.categoryAxis.labels.boxAnchor = "ne"
    bc.categoryAxis.labels.dx = 2
    bc.categoryAxis.labels.dy = -2
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(values) * 1.1 if max(values) else 1
    drawing.add(bc)
    return drawing
