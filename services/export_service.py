"""Export facade — delegates to specialized exporters (SRP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from services.exporters import (
    CSVExporter,
    ExcelReportExporter,
    ExportManager,
    JSONExporter,
    MarkdownExporter,
    PDFReportExporter,
)
from utils.helpers import ensure_dirs
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ExportService:
    """Facade over ExportManager + specialized exporters."""

    def __init__(self, export_dir: str = "exports") -> None:
        self.export_dir = Path(export_dir)
        ensure_dirs(self.export_dir)
        self._manager = ExportManager(self.export_dir)
        self.excel_exporter = ExcelReportExporter(self.export_dir)
        self.pdf_exporter = PDFReportExporter(self.export_dir)
        self.csv_exporter = CSVExporter(self.export_dir)
        self.json_exporter = JSONExporter(self.export_dir)
        self.markdown_exporter = MarkdownExporter(self.export_dir)

    def to_csv(self, df: pd.DataFrame, label: str) -> str:
        return self.csv_exporter.run(df, label)

    def to_json(self, df: pd.DataFrame, label: str) -> str:
        return self.json_exporter.run(df, label)

    def to_markdown(self, df: pd.DataFrame, label: str) -> str:
        return self.markdown_exporter.run(df, label)

    def to_excel(self, df: pd.DataFrame, label: str) -> str:
        return self.to_excel_report(df, label)

    def to_excel_report(
        self,
        df: pd.DataFrame,
        label: str,
        *,
        question: str = "",
        sql: str = "",
        insights: str = "",
        meta: dict[str, Any] | None = None,
        fig: Any | None = None,
        recommendations: str = "",
    ) -> str:
        return self.excel_exporter.run(
            df,
            label,
            question=question,
            sql=sql,
            insights=insights,
            recommendations=recommendations,
            meta=meta,
        )

    def build_pdf_report(
        self,
        *,
        title: str,
        question: str,
        sql: str,
        insights: str,
        df: pd.DataFrame,
        meta: dict[str, Any] | None = None,
        fig: Any | None = None,
        recommendations: str = "",
    ) -> str:
        return self.pdf_exporter.run(
            df,
            title,
            question=question,
            sql=sql,
            insights=insights,
            recommendations=recommendations,
            meta=meta,
        )

    def export_all(
        self,
        df: pd.DataFrame,
        label: str,
        *,
        question: str = "",
        sql: str = "",
        insights: str = "",
        include_pdf: bool = True,
        recommendations: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        return self._manager.run(
            df,
            label,
            question=question,
            sql=sql,
            insights=insights,
            recommendations=recommendations,
            meta=meta or {"rows": len(df) if df is not None else 0},
            include_pdf=include_pdf,
        )
