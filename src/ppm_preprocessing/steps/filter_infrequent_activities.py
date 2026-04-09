from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterInfrequentActivitiesConfig:
    # Remove activity labels that appear in fewer than this fraction of traces
    min_trace_freq: float = 0.005  # 0.5% of traces
    activity_col: str = "activity"
    case_col: str = "case_id"
    report_filename: str = "05c_filter_infrequent_activities_qc.json"


class FilterInfrequentActivitiesStep(Step):
    """
    Removes events whose activity label appears in fewer than
    `min_trace_freq` fraction of all traces.

    Rationale (Fani Sani et al. 2020): infrequent activities are often
    noise or outliers and degrade sequence models. Removing them reduces
    vocabulary size while retaining the dominant process behaviour.

    Side effects:
      - updates ctx.log.df (rows with rare activities are dropped)
      - writes QC to ctx.artifacts["filter_infrequent_activities_qc"]
    """
    name = "filter_infrequent_activities"

    def __init__(self, config: FilterInfrequentActivitiesConfig | None = None):
        self.config = config or FilterInfrequentActivitiesConfig()

    def _report_dir(self, ctx: PipelineContext) -> Path:
        base = ctx.artifacts.get("out_dir") or getattr(ctx, "output_dir", None)
        if base:
            return Path(base) / "reports"
        return Path("outputs") / "reports"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None — run NormalizeSchemaStep first.")

        c = self.config
        df = ctx.log.df

        n_traces = int(df[c.case_col].nunique())
        before_rows = int(len(df))
        before_activities = int(df[c.activity_col].nunique())

        # Count how many distinct traces each activity appears in
        trace_counts = (
            df.groupby(c.activity_col)[c.case_col]
            .nunique()
            .reset_index()
            .rename(columns={c.case_col: "trace_count"})
        )
        trace_counts["trace_freq"] = trace_counts["trace_count"] / n_traces if n_traces else 0.0

        min_count = max(1, int(c.min_trace_freq * n_traces))
        rare_activities: List[str] = list(
            trace_counts.loc[trace_counts["trace_count"] < min_count, c.activity_col]
        )

        if rare_activities:
            df = df[~df[c.activity_col].isin(rare_activities)].reset_index(drop=True)

        after_rows = int(len(df))
        after_activities = int(df[c.activity_col].nunique())

        qc: Dict[str, Any] = {
            "step": self.name,
            "min_trace_freq": c.min_trace_freq,
            "min_trace_count_threshold": min_count,
            "n_traces": n_traces,
            "before_rows": before_rows,
            "before_activities": before_activities,
            "rare_activities_removed": len(rare_activities),
            "rare_activity_names": rare_activities[:20],  # cap for readability
            "after_rows": after_rows,
            "after_activities": after_activities,
            "rows_dropped": before_rows - after_rows,
        }

        ctx.log.df = df
        ctx.artifacts["filter_infrequent_activities_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
